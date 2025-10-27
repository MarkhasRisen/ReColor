"""High-level orchestration of the adaptive color correction pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from .clustering import KMeansSegmenter
from .daltonization import Daltonizer, blend
from .cnn_inference import TFLiteColorCorrector
from .profile import VisionProfile
from ..services.inference import OffloadInferenceClient
from ..utils.resource import ResourceMonitorProtocol
from ..utils.centroid_loader import get_centroid_bias_for_profile


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    n_clusters: int = 9  # Match precomputed centroids (9 clusters)
    blend_alpha: float = 0.6
    apply_cnn: bool = True
    merge_weight: float = 0.5
    offload_enabled: bool = True
    resource_monitor: Optional[Union[Callable[[], bool], ResourceMonitorProtocol]] = None
    offload_endpoint: Optional[str] = None
    offload_timeout: float = 5.0


@dataclass
class AdaptiveColorPipeline:
    """Compose clustering, daltonization, and optional CNN transforms."""

    config: PipelineConfig
    model_dir: Path
    profile: VisionProfile
    segmenter: Optional[KMeansSegmenter] = None
    corrector: Optional[TFLiteColorCorrector] = None
    offload_client: Optional[OffloadInferenceClient] = None

    def __post_init__(self) -> None:
        # Initialize K-Means with LAB color space
        self.segmenter = self.segmenter or KMeansSegmenter(
            n_clusters=self.config.n_clusters,
            use_lab_space=True  # Use LAB for perceptually uniform clustering
        )
        
        # Load precomputed centroids for profile-aware initialization
        self._centroid_bias = get_centroid_bias_for_profile(
            self.profile.deficiency,
            self.profile.severity
        )
        
        if self.profile.deficiency != "normal" and self.config.apply_cnn:
            model_path = self._resolve_model_path()
            if model_path:
                self.corrector = TFLiteColorCorrector(model_path=model_path)
        if self.config.offload_enabled and self.config.offload_endpoint and not self.offload_client:
            self.offload_client = OffloadInferenceClient(
                endpoint=self.config.offload_endpoint,
                timeout=self.config.offload_timeout,
            )

    def run(self, frame: np.ndarray) -> np.ndarray:
        """Execute the adaptive pipeline on an RGB frame (H, W, 3)."""
        normalized = self._normalize(frame)

        clustered_frame = self._run_kmeans(normalized)
        cnn_frame = self._run_cnn(normalized)
        merged = self._merge_paths(clustered_frame, cnn_frame)

        daltonized = self._apply_daltonization(merged)
        if self.profile.deficiency == "normal" or self.profile.severity == 0.0:
            output = normalized
        else:
            output = blend(normalized, daltonized, self.config.blend_alpha)
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)

    def _resolve_model_path(self) -> Optional[Path]:
        candidate = self.model_dir / f"{self.profile.deficiency}_v1.tflite"
        return candidate if candidate.exists() else None

    @staticmethod
    def _normalize(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame.astype(np.float32) / 255.0
        if frame.dtype == np.float32:
            return np.clip(frame, 0.0, 1.0)
        raise ValueError("Unsupported frame dtype")

    def _run_kmeans(self, normalized: np.ndarray) -> np.ndarray:
        """Run K-Means clustering with profile-specific centroid initialization."""
        flattened = normalized.reshape(-1, 3)
        labels, centroids = self.segmenter.fit_predict(
            flattened,
            centroid_bias=self._centroid_bias
        )
        
        # Convert centroids back to RGB if using LAB space
        if self.segmenter.use_lab_space:
            from skimage import color
            centroids_rgb = color.lab2rgb(centroids.reshape(-1, 1, 3)).reshape(-1, 3)
            frame = centroids_rgb[labels].reshape(normalized.shape)
        else:
            frame = centroids[labels].reshape(normalized.shape)
        
        return frame

    def _run_cnn(self, normalized: np.ndarray) -> Optional[np.ndarray]:
        if self._should_offload():
            return self._offload_inference(normalized)
        if not self.corrector:
            return None
        result = self.corrector.run(normalized[np.newaxis, ...])[0]
        return np.clip(result, 0.0, 1.0)

    def _merge_paths(self, clustered: np.ndarray, cnn_frame: Optional[np.ndarray]) -> np.ndarray:
        if cnn_frame is None:
            return clustered
        weight = np.clip(self.config.merge_weight, 0.0, 1.0)
        return np.clip((1 - weight) * clustered + weight * cnn_frame, 0.0, 1.0)

    def _apply_daltonization(self, frame: np.ndarray) -> np.ndarray:
        if self.profile.deficiency == "normal" or self.profile.severity == 0.0:
            return frame
        daltonizer = Daltonizer(
            deficiency=self.profile.deficiency,
            severity=self.profile.severity,
        )
        flattened = frame.reshape(-1, 3)
        corrected = daltonizer.apply(flattened)
        return corrected.reshape(frame.shape)

    def _should_offload(self) -> bool:
        if not self.config.offload_enabled:
            return False
        monitor = self.config.resource_monitor
        if monitor is None:
            return False
        if hasattr(monitor, "should_offload"):
            return bool(monitor.should_offload())  # type: ignore[attr-defined]
        if callable(monitor):
            return bool(monitor())
        return False

    def _offload_inference(self, normalized: np.ndarray) -> Optional[np.ndarray]:
        if not self.offload_client:
            return None
        try:
            result = self.offload_client.infer(normalized, self.profile)
            return result
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Remote inference failed: %s", exc)
        return None
