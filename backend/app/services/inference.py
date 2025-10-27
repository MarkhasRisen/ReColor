"""Client for delegating CNN inference to a remote service."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests

from ..pipeline.profile import VisionProfile


@dataclass
class OffloadInferenceClient:
    """HTTP client for remote TensorFlow Lite inference."""

    endpoint: str
    timeout: float = 5.0

    def infer(self, frame: np.ndarray, profile: VisionProfile) -> Optional[np.ndarray]:
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("frame must be shaped (H, W, 3)")

        payload = {
            "profile": {
                "deficiency": profile.deficiency,
                "severity": profile.severity,
                "confidence": profile.confidence,
            },
            "image_base64": self._encode_frame(frame),
        }
        response = requests.post(
            self.endpoint,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return self._decode_frame(data["data"], frame.shape)

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        buffer = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    @staticmethod
    def _decode_frame(encoded: str, shape) -> np.ndarray:
        raw = base64.b64decode(encoded)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        return frame.astype(np.float32) / 255.0
