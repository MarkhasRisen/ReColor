import numpy as np

from backend.app.pipeline.processing import AdaptiveColorPipeline, PipelineConfig
from backend.app.pipeline.profile import VisionProfile


class _AlwaysOffloadMonitor:
    def should_offload(self) -> bool:
        return True


class _MockOffloadClient:
    def __init__(self, value: float) -> None:
        self.value = value
        self.called = False

    def infer(self, frame: np.ndarray, profile: VisionProfile) -> np.ndarray:
        self.called = True
        return np.full_like(frame, self.value)


def test_pipeline_runs_without_cnn(tmp_path):
    profile = VisionProfile(deficiency="normal", severity=0.0, confidence=1.0)
    pipeline = AdaptiveColorPipeline(
        config=PipelineConfig(apply_cnn=False),
        model_dir=tmp_path,
        profile=profile,
    )

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = pipeline.run(frame)

    assert result.shape == frame.shape


def test_pipeline_offloads_when_monitor_triggers(tmp_path):
    profile = VisionProfile(deficiency="deutan", severity=0.4, confidence=0.9)
    monitor = _AlwaysOffloadMonitor()
    offload = _MockOffloadClient(0.5)

    pipeline = AdaptiveColorPipeline(
        config=PipelineConfig(
            apply_cnn=True,
            merge_weight=1.0,
            offload_enabled=True,
            resource_monitor=monitor,
        ),
        model_dir=tmp_path,
        profile=profile,
        offload_client=offload,
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    result = pipeline.run(frame)

    assert offload.called is True
    assert result.shape == frame.shape
