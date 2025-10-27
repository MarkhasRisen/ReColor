"""TensorFlow Lite inference helper for adaptive color transforms."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TFLiteColorCorrector:
    """Run CNN-based color correction via TensorFlow Lite."""

    model_path: Path
    num_threads: Optional[int] = None

    def __post_init__(self) -> None:
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency at runtime
            raise RuntimeError("TensorFlow is required for TFLite inference") from exc

        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        if self.num_threads is not None:
            self.interpreter._interpreter.SetNumThreads(self.num_threads)  # pylint: disable=protected-access
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run(self, batch: np.ndarray) -> np.ndarray:
        """Execute inference on a normalized RGB batch shaped (n, h, w, 3)."""
        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)
        if batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError("batch must be shaped (n, h, w, 3)")

        self.interpreter.set_tensor(self.input_details[0]["index"], batch)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])
