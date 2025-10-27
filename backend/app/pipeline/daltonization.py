"""Daltonization utilities for remapping colors along confusion lines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

ConfusionType = Literal["protan", "deutan", "tritan"]

# Matrices adapted from standard Brettel daltonization references.
_SIMULATION_MATRICES = {
    "protan": np.array(
        [[0.0, 2.02344, -2.52581], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),
    "deutan": np.array(
        [[1.0, 0.0, 0.0], [0.494207, 0.0, 1.24827], [0.0, 0.0, 1.0]], dtype=np.float32
    ),
    "tritan": np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.395913, 0.801109, 0.0]], dtype=np.float32
    ),
}

_CORRECTION_MATRICES = {
    "protan": np.array(
        [[0.0, 0.7, 0.7], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),
    "deutan": np.array(
        [[1.0, 0.0, 0.0], [0.7, 0.0, 0.7], [0.0, 0.0, 1.0]], dtype=np.float32
    ),
    "tritan": np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]], dtype=np.float32
    ),
}


@dataclass
class Daltonizer:
    """Apply daltonization given a user profile severity."""

    deficiency: ConfusionType
    severity: float

    def apply(self, rgb_pixels: np.ndarray) -> np.ndarray:
        """Apply daltonization adjustments to an array of RGB pixels."""
        if rgb_pixels.ndim != 2 or rgb_pixels.shape[1] != 3:
            raise ValueError("rgb_pixels must be shaped (n, 3)")

        sim = rgb_pixels @ _SIMULATION_MATRICES[self.deficiency].T
        error = rgb_pixels - sim
        correction = rgb_pixels + self.severity * (error @ _CORRECTION_MATRICES[self.deficiency].T)
        return np.clip(correction, 0.0, 1.0)

    def adjust_centroids(self, centroids: np.ndarray) -> np.ndarray:
        """Return corrected centroids to minimize recomputation downstream."""
        return self.apply(centroids)


def blend(original: np.ndarray, corrected: np.ndarray, alpha: float) -> np.ndarray:
    """Blend original and corrected pixels to avoid overcorrection artifacts."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be within [0, 1]")
    return np.clip((1 - alpha) * original + alpha * corrected, 0.0, 1.0)
