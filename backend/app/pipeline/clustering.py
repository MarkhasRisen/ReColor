"""Adaptive pixel clustering utilities for the color correction pipeline.

Uses LAB color space for perceptually uniform clustering, critical for 
accurate color discrimination in color blindness correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from skimage import color


@dataclass
class KMeansSegmenter:
    """Wrap sklearn K-Means with LAB color space and profile-aware initialization.
    
    Uses LAB color space (perceptually uniform) instead of RGB for clustering,
    which provides better color discrimination for daltonization algorithms.
    """

    n_clusters: int
    max_iter: int = 50
    random_state: Optional[int] = None
    inertia_tolerance: float = 1e-2
    use_lab_space: bool = True  # Use LAB instead of RGB
    cached_centroids: Optional[np.ndarray] = field(default=None, init=False)

    def fit_predict(
        self, 
        pixels: np.ndarray, 
        centroid_bias: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster pixels and return assignments alongside centroids.
        
        Args:
            pixels: RGB pixel array shaped (num_pixels, 3) with values in [0, 1]
            centroid_bias: Optional precomputed centroids in LAB space
            
        Returns:
            labels: Cluster assignments for each pixel
            centroids: Cluster centroids in LAB space (if use_lab_space=True)
        """
        if pixels.ndim != 2:
            raise ValueError("pixels must be shaped (num_pixels, num_channels)")

        # Convert RGB to LAB for perceptually uniform clustering
        if self.use_lab_space:
            pixels_lab = color.rgb2lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
        else:
            pixels_lab = pixels

        init = self._resolve_initialization(pixels_lab, centroid_bias)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=init,
            n_init=1,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        labels = kmeans.fit_predict(pixels_lab)
        self.cached_centroids = kmeans.cluster_centers_
        return labels, self.cached_centroids

    def _resolve_initialization(
        self, 
        pixels: np.ndarray, 
        centroid_bias: Optional[np.ndarray]
    ) -> np.ndarray:
        """Determine initialization strategy for K-Means."""
        if centroid_bias is not None:
            if centroid_bias.shape != (self.n_clusters, pixels.shape[1]):
                raise ValueError("centroid_bias has incompatible shape")
            return centroid_bias

        if self.cached_centroids is not None:
            return self.cached_centroids

        idx = np.random.default_rng(self.random_state).choice(
            pixels.shape[0], size=self.n_clusters, replace=False
        )
        return pixels[idx]
