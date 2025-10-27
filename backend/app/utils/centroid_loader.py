"""Utilities for loading and managing precomputed K-Means centroids."""
from pathlib import Path
from typing import Optional

import numpy as np


def load_centroids(
    deficiency: str,
    centroids_dir: Optional[Path] = None
) -> Optional[np.ndarray]:
    """Load precomputed LAB centroids for a deficiency type.
    
    Args:
        deficiency: One of 'protan', 'deutan', 'tritan', 'normal'
        centroids_dir: Directory containing centroid files (defaults to training/models/centroids)
        
    Returns:
        LAB centroids array (n_clusters, 3) or None if not found
    """
    if centroids_dir is None:
        centroids_dir = Path(__file__).parent.parent.parent.parent / "training" / "models" / "centroids"
    
    centroid_file = centroids_dir / f"{deficiency}_centroids_lab.npy"
    
    if not centroid_file.exists():
        return None
    
    try:
        centroids = np.load(centroid_file)
        return centroids
    except Exception as e:
        print(f"⚠️  Failed to load centroids from {centroid_file}: {e}")
        return None


def get_centroid_bias_for_profile(deficiency: str, severity: float) -> Optional[np.ndarray]:
    """Get centroid bias weighted by severity.
    
    For low severity, returns None (use random initialization).
    For higher severity, returns precomputed centroids to bias clustering.
    
    Args:
        deficiency: Deficiency type
        severity: Severity score in [0, 1]
        
    Returns:
        LAB centroids or None
    """
    # Only use bias for moderate to severe deficiencies
    if severity < 0.3:
        return None
    
    return load_centroids(deficiency)
