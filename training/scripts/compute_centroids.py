"""Precompute K-Means centroids for profile-specific initialization.

This script trains K-Means on the dataset in LAB color space (perceptually uniform)
and stores centroids for each deficiency type. These centroids are used to:
1. Initialize K-Means faster at inference time
2. Bias clustering toward colors critical for each deficiency type
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import joblib
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color as skcolor
from tqdm import tqdm


# Deficiency-specific color priorities
DEFICIENCY_COLOR_PRIORITIES = {
    "protan": ["red", "orange", "green"],  # Red-green confusion
    "deutan": ["green", "red", "yellow"],  # Green-red confusion
    "tritan": ["blue", "yellow", "cyan"],  # Blue-yellow confusion
    "normal": ["red", "green", "blue", "yellow", "cyan", "orange", "indigo", "violet"],
}

N_CLUSTERS = 9  # Match existing pipeline configuration
RANDOM_STATE = 42


def load_dataset_pixels(dataset_dir: Path, color_families: List[str]) -> np.ndarray:
    """Load all images from specified color families and convert to LAB."""
    pixels_rgb = []
    
    print(f"📂 Loading dataset from {dataset_dir}...")
    for family in color_families:
        folder = dataset_dir / family
        if not folder.exists():
            print(f"⚠️  Warning: {family} folder not found, skipping...")
            continue
        
        images = list(folder.glob("*.png"))
        print(f"  Loading {len(images)} images from {family}...")
        
        for img_path in tqdm(images, desc=f"  {family}", ncols=80):
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img) / 255.0  # Normalize to [0, 1]
            pixels_rgb.append(img_np.reshape(-1, 3))
    
    # Stack all pixels and convert to LAB
    pixels_rgb = np.vstack(pixels_rgb)
    print(f"  Converting {len(pixels_rgb)} pixels to LAB color space...")
    
    # Convert to LAB (perceptually uniform)
    pixels_lab = skcolor.rgb2lab(pixels_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    
    return pixels_lab


def compute_centroids_for_deficiency(
    pixels_lab: np.ndarray,
    deficiency: str,
    n_clusters: int = N_CLUSTERS
) -> Tuple[np.ndarray, np.ndarray]:
    """Train K-Means in LAB space and return centroids in both LAB and RGB."""
    print(f"\n🔬 Computing centroids for {deficiency}...")
    
    # Train K-Means in LAB space
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(pixels_lab)
    
    centroids_lab = kmeans.cluster_centers_
    
    # Convert centroids back to RGB for visualization/storage
    centroids_rgb = skcolor.lab2rgb(centroids_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    
    print(f"  ✓ Computed {n_clusters} centroids")
    print(f"    LAB range: L=[{centroids_lab[:, 0].min():.1f}, {centroids_lab[:, 0].max():.1f}], "
          f"a=[{centroids_lab[:, 1].min():.1f}, {centroids_lab[:, 1].max():.1f}], "
          f"b=[{centroids_lab[:, 2].min():.1f}, {centroids_lab[:, 2].max():.1f}]")
    
    return centroids_lab, centroids_rgb


def save_centroids(
    output_dir: Path,
    deficiency: str,
    centroids_lab: np.ndarray,
    centroids_rgb: np.ndarray
):
    """Save centroids in multiple formats for flexibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays (for fast loading)
    np.save(output_dir / f"{deficiency}_centroids_lab.npy", centroids_lab)
    np.save(output_dir / f"{deficiency}_centroids_rgb.npy", centroids_rgb)
    
    # Save as JSON (for Firebase/human-readable storage)
    centroid_data = {
        "deficiency": deficiency,
        "n_clusters": len(centroids_lab),
        "centroids_lab": centroids_lab.tolist(),
        "centroids_rgb": centroids_rgb.tolist(),
    }
    
    with open(output_dir / f"{deficiency}_centroids.json", "w") as f:
        json.dump(centroid_data, f, indent=2)
    
    # Save scikit-learn model (for backward compatibility)
    kmeans = KMeans(n_clusters=len(centroids_lab), init=centroids_lab, n_init=1)
    kmeans.cluster_centers_ = centroids_lab
    kmeans._n_features_out = 3
    joblib.dump(kmeans, output_dir / f"{deficiency}_kmeans_model.pkl")
    
    print(f"  💾 Saved centroids to {output_dir}/{deficiency}_centroids.*")


def main():
    """Compute centroids for all deficiency types."""
    # Paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / "datasets" / "color_varied"
    output_dir = script_dir.parent / "models" / "centroids"
    
    if not dataset_dir.exists():
        print(f"❌ Dataset not found at {dataset_dir}")
        print("   Run generate_dataset.py first!")
        return
    
    print("=" * 80)
    print("🎯 K-Means Centroid Precomputation for Adaptive Color Correction")
    print("=" * 80)
    
    # Compute centroids for each deficiency type
    for deficiency, priority_families in DEFICIENCY_COLOR_PRIORITIES.items():
        print(f"\n{'─' * 80}")
        print(f"Processing {deficiency.upper()} deficiency")
        print(f"Priority colors: {', '.join(priority_families)}")
        print(f"{'─' * 80}")
        
        # Load dataset pixels (filtered by priority)
        pixels_lab = load_dataset_pixels(dataset_dir, priority_families)
        
        # Compute centroids
        centroids_lab, centroids_rgb = compute_centroids_for_deficiency(
            pixels_lab, deficiency
        )
        
        # Save in multiple formats
        save_centroids(output_dir, deficiency, centroids_lab, centroids_rgb)
    
    print("\n" + "=" * 80)
    print("✅ Centroid computation complete!")
    print(f"   Models saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
