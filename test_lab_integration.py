"""Quick test to verify LAB color space integration."""
import numpy as np
from backend.app.pipeline.clustering import KMeansSegmenter
from skimage import color

# Create a simple test image with red and green pixels
test_pixels_rgb = np.array([
    [1.0, 0.0, 0.0],  # Pure red
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],  # Pure green
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],  # Pure blue
    [0.0, 0.0, 1.0],
])

print("🧪 Testing LAB-based K-Means clustering...")
print(f"Input: {len(test_pixels_rgb)} RGB pixels")

# Initialize segmenter with LAB space
segmenter = KMeansSegmenter(n_clusters=3, use_lab_space=True, random_state=42)

# Run clustering
labels, centroids_lab = segmenter.fit_predict(test_pixels_rgb)

print(f"\n✓ Clustering successful!")
print(f"  Labels: {labels}")
print(f"  Centroids (LAB):")
for i, centroid in enumerate(centroids_lab):
    print(f"    Cluster {i}: L={centroid[0]:.1f}, a={centroid[1]:.1f}, b={centroid[2]:.1f}")

# Convert centroids back to RGB
centroids_rgb = color.lab2rgb(centroids_lab.reshape(-1, 1, 3)).reshape(-1, 3)
print(f"\n  Centroids (RGB):")
for i, centroid in enumerate(centroids_rgb):
    print(f"    Cluster {i}: R={centroid[0]:.2f}, G={centroid[1]:.2f}, B={centroid[2]:.2f}")

print("\n✅ LAB integration test passed!")
