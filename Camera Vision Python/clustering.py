#!/usr/bin/env python3
"""
Clustering Module for ReColor
============================
Implements MiniBatch K-Means clustering for efficient color segmentation and palette extraction.
Optimized for real-time mobile processing with minimal memory footprint.

Author: ReColor Development Team
Date: October 2025
License: MIT
"""

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle
from typing import Tuple, Dict, Optional, List
import time
import threading
from concurrent.futures import ThreadPoolExecutor

class RealtimeColorClusterer:
    """
    Real-time color clustering system using MiniBatch K-Means.
    
    Optimized for mobile processing with adaptive clustering parameters
    and efficient memory management for continuous video processing.
    """
    
    def __init__(self, n_clusters: int = 8, batch_size: int = 1000, 
                 max_iter: int = 10, random_state: int = 42):
        """
        Initialize Real-time Color Clusterer.
        
        Args:
            n_clusters: Number of color clusters (3-16 recommended for CVD)
            batch_size: Batch size for MiniBatch K-Means (larger = more accurate, slower)
            max_iter: Maximum iterations per batch (lower = faster, less accurate)
            random_state: Random seed for reproducible results
        """
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Initialize MiniBatch K-Means
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=random_state,
            n_init=3,  # Reduced for speed
            max_no_improvement=5
        )
        
        # Performance tracking
        self.processing_times = []
        self.cluster_stability_history = []
        self.fitted = False
        
        # Threading for background processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Adaptive parameters
        self.adaptive_mode = True
        self.min_clusters = 3
        self.max_clusters = 16
        
        print(f"🎯 Real-time Color Clusterer initialized (K={n_clusters})")
    
    def preprocess_image(self, image: np.ndarray, resize_factor: float = 0.5) -> np.ndarray:
        """
        Preprocess image for faster clustering.
        
        Args:
            image: Input RGB image
            resize_factor: Factor to resize image (smaller = faster)
            
        Returns:
            Preprocessed image
        """
        if resize_factor != 1.0:
            h, w = image.shape[:2]
            new_h, new_w = int(h * resize_factor), int(w * resize_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def extract_color_samples(self, image: np.ndarray, sample_rate: float = 0.1) -> np.ndarray:
        """
        Extract representative color samples from image for clustering.
        
        Args:
            image: Input RGB image
            sample_rate: Fraction of pixels to sample (0.1 = 10% of pixels)
            
        Returns:
            Array of color samples (N, 3)
        """
        # Reshape image to pixel array
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Random sampling for speed
        n_samples = int(len(pixels) * sample_rate)
        n_samples = max(min(n_samples, self.batch_size * 2), 100)  # Ensure reasonable sample size
        
        if len(pixels) > n_samples:
            # Use sklearn's shuffle for better randomization
            samples = shuffle(pixels, n_samples=n_samples, random_state=self.random_state)
        else:
            samples = pixels
        
        return samples
    
    def adaptive_cluster_count(self, image: np.ndarray) -> int:
        """
        Adaptively determine optimal cluster count based on image characteristics.
        
        Args:
            image: Input RGB image
            
        Returns:
            Optimal number of clusters
        """
        if not self.adaptive_mode:
            return self.n_clusters
        
        # Calculate image complexity metrics
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge density (measure of detail)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color variance (measure of color diversity)
        pixels = image.reshape(-1, 3).astype(np.float32)
        color_std = np.mean(np.std(pixels, axis=0))
        
        # Adaptive cluster count based on complexity
        base_clusters = self.n_clusters
        
        # Adjust based on edge density
        if edge_density > 0.15:  # High detail image
            base_clusters = min(base_clusters + 2, self.max_clusters)
        elif edge_density < 0.05:  # Low detail image  
            base_clusters = max(base_clusters - 2, self.min_clusters)
        
        # Adjust based on color variance
        if color_std > 80:  # High color diversity
            base_clusters = min(base_clusters + 1, self.max_clusters)
        elif color_std < 30:  # Low color diversity
            base_clusters = max(base_clusters - 1, self.min_clusters)
        
        return base_clusters
    
    def fit_incremental(self, image: np.ndarray) -> 'RealtimeColorClusterer':
        """
        Incrementally fit the K-Means model with new image data.
        
        Args:
            image: New RGB image to learn from
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        
        # Preprocess and sample
        processed = self.preprocess_image(image)
        samples = self.extract_color_samples(processed)
        
        # Adaptive clustering
        if self.adaptive_mode:
            optimal_k = self.adaptive_cluster_count(processed)
            if optimal_k != self.kmeans.n_clusters:
                # Reinitialize with new cluster count
                self.kmeans = MiniBatchKMeans(
                    n_clusters=optimal_k,
                    batch_size=self.batch_size,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    n_init=3
                )
                self.n_clusters = optimal_k
        
        # Incremental fitting
        if not self.fitted:
            self.kmeans.fit(samples)
            self.fitted = True
        else:
            self.kmeans.partial_fit(samples)
        
        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 50:
            self.processing_times.pop(0)
        
        return self
    
    def predict_clusters(self, image: np.ndarray, return_centers: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict cluster assignments for image pixels.
        
        Args:
            image: Input RGB image
            return_centers: Whether to return cluster centers
            
        Returns:
            Dictionary with cluster assignments and optionally centers
        """
        if not self.fitted:
            raise ValueError("Clusterer must be fitted before prediction. Call fit_incremental() first.")
        
        start_time = time.time()
        
        # Store original shape
        original_shape = image.shape
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Predict clusters
        cluster_labels = self.kmeans.predict(pixels)
        
        # Reshape back to image dimensions
        cluster_map = cluster_labels.reshape(original_shape[:2])
        
        # Create results dictionary
        results = {
            'cluster_map': cluster_map,
            'n_clusters': self.kmeans.n_clusters
        }
        
        if return_centers:
            results['cluster_centers'] = self.kmeans.cluster_centers_.astype(np.uint8)
            results['cluster_image'] = self._create_cluster_image(cluster_map, results['cluster_centers'])
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return results
    
    def _create_cluster_image(self, cluster_map: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Create image where each pixel is replaced by its cluster center color."""
        height, width = cluster_map.shape
        cluster_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(self.kmeans.n_clusters):
            mask = cluster_map == i
            cluster_image[mask] = centers[i]
        
        return cluster_image
    
    def extract_dominant_colors(self, image: np.ndarray, n_colors: Optional[int] = None) -> List[Tuple[np.ndarray, float]]:
        """
        Extract dominant colors with their frequencies.
        
        Args:
            image: Input RGB image
            n_colors: Number of dominant colors to extract (uses n_clusters if None)
            
        Returns:
            List of (color, frequency) tuples sorted by frequency
        """
        if not self.fitted:
            self.fit_incremental(image)
        
        results = self.predict_clusters(image, return_centers=True)
        cluster_map = results['cluster_map']
        centers = results['cluster_centers']
        
        # Calculate cluster frequencies
        unique, counts = np.unique(cluster_map, return_counts=True)
        total_pixels = cluster_map.size
        
        # Create color-frequency pairs
        color_frequencies = []
        for cluster_id, count in zip(unique, counts):
            color = centers[cluster_id]
            frequency = count / total_pixels
            color_frequencies.append((color, frequency))
        
        # Sort by frequency (most dominant first)
        color_frequencies.sort(key=lambda x: x[1], reverse=True)
        
        # Return requested number of colors
        if n_colors:
            color_frequencies = color_frequencies[:n_colors]
        
        return color_frequencies
    
    def create_color_palette(self, image: np.ndarray, palette_size: Tuple[int, int] = (400, 80)) -> np.ndarray:
        """
        Create visual color palette from dominant colors.
        
        Args:
            image: Input RGB image
            palette_size: (width, height) of palette image
            
        Returns:
            Color palette image
        """
        dominant_colors = self.extract_dominant_colors(image)
        
        palette_width, palette_height = palette_size
        n_colors = len(dominant_colors)
        
        if n_colors == 0:
            return np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
        
        # Create palette
        palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
        color_width = palette_width // n_colors
        
        for i, (color, frequency) in enumerate(dominant_colors):
            start_x = i * color_width
            end_x = start_x + color_width if i < n_colors - 1 else palette_width
            palette[:, start_x:end_x] = color
            
            # Add frequency text
            text = f"{frequency:.1%}"
            cv2.putText(palette, text, (start_x + 5, palette_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return palette
    
    def segment_by_color_similarity(self, image: np.ndarray, target_color: np.ndarray, 
                                  tolerance: float = 30.0) -> np.ndarray:
        """
        Segment image regions similar to target color using clustering.
        
        Args:
            image: Input RGB image
            target_color: Target RGB color (3,) array
            tolerance: Color similarity tolerance
            
        Returns:
            Binary mask of similar color regions
        """
        if not self.fitted:
            self.fit_incremental(image)
        
        results = self.predict_clusters(image, return_centers=True)
        cluster_centers = results['cluster_centers']
        cluster_map = results['cluster_map']
        
        # Find closest cluster to target color
        distances = np.linalg.norm(cluster_centers - target_color, axis=1)
        closest_clusters = np.where(distances <= tolerance)[0]
        
        # Create mask for similar colors
        mask = np.isin(cluster_map, closest_clusters)
        
        return mask.astype(np.uint8) * 255
    
    def analyze_color_distribution(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze color distribution characteristics.
        
        Args:
            image: Input RGB image
            
        Returns:
            Dictionary with distribution metrics
        """
        if not self.fitted:
            self.fit_incremental(image)
        
        results = self.predict_clusters(image)
        cluster_map = results['cluster_map']
        
        # Calculate distribution metrics
        unique, counts = np.unique(cluster_map, return_counts=True)
        frequencies = counts / cluster_map.size
        
        # Distribution entropy (measure of color diversity)
        entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10))
        
        # Dominant cluster percentage
        max_frequency = np.max(frequencies)
        
        # Color uniformity (inverse of entropy, normalized)
        uniformity = 1.0 - (entropy / np.log2(len(unique)))
        
        return {
            'entropy': entropy,
            'uniformity': uniformity,
            'dominant_cluster_percentage': max_frequency * 100,
            'effective_colors': len(unique),
            'color_diversity_score': entropy / np.log2(self.max_clusters)
        }
    
    def process_video_frame(self, frame: np.ndarray, update_model: bool = True) -> Dict[str, np.ndarray]:
        """
        Process single video frame with optional model updating.
        
        Args:
            frame: Input video frame
            update_model: Whether to update clustering model with this frame
            
        Returns:
            Dictionary with processed frame data
        """
        # Incremental learning
        if update_model:
            self.fit_incremental(frame)
        
        # Get clustering results
        results = self.predict_clusters(frame, return_centers=True)
        
        # Add palette for visualization
        results['color_palette'] = self.create_color_palette(frame)
        results['color_analysis'] = self.analyze_color_distribution(frame)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get clustering performance statistics."""
        if not self.processing_times:
            return {'avg_time': 0, 'fps': 0, 'clusters': self.n_clusters}
        
        times = np.array(self.processing_times)
        return {
            'avg_time_ms': np.mean(times) * 1000,
            'fps': 1.0 / np.mean(times),
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'current_clusters': self.n_clusters,
            'model_fitted': self.fitted
        }
    
    def reset_model(self):
        """Reset the clustering model to initial state."""
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=3
        )
        self.fitted = False
        print("🔄 Clustering model reset")
    
    def set_cluster_count(self, n_clusters: int):
        """
        Dynamically change the number of clusters.
        
        Args:
            n_clusters: New number of clusters (3-16)
        """
        n_clusters = np.clip(n_clusters, self.min_clusters, self.max_clusters)
        if n_clusters != self.n_clusters:
            self.n_clusters = n_clusters
            self.reset_model()
            print(f"🎯 Cluster count changed to {n_clusters}")

# Utility functions for easy integration
def quick_cluster_colors(image: np.ndarray, n_clusters: int = 8) -> Dict[str, np.ndarray]:
    """
    Quick color clustering function.
    
    Args:
        image: RGB image to cluster
        n_clusters: Number of color clusters
        
    Returns:
        Dictionary with clustering results
    """
    clusterer = RealtimeColorClusterer(n_clusters=n_clusters)
    clusterer.fit_incremental(image)
    return clusterer.predict_clusters(image, return_centers=True)

def extract_color_palette(image: np.ndarray, n_colors: int = 5) -> List[np.ndarray]:
    """
    Extract dominant color palette from image.
    
    Args:
        image: RGB image
        n_colors: Number of colors in palette
        
    Returns:
        List of dominant RGB colors
    """
    clusterer = RealtimeColorClusterer(n_clusters=max(n_colors, 8))
    dominant_colors = clusterer.extract_dominant_colors(image, n_colors)
    return [color for color, frequency in dominant_colors]

def create_color_segmentation(image: np.ndarray, n_segments: int = 6) -> np.ndarray:
    """
    Create color-based image segmentation.
    
    Args:
        image: Input RGB image
        n_segments: Number of color segments
        
    Returns:
        Segmented image where each region has uniform color
    """
    results = quick_cluster_colors(image, n_segments)
    return results['cluster_image']

if __name__ == "__main__":
    # Demo and testing
    print("🎯 Real-time Color Clusterer Module - Testing")
    
    # Create test image with multiple colors
    test_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    # Add some distinct color regions
    test_img[:100, :133] = [255, 0, 0]    # Red
    test_img[:100, 133:266] = [0, 255, 0]  # Green  
    test_img[:100, 266:] = [0, 0, 255]    # Blue
    
    # Test clusterer
    clusterer = RealtimeColorClusterer(n_clusters=8, adaptive_mode=True)
    
    # Fit and predict
    clusterer.fit_incremental(test_img)
    results = clusterer.predict_clusters(test_img, return_centers=True)
    
    print(f"📊 Clustering Results:")
    print(f"   Detected clusters: {results['n_clusters']}")
    print(f"   Cluster centers shape: {results['cluster_centers'].shape}")
    
    # Test dominant colors
    dominant_colors = clusterer.extract_dominant_colors(test_img, n_colors=5)
    print(f"🎨 Top 3 Dominant Colors:")
    for i, (color, freq) in enumerate(dominant_colors[:3]):
        print(f"   {i+1}. RGB{tuple(color)} - {freq:.1%}")
    
    # Performance stats
    stats = clusterer.get_performance_stats()
    print("⚡ Performance Stats:")
    print(f"   Average: {stats['avg_time_ms']:.1f}ms")
    print(f"   FPS: {stats['fps']:.1f}")
    print(f"   Current clusters: {stats['current_clusters']}")
    
    # Color analysis
    analysis = clusterer.analyze_color_distribution(test_img)
    print("📈 Color Distribution Analysis:")
    print(f"   Diversity Score: {analysis['color_diversity_score']:.3f}")
    print(f"   Dominant Color: {analysis['dominant_cluster_percentage']:.1f}%")
    
    print("✅ Real-time Color Clusterer ready for integration")