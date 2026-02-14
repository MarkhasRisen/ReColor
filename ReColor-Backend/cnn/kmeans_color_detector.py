"""
K-Means Color Detection System for ReColor Backend.
Integrates K-means clustering with LAB and HSV color space for accurate color detection.
Based on the kmeans.ipynb notebook implementation.
"""

import numpy as np
import cv2
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans
from skimage import color as skcolor
from collections import Counter
import colorsys


@dataclass
class KMeansColorInfo:
    """K-means color detection information."""
    name: str
    rgb: Tuple[int, int, int]
    hex: str
    lab: Tuple[float, float, float]
    hsv: Tuple[float, float, float]
    confidence: float
    cluster_id: int
    centroid_rgb: Tuple[int, int, int]
    detection_method: str


class KMeansColorDetector:
    """
    K-means based color detector using LAB color space for better perceptual accuracy.
    """
    
    def __init__(self, model_path: Optional[str] = None, n_clusters: int = 9):
        """
        Initialize K-means color detector.
        
        Args:
            model_path: Path to pre-trained K-means model (optional)
            n_clusters: Number of color clusters (default: 9 for main colors)
        """
        self.logger = logging.getLogger(__name__)
        self.n_clusters = n_clusters
        self.kmeans_model = None
        self.model_path = model_path
        
        # Define named colors (9 main color families)
        self.named_colors = {
            "Red": (255, 0, 0),
            "Orange": (255, 165, 0),
            "Yellow": (255, 255, 0),
            "Green": (0, 128, 0),
            "Cyan": (0, 255, 255),
            "Blue": (0, 0, 255),
            "Purple": (128, 0, 128),
            "White": (255, 255, 255),
            "Black": (0, 0, 0)
        }
        
        # Convert named colors to LAB space for better matching
        self.lab_values = self._convert_colors_to_lab()
        self.color_names = list(self.named_colors.keys())
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._create_default_model()
    
    def _convert_colors_to_lab(self) -> np.ndarray:
        """Convert named colors from RGB to LAB color space."""
        lab_values = []
        for rgb in self.named_colors.values():
            # Normalize RGB to [0, 1]
            rgb_norm = np.array([[np.array(rgb) / 255.0]])
            # Convert to LAB
            lab = skcolor.rgb2lab(rgb_norm)[0, 0]
            lab_values.append(lab)
        return np.array(lab_values)
    
    def _create_default_model(self):
        """Create default K-means model with predefined centroids."""
        try:
            # Use predefined centroids (RGB values of main colors)
            centroids = np.array([
                [255, 0, 0],      # Red
                [255, 165, 0],    # Orange
                [255, 255, 0],    # Yellow
                [0, 128, 0],      # Green
                [0, 255, 255],    # Cyan
                [0, 0, 255],      # Blue
                [128, 0, 128],    # Purple
                [255, 255, 255],  # White
                [0, 0, 0]         # Black
            ], dtype=np.float32)
            
            # Create K-means with predefined centroids
            self.kmeans_model = KMeans(n_clusters=self.n_clusters, init=centroids, n_init=1, random_state=42)
            
            # Fit with dummy data to initialize
            self.kmeans_model.fit(centroids)
            
            self.logger.info("Default K-means model created with predefined centroids")
            
        except Exception as e:
            self.logger.error(f"Error creating default K-means model: {e}")
            self.kmeans_model = None
    
    def train_on_image(self, image: np.ndarray, save_path: Optional[str] = None) -> bool:
        """
        Train K-means model on a single image using LAB color space.
        
        Args:
            image: Input image as numpy array (RGB format)
            save_path: Optional path to save trained model
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            if image is None or image.size == 0:
                self.logger.error("Invalid image for training")
                return False
            
            # Convert image to LAB color space
            image_normalized = image.astype(np.float32) / 255.0
            lab_image = skcolor.rgb2lab(image_normalized)
            
            # Reshape to pixel array
            pixels = lab_image.reshape(-1, 3)
            
            # Train K-means
            self.logger.info(f"Training K-means on {len(pixels)} pixels...")
            self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.kmeans_model.fit(pixels)
            
            self.logger.info("K-means training completed")
            
            # Save model if path provided
            if save_path:
                self.save_model(save_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training K-means model: {e}")
            return False
    
    def detect_color_kmeans(self, image: np.ndarray) -> KMeansColorInfo:
        """
        Detect dominant color using K-means clustering in LAB space.
        
        Args:
            image: Input image region (ROI) as numpy array
            
        Returns:
            K-means color information
        """
        try:
            if image is None or image.size == 0:
                return self._create_fallback_color_info()
            
            if self.kmeans_model is None:
                self.logger.warning("K-means model not initialized, creating default model")
                self._create_default_model()
            
            # Resize for faster processing if needed
            if image.shape[0] > 100 or image.shape[1] > 100:
                image = cv2.resize(image, (100, 100))
            
            # Get dominant color using K-means
            dominant_rgb, cluster_id, centroid_rgb = self._get_dominant_color_kmeans(image)
            
            # Convert to LAB
            rgb_norm = np.array([[dominant_rgb]]) / 255.0
            lab = skcolor.rgb2lab(rgb_norm)[0, 0]
            
            # Convert to HSV
            hsv = colorsys.rgb_to_hsv(dominant_rgb[0]/255.0, dominant_rgb[1]/255.0, dominant_rgb[2]/255.0)
            hsv_degrees = (hsv[0] * 360, hsv[1] * 100, hsv[2] * 100)
            
            # Find closest named color
            color_name = self._closest_named_color(lab)
            
            # Calculate confidence based on cluster consistency
            confidence = self._calculate_cluster_confidence(image, cluster_id)
            
            return KMeansColorInfo(
                name=color_name,
                rgb=tuple(map(int, dominant_rgb)),
                hex=f"#{int(dominant_rgb[0]):02x}{int(dominant_rgb[1]):02x}{int(dominant_rgb[2]):02x}",
                lab=tuple(lab),
                hsv=hsv_degrees,
                confidence=confidence,
                cluster_id=cluster_id,
                centroid_rgb=tuple(map(int, centroid_rgb)),
                detection_method="kmeans_lab"
            )
            
        except Exception as e:
            self.logger.error(f"Error in K-means color detection: {e}")
            return self._create_fallback_color_info()
    
    def _get_dominant_color_kmeans(self, image: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Get dominant color using K-means clustering.
        
        Returns:
            Tuple of (dominant_rgb, cluster_id, centroid_rgb)
        """
        try:
            # Convert to LAB for clustering
            image_normalized = image.astype(np.float32) / 255.0
            lab_image = skcolor.rgb2lab(image_normalized)
            pixels_lab = lab_image.reshape(-1, 3)
            
            # Predict cluster labels
            labels = self.kmeans_model.predict(pixels_lab)
            
            # Find most common cluster
            counts = Counter(labels)
            dominant_cluster = counts.most_common(1)[0][0]
            
            # Get centroid in LAB space
            centroid_lab = self.kmeans_model.cluster_centers_[dominant_cluster]
            
            # Convert centroid back to RGB
            centroid_rgb_norm = skcolor.lab2rgb(centroid_lab.reshape(1, 1, 3)).reshape(-1)
            centroid_rgb = np.clip(centroid_rgb_norm * 255, 0, 255)
            
            # Calculate average color in the dominant cluster
            cluster_mask = labels == dominant_cluster
            cluster_pixels_lab = pixels_lab[cluster_mask]
            
            if len(cluster_pixels_lab) > 0:
                avg_lab = np.mean(cluster_pixels_lab, axis=0)
                avg_rgb_norm = skcolor.lab2rgb(avg_lab.reshape(1, 1, 3)).reshape(-1)
                avg_rgb = np.clip(avg_rgb_norm * 255, 0, 255)
            else:
                avg_rgb = centroid_rgb
            
            return avg_rgb, dominant_cluster, centroid_rgb
            
        except Exception as e:
            self.logger.error(f"Error getting dominant color: {e}")
            # Fallback to simple mean
            mean_rgb = np.mean(image.reshape(-1, 3), axis=0)
            return mean_rgb, 0, mean_rgb
    
    def _closest_named_color(self, lab_color: np.ndarray) -> str:
        """
        Find closest named color using LAB color space for perceptual accuracy.
        
        Args:
            lab_color: Color in LAB space
            
        Returns:
            Name of closest color
        """
        try:
            # Calculate Euclidean distances in LAB space
            distances = np.linalg.norm(self.lab_values - lab_color, axis=1)
            closest_idx = np.argmin(distances)
            return self.color_names[closest_idx]
            
        except Exception as e:
            self.logger.error(f"Error finding closest named color: {e}")
            return "Unknown"
    
    def _calculate_cluster_confidence(self, image: np.ndarray, cluster_id: int) -> float:
        """
        Calculate confidence based on cluster consistency.
        
        Args:
            image: Input image
            cluster_id: Dominant cluster ID
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            # Convert to LAB
            image_normalized = image.astype(np.float32) / 255.0
            lab_image = skcolor.rgb2lab(image_normalized)
            pixels_lab = lab_image.reshape(-1, 3)
            
            # Predict labels
            labels = self.kmeans_model.predict(pixels_lab)
            
            # Calculate percentage of pixels in dominant cluster
            cluster_percentage = np.sum(labels == cluster_id) / len(labels)
            
            return min(cluster_percentage * 1.2, 1.0)  # Boost confidence slightly, cap at 1.0
            
        except Exception:
            return 0.5  # Default medium confidence
    
    def detect_multiple_colors(self, image: np.ndarray, top_n: int = 3) -> List[KMeansColorInfo]:
        """
        Detect multiple dominant colors in an image.
        
        Args:
            image: Input image
            top_n: Number of top colors to return
            
        Returns:
            List of KMeansColorInfo objects
        """
        try:
            if image is None or image.size == 0:
                return [self._create_fallback_color_info()]
            
            if self.kmeans_model is None:
                self._create_default_model()
            
            # Convert to LAB
            image_normalized = image.astype(np.float32) / 255.0
            lab_image = skcolor.rgb2lab(image_normalized)
            pixels_lab = lab_image.reshape(-1, 3)
            
            # Predict labels
            labels = self.kmeans_model.predict(pixels_lab)
            
            # Get cluster counts
            counts = Counter(labels)
            top_clusters = counts.most_common(top_n)
            
            # Get color info for each cluster
            colors = []
            for cluster_id, count in top_clusters:
                # Get centroid
                centroid_lab = self.kmeans_model.cluster_centers_[cluster_id]
                centroid_rgb_norm = skcolor.lab2rgb(centroid_lab.reshape(1, 1, 3)).reshape(-1)
                centroid_rgb = np.clip(centroid_rgb_norm * 255, 0, 255)
                
                # Convert to HSV
                hsv = colorsys.rgb_to_hsv(centroid_rgb[0]/255.0, centroid_rgb[1]/255.0, centroid_rgb[2]/255.0)
                hsv_degrees = (hsv[0] * 360, hsv[1] * 100, hsv[2] * 100)
                
                # Find closest named color
                color_name = self._closest_named_color(centroid_lab)
                
                # Calculate confidence
                confidence = count / len(labels)
                
                color_info = KMeansColorInfo(
                    name=color_name,
                    rgb=tuple(map(int, centroid_rgb)),
                    hex=f"#{int(centroid_rgb[0]):02x}{int(centroid_rgb[1]):02x}{int(centroid_rgb[2]):02x}",
                    lab=tuple(centroid_lab),
                    hsv=hsv_degrees,
                    confidence=confidence,
                    cluster_id=cluster_id,
                    centroid_rgb=tuple(map(int, centroid_rgb)),
                    detection_method="kmeans_lab_multi"
                )
                colors.append(color_info)
            
            return colors
            
        except Exception as e:
            self.logger.error(f"Error detecting multiple colors: {e}")
            return [self._create_fallback_color_info()]
    
    def save_model(self, path: str) -> bool:
        """
        Save K-means model to disk.
        
        Args:
            path: Path to save model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.kmeans_model is None:
                self.logger.error("No model to save")
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            joblib.dump(self.kmeans_model, path)
            self.logger.info(f"K-means model saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving K-means model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load K-means model from disk.
        
        Args:
            path: Path to load model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(path):
                self.logger.error(f"Model file not found: {path}")
                return False
            
            self.kmeans_model = joblib.load(path)
            self.model_path = path
            self.logger.info(f"K-means model loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading K-means model: {e}")
            return False
    
    def _create_fallback_color_info(self) -> KMeansColorInfo:
        """Create fallback color information."""
        return KMeansColorInfo(
            name="Unknown",
            rgb=(128, 128, 128),
            hex="#808080",
            lab=(53.0, 0.0, 0.0),
            hsv=(0.0, 0.0, 50.0),
            confidence=0.0,
            cluster_id=-1,
            centroid_rgb=(128, 128, 128),
            detection_method="fallback"
        )


# Integration functions
def get_kmeans_color_info(image: np.ndarray, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get color information using K-means detection.
    
    Args:
        image: Input image region
        model_path: Optional path to K-means model
        
    Returns:
        Dictionary with color information
    """
    detector = KMeansColorDetector(model_path=model_path)
    color_info = detector.detect_color_kmeans(image)
    
    return {
        'name': color_info.name,
        'rgb': color_info.rgb,
        'hex': color_info.hex,
        'lab': color_info.lab,
        'hsv': color_info.hsv,
        'confidence': color_info.confidence,
        'cluster_id': color_info.cluster_id,
        'centroid_rgb': color_info.centroid_rgb,
        'detection_method': color_info.detection_method
    }


def get_multiple_kmeans_colors(image: np.ndarray, top_n: int = 3, 
                               model_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get multiple dominant colors using K-means detection.
    
    Args:
        image: Input image
        top_n: Number of top colors to detect
        model_path: Optional path to K-means model
        
    Returns:
        List of color information dictionaries
    """
    detector = KMeansColorDetector(model_path=model_path)
    colors = detector.detect_multiple_colors(image, top_n=top_n)
    
    return [{
        'name': c.name,
        'rgb': c.rgb,
        'hex': c.hex,
        'lab': c.lab,
        'hsv': c.hsv,
        'confidence': c.confidence,
        'cluster_id': c.cluster_id,
        'centroid_rgb': c.centroid_rgb,
        'detection_method': c.detection_method
    } for c in colors]


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing K-means Color Detector...")
    
    # Create test image (red color)
    test_image = np.zeros((64, 64, 3), dtype=np.uint8)
    test_image[:, :] = (255, 50, 50)  # Reddish
    
    # Test detection
    detector = KMeansColorDetector()
    color_info = detector.detect_color_kmeans(test_image)
    
    print(f"\nDetected color: {color_info.name}")
    print(f"RGB: {color_info.rgb}")
    print(f"Hex: {color_info.hex}")
    print(f"Confidence: {color_info.confidence:.3f}")
    print(f"Cluster ID: {color_info.cluster_id}")
    
    print("\nK-means Color Detector test completed!")
