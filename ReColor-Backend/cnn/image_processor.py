"""
Real-time Image Processing Module for ReColor
Handles camera capture, K-means clustering, and before/after visualization
Uses real camera data instead of synthetic datasets
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import io
import base64
from datetime import datetime
import os
from sklearn.cluster import KMeans
from skimage import color as skcolor
import joblib

from kmeans_color_detector import KMeansColorDetector


class ImageProcessor:
    """Process real images with before/after color detection and clustering."""
    
    def __init__(self, save_images: bool = True, output_dir: str = "processed_images"):
        """
        Initialize image processor.
        
        Args:
            save_images: Whether to save processed images
            output_dir: Directory to save processed images
        """
        self.logger = logging.getLogger(__name__)
        self.save_images = save_images
        self.output_dir = output_dir
        
        if save_images:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "before"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "after"), exist_ok=True)
        
        # Initialize K-means detector
        self.kmeans_detector = KMeansColorDetector()
        
        # Processing history
        self.processing_history = []
    
    def process_camera_image(self, image: np.ndarray, n_colors: int = 9) -> Dict[str, Any]:
        """
        Process a real camera image with K-means clustering.
        
        Args:
            image: Input image from camera (RGB format)
            n_colors: Number of color clusters
            
        Returns:
            Dictionary with before/after images and detected colors
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Store original image
            original_image = image.copy()
            
            # Detect dominant colors using K-means
            detected_colors = self.kmeans_detector.detect_multiple_colors(image, top_n=n_colors)
            
            # Apply K-means clustering for visualization
            clustered_image = self._apply_kmeans_clustering(image, n_colors)
            
            # Detect individual color regions
            color_map = self._create_color_map(image, clustered_image, detected_colors)
            
            # Create side-by-side comparison
            comparison_image = self._create_comparison_image(original_image, clustered_image, color_map)
            
            # Save images if enabled
            if self.save_images:
                self._save_images(timestamp, original_image, clustered_image, comparison_image)
            
            # Convert images to base64 for API response
            before_base64 = self._image_to_base64(original_image)
            after_base64 = self._image_to_base64(clustered_image)
            comparison_base64 = self._image_to_base64(comparison_image)
            
            # Prepare color data
            colors_data = [{
                'name': c.name,
                'rgb': c.rgb,
                'hex': c.hex,
                'percentage': c.confidence * 100,
                'cluster_id': c.cluster_id
            } for c in detected_colors]
            
            # Store in history
            processing_result = {
                'timestamp': timestamp,
                'colors': colors_data,
                'n_colors': n_colors,
                'image_size': original_image.shape[:2]
            }
            self.processing_history.append(processing_result)
            
            return {
                'success': True,
                'timestamp': timestamp,
                'before_image': before_base64,
                'after_image': after_base64,
                'comparison_image': comparison_base64,
                'detected_colors': colors_data,
                'statistics': {
                    'total_colors': len(colors_data),
                    'dominant_color': colors_data[0]['name'] if colors_data else 'Unknown',
                    'image_dimensions': f"{original_image.shape[1]}x{original_image.shape[0]}"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing camera image: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _apply_kmeans_clustering(self, image: np.ndarray, n_colors: int) -> np.ndarray:
        """Apply K-means clustering to image for visualization."""
        try:
            # Convert to LAB color space for better perceptual clustering
            image_float = image.astype(np.float32) / 255.0
            lab_image = skcolor.rgb2lab(image_float)
            
            # Reshape for clustering
            h, w = image.shape[:2]
            pixels = lab_image.reshape(-1, 3)
            
            # Apply K-means
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # Get cluster centers and convert back to RGB
            centers_lab = kmeans.cluster_centers_
            centers_rgb = []
            
            for center in centers_lab:
                center_rgb_norm = skcolor.lab2rgb(center.reshape(1, 1, 3)).reshape(-1)
                center_rgb = np.clip(center_rgb_norm * 255, 0, 255)
                centers_rgb.append(center_rgb)
            
            centers_rgb = np.array(centers_rgb)
            
            # Create clustered image
            clustered_pixels = centers_rgb[labels]
            clustered_image = clustered_pixels.reshape(h, w, 3).astype(np.uint8)
            
            return clustered_image
            
        except Exception as e:
            self.logger.error(f"Error in K-means clustering: {e}")
            return image
    
    def _create_color_map(self, original: np.ndarray, clustered: np.ndarray, 
                         colors: List) -> np.ndarray:
        """Create a color map showing detected color regions."""
        try:
            h, w = original.shape[:2]
            color_map = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Create a simple color legend/map
            legend_height = 60
            color_map = np.ones((legend_height, w, 3), dtype=np.uint8) * 255
            
            # Draw color swatches
            num_colors = len(colors)
            if num_colors > 0:
                swatch_width = w // num_colors
                
                for i, color in enumerate(colors):
                    x_start = i * swatch_width
                    x_end = (i + 1) * swatch_width if i < num_colors - 1 else w
                    color_map[:, x_start:x_end] = color.rgb
                    
                    # Add text label
                    cv2.putText(color_map, f"{color.name}", 
                              (x_start + 5, 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(color_map, f"{color.confidence*100:.1f}%", 
                              (x_start + 5, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            return color_map
            
        except Exception as e:
            self.logger.error(f"Error creating color map: {e}")
            return np.ones((60, original.shape[1], 3), dtype=np.uint8) * 128
    
    def _create_comparison_image(self, before: np.ndarray, after: np.ndarray, 
                                color_map: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison with color map."""
        try:
            h, w = before.shape[:2]
            
            # Ensure same dimensions
            if after.shape[:2] != (h, w):
                after = cv2.resize(after, (w, h))
            
            # Add labels
            before_labeled = before.copy()
            after_labeled = after.copy()
            
            cv2.putText(before_labeled, "BEFORE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(after_labeled, "AFTER (K-Means)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Stack horizontally
            comparison = np.hstack([before_labeled, after_labeled])
            
            # Resize color map to match width
            color_map_resized = cv2.resize(color_map, (comparison.shape[1], color_map.shape[0]))
            
            # Stack vertically with color map
            final_comparison = np.vstack([comparison, color_map_resized])
            
            return final_comparison
            
        except Exception as e:
            self.logger.error(f"Error creating comparison image: {e}")
            return np.hstack([before, after])
    
    def _save_images(self, timestamp: str, before: np.ndarray, 
                    after: np.ndarray, comparison: np.ndarray):
        """Save processed images to disk."""
        try:
            before_path = os.path.join(self.output_dir, "before", f"before_{timestamp}.jpg")
            after_path = os.path.join(self.output_dir, "after", f"after_{timestamp}.jpg")
            comparison_path = os.path.join(self.output_dir, f"comparison_{timestamp}.jpg")
            
            cv2.imwrite(before_path, cv2.cvtColor(before, cv2.COLOR_RGB2BGR))
            cv2.imwrite(after_path, cv2.cvtColor(after, cv2.COLOR_RGB2BGR))
            cv2.imwrite(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            
            self.logger.info(f"Images saved for timestamp {timestamp}")
            
        except Exception as e:
            self.logger.error(f"Error saving images: {e}")
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            self.logger.error(f"Error converting image to base64: {e}")
            return ""
    
    def process_real_time_stream(self, camera_index: int = 0, 
                                duration_seconds: int = 10,
                                n_colors: int = 9) -> List[Dict[str, Any]]:
        """
        Process real-time camera stream for a specified duration.
        
        Args:
            camera_index: Camera device index
            duration_seconds: How long to capture (seconds)
            n_colors: Number of colors to detect
            
        Returns:
            List of processing results
        """
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                self.logger.error("Cannot open camera")
                return []
            
            results = []
            start_time = datetime.now()
            frame_count = 0
            
            self.logger.info(f"Starting real-time capture for {duration_seconds} seconds...")
            
            while (datetime.now() - start_time).seconds < duration_seconds:
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    continue
                
                # Process every 30th frame to avoid overload
                frame_count += 1
                if frame_count % 30 != 0:
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                result = self.process_camera_image(frame_rgb, n_colors=n_colors)
                
                if result['success']:
                    results.append(result)
                    self.logger.info(f"Processed frame {frame_count}, detected {len(result['detected_colors'])} colors")
            
            cap.release()
            self.logger.info(f"Capture complete. Processed {len(results)} frames")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in real-time stream processing: {e}")
            return []
    
    def get_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent processing history."""
        return self.processing_history[-limit:]
    
    def clear_history(self):
        """Clear processing history."""
        self.processing_history = []
        self.logger.info("Processing history cleared")


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Image Processor...")
    
    processor = ImageProcessor(save_images=True)
    
    # Test with a sample image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some color regions for testing
    test_image[0:160, :] = [255, 100, 100]  # Red region
    test_image[160:320, :] = [100, 255, 100]  # Green region
    test_image[320:480, :] = [100, 100, 255]  # Blue region
    
    result = processor.process_camera_image(test_image, n_colors=3)
    
    if result['success']:
        print(f"\n✅ Processing successful!")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Detected colors: {len(result['detected_colors'])}")
        for color in result['detected_colors']:
            print(f"  - {color['name']}: {color['percentage']:.1f}%")
    else:
        print(f"❌ Processing failed: {result.get('error')}")
    
    print("\nImage Processor test completed!")
