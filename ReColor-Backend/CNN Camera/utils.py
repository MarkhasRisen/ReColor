"""
Utility functions for ReColor TensorFlow colorblind detection system.
Contains helper functions for color conversion, GPU detection, K-means clustering, and common utilities.
"""

import numpy as np
import tensorflow as tf
import cv2
import logging
from typing import Tuple, Dict, List, Optional
import colorsys
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


def setup_gpu() -> Dict[str, any]:
    """
    Set up GPU for TensorFlow and return device information.
    
    Returns:
        Dictionary containing GPU information and configuration
    """
    gpu_info = {
        'gpu_available': False,
        'gpu_name': None,
        'gpu_memory': None,
        'device': '/CPU:0'
    }
    
    try:
        # Check if GPU is available
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if physical_devices:
            gpu_info['gpu_available'] = True
            gpu_info['device'] = '/GPU:0'
            
            # Get GPU details
            gpu_details = tf.config.experimental.get_device_details(physical_devices[0])
            gpu_info['gpu_name'] = gpu_details.get('device_name', 'Unknown GPU')
            
            # Configure GPU memory growth to avoid allocation issues
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            
            # Get GPU memory info if available
            try:
                gpu_info['gpu_memory'] = f"{gpu_details.get('compute_capability', 'Unknown')} Compute Capability"
            except:
                gpu_info['gpu_memory'] = "Memory info unavailable"
                
            logging.info(f"✅ GPU detected: {gpu_info['gpu_name']}")
            logging.info(f"GPU Memory: {gpu_info['gpu_memory']}")
            
        else:
            logging.warning("⚠️ No GPU found. Using CPU for inference.")
            
    except Exception as e:
        logging.error(f"Error setting up GPU: {e}")
        logging.warning("Falling back to CPU.")
    
    return gpu_info


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hexadecimal string.
    
    Args:
        rgb: RGB color tuple (r, g, b)
        
    Returns:
        Hexadecimal color string (e.g., "#FF0000")
    """
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}".upper()


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hexadecimal string to RGB tuple.
    
    Args:
        hex_color: Hexadecimal color string (e.g., "#FF0000" or "FF0000")
        
    Returns:
        RGB color tuple (r, g, b)
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert to RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_dominant_color(image: np.ndarray, method: str = "mean") -> Tuple[int, int, int]:
    """
    Extract dominant color from image using specified method.
    
    Args:
        image: Input image as numpy array (H, W, C)
        method: Method to use ("mean", "median", "histogram")
        
    Returns:
        Dominant RGB color tuple
    """
    try:
        if method == "mean":
            # Calculate mean color
            mean_color = np.mean(image, axis=(0, 1))
            return tuple(map(int, mean_color))
            
        elif method == "median":
            # Calculate median color
            median_color = np.median(image, axis=(0, 1))
            return tuple(map(int, median_color))
            
        elif method == "histogram":
            # Use histogram to find most common color
            # Reshape image to list of pixels
            pixels = image.reshape(-1, 3)
            
            # Quantize colors to reduce complexity
            quantized = (pixels // 32) * 32
            
            # Find most common color
            unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
            dominant_idx = np.argmax(counts)
            dominant_color = unique_colors[dominant_idx]
            
            return tuple(map(int, dominant_color))
            
        else:
            # Default to mean
            mean_color = np.mean(image, axis=(0, 1))
            return tuple(map(int, mean_color))
            
    except Exception as e:
        logging.error(f"Error extracting dominant color: {e}")
        return (128, 128, 128)  # Default gray


def get_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Get approximate color name from RGB values.
    
    Args:
        rgb: RGB color tuple
        
    Returns:
        Approximate color name string
    """
    r, g, b = rgb
    
    # Convert to HSV for better color classification
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    h = h * 360  # Convert to degrees
    s = s * 100  # Convert to percentage
    v = v * 100  # Convert to percentage
    
    # Define color ranges
    if v < 20:
        return "Black"
    elif s < 10:
        if v < 30:
            return "Dark Gray"
        elif v < 70:
            return "Gray"
        else:
            return "White"
    elif s < 30:
        if v < 50:
            return "Dark Gray"
        else:
            return "Light Gray"
    else:
        # Color classification based on hue
        if h < 15 or h >= 345:
            return "Red"
        elif h < 45:
            return "Orange"
        elif h < 75:
            return "Yellow"
        elif h < 150:
            return "Green"
        elif h < 210:
            return "Cyan"
        elif h < 270:
            return "Blue"
        elif h < 330:
            return "Purple"
        else:
            return "Pink"


def resize_image_with_aspect_ratio(image: np.ndarray, 
                                 target_size: Tuple[int, int],
                                 keep_aspect: bool = True) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio if requested.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        keep_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    try:
        if not keep_aspect:
            return cv2.resize(image, target_size)
        
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas with target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image on canvas
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        return canvas
        
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        return cv2.resize(image, target_size)


def create_color_swatch(color: Tuple[int, int, int], 
                       size: Tuple[int, int] = (100, 100),
                       border_color: Tuple[int, int, int] = (255, 255, 255),
                       border_thickness: int = 2) -> np.ndarray:
    """
    Create a color swatch image.
    
    Args:
        color: RGB color tuple
        size: Swatch size (width, height)
        border_color: Border color (BGR format for OpenCV)
        border_thickness: Border thickness in pixels
        
    Returns:
        Color swatch as numpy array
    """
    try:
        width, height = size
        
        # Create swatch with specified color (convert RGB to BGR for OpenCV)
        bgr_color = (color[2], color[1], color[0])
        swatch = np.full((height, width, 3), bgr_color, dtype=np.uint8)
        
        # Add border
        if border_thickness > 0:
            cv2.rectangle(swatch, (0, 0), (width-1, height-1), border_color, border_thickness)
        
        return swatch
        
    except Exception as e:
        logging.error(f"Error creating color swatch: {e}")
        # Return a gray swatch as fallback
        return np.full((size[1], size[0], 3), (128, 128, 128), dtype=np.uint8)


def validate_rgb(rgb: Tuple[int, int, int]) -> bool:
    """
    Validate RGB color values.
    
    Args:
        rgb: RGB color tuple
        
    Returns:
        True if valid, False otherwise
    """
    try:
        r, g, b = rgb
        return all(0 <= val <= 255 for val in [r, g, b])
    except:
        return False


def clamp_rgb(rgb: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    Clamp RGB values to valid range and convert to integers.
    
    Args:
        rgb: RGB color tuple (may contain floats)
        
    Returns:
        Clamped RGB tuple with integers
    """
    return tuple(max(0, min(255, int(val))) for val in rgb)


def get_contrast_text_color(background_rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Get contrasting text color (black or white) for given background.
    
    Args:
        background_rgb: Background RGB color
        
    Returns:
        Text color RGB tuple (either black or white)
    """
    # Calculate luminance
    r, g, b = background_rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Return white text for dark backgrounds, black for light backgrounds
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)


def format_timestamp(timestamp: float) -> str:
    """
    Format timestamp for logging.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Formatted timestamp string
    """
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        import os
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error creating directory {directory_path}: {e}")
        return False


# Test utility functions
if __name__ == "__main__":
    # Test color conversion
    print("Testing utility functions...")
    
    # Test RGB to Hex conversion
    rgb_color = (255, 128, 64)
    hex_color = rgb_to_hex(rgb_color)
    back_to_rgb = hex_to_rgb(hex_color)
    print(f"RGB {rgb_color} → HEX {hex_color} → RGB {back_to_rgb}")
    
    # Test color naming
    test_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 128, 128)]
    for color in test_colors:
        name = get_color_name(color)
        print(f"RGB {color} → {name}")
    
    # Test GPU setup
    gpu_info = setup_gpu()
    print(f"GPU Info: {gpu_info}")
    
    # Test dominant color extraction
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dominant = get_dominant_color(test_image, "mean")
    print(f"Dominant color: RGB {dominant} → {get_color_name(dominant)}")
    
    print("Utility functions test completed!")


def extract_dominant_colors_kmeans(image: np.ndarray, k: int = 5, sample_ratio: float = 0.1) -> Tuple[List[Tuple[int, int, int]], List[float]]:
    """
    Extract dominant colors from image using K-means clustering.
    
    Args:
        image: Input image as numpy array (H, W, C) in RGB format
        k: Number of color clusters to extract
        sample_ratio: Ratio of pixels to sample for clustering (0.1 = 10%)
        
    Returns:
        Tuple of (dominant_colors, percentages) where:
        - dominant_colors: List of RGB tuples representing cluster centers
        - percentages: List of percentages for each color cluster
    """
    try:
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB format (H, W, 3)")
        
        # Reshape image to pixel array
        pixels = image.reshape(-1, 3)
        
        # Sample pixels for performance (K-means on all pixels can be slow)
        if sample_ratio < 1.0:
            pixels = shuffle(pixels, random_state=42)
            n_samples = int(len(pixels) * sample_ratio)
            pixels = pixels[:n_samples]
        
        # Check unique colors and adjust k if necessary
        unique_pixels = np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0)
        n_unique_colors = len(unique_pixels)
        
        # Adjust k to not exceed the number of unique colors
        effective_k = min(k, n_unique_colors)
        
        if effective_k < k:
            logging.debug(f"Reduced K-means clusters from {k} to {effective_k} due to limited unique colors ({n_unique_colors})")
        
        # Apply K-means clustering with adjusted k
        if effective_k <= 1:
            # Handle case with only one unique color
            colors = unique_pixels[:1].astype(int)
            percentages = [100.0]
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*Number of distinct clusters.*")
                kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Get cluster centers (dominant colors)
                colors = kmeans.cluster_centers_.astype(int)
                
                # Calculate percentages for each cluster
                labels = kmeans.labels_
                unique_labels, counts = np.unique(labels, return_counts=True)
                percentages = (counts / len(labels) * 100).tolist()
        
        # Convert to list of RGB tuples
        dominant_colors = [(int(color[0]), int(color[1]), int(color[2])) for color in colors]
        
        # Sort by percentage (descending)
        color_data = list(zip(dominant_colors, percentages))
        color_data.sort(key=lambda x: x[1], reverse=True)
        
        dominant_colors, percentages = zip(*color_data)
        
        return list(dominant_colors), list(percentages)
        
    except Exception as e:
        logging.error(f"Error in K-means color extraction: {e}")
        # Fallback to single dominant color
        dominant = get_dominant_color(image, method="mean")
        return [dominant], [100.0]


def simplify_image_kmeans(image: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Simplify image colors using K-means clustering (posterization effect).
    
    Args:
        image: Input image as numpy array (H, W, C) in RGB format
        k: Number of color clusters for simplification
        
    Returns:
        Simplified image with reduced color palette
    """
    try:
        original_shape = image.shape
        
        # Reshape image to pixel array
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Check unique colors and adjust k if necessary
        unique_pixels = np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0)
        n_unique_colors = len(unique_pixels)
        
        # Adjust k to not exceed the number of unique colors
        effective_k = min(k, n_unique_colors)
        
        if effective_k < k:
            logging.debug(f"Reduced K-means clusters from {k} to {effective_k} due to limited unique colors ({n_unique_colors})")
        
        # Apply K-means clustering with adjusted k
        if effective_k <= 1:
            # Handle case with only one unique color - return original image
            simplified_pixels = pixels
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*Number of distinct clusters.*")
                kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Replace each pixel with its cluster center
                simplified_pixels = kmeans.cluster_centers_[kmeans.labels_]
        
        # Reshape back to image
        simplified_image = simplified_pixels.reshape(original_shape).astype(np.uint8)
        
        return simplified_image
        
    except Exception as e:
        logging.error(f"Error in K-means image simplification: {e}")
        return image.copy()


def analyze_color_distribution(image: np.ndarray, k: int = 10) -> Dict:
    """
    Analyze color distribution in image using K-means clustering.
    
    Args:
        image: Input image as numpy array (H, W, C) in RGB format
        k: Number of color clusters for analysis
        
    Returns:
        Dictionary containing color analysis results
    """
    try:
        dominant_colors, percentages = extract_dominant_colors_kmeans(image, k)
        
        # Get color names for dominant colors
        color_names = [get_color_name(color) for color in dominant_colors]
        
        # Calculate color diversity metrics
        total_colors = len(dominant_colors)
        dominant_color_percentage = max(percentages) if percentages else 0
        color_diversity = 100 - dominant_color_percentage  # Higher when colors are more evenly distributed
        
        analysis = {
            'dominant_colors': dominant_colors,
            'percentages': percentages,
            'color_names': color_names,
            'total_clusters': total_colors,
            'most_dominant_percentage': dominant_color_percentage,
            'color_diversity_score': color_diversity,
            'primary_color': dominant_colors[0] if dominant_colors else (128, 128, 128),
            'primary_color_name': color_names[0] if color_names else "Gray"
        }
        
        return analysis
        
    except Exception as e:
        logging.error(f"Error in color distribution analysis: {e}")
        return {
            'dominant_colors': [(128, 128, 128)],
            'percentages': [100.0],
            'color_names': ["Gray"],
            'total_clusters': 1,
            'most_dominant_percentage': 100.0,
            'color_diversity_score': 0.0,
            'primary_color': (128, 128, 128),
            'primary_color_name': "Gray"
        }