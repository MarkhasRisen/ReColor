"""
Improved Enhanced Color Detection System for ReColor Backend.
Optimized for real-world accuracy with better performance and color matching.
"""

import numpy as np
import cv2
import colorsys
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans
from collections import Counter
import webcolors


@dataclass
class ImprovedColorInfo:
    """Improved color information structure."""
    name: str
    rgb: Tuple[int, int, int]
    hex: str
    hsv: Tuple[float, float, float]
    confidence: float
    detection_method: str
    closest_web_color: str
    color_temperature: str
    saturation_level: str
    brightness_level: str
    alternative_names: List[str]


class ImprovedColorDetector:
    """Improved color detection optimized for real-world accuracy."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Simplified but more accurate color ranges
        self.color_ranges = self._build_optimized_color_ranges()
        
        # Performance-optimized settings
        self.kmeans_samples = 1000  # Reduced for speed
        self.roi_reduction_factor = 4  # Process every 4th pixel for speed
        
    def _build_optimized_color_ranges(self) -> Dict[str, Dict]:
        """Build optimized color ranges based on HSV color space."""
        
        # HSV ranges for better color classification
        # H: 0-360, S: 0-100, V: 0-100
        ranges = {
            "Red": {
                "hsv_ranges": [(0, 15, 30, 100, 30, 100), (345, 360, 30, 100, 30, 100)],
                "rgb_center": (255, 0, 0),
                "alternatives": ["Crimson", "Cherry", "Rose"]
            },
            "Orange": {
                "hsv_ranges": [(15, 45, 40, 100, 40, 100)],
                "rgb_center": (255, 165, 0),
                "alternatives": ["Amber", "Peach", "Coral"]
            },
            "Yellow": {
                "hsv_ranges": [(45, 75, 40, 100, 50, 100)],
                "rgb_center": (255, 255, 0),
                "alternatives": ["Gold", "Lemon", "Cream"]
            },
            "Green": {
                "hsv_ranges": [(75, 150, 30, 100, 20, 100)],
                "rgb_center": (0, 255, 0),
                "alternatives": ["Lime", "Forest", "Emerald"]
            },
            "Cyan": {
                "hsv_ranges": [(150, 210, 40, 100, 30, 100)],
                "rgb_center": (0, 255, 255),
                "alternatives": ["Aqua", "Turquoise", "Teal"]
            },
            "Blue": {
                "hsv_ranges": [(210, 270, 30, 100, 20, 100)],
                "rgb_center": (0, 0, 255),
                "alternatives": ["Navy", "Sky", "Royal"]
            },
            "Purple": {
                "hsv_ranges": [(270, 330, 30, 100, 20, 100)],
                "rgb_center": (128, 0, 128),
                "alternatives": ["Violet", "Indigo", "Magenta"]
            },
            "Pink": {
                "hsv_ranges": [(330, 360, 20, 60, 50, 100), (0, 15, 20, 60, 50, 100)],
                "rgb_center": (255, 192, 203),
                "alternatives": ["Rose", "Salmon", "Blush"]
            },
            "Brown": {
                "hsv_ranges": [(10, 40, 20, 80, 10, 60)],
                "rgb_center": (165, 42, 42),
                "alternatives": ["Tan", "Beige", "Chocolate"]
            },
            "Black": {
                "hsv_ranges": [(0, 360, 0, 100, 0, 25)],
                "rgb_center": (0, 0, 0),
                "alternatives": ["Charcoal", "Ebony", "Jet"]
            },
            "White": {
                "hsv_ranges": [(0, 360, 0, 20, 80, 100)],
                "rgb_center": (255, 255, 255),
                "alternatives": ["Ivory", "Cream", "Snow"]
            },
            "Gray": {
                "hsv_ranges": [(0, 360, 0, 20, 25, 80)],
                "rgb_center": (128, 128, 128),
                "alternatives": ["Silver", "Ash", "Slate"]
            }
        }
        
        return ranges
    
    def detect_color_improved(self, image: np.ndarray) -> ImprovedColorInfo:
        """
        Improved color detection with optimized performance and accuracy.
        
        Args:
            image: Input image region (ROI)
            
        Returns:
            Improved color information
        """
        try:
            if image is None or image.size == 0:
                return self._create_fallback_color_info()
            
            # Fast preprocessing - reduce image size for speed
            if image.shape[0] > 50 or image.shape[1] > 50:
                # Resize to maximum 50x50 for speed while maintaining accuracy
                h, w = image.shape[:2]
                scale = min(50/h, 50/w)
                if scale < 1:
                    new_h, new_w = int(h * scale), int(w * scale)
                    image = cv2.resize(image, (new_w, new_h))
            
            # Get dominant color using optimized K-means
            dominant_rgb = self._get_fast_dominant_color(image)
            
            # Enhance color for better accuracy
            enhanced_rgb = self._enhance_color_simple(dominant_rgb, image)
            
            # Classify color using HSV ranges
            color_info = self._classify_color_hsv(enhanced_rgb)
            
            return color_info
            
        except Exception as e:
            self.logger.error(f"Error in improved color detection: {e}")
            return self._create_fallback_color_info()
    
    def _get_fast_dominant_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """Get dominant color using fast K-means clustering."""
        try:
            # Subsample pixels for speed
            pixels = image.reshape(-1, 3)
            
            # Take every nth pixel for speed (but ensure we have enough samples)
            step = max(1, len(pixels) // self.kmeans_samples)
            sampled_pixels = pixels[::step].astype(np.float32)
            
            if len(sampled_pixels) < 5:
                # Fallback to mean if too few pixels
                return tuple(map(int, np.mean(pixels, axis=0)))
            
            # Apply fast K-means with fewer clusters
            n_clusters = min(3, len(sampled_pixels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=50)
            kmeans.fit(sampled_pixels)
            
            # Get the most populous cluster
            labels = kmeans.predict(sampled_pixels)
            label_counts = Counter(labels)
            dominant_label = label_counts.most_common(1)[0][0]
            
            dominant_color = kmeans.cluster_centers_[dominant_label]
            return tuple(map(int, np.clip(dominant_color, 0, 255)))
            
        except Exception:
            # Fallback to simple mean
            pixels = image.reshape(-1, 3)
            return tuple(map(int, np.mean(pixels, axis=0)))
    
    def _enhance_color_simple(self, rgb: Tuple[int, int, int], 
                            image: np.ndarray) -> Tuple[int, int, int]:
        """Simple color enhancement for better accuracy."""
        try:
            r, g, b = rgb
            
            # Simple contrast enhancement
            # Convert to HSV
            h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            
            # Slightly boost saturation for better color discrimination
            s = min(1.0, s * 1.15)
            
            # Convert back to RGB
            enhanced_r, enhanced_g, enhanced_b = colorsys.hsv_to_rgb(h, s, v)
            
            return (int(enhanced_r * 255), int(enhanced_g * 255), int(enhanced_b * 255))
            
        except Exception:
            return rgb
    
    def _classify_color_hsv(self, rgb: Tuple[int, int, int]) -> ImprovedColorInfo:
        """Classify color using HSV ranges for better accuracy."""
        try:
            r, g, b = rgb
            
            # Convert to HSV
            h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            h_deg = h * 360
            s_percent = s * 100
            v_percent = v * 100
            
            # Find best matching color
            best_match = "Gray"  # Default
            best_score = 0.0
            alternatives = []
            
            # Special case for very low saturation (grays, blacks, whites)
            if s_percent < 15:
                if v_percent < 20:
                    best_match = "Black"
                    best_score = 0.9
                elif v_percent > 85:
                    best_match = "White" 
                    best_score = 0.9
                else:
                    best_match = "Gray"
                    best_score = 0.8
            else:
                # Check against color ranges
                for color_name, color_data in self.color_ranges.items():
                    score = self._calculate_hsv_match_score(
                        (h_deg, s_percent, v_percent), 
                        color_data["hsv_ranges"]
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_match = color_name
                        alternatives = color_data.get("alternatives", [])
            
            # Get additional properties
            color_temp = self._analyze_color_temperature_simple(rgb)
            saturation_level = self._get_saturation_level(s_percent)
            brightness_level = self._get_brightness_level(v_percent)
            
            # Get closest web color
            try:
                closest_web_color = webcolors.rgb_to_name(rgb)
            except ValueError:
                closest_web_color = self._find_closest_web_color_fast(rgb)
            
            return ImprovedColorInfo(
                name=best_match,
                rgb=rgb,
                hex=f"#{r:02x}{g:02x}{b:02x}",
                hsv=(h_deg, s_percent, v_percent),
                confidence=best_score,
                detection_method="hsv_optimized",
                closest_web_color=closest_web_color,
                color_temperature=color_temp,
                saturation_level=saturation_level,
                brightness_level=brightness_level,
                alternative_names=alternatives[:3]  # Top 3 alternatives
            )
            
        except Exception as e:
            self.logger.error(f"Error in HSV color classification: {e}")
            return self._create_fallback_color_info()
    
    def _calculate_hsv_match_score(self, target_hsv: Tuple[float, float, float],
                                 ranges: List[Tuple[float, ...]]) -> float:
        """Calculate how well target HSV matches color ranges."""
        h_target, s_target, v_target = target_hsv
        best_score = 0.0
        
        for hsv_range in ranges:
            if len(hsv_range) == 6:  # (h_min, h_max, s_min, s_max, v_min, v_max)
                h_min, h_max, s_min, s_max, v_min, v_max = hsv_range
                
                # Check hue (handle wrap-around for red)
                if h_min <= h_max:
                    h_score = 1.0 if h_min <= h_target <= h_max else 0.0
                else:  # Wrap around case (e.g., red: 345-360, 0-15)
                    h_score = 1.0 if h_target >= h_min or h_target <= h_max else 0.0
                
                # Check saturation
                s_score = 1.0 if s_min <= s_target <= s_max else max(0.0, 1.0 - abs(s_target - ((s_min + s_max) / 2)) / 50)
                
                # Check value (brightness)
                v_score = 1.0 if v_min <= v_target <= v_max else max(0.0, 1.0 - abs(v_target - ((v_min + v_max) / 2)) / 50)
                
                # Combined score (all components must match reasonably well)
                score = h_score * s_score * v_score
                best_score = max(best_score, score)
        
        return best_score
    
    def _analyze_color_temperature_simple(self, rgb: Tuple[int, int, int]) -> str:
        """Simple color temperature analysis."""
        r, g, b = rgb
        
        # Simple heuristic based on RGB ratios
        total = r + g + b
        if total == 0:
            return "neutral"
        
        r_ratio = r / total
        b_ratio = b / total
        
        if r_ratio > 0.4:
            return "warm"
        elif b_ratio > 0.4:
            return "cool"
        else:
            return "neutral"
    
    def _get_saturation_level(self, saturation: float) -> str:
        """Get saturation level description."""
        if saturation < 20:
            return "low"
        elif saturation < 60:
            return "medium"
        else:
            return "high"
    
    def _get_brightness_level(self, value: float) -> str:
        """Get brightness level description."""
        if value < 30:
            return "dark"
        elif value < 70:
            return "medium"
        else:
            return "bright"
    
    def _find_closest_web_color_fast(self, rgb: Tuple[int, int, int]) -> str:
        """Fast closest web color finding."""
        try:
            # Use a simplified approach with common web colors
            web_colors = {
                "red": (255, 0, 0),
                "green": (0, 128, 0),
                "blue": (0, 0, 255),
                "yellow": (255, 255, 0),
                "orange": (255, 165, 0),
                "purple": (128, 0, 128),
                "pink": (255, 192, 203),
                "brown": (165, 42, 42),
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "gray": (128, 128, 128)
            }
            
            min_distance = float('inf')
            closest_color = "gray"
            
            for color_name, color_rgb in web_colors.items():
                distance = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name
            
            return closest_color
            
        except Exception:
            return "gray"
    
    def _create_fallback_color_info(self) -> ImprovedColorInfo:
        """Create fallback color information."""
        return ImprovedColorInfo(
            name="Unknown",
            rgb=(128, 128, 128),
            hex="#808080",
            hsv=(0.0, 0.0, 50.0),
            confidence=0.0,
            detection_method="fallback",
            closest_web_color="gray",
            color_temperature="neutral",
            saturation_level="low",
            brightness_level="medium",
            alternative_names=[]
        )


# Integration functions
def get_improved_color_info(image: np.ndarray) -> Dict[str, Any]:
    """Get improved color information with better accuracy and performance."""
    detector = ImprovedColorDetector()
    color_info = detector.detect_color_improved(image)
    
    return {
        'name': color_info.name,
        'rgb': color_info.rgb,
        'hex': color_info.hex,
        'hsv': color_info.hsv,
        'confidence': color_info.confidence,
        'closest_web_color': color_info.closest_web_color,
        'color_temperature': color_info.color_temperature,
        'saturation_level': color_info.saturation_level,
        'brightness_level': color_info.brightness_level,
        'alternatives': color_info.alternative_names,
        'detection_method': color_info.detection_method
    }


def analyze_color_consistency(image: np.ndarray, roi_size: int = 50) -> Dict[str, Any]:
    """Analyze color consistency across multiple regions."""
    detector = ImprovedColorDetector()
    
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Test multiple regions around center
    regions = []
    for offset_x, offset_y in [(0, 0), (-10, -10), (10, 10), (-10, 10), (10, -10)]:
        x1 = max(0, center_x + offset_x - roi_size // 2)
        y1 = max(0, center_y + offset_y - roi_size // 2)
        x2 = min(w, x1 + roi_size)
        y2 = min(h, y1 + roi_size)
        
        roi = image[y1:y2, x1:x2]
        color_info = detector.detect_color_improved(roi)
        regions.append({
            'color': color_info.name,
            'confidence': color_info.confidence,
            'coords': (x1, y1, x2, y2)
        })
    
    # Find consensus
    colors = [region['color'] for region in regions]
    from collections import Counter
    color_counts = Counter(colors)
    consensus_color = color_counts.most_common(1)[0][0]
    consensus_strength = color_counts[consensus_color] / len(colors)
    
    return {
        'consensus_color': consensus_color,
        'consensus_strength': consensus_strength,
        'color_consistency': consensus_strength >= 0.6,  # 60% agreement
        'regions': regions,
        'color_variation': len(set(colors))
    }