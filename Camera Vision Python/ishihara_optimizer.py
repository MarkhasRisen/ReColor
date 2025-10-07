#!/usr/bin/env python3
"""
Ishihara Test Optimization Module for ReColor
============================================
Specialized optimization for Ishihara color blindness test plates.
Enhances color discrimination for red-green deficiencies.

Based on Ishihara test characteristics:
- Red dots on green backgrounds (Plates 1-17)
- Green dots on red backgrounds (Plates 18-21) 
- Color confusion lines optimization
- Enhanced contrast for number visibility

Author: ReColor Development Team
Date: October 2025
License: MIT
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
import time

class IshiharaOptimizer:
    """
    Specialized optimization for Ishihara color blindness test plates.
    
    Focuses on enhancing red-green color discrimination and improving
    visibility of hidden numbers in Ishihara test plates.
    """
    
    def __init__(self):
        """Initialize Ishihara Test Optimizer."""
        
        # Ishihara test color ranges (RGB)
        self.ishihara_colors = {
            'red_foreground': np.array([[150, 50, 50], [255, 120, 120]]),      # Red numbers
            'green_background': np.array([[50, 120, 50], [120, 200, 120]]),    # Green background
            'green_foreground': np.array([[40, 100, 40], [100, 180, 100]]),    # Green numbers
            'red_background': np.array([[120, 40, 40], [200, 100, 100]]),      # Red background
            'yellow_dots': np.array([[180, 180, 50], [255, 255, 150]]),        # Yellow confusion dots
            'orange_dots': np.array([[200, 120, 50], [255, 180, 100]])         # Orange confusion dots
        }
        
        # CVD-specific enhancement parameters
        self.cvd_enhancements = {
            'protanopia': {
                'red_boost': 2.5,        # Boost red channel significantly
                'green_suppress': 0.7,   # Slightly suppress green
                'blue_maintain': 1.0,    # Maintain blue
                'contrast_boost': 1.8    # High contrast boost
            },
            'deuteranopia': {
                'red_boost': 2.2,        # Strong red boost
                'green_suppress': 0.6,   # Suppress green more
                'blue_maintain': 1.1,    # Slight blue boost
                'contrast_boost': 2.0    # Maximum contrast boost
            },
            'tritanopia': {
                'red_boost': 1.3,        # Moderate red boost
                'green_suppress': 1.2,   # Slight green boost
                'blue_maintain': 2.5,    # Strong blue boost
                'contrast_boost': 1.5    # Moderate contrast
            }
        }
        
        # Confusion line matrices for targeted correction
        self.confusion_matrices = self._initialize_confusion_matrices()
        
        # Performance tracking
        self.processing_times = []
        
        print("🎯 Ishihara Test Optimizer initialized")
    
    def _initialize_confusion_matrices(self) -> Dict[str, np.ndarray]:
        """Initialize confusion line correction matrices."""
        
        matrices = {}
        
        # Protanopia confusion line (red-green confusion)
        matrices['protanopia'] = np.array([
            [0.0, 2.02344, -2.52581],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Deuteranopia confusion line (red-green confusion, different axis)
        matrices['deuteranopia'] = np.array([
            [1.0, 0.0, 0.0],
            [0.49421, 0.0, 1.24827],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Tritanopia confusion line (blue-yellow confusion)
        matrices['tritanopia'] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.39578, 0.80155, 0.0]
        ], dtype=np.float32)
        
        return matrices
    
    def detect_ishihara_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect potential Ishihara test regions in the image.
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary with detected region masks
        """
        
        masks = {}
        
        for color_name, color_range in self.ishihara_colors.items():
            lower_bound = color_range[0]
            upper_bound = color_range[1]
            
            # Create mask for this color range
            mask = np.all((image >= lower_bound) & (image <= upper_bound), axis=2)
            masks[color_name] = mask.astype(np.uint8) * 255
        
        return masks
    
    def enhance_red_green_discrimination(self, image: np.ndarray, cvd_type: str) -> np.ndarray:
        """
        Enhance red-green color discrimination for Ishihara tests.
        
        Args:
            image: RGB image array
            cvd_type: Type of CVD ('protanopia', 'deuteranopia', 'tritanopia')
            
        Returns:
            Enhanced RGB image
        """
        
        if cvd_type not in self.cvd_enhancements:
            return image
        
        enhanced = image.astype(np.float32)
        params = self.cvd_enhancements[cvd_type]
        
        # Apply channel-specific enhancements
        enhanced[:, :, 0] *= params['red_boost']      # Red channel
        enhanced[:, :, 1] *= params['green_suppress'] # Green channel
        enhanced[:, :, 2] *= params['blue_maintain']  # Blue channel
        
        # Apply confusion line correction
        if cvd_type in self.confusion_matrices:
            confusion_matrix = self.confusion_matrices[cvd_type]
            
            # Reshape for matrix multiplication
            h, w, c = enhanced.shape
            enhanced_flat = enhanced.reshape(-1, 3)
            
            # Apply confusion correction
            corrected_flat = np.dot(enhanced_flat, confusion_matrix.T)
            enhanced = corrected_flat.reshape(h, w, c)
        
        # Contrast enhancement
        enhanced = self._enhance_contrast(enhanced, params['contrast_boost'])
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _enhance_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Enhance contrast using adaptive histogram equalization.
        
        Args:
            image: Float32 image array
            factor: Contrast enhancement factor
            
        Returns:
            Contrast-enhanced image
        """
        
        # Convert to LAB for better contrast control
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        
        # Extract L channel
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Apply contrast enhancement to L channel only
        l_enhanced = (l_channel - 128) * factor + 128
        l_enhanced = np.clip(l_enhanced, 0, 255)
        
        # Put back enhanced L channel
        lab[:, :, 0] = l_enhanced.astype(np.uint8)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb.astype(np.float32)
    
    def optimize_ishihara_visibility(self, image: np.ndarray, cvd_type: str, 
                                   enhancement_strength: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Comprehensive Ishihara test optimization.
        
        Args:
            image: RGB image array
            cvd_type: Type of CVD
            enhancement_strength: Enhancement strength (0.0 to 2.0)
            
        Returns:
            Dictionary with optimization results
        """
        
        start_time = time.time()
        
        # Step 1: Detect Ishihara-like regions
        region_masks = self.detect_ishihara_regions(image)
        
        # Step 2: Enhanced red-green discrimination
        enhanced_base = self.enhance_red_green_discrimination(image, cvd_type)
        
        # Step 3: Targeted enhancement based on detected regions
        enhanced_targeted = enhanced_base.copy().astype(np.float32)
        
        # Enhance red foreground on green background (typical Ishihara)
        if 'red_foreground' in region_masks and 'green_background' in region_masks:
            red_regions = region_masks['red_foreground'] > 0
            green_regions = region_masks['green_background'] > 0
            
            # Boost red in red regions
            enhanced_targeted[red_regions, 0] *= (1.5 * enhancement_strength)
            
            # Suppress green in green regions for better contrast
            enhanced_targeted[green_regions, 1] *= (0.7 / enhancement_strength)
        
        # Step 4: Color space transformation for better discrimination
        enhanced_final = self._apply_discriminative_transform(
            enhanced_targeted, cvd_type, enhancement_strength
        )
        
        # Step 5: Adaptive sharpening for number visibility
        sharpened = self._adaptive_sharpen(enhanced_final)
        
        # Performance tracking
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        return {
            'original': image,
            'enhanced_base': enhanced_base,
            'enhanced_targeted': enhanced_targeted.astype(np.uint8),
            'final_optimized': sharpened,
            'region_masks': region_masks,
            'processing_time_ms': processing_time
        }
    
    def _apply_discriminative_transform(self, image: np.ndarray, cvd_type: str, 
                                      strength: float) -> np.ndarray:
        """
        Apply color space transformation for better color discrimination.
        
        Args:
            image: Float32 RGB image
            cvd_type: CVD type
            strength: Enhancement strength
            
        Returns:
            Transformed image
        """
        
        # Convert to HSV for better color manipulation
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        if cvd_type in ['protanopia', 'deuteranopia']:
            # Enhance red-green discrimination
            # Shift hues to increase separation
            red_hue_mask = (hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 150)  # Red hues
            green_hue_mask = (hsv[:, :, 0] >= 60) & (hsv[:, :, 0] <= 120)  # Green hues
            
            # Boost saturation for red and green regions
            hsv[red_hue_mask, 1] *= (1.3 * strength)
            hsv[green_hue_mask, 1] *= (1.2 * strength)
            
            # Adjust hues for better separation
            hsv[red_hue_mask, 0] = np.clip(hsv[red_hue_mask, 0] - 5 * strength, 0, 179)
            hsv[green_hue_mask, 0] = np.clip(hsv[green_hue_mask, 0] + 5 * strength, 0, 179)
        
        elif cvd_type == 'tritanopia':
            # Enhance blue-yellow discrimination
            blue_hue_mask = (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130)  # Blue hues
            yellow_hue_mask = (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 40)  # Yellow hues
            
            hsv[blue_hue_mask, 1] *= (1.4 * strength)
            hsv[yellow_hue_mask, 1] *= (1.3 * strength)
        
        # Ensure values are in valid range
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
        
        # Convert back to RGB
        hsv_uint8 = hsv.astype(np.uint8)
        rgb_transformed = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB)
        
        return rgb_transformed.astype(np.float32)
    
    def _adaptive_sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive sharpening to enhance number visibility.
        
        Args:
            image: Float32 RGB image
            
        Returns:
            Sharpened image
        """
        
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        
        # Detect edges using Sobel operator
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize edge magnitude
        edge_magnitude = (edge_magnitude / edge_magnitude.max() * 255).astype(np.uint8)
        
        # Create adaptive sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]], dtype=np.float32)
        
        # Apply sharpening with edge-based weighting
        sharpened = cv2.filter2D(image_uint8, -1, kernel)
        
        # Blend original and sharpened based on edge strength
        edge_weight = (edge_magnitude / 255.0)[:, :, np.newaxis]
        result = image_uint8 * (1 - edge_weight * 0.5) + sharpened * (edge_weight * 0.5)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def create_ishihara_test_simulation(self, width: int = 400, height: int = 400,
                                      number: str = "8") -> np.ndarray:
        """
        Create a simulated Ishihara test plate for testing.
        
        Args:
            width: Image width
            height: Image height  
            number: Number to display
            
        Returns:
            Simulated Ishihara test plate
        """
        
        # Create background with green dots
        background = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate random dot pattern
        np.random.seed(42)  # For reproducibility
        
        # Background dots (green)
        for _ in range(2000):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            radius = np.random.randint(3, 8)
            
            green_intensity = np.random.randint(100, 180)
            color = (np.random.randint(40, 80), green_intensity, np.random.randint(40, 80))
            
            cv2.circle(background, (x, y), radius, color, -1)
        
        # Create number mask
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 8
        thickness = 20
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(number, font, font_scale, thickness)
        
        # Center the text
        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2
        
        # Create number mask
        number_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.putText(number_mask, number, (text_x, text_y), font, font_scale, 255, thickness)
        
        # Add red dots in number region
        number_pixels = np.where(number_mask > 0)
        
        for i in range(0, len(number_pixels[0]), 15):  # Sparse sampling
            y, x = number_pixels[0][i], number_pixels[1][i]
            
            # Add some randomness
            x += np.random.randint(-5, 5)
            y += np.random.randint(-5, 5)
            
            if 0 <= x < width and 0 <= y < height:
                radius = np.random.randint(3, 7)
                red_intensity = np.random.randint(120, 200)
                color = (red_intensity, np.random.randint(40, 80), np.random.randint(40, 80))
                
                cv2.circle(background, (x, y), radius, color, -1)
        
        return background
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        
        if not self.processing_times:
            return {'avg_time_ms': 0.0, 'fps': 0.0, 'samples': 0}
        
        avg_time = np.mean(self.processing_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_time_ms': avg_time,
            'fps': fps,
            'samples': len(self.processing_times)
        }

def test_ishihara_optimizer():
    """Test the Ishihara optimizer with simulated test plates."""
    
    print("🧪 Testing Ishihara Optimizer...")
    
    # Initialize optimizer
    optimizer = IshiharaOptimizer()
    
    # Create test Ishihara plates
    test_numbers = ["8", "3", "5", "2"]
    
    for number in test_numbers:
        print(f"\n🔢 Testing with number: {number}")
        
        # Create simulated test plate
        test_plate = optimizer.create_ishihara_test_simulation(number=number)
        
        # Test with different CVD types
        for cvd_type in ['protanopia', 'deuteranopia', 'tritanopia']:
            print(f"   Testing {cvd_type}...")
            
            # Optimize for this CVD type
            results = optimizer.optimize_ishihara_visibility(test_plate, cvd_type)
            
            print(f"   ✅ Processing time: {results['processing_time_ms']:.1f}ms")
    
    # Performance stats
    stats = optimizer.get_performance_stats()
    print(f"\n📊 Performance Summary:")
    print(f"   Average time: {stats['avg_time_ms']:.1f}ms")
    print(f"   FPS: {stats['fps']:.1f}")
    print(f"   Samples: {stats['samples']}")
    
    print("✅ Ishihara optimizer test complete")
    return optimizer

if __name__ == "__main__":
    test_ishihara_optimizer()