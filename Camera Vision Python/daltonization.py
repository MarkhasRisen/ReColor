#!/usr/bin/env python3
"""
Daltonization Module for ReColor
===============================
Implements adaptive daltonization for Color Vision Deficiency correction.
Uses confusion line analysis and perceptual contrast enhancement.

Author: ReColor Development Team  
Date: October 2025
License: MIT
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
import time
from cvd_simulation import CVDSimulator

class AdaptiveDaltonizer:
    """
    Adaptive daltonization system for CVD color correction.
    
    Based on Fidaner et al. (2005) daltonization algorithm with adaptive enhancements
    for real-time mobile processing and personalized correction.
    """
    
    def __init__(self, cvd_simulator: Optional[CVDSimulator] = None, adaptation_level: str = 'medium'):
        """
        Initialize Adaptive Daltonizer.
        
        Args:
            cvd_simulator: CVD simulator instance (creates new if None)
            adaptation_level: 'low', 'medium', 'high' - controls correction strength
        """
        self.cvd_simulator = cvd_simulator if cvd_simulator else CVDSimulator()
        self.adaptation_level = adaptation_level
        
        # Correction strength based on adaptation level
        self.correction_strengths = {
            'low': 0.3,
            'medium': 0.7, 
            'high': 1.0
        }
        
        self.correction_strength = self.correction_strengths.get(adaptation_level, 0.7)
        
        # Daltonization matrices for each CVD type
        self._initialize_dalton_matrices()
        
        # Performance tracking
        self.processing_times = []
        self.contrast_improvements = []
        
        print(f"✅ Adaptive Daltonizer initialized (adaptation: {adaptation_level})")
    
    def _initialize_dalton_matrices(self):
        """Initialize daltonization correction matrices."""
        
        # Protanopia daltonization matrix (Fidaner et al.)
        self.protanopia_dalton = np.array([
            [0.0, 0.0, 0.0],
            [0.7, 1.0, 0.0], 
            [0.7, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Deuteranopia daltonization matrix
        self.deuteranopia_dalton = np.array([
            [1.0, 0.7, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.7, 1.0]
        ], dtype=np.float32)
        
        # Tritanopia daltonization matrix
        self.tritanopia_dalton = np.array([
            [1.0, 0.0, 0.7],
            [0.0, 1.0, 0.7],
            [0.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        self.dalton_matrices = {
            'protanopia': self.protanopia_dalton,
            'deuteranopia': self.deuteranopia_dalton,
            'tritanopia': self.tritanopia_dalton,
            'protan': self.protanopia_dalton,
            'deutan': self.deuteranopia_dalton, 
            'tritan': self.tritanopia_dalton
        }
    
    def calculate_confusion_lines(self, image: np.ndarray, cvd_type: str) -> np.ndarray:
        """
        Calculate confusion lines for the given CVD type.
        
        Args:
            image: Input RGB image
            cvd_type: Type of CVD
            
        Returns:
            Confusion line map showing problem areas
        """
        # Simulate CVD to find problematic colors
        cvd_simulated = self.cvd_simulator.simulate_cvd(image, cvd_type)
        
        # Calculate per-pixel difference 
        difference = np.abs(image.astype(np.float32) - cvd_simulated.astype(np.float32))
        
        # Create confusion strength map (grayscale)
        confusion_map = np.mean(difference, axis=2)
        
        # Normalize to 0-255 range
        if confusion_map.max() > 0:
            confusion_map = (confusion_map / confusion_map.max() * 255).astype(np.uint8)
        else:
            confusion_map = np.zeros_like(confusion_map, dtype=np.uint8)
        
        return confusion_map
    
    def enhance_perceptual_contrast(self, image: np.ndarray, confusion_map: np.ndarray) -> np.ndarray:
        """
        Enhance perceptual contrast in confusion areas.
        
        Args:
            image: Input RGB image
            confusion_map: Confusion line map from calculate_confusion_lines
            
        Returns:
            Contrast-enhanced image
        """
        enhanced = image.astype(np.float32)
        
        # Normalize confusion map to 0-1 range
        confusion_normalized = confusion_map.astype(np.float32) / 255.0
        
        # Apply contrast enhancement where confusion is high
        for channel in range(3):
            channel_data = enhanced[:, :, channel]
            
            # Increase contrast using histogram stretching in problem areas
            min_val = np.percentile(channel_data, 5)
            max_val = np.percentile(channel_data, 95)
            
            if max_val > min_val:
                # Stretch histogram
                stretched = (channel_data - min_val) / (max_val - min_val) * 255
                stretched = np.clip(stretched, 0, 255)
                
                # Blend based on confusion strength
                alpha = confusion_normalized * self.correction_strength
                enhanced[:, :, channel] = (1 - alpha) * channel_data + alpha * stretched
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def apply_daltonization(self, image: np.ndarray, cvd_type: str) -> np.ndarray:
        """
        Apply daltonization correction for specified CVD type.
        
        Args:
            image: Input RGB image (H, W, 3)
            cvd_type: Type of CVD to correct
            
        Returns:
            Daltonized RGB image
        """
        start_time = time.time()
        
        if cvd_type.lower() not in self.dalton_matrices:
            available_types = list(self.dalton_matrices.keys())
            raise ValueError(f"Unsupported CVD type: {cvd_type}. Available: {available_types}")
        
        # Step 1: Simulate CVD to understand the deficit
        cvd_simulated = self.cvd_simulator.simulate_cvd(image, cvd_type)
        
        # Step 2: Calculate the error (what CVD person cannot see)
        error = image.astype(np.float32) - cvd_simulated.astype(np.float32)
        
        # Step 3: Apply daltonization matrix to redistribute error
        dalton_matrix = self.dalton_matrices[cvd_type.lower()]
        
        height, width, channels = error.shape
        error_flat = error.reshape(-1, channels)
        
        # Apply daltonization transformation
        correction_flat = np.dot(error_flat, dalton_matrix.T)
        correction = correction_flat.reshape(height, width, channels)
        
        # Step 4: Add correction to original image
        daltonized = image.astype(np.float32) + correction * self.correction_strength
        
        # Step 5: Clamp values to valid range
        daltonized = np.clip(daltonized, 0, 255).astype(np.uint8)
        
        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 50:
            self.processing_times.pop(0)
        
        return daltonized
    
    def adaptive_daltonization(self, image: np.ndarray, cvd_type: str, 
                             region_based: bool = True) -> np.ndarray:
        """
        Apply adaptive daltonization with region-specific corrections.
        
        Args:
            image: Input RGB image
            cvd_type: Type of CVD to correct  
            region_based: Apply region-specific adaptive corrections
            
        Returns:
            Adaptively corrected image
        """
        if not region_based:
            return self.apply_daltonization(image, cvd_type)
        
        # Step 1: Calculate confusion lines 
        confusion_map = self.calculate_confusion_lines(image, cvd_type)
        
        # Step 2: Apply basic daltonization
        daltonized = self.apply_daltonization(image, cvd_type)
        
        # Step 3: Enhance contrast in problematic areas
        enhanced = self.enhance_perceptual_contrast(daltonized, confusion_map)
        
        # Step 4: Adaptive blending based on confusion strength
        confusion_normalized = confusion_map.astype(np.float32) / 255.0
        
        # Blend between original daltonized and enhanced versions
        alpha = np.expand_dims(confusion_normalized, axis=2) * 0.5  # Max 50% enhancement
        result = (1 - alpha) * daltonized.astype(np.float32) + alpha * enhanced.astype(np.float32)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def optimize_for_ishihara(self, image: np.ndarray, cvd_type: str) -> np.ndarray:
        """
        Optimize daltonization specifically for Ishihara test visibility.
        
        Args:
            image: Input RGB image (likely Ishihara test)
            cvd_type: Type of CVD to optimize for
            
        Returns:
            Optimized image for better number recognition
        """
        # Apply stronger daltonization for test images
        original_strength = self.correction_strength
        self.correction_strength = min(self.correction_strength * 1.5, 1.0)
        
        # Enhanced region-based processing
        daltonized = self.adaptive_daltonization(image, cvd_type, region_based=True)
        
        # Additional contrast enhancement for test patterns
        if cvd_type.lower() in ['deuteranopia', 'deutan']:
            # Boost red-green contrast for deuteranopia
            enhanced = self._enhance_red_green_contrast(daltonized)
        elif cvd_type.lower() in ['protanopia', 'protan']:
            # Boost red contrast for protanopia
            enhanced = self._enhance_red_contrast(daltonized)
        else:
            # Tritanopia - boost blue-yellow contrast
            enhanced = self._enhance_blue_yellow_contrast(daltonized)
        
        # Restore original strength
        self.correction_strength = original_strength
        
        return enhanced
    
    def _enhance_red_green_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance red-green contrast for deuteranopia."""
        enhanced = image.astype(np.float32)
        
        # Calculate red-green difference
        red_green_diff = enhanced[:, :, 0] - enhanced[:, :, 1]
        
        # Amplify difference where it exists
        amplification = 1.3
        enhanced[:, :, 0] += red_green_diff * (amplification - 1) * 0.5
        enhanced[:, :, 1] -= red_green_diff * (amplification - 1) * 0.5
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _enhance_red_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance red contrast for protanopia."""
        enhanced = image.copy().astype(np.float32)
        
        # Boost red channel where it's significant
        red_mask = enhanced[:, :, 0] > 100
        enhanced[red_mask, 0] *= 1.2
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _enhance_blue_yellow_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance blue-yellow contrast for tritanopia.""" 
        enhanced = image.astype(np.float32)
        
        # Convert to LAB color space for better yellow manipulation
        lab = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab = lab.astype(np.float32)
        
        # Enhance b* channel (blue-yellow axis)
        lab[:, :, 2] = lab[:, :, 2] * 1.2
        
        # Convert back to RGB
        lab_clipped = np.clip(lab, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(lab_clipped, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def batch_daltonize(self, images: List[np.ndarray], cvd_type: str) -> List[np.ndarray]:
        """
        Apply daltonization to multiple images efficiently.
        
        Args:
            images: List of RGB images
            cvd_type: CVD type to correct
            
        Returns:
            List of daltonized images
        """
        results = []
        
        for image in images:
            daltonized = self.adaptive_daltonization(image, cvd_type)
            results.append(daltonized)
        
        return results
    
    def evaluate_correction_effectiveness(self, original: np.ndarray, corrected: np.ndarray, 
                                        cvd_type: str) -> Dict[str, float]:
        """
        Evaluate the effectiveness of daltonization correction.
        
        Args:
            original: Original image
            corrected: Daltonized image  
            cvd_type: CVD type that was corrected
            
        Returns:
            Dictionary with effectiveness metrics
        """
        # Simulate how CVD person sees both images
        orig_cvd = self.cvd_simulator.simulate_cvd(original, cvd_type)
        corr_cvd = self.cvd_simulator.simulate_cvd(corrected, cvd_type)
        
        # Calculate improvements
        orig_error = np.mean(np.abs(original.astype(float) - orig_cvd.astype(float)))
        corr_error = np.mean(np.abs(original.astype(float) - corr_cvd.astype(float)))
        
        # Contrast improvement
        orig_contrast = self._calculate_contrast(orig_cvd)
        corr_contrast = self._calculate_contrast(corr_cvd)
        
        contrast_improvement = (corr_contrast - orig_contrast) / orig_contrast if orig_contrast > 0 else 0
        error_reduction = (orig_error - corr_error) / orig_error if orig_error > 0 else 0
        
        self.contrast_improvements.append(contrast_improvement)
        
        return {
            'error_reduction_percent': error_reduction * 100,
            'contrast_improvement_percent': contrast_improvement * 100,
            'original_error': orig_error,
            'corrected_error': corr_error,
            'effectiveness_score': (error_reduction + max(0, contrast_improvement)) / 2
        }
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast using RMS method."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.std(gray)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.processing_times:
            return {'avg_time': 0, 'fps': 0}
        
        times = np.array(self.processing_times)
        return {
            'avg_time_ms': np.mean(times) * 1000,
            'fps': 1.0 / np.mean(times),
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000
        }
    
    def set_correction_strength(self, strength: float):
        """
        Dynamically adjust correction strength.
        
        Args:
            strength: Correction strength (0.0 to 1.0)
        """
        self.correction_strength = np.clip(strength, 0.0, 1.0)
        print(f"🔧 Correction strength set to {self.correction_strength:.2f}")

# Utility functions
def quick_daltonize(image: np.ndarray, cvd_type: str, strength: str = 'medium') -> np.ndarray:
    """
    Quick daltonization function.
    
    Args:
        image: RGB image to correct
        cvd_type: CVD type to correct for
        strength: Correction strength ('low', 'medium', 'high')
        
    Returns:
        Daltonized image
    """
    daltonizer = AdaptiveDaltonizer(adaptation_level=strength)
    return daltonizer.adaptive_daltonization(image, cvd_type)

def create_before_after_comparison(image: np.ndarray, cvd_type: str) -> np.ndarray:
    """
    Create before/after comparison for daltonization.
    
    Args:
        image: Original image
        cvd_type: CVD type to demonstrate
        
    Returns:
        Side-by-side comparison image
    """
    daltonizer = AdaptiveDaltonizer()
    
    # Create comparison
    daltonized = daltonizer.adaptive_daltonization(image, cvd_type)
    
    # Side-by-side layout
    h, w, c = image.shape
    comparison = np.zeros((h, w * 2, c), dtype=np.uint8)
    comparison[:, :w] = image
    comparison[:, w:] = daltonized
    
    return comparison

if __name__ == "__main__":
    # Demo and testing
    print("🎨 Adaptive Daltonizer Module - Testing")
    
    # Create test image with red-green pattern
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    test_img[:, :100, 0] = 255  # Red left half
    test_img[:, 100:, 1] = 255  # Green right half
    
    # Test daltonizer
    daltonizer = AdaptiveDaltonizer()
    
    for cvd_type in ['protanopia', 'deuteranopia', 'tritanopia']:
        daltonized = daltonizer.adaptive_daltonization(test_img, cvd_type)
        effectiveness = daltonizer.evaluate_correction_effectiveness(test_img, daltonized, cvd_type)
        
        print(f"📊 {cvd_type.title()}:")
        print(f"   Error Reduction: {effectiveness['error_reduction_percent']:.1f}%")
        print(f"   Contrast Improvement: {effectiveness['contrast_improvement_percent']:.1f}%")
        print(f"   Effectiveness Score: {effectiveness['effectiveness_score']:.3f}")
    
    stats = daltonizer.get_performance_stats()
    print("⚡ Performance Stats:")
    print(f"   Average: {stats['avg_time_ms']:.1f}ms")
    print(f"   FPS: {stats['fps']:.1f}")
    print("✅ Adaptive Daltonizer ready for integration")