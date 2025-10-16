"""
ColorBlindnessSimulator for ReColor TensorFlow colorblind detection system.
Implements scientifically accurate color vision deficiency simulation using transformation matrices.
"""

import numpy as np
import cv2
import logging
from typing import Dict, Tuple, Optional
from enum import Enum


class CVDType(Enum):
    """Enumeration of color vision deficiency types."""
    NORMAL = "normal"
    PROTANOPIA = "protanopia"
    DEUTERANOPIA = "deuteranopia"
    TRITANOPIA = "tritanopia"
    PROTANOMALY = "protanomaly"
    DEUTERANOMALY = "deuteranomaly"
    TRITANOMALY = "tritanomaly"


class CVDSeverity(Enum):
    """CVD severity levels for anomalous trichromacy."""
    MILD = 0.3
    MODERATE = 0.6
    SEVERE = 0.9


class ColorBlindnessSimulator:
    """
    Simulates color vision deficiencies using scientifically accurate transformation matrices.
    Based on research by Brettel, Viénot and Mollon (1997) and Machado, Oliveira and Fernandes (2009).
    """
    
    def __init__(self):
        """Initialize ColorBlindnessSimulator with transformation matrices."""
        self.logger = logging.getLogger(__name__)
        
        # Scientific color transformation matrices for different CVD types
        # These matrices are based on the Brettel-Viénot-Mollon model with realistic confusion
        self.transformation_matrices = {
            # Complete dichromacy (missing cone type)
            CVDType.PROTANOPIA: np.array([
                [0.170, 0.830, 0.000],  # More realistic red-green confusion
                [0.170, 0.830, 0.000],  # Complete inability to distinguish red-green
                [0.000, 0.000, 1.000]   # Blue perception intact
            ], dtype=np.float32),
            
            CVDType.DEUTERANOPIA: np.array([
                [0.330, 0.670, 0.000],  # Different red-green confusion pattern
                [0.330, 0.670, 0.000],  # Complete green blindness
                [0.000, 0.000, 1.000]   # Blue perception intact
            ], dtype=np.float32),
            
            CVDType.TRITANOPIA: np.array([
                [1.000, 0.000, 0.000],  # Red perception intact
                [0.000, 1.000, 0.000],  # Green perception intact
                [0.000, 0.000, 0.000]   # Complete blue blindness (very rare)
            ], dtype=np.float32),
            
            # Anomalous trichromacy (shifted cone sensitivity)
            CVDType.PROTANOMALY: np.array([
                [0.817, 0.183, 0.000],  # Mild red-green confusion
                [0.333, 0.667, 0.000],
                [0.000, 0.125, 0.875]
            ], dtype=np.float32),
            
            CVDType.DEUTERANOMALY: np.array([
                [0.800, 0.200, 0.000],  # Most common CVD type
                [0.258, 0.742, 0.000],
                [0.000, 0.142, 0.858]
            ], dtype=np.float32),
            
            CVDType.TRITANOMALY: np.array([
                [0.967, 0.033, 0.000],  # Very rare blue-yellow confusion
                [0.000, 0.733, 0.267],
                [0.000, 0.183, 0.817]
            ], dtype=np.float32)
        }
        
        # Ishihara test confusion matrices for realistic simulation
        self.ishihara_confusion_patterns = {
            CVDType.PROTANOPIA: {
                # Colors that become indistinguishable (RGB values)
                'confusion_pairs': [
                    ((255, 0, 0), (0, 128, 0)),      # Red-Green
                    ((255, 165, 0), (255, 255, 0)),  # Orange-Yellow
                    ((128, 0, 128), (0, 0, 255)),    # Purple-Blue
                ],
                'invisible_numbers': [12, 8, 29, 70]  # Numbers invisible to protanopes
            },
            CVDType.DEUTERANOPIA: {
                'confusion_pairs': [
                    ((255, 0, 0), (0, 255, 0)),      # Red-Green (different pattern)
                    ((255, 105, 180), (128, 128, 128)), # Pink-Gray
                    ((165, 42, 42), (0, 128, 0)),    # Brown-Green
                ],
                'invisible_numbers': [12, 5, 3, 15]   # Numbers invisible to deuteranopes
            },
            CVDType.TRITANOPIA: {
                'confusion_pairs': [
                    ((0, 0, 255), (0, 255, 0)),      # Blue-Green
                    ((255, 255, 0), (255, 192, 203)), # Yellow-Pink
                    ((128, 0, 128), (255, 0, 0)),    # Purple-Red
                ],
                'invisible_numbers': [8, 2, 6]        # Numbers invisible to tritanopes
            }
        }
        
        # Current simulation type
        self.current_cvd_type = CVDType.NORMAL
        
        # Cache for performance optimization
        self.last_transformation = None
        self.cache_enabled = True
        
        self.logger.info("ColorBlindnessSimulator initialized")
        self.logger.info(f"Available CVD types: {[cvd.value for cvd in CVDType]}")
    
    def simulate(self, image: np.ndarray, cvd_type: CVDType) -> np.ndarray:
        """
        Simulate color vision deficiency on input image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            cvd_type: Type of color vision deficiency to simulate
            
        Returns:
            Simulated image with color vision deficiency applied
        """
        try:
            # Return original image for normal vision
            if cvd_type == CVDType.NORMAL:
                return image.copy()
            
            # Get transformation matrix
            if cvd_type not in self.transformation_matrices:
                self.logger.warning(f"Unknown CVD type: {cvd_type}. Returning original image.")
                return image.copy()
            
            transform_matrix = self.transformation_matrices[cvd_type]
            
            # Apply transformation
            simulated_image = self._apply_transformation(image, transform_matrix)
            
            return simulated_image
            
        except Exception as e:
            self.logger.error(f"Error simulating {cvd_type.value}: {e}")
            return image.copy()
    
    def _apply_transformation(self, image: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply color transformation matrix to image.
        
        Args:
            image: Input image (H, W, C) in RGB format
            transform_matrix: 3x3 transformation matrix
            
        Returns:
            Transformed image
        """
        try:
            # Ensure image is in correct format
            if image.dtype != np.float32:
                image_float = image.astype(np.float32) / 255.0
            else:
                image_float = image.copy()
            
            # Get image dimensions
            original_shape = image_float.shape
            
            # Reshape image to (N, 3) for matrix multiplication
            pixels = image_float.reshape(-1, 3)
            
            # Apply transformation matrix
            # Each pixel (RGB) is multiplied by the 3x3 transformation matrix
            transformed_pixels = np.dot(pixels, transform_matrix.T)
            
            # Reshape back to original image shape
            transformed_image = transformed_pixels.reshape(original_shape)
            
            # Clamp values to [0, 1] range
            transformed_image = np.clip(transformed_image, 0.0, 1.0)
            
            # Convert back to uint8 if input was uint8
            if image.dtype == np.uint8:
                transformed_image = (transformed_image * 255).astype(np.uint8)
            
            return transformed_image
            
        except Exception as e:
            self.logger.error(f"Error applying transformation: {e}")
            return image
    
    def set_cvd_type(self, cvd_type: CVDType) -> None:
        """
        Set current CVD simulation type.
        
        Args:
            cvd_type: Color vision deficiency type to set
        """
        if isinstance(cvd_type, str):
            # Convert string to CVDType enum
            try:
                cvd_type = CVDType(cvd_type.lower())
            except ValueError:
                self.logger.error(f"Invalid CVD type: {cvd_type}")
                return
        
        self.current_cvd_type = cvd_type
        self.logger.info(f"CVD type set to: {cvd_type.value}")
    
    def get_current_cvd_type(self) -> CVDType:
        """Get current CVD simulation type."""
        return self.current_cvd_type
    
    def cycle_cvd_type(self) -> CVDType:
        """
        Cycle to next CVD type.
        
        Returns:
            New current CVD type
        """
        cvd_types = list(CVDType)
        current_index = cvd_types.index(self.current_cvd_type)
        next_index = (current_index + 1) % len(cvd_types)
        
        self.current_cvd_type = cvd_types[next_index]
        self.logger.info(f"Cycled to CVD type: {self.current_cvd_type.value}")
        
        return self.current_cvd_type
    
    def simulate_current(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate using current CVD type.
        
        Args:
            image: Input image
            
        Returns:
            Simulated image with current CVD type
        """
        return self.simulate(image, self.current_cvd_type)
    
    def check_ishihara_visibility(self, number: int, cvd_type: CVDType) -> bool:
        """
        Check if a specific Ishihara test number would be visible for a given CVD type.
        
        Args:
            number: Ishihara test plate number
            cvd_type: CVD type to check
            
        Returns:
            True if number is visible, False if invisible/confusing
        """
        if cvd_type == CVDType.NORMAL:
            return True
        
        if cvd_type in self.ishihara_confusion_patterns:
            invisible_numbers = self.ishihara_confusion_patterns[cvd_type]['invisible_numbers']
            return number not in invisible_numbers
        
        return True
    
    def daltonize(self, image: np.ndarray, cvd_type: CVDType, enhancement_factor: float = 1.5) -> np.ndarray:
        """
        Apply daltonization to enhance color discrimination for colorblind users.
        Daltonization redistributes colors to make them more distinguishable for specific CVD types.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            cvd_type: Type of CVD to optimize for
            enhancement_factor: Strength of enhancement (1.0 = normal, higher = more enhancement)
            
        Returns:
            Daltonized image with enhanced color discrimination
        """
        try:
            if cvd_type == CVDType.NORMAL:
                return image.copy()
            
            # Convert to float for processing
            image_float = image.astype(np.float32) / 255.0
            
            # Simulate how the image appears to someone with CVD
            cvd_simulation = self.simulate(image_float, cvd_type)
            
            # Calculate the error (what the CVD person cannot see)
            error = image_float - cvd_simulation
            
            # Daltonization matrices for redistributing colors
            if cvd_type in [CVDType.PROTANOPIA, CVDType.PROTANOMALY]:
                # Shift reds toward blues and greens
                daltonization_matrix = np.array([
                    [0.0, 2.02344, -2.52581],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
            elif cvd_type in [CVDType.DEUTERANOPIA, CVDType.DEUTERANOMALY]:
                # Shift greens toward reds and blues
                daltonization_matrix = np.array([
                    [1.0, 0.0, 0.0],
                    [0.494207, 0.0, 1.24827],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
            elif cvd_type in [CVDType.TRITANOPIA, CVDType.TRITANOMALY]:
                # Shift blues toward reds and greens
                daltonization_matrix = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-0.395913, 0.801109, 0.0]
                ], dtype=np.float32)
            else:
                return image.copy()
            
            # Apply daltonization to the error
            error_corrected = self._apply_transformation(error, daltonization_matrix)
            
            # Add the corrected error back to the original simulation
            daltonized = cvd_simulation + (error_corrected * enhancement_factor)
            
            # Ensure values stay in valid range
            daltonized = np.clip(daltonized, 0.0, 1.0)
            
            # Convert back to uint8
            daltonized_image = (daltonized * 255).astype(np.uint8)
            
            return daltonized_image
            
        except Exception as e:
            self.logger.error(f"Error in daltonization for {cvd_type.value}: {e}")
            return image.copy()


# Test the ColorBlindnessSimulator
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ColorBlindnessSimulator...")
    
    try:
        # Initialize simulator
        simulator = ColorBlindnessSimulator()
        
        # Create test image
        test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        print(f"Test image shape: {test_image.shape}")
        
        # Test all CVD types
        for cvd_type in CVDType:
            print(f"\nTesting {cvd_type.value.upper()}:")
            
            # Test simulation
            simulated = simulator.simulate(test_image, cvd_type)
            print(f"  Simulation successful: {simulated.shape}")
            
            if cvd_type != CVDType.NORMAL:
                # Calculate difference
                difference = np.mean(np.abs(test_image.astype(float) - simulated.astype(float)))
                print(f"  Average pixel difference: {difference:.2f}")
                
                # Test Ishihara visibility
                visible = simulator.check_ishihara_visibility(12, cvd_type)
                print(f"  Ishihara #12 visible: {visible}")
        
        print("\n✅ ColorBlindnessSimulator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()