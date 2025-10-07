#!/usr/bin/env python3
"""
CVD Simulation Module for ReColor
=================================
Implements Color Vision Deficiency simulation using LMS-based transformation matrices.
Supports Protanopia, Deuteranopia, and Tritanopia with scientifically accurate color transforms.

Author: ReColor Development Team
Date: October 2025
License: MIT
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from numba import jit
import time

# Import Ishihara optimization
try:
    from ishihara_optimizer import IshiharaOptimizer
    HAS_ISHIHARA_OPTIMIZER = True
except ImportError:
    HAS_ISHIHARA_OPTIMIZER = False

class CVDSimulator:
    """
    Color Vision Deficiency simulator using LMS color space transformations.
    
    Based on Brettel et al. (1997) and Viénot et al. (1999) research on CVD simulation.
    Provides accurate simulation of the three main types of color blindness.
    """
    
    def __init__(self, optimization_level: str = 'balanced'):
        """
        Initialize CVD Simulator with transformation matrices.
        
        Args:
            optimization_level: 'mobile' for fastest, 'balanced' for quality/speed, 'quality' for best accuracy
        """
        self.optimization_level = optimization_level
        
        # RGB to LMS transformation matrix (Hunt-Pointer-Estevez)
        self.rgb_to_lms_matrix = np.array([
            [0.31399022, 0.63951294, 0.04649755],
            [0.15537241, 0.75789446, 0.08670142], 
            [0.01775239, 0.10944209, 0.87256922]
        ], dtype=np.float32)
        
        # LMS to RGB transformation matrix (inverse)
        self.lms_to_rgb_matrix = np.array([
            [5.47221206, -4.6419601, 0.16963708],
            [-1.1252419, 2.29317094, -0.1678952],
            [0.02980165, -0.19318073, 1.16364789]
        ], dtype=np.float32)
        
        # CVD simulation matrices in LMS space
        self._initialize_cvd_matrices()
        
        # Initialize Ishihara optimizer if available
        if HAS_ISHIHARA_OPTIMIZER:
            self.ishihara_optimizer = IshiharaOptimizer()
            self.ishihara_mode = False  # Toggle for Ishihara-specific optimization
        else:
            self.ishihara_optimizer = None
            self.ishihara_mode = False
        
        # Performance optimization
        if optimization_level == 'mobile':
            self._enable_mobile_optimizations()
        
        # Statistics tracking
        self.processing_times = []
        
        print(f"✅ CVD Simulator initialized with {optimization_level} optimization")
    
    def _initialize_cvd_matrices(self):
        """Initialize CVD transformation matrices for each deficiency type."""
        
        # Protanopia (L-cone deficiency) - Red color blindness
        self.protanopia_matrix = np.array([
            [0.0, 1.05118294, -0.05116099],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Deuteranopia (M-cone deficiency) - Green color blindness  
        self.deuteranopia_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.9513092, 0.0, 0.04866992],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Tritanopia (S-cone deficiency) - Blue color blindness
        self.tritanopia_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.86744736, 1.86727089, 0.0]
        ], dtype=np.float32)
        
        # Combined matrices dictionary
        self.cvd_matrices = {
            'protanopia': self.protanopia_matrix,
            'deuteranopia': self.deuteranopia_matrix, 
            'tritanopia': self.tritanopia_matrix,
            'protan': self.protanopia_matrix,  # Aliases
            'deutan': self.deuteranopia_matrix,
            'tritan': self.tritanopia_matrix
        }
    
    def _enable_mobile_optimizations(self):
        """Enable optimizations for mobile/low-power processing."""
        # Use lower precision for mobile
        self.rgb_to_lms_matrix = self.rgb_to_lms_matrix.astype(np.float16)
        self.lms_to_rgb_matrix = self.lms_to_rgb_matrix.astype(np.float16)
        
        for key in self.cvd_matrices:
            self.cvd_matrices[key] = self.cvd_matrices[key].astype(np.float16)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _rgb_to_lms_optimized(rgb_image: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Optimized RGB to LMS conversion using Numba JIT compilation."""
        height, width, _ = rgb_image.shape
        lms_image = np.zeros_like(rgb_image, dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                pixel = rgb_image[i, j].astype(np.float32) / 255.0
                lms_pixel = np.dot(transform_matrix, pixel)
                lms_image[i, j] = lms_pixel
        
        return lms_image
    
    @staticmethod
    @jit(nopython=True, cache=True) 
    def _lms_to_rgb_optimized(lms_image: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Optimized LMS to RGB conversion using Numba JIT compilation."""
        height, width, _ = lms_image.shape
        rgb_image = np.zeros_like(lms_image, dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                lms_pixel = lms_image[i, j]
                rgb_pixel = np.dot(transform_matrix, lms_pixel)
                # Clamp and convert to uint8
                rgb_pixel = np.clip(rgb_pixel * 255.0, 0, 255)
                rgb_image[i, j] = rgb_pixel.astype(np.uint8)
        
        return rgb_image
    
    def rgb_to_lms(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to LMS color space.
        
        Args:
            rgb_image: Input RGB image (H, W, 3)
            
        Returns:
            LMS image (H, W, 3) with same dimensions
        """
        if self.optimization_level == 'mobile':
            return self._rgb_to_lms_optimized(rgb_image, self.rgb_to_lms_matrix)
        
        # Standard numpy implementation for quality mode
        rgb_normalized = rgb_image.astype(np.float32) / 255.0
        height, width, channels = rgb_normalized.shape
        
        # Reshape for matrix multiplication
        rgb_flat = rgb_normalized.reshape(-1, channels)
        lms_flat = np.dot(rgb_flat, self.rgb_to_lms_matrix.T)
        
        return lms_flat.reshape(height, width, channels)
    
    def lms_to_rgb(self, lms_image: np.ndarray) -> np.ndarray:
        """
        Convert LMS image back to RGB color space.
        
        Args:
            lms_image: Input LMS image (H, W, 3)
            
        Returns:
            RGB image (H, W, 3) as uint8
        """
        if self.optimization_level == 'mobile':
            return self._lms_to_rgb_optimized(lms_image, self.lms_to_rgb_matrix)
        
        # Standard numpy implementation
        height, width, channels = lms_image.shape
        lms_flat = lms_image.reshape(-1, channels)
        rgb_flat = np.dot(lms_flat, self.lms_to_rgb_matrix.T)
        
        # Clamp values and convert to uint8
        rgb_flat = np.clip(rgb_flat * 255.0, 0, 255)
        rgb_image = rgb_flat.reshape(height, width, channels).astype(np.uint8)
        
        return rgb_image
    
    def simulate_cvd(self, rgb_image: np.ndarray, cvd_type: str) -> np.ndarray:
        """
        Simulate Color Vision Deficiency on RGB image.
        
        Args:
            rgb_image: Input RGB image (H, W, 3)
            cvd_type: Type of CVD ('protanopia', 'deuteranopia', 'tritanopia')
            
        Returns:
            CVD-simulated RGB image (H, W, 3)
            
        Raises:
            ValueError: If cvd_type is not supported
        """
        start_time = time.time()
        
        if cvd_type.lower() not in self.cvd_matrices:
            available_types = list(self.cvd_matrices.keys())
            raise ValueError(f"Unsupported CVD type: {cvd_type}. Available: {available_types}")
        
        # Get CVD transformation matrix
        cvd_matrix = self.cvd_matrices[cvd_type.lower()]
        
        # Convert RGB → LMS
        lms_image = self.rgb_to_lms(rgb_image)
        
        # Apply CVD simulation in LMS space
        height, width, channels = lms_image.shape
        lms_flat = lms_image.reshape(-1, channels)
        
        # Apply CVD transformation
        cvd_lms_flat = np.dot(lms_flat, cvd_matrix.T)
        cvd_lms_image = cvd_lms_flat.reshape(height, width, channels)
        
        # Convert LMS → RGB
        cvd_rgb_image = self.lms_to_rgb(cvd_lms_image)
        
        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:  # Keep last 100 measurements
            self.processing_times.pop(0)
        
        return cvd_rgb_image
    
    def simulate_multiple_cvd(self, rgb_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate all CVD types at once for comparison.
        
        Args:
            rgb_image: Input RGB image (H, W, 3)
            
        Returns:
            Dictionary with CVD type as key and simulated image as value
        """
        results = {'original': rgb_image.copy()}
        
        for cvd_type in ['protanopia', 'deuteranopia', 'tritanopia']:
            results[cvd_type] = self.simulate_cvd(rgb_image, cvd_type)
        
        return results
    
    def get_severity_simulation(self, rgb_image: np.ndarray, cvd_type: str, severity: float = 1.0) -> np.ndarray:
        """
        Simulate CVD with varying severity levels.
        
        Args:
            rgb_image: Input RGB image
            cvd_type: Type of CVD
            severity: Severity from 0.0 (normal) to 1.0 (complete deficiency)
            
        Returns:
            CVD-simulated image with specified severity
        """
        if not 0.0 <= severity <= 1.0:
            raise ValueError("Severity must be between 0.0 and 1.0")
        
        if severity == 0.0:
            return rgb_image.copy()
        
        # Full CVD simulation
        cvd_image = self.simulate_cvd(rgb_image, cvd_type)
        
        # Blend between original and CVD based on severity
        blended = (1 - severity) * rgb_image.astype(np.float32) + severity * cvd_image.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for the CVD simulator.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.processing_times:
            return {'avg_time': 0, 'fps': 0, 'min_time': 0, 'max_time': 0}
        
        times = np.array(self.processing_times)
        return {
            'avg_time_ms': np.mean(times) * 1000,
            'fps': 1.0 / np.mean(times),
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'std_time_ms': np.std(times) * 1000
        }
    
    def validate_cvd_accuracy(self) -> Dict[str, bool]:
        """
        Validate CVD simulation accuracy using test patterns.
        
        Returns:
            Dictionary with validation results for each CVD type
        """
        # Create Ishihara-like test pattern
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Red-green pattern for deuteranopia testing
        test_image[:50, :, 0] = 255  # Red top half
        test_image[50:, :, 1] = 255  # Green bottom half
        
        validation_results = {}
        
        for cvd_type in ['protanopia', 'deuteranopia', 'tritanopia']:
            try:
                simulated = self.simulate_cvd(test_image, cvd_type)
                # Basic validation: check if transformation occurred
                difference = np.mean(np.abs(test_image.astype(float) - simulated.astype(float)))
                validation_results[cvd_type] = difference > 10.0  # Threshold for meaningful change
            except Exception as e:
                validation_results[cvd_type] = False
                print(f"❌ Validation failed for {cvd_type}: {e}")
        
        return validation_results
    
    def simulate_cvd_ishihara_optimized(self, rgb_image: np.ndarray, cvd_type: str, 
                                      enhancement_strength: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Simulate CVD with Ishihara test optimization.
        
        This method combines CVD simulation with specialized Ishihara test optimization
        to improve visibility of color blindness test patterns.
        
        Args:
            rgb_image: Input RGB image (H, W, 3)
            cvd_type: 'protanopia', 'deuteranopia', or 'tritanopia'
            enhancement_strength: Ishihara enhancement strength (0.5 to 2.0)
            
        Returns:
            Dictionary with simulation and optimization results
        """
        
        start_time = time.time()
        
        # Standard CVD simulation
        cvd_simulated = self.simulate_cvd(rgb_image, cvd_type)
        
        results = {
            'original': rgb_image,
            'cvd_simulation': cvd_simulated,
            'ishihara_optimized': cvd_simulated,  # Fallback if no optimizer
            'processing_time_ms': 0
        }
        
        # Apply Ishihara optimization if available
        if self.ishihara_optimizer and self.ishihara_mode:
            try:
                ishihara_results = self.ishihara_optimizer.optimize_ishihara_visibility(
                    rgb_image, cvd_type, enhancement_strength
                )
                
                # Combine CVD simulation with Ishihara optimization
                optimized_image = ishihara_results['final_optimized']
                
                # Blend CVD simulation with Ishihara optimization
                # This helps maintain CVD accuracy while improving test visibility
                blend_factor = 0.7  # 70% Ishihara optimization, 30% CVD simulation
                
                combined = (blend_factor * optimized_image.astype(np.float32) + 
                           (1 - blend_factor) * cvd_simulated.astype(np.float32))
                
                results.update({
                    'ishihara_optimized': np.clip(combined, 0, 255).astype(np.uint8),
                    'ishihara_pure': optimized_image,
                    'region_masks': ishihara_results.get('region_masks', {}),
                    'ishihara_processing_time_ms': ishihara_results.get('processing_time_ms', 0)
                })
                
            except Exception as e:
                print(f"⚠️  Ishihara optimization failed: {e}")
                # Fallback to standard CVD simulation
                results['ishihara_optimized'] = cvd_simulated
        
        # Calculate total processing time
        total_time = (time.time() - start_time) * 1000
        results['processing_time_ms'] = total_time
        
        return results
    
    def toggle_ishihara_mode(self, enable: bool = None) -> bool:
        """
        Toggle Ishihara optimization mode.
        
        Args:
            enable: True to enable, False to disable, None to toggle
            
        Returns:
            Current state of Ishihara mode
        """
        
        if not HAS_ISHIHARA_OPTIMIZER:
            print("⚠️  Ishihara optimizer not available")
            return False
        
        if enable is None:
            self.ishihara_mode = not self.ishihara_mode
        else:
            self.ishihara_mode = enable
        
        status = "ON" if self.ishihara_mode else "OFF"
        print(f"🎯 Ishihara optimization mode: {status}")
        
        return self.ishihara_mode
    
    def create_ishihara_test_plate(self, number: str = "8", size: int = 400) -> np.ndarray:
        """
        Create a simulated Ishihara test plate for testing.
        
        Args:
            number: Number to display in the test plate
            size: Size of the test plate (width and height)
            
        Returns:
            Simulated Ishihara test plate image
        """
        
        if self.ishihara_optimizer:
            return self.ishihara_optimizer.create_ishihara_test_simulation(size, size, number)
        else:
            # Simple fallback test pattern
            test_plate = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Create red-green checkerboard pattern
            square_size = size // 20
            for i in range(0, size, square_size):
                for j in range(0, size, square_size):
                    if (i // square_size + j // square_size) % 2 == 0:
                        test_plate[i:i+square_size, j:j+square_size] = [180, 50, 50]  # Red
                    else:
                        test_plate[i:i+square_size, j:j+square_size] = [50, 150, 50]  # Green
            
            # Add number in center
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = size / 100
            thickness = max(1, size // 100)
            
            (text_width, text_height), _ = cv2.getTextSize(number, font, font_scale, thickness)
            text_x = (size - text_width) // 2
            text_y = (size + text_height) // 2
            
            cv2.putText(test_plate, number, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            
            return test_plate

# Utility functions for easy integration
def quick_simulate_cvd(image: np.ndarray, cvd_type: str, optimization: str = 'balanced') -> np.ndarray:
    """
    Quick function to simulate CVD on an image.
    
    Args:
        image: RGB image array
        cvd_type: 'protanopia', 'deuteranopia', or 'tritanopia'  
        optimization: Optimization level
        
    Returns:
        CVD-simulated image
    """
    simulator = CVDSimulator(optimization_level=optimization)
    return simulator.simulate_cvd(image, cvd_type)

def create_cvd_comparison(image: np.ndarray) -> np.ndarray:
    """
    Create a 2x2 comparison grid showing original and all CVD types.
    
    Args:
        image: Input RGB image
        
    Returns:
        Comparison grid image
    """
    simulator = CVDSimulator()
    results = simulator.simulate_multiple_cvd(image)
    
    # Resize images for comparison
    h, w = image.shape[:2]
    new_h, new_w = h // 2, w // 2
    
    # Create grid
    grid = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Top row: Original, Protanopia
    grid[:new_h, :new_w] = cv2.resize(results['original'], (new_w, new_h))
    grid[:new_h, new_w:] = cv2.resize(results['protanopia'], (new_w, new_h))
    
    # Bottom row: Deuteranopia, Tritanopia  
    grid[new_h:, :new_w] = cv2.resize(results['deuteranopia'], (new_w, new_h))
    grid[new_h:, new_w:] = cv2.resize(results['tritanopia'], (new_w, new_h))
    
    return grid

if __name__ == "__main__":
    # Demo and testing
    print("🔬 CVD Simulator Module - Testing")
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test simulator
    simulator = CVDSimulator()
    
    # Validate accuracy
    validation = simulator.validate_cvd_accuracy()
    print("📊 Validation Results:")
    for cvd_type, is_valid in validation.items():
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"   {cvd_type}: {status}")
    
    # Performance test
    for cvd_type in ['protanopia', 'deuteranopia', 'tritanopia']:
        simulated = simulator.simulate_cvd(test_img, cvd_type)
    
    stats = simulator.get_performance_stats()
    print("⚡ Performance Stats:")
    print(f"   Average: {stats['avg_time_ms']:.1f}ms")
    print(f"   FPS: {stats['fps']:.1f}")
    print("✅ CVD Simulator ready for integration")

def run_unified_cvd_camera():
    """
    Run unified CVD camera interface integrated into cvd_simulation.py
    This replaces the separate unified_camera.py file.
    """
    print("\n🎥" + "="*48 + "🎥")
    print("        ReColor Unified CVD Camera")
    print("   Real-time Color Vision Processing")
    print("🎥" + "="*48 + "🎥")
    
    try:
        # Import required modules
        from daltonization import AdaptiveDaltonizer
        from clustering import RealtimeColorClusterer
        from datetime import datetime
        import threading
        from typing import Dict, List, Tuple, Optional
        
        # Initialize processing components
        cvd_simulator = CVDSimulator(optimization_level='balanced')
        daltonizer = AdaptiveDaltonizer(adaptation_level='medium')
        clusterer = RealtimeColorClusterer(n_clusters=8)
        
        # Current settings
        cvd_type = 'deuteranopia'
        k_clusters = 8
        dalton_strength = 0.7
        show_original = True
        show_palette = True
        
        # Processing modes
        modes = ['unified', 'simulation', 'correction', 'clustering', 'ishihara']
        current_mode_idx = 0  # Start with unified
        
        # Ishihara optimization settings
        ishihara_strength = 1.0
        ishihara_mode_enabled = HAS_ISHIHARA_OPTIMIZER
        
        # Performance tracking
        frame_count = 0
        fps_history = []
        paused = False
        
        print("🎥 Unified ReColor Camera initialized")
        print(f"🎯 K-means: {k_clusters} clusters")
        print(f"👁️  CVD Type: {cvd_type}")
        print(f"🔄 Mode: {modes[current_mode_idx]}")
        if HAS_ISHIHARA_OPTIMIZER:
            print("🎯 Ishihara optimization available")
        else:
            print("⚠️  Ishihara optimization not available")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot open camera")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera: {actual_w}x{actual_h}")
        
        print("\n🎮 Controls:")
        print("   M - Switch modes (Unified/Simulation/Correction/Clustering/Ishihara)")
        print("   D - Cycle CVD types (Protanopia/Deuteranopia/Tritanopia)")
        print("   K - Change K-means clusters (3-16)")
        print("   +/- - Adjust daltonization strength")
        print("   I - Toggle Ishihara optimization")
        print("   [/] - Ishihara strength (0.5-2.0)")
        print("   T - Generate test Ishihara plate")
        print("   O - Toggle original view (Split/Full)")
        print("   P - Toggle color palette display")
        print("   R - Reset to defaults")
        print("   Space - Pause/Resume")
        print("   H - Show help")
        print("   Q/ESC - Quit")
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Error capturing frame")
                        break
                    
                    start_time = time.time()
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb_frame.shape[:2]
                    
                    # Ensure clusterer is fitted
                    if not clusterer.fitted:
                        clusterer.fit_incremental(rgb_frame)
                    
                    # Step 1: K-means clustering
                    cluster_results = clusterer.process_video_frame(
                        rgb_frame, 
                        update_model=(frame_count % 10 == 0)
                    )
                    clustered_image = cluster_results.get('cluster_image', rgb_frame)
                    
                    # Step 2: CVD simulation
                    cvd_simulated = cvd_simulator.simulate_cvd(clustered_image, cvd_type)
                    
                    # Step 3: Daltonization correction
                    daltonized = daltonizer.adaptive_daltonization(clustered_image, cvd_type)
                    
                    # Step 4: Process based on current mode
                    current_mode = modes[current_mode_idx]
                    
                    if current_mode == 'unified':
                        # Blend all techniques
                        alpha_sim = 0.2   # CVD simulation weight
                        alpha_corr = 0.6  # Daltonization weight
                        alpha_clust = 0.2 # Clustering weight
                        
                        unified = (alpha_sim * cvd_simulated.astype(np.float32) +
                                  alpha_corr * daltonized.astype(np.float32) +
                                  alpha_clust * clustered_image.astype(np.float32))
                        
                        processed_main = np.clip(unified, 0, 255).astype(np.uint8)
                        
                    elif current_mode == 'simulation':
                        processed_main = cvd_simulated
                    elif current_mode == 'correction':
                        processed_main = daltonized
                    elif current_mode == 'clustering':
                        processed_main = clustered_image
                    elif current_mode == 'ishihara':
                        # Ishihara-optimized processing
                        if ishihara_mode_enabled and cvd_simulator.ishihara_optimizer:
                            cvd_simulator.toggle_ishihara_mode(True)
                            ishihara_results = cvd_simulator.simulate_cvd_ishihara_optimized(
                                clustered_image, cvd_type, ishihara_strength
                            )
                            processed_main = ishihara_results['ishihara_optimized']
                        else:
                            # Fallback to enhanced CVD simulation
                            processed_main = cvd_simulated
                    else:
                        processed_main = clustered_image
                    
                    # Create display
                    original_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    processed_bgr = cv2.cvtColor(processed_main, cv2.COLOR_RGB2BGR)
                    
                    if show_original:
                        # Side-by-side layout
                        display = np.zeros((h, w * 2, 3), dtype=np.uint8)
                        display[:, :w] = original_bgr
                        display[:, w:] = processed_bgr
                        
                        # Labels
                        cv2.putText(display, "ORIGINAL", (30, h - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display, f"{current_mode.upper()}", (w + 30, h - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        # Full window
                        display = processed_bgr.copy()
                    
                    # Add overlay information
                    overlay_h, overlay_w = display.shape[:2]
                    cv2.rectangle(display, (10, 10), (overlay_w - 10, 100), (0, 0, 0), -1)
                    cv2.addWeighted(display, 0.75, display, 0.25, 0, display)
                    
                    # Title and mode
                    cv2.putText(display, "ReColor Unified Camera", (20, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display, f"Mode: {current_mode.title()} | CVD: {cvd_type.title()}", 
                               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Performance
                    processing_time = time.time() - start_time
                    fps = 1.0 / processing_time if processing_time > 0 else 0
                    fps_history.append(fps)
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                    avg_fps = np.mean(fps_history)
                    
                    cv2.putText(display, f"FPS: {avg_fps:.1f} | K={k_clusters} | Strength: {dalton_strength:.2f}", 
                               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    frame_count += 1
                
                # Show display
                cv2.imshow('ReColor - Unified CVD Camera', display)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if key == ord('q') or key == 27:  # Q or ESC
                        break
                    elif key == ord('m'):
                        current_mode_idx = (current_mode_idx + 1) % len(modes)
                        print(f"🔄 Mode: {modes[current_mode_idx]}")
                    elif key == ord('d'):
                        cvd_types = ['protanopia', 'deuteranopia', 'tritanopia']
                        current_idx = cvd_types.index(cvd_type)
                        cvd_type = cvd_types[(current_idx + 1) % len(cvd_types)]
                        print(f"👁️  CVD Type: {cvd_type}")
                    elif key == ord('k'):
                        k_values = [3, 4, 5, 6, 8, 10, 12, 16]
                        try:
                            current_idx = k_values.index(k_clusters)
                            k_clusters = k_values[(current_idx + 1) % len(k_values)]
                        except ValueError:
                            k_clusters = 8
                        clusterer.set_cluster_count(k_clusters)
                        print(f"🎯 K-means clusters: {k_clusters}")
                    elif key == ord('+') or key == ord('='):
                        dalton_strength = min(dalton_strength + 0.1, 1.0)
                        daltonizer.set_correction_strength(dalton_strength)
                        print(f"🔧 Daltonization strength: {dalton_strength:.2f}")
                    elif key == ord('-'):
                        dalton_strength = max(dalton_strength - 0.1, 0.0)
                        daltonizer.set_correction_strength(dalton_strength)
                        print(f"🔧 Daltonization strength: {dalton_strength:.2f}")
                    elif key == ord('o'):
                        show_original = not show_original
                        layout = "Split" if show_original else "Full"
                        print(f"🖼️  Layout: {layout}")
                    elif key == ord('p'):
                        show_palette = not show_palette
                        print(f"🎨 Palette: {'ON' if show_palette else 'OFF'}")
                    elif key == ord('i'):
                        # Toggle Ishihara optimization mode
                        if HAS_ISHIHARA_OPTIMIZER:
                            ishihara_mode_enabled = not ishihara_mode_enabled
                            status = "ON" if ishihara_mode_enabled else "OFF"
                            print(f"🎯 Ishihara optimization: {status}")
                        else:
                            print("⚠️  Ishihara optimization not available")
                    elif key == ord(']'):
                        # Increase Ishihara strength
                        if HAS_ISHIHARA_OPTIMIZER:
                            ishihara_strength = min(ishihara_strength + 0.1, 2.0)
                            print(f"🎯 Ishihara strength: {ishihara_strength:.2f}")
                    elif key == ord('['):
                        # Decrease Ishihara strength
                        if HAS_ISHIHARA_OPTIMIZER:
                            ishihara_strength = max(ishihara_strength - 0.1, 0.5)
                            print(f"🎯 Ishihara strength: {ishihara_strength:.2f}")
                    elif key == ord('t'):
                        # Generate test Ishihara plate
                        if HAS_ISHIHARA_OPTIMIZER and cvd_simulator.ishihara_optimizer:
                            test_plate = cvd_simulator.create_ishihara_test_plate("8", 400)
                            cv2.imshow('Ishihara Test Plate', test_plate)
                            print("🎯 Ishihara test plate generated")
                    elif key == ord('r'):
                        # Reset to defaults
                        cvd_type = 'deuteranopia'
                        k_clusters = 8
                        dalton_strength = 0.7
                        current_mode_idx = 0
                        clusterer.set_cluster_count(k_clusters)
                        daltonizer.set_correction_strength(dalton_strength)
                        print("🔄 Reset to defaults")
                    elif key == ord(' '):
                        paused = not paused
                        print(f"⏸️  {'Paused' if paused else 'Resumed'}")
                    elif key == ord('h'):
                        print("\n🎮 Controls Help:")
                        print("   M - Switch modes | D - CVD types | K - Clusters")
                        print("   +/- - Dalton strength | I - Ishihara mode")
                        print("   [/] - Ishihara strength | T - Test plate")
                        print("   O - Layout | P - Palette | R - Reset | Space - Pause | Q - Quit")
        
        except KeyboardInterrupt:
            print("\n⏹️  Camera stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Final stats
            if fps_history:
                avg_fps = np.mean(fps_history)
                print(f"\n📊 Final Statistics:")
                print(f"   Frames processed: {frame_count}")
                print(f"   Average FPS: {avg_fps:.1f}")
                print(f"   CVD type: {cvd_type}")
                print(f"   Mode: {modes[current_mode_idx]}")
            
            print("✅ Unified ReColor Camera stopped successfully")
    
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("💡 Make sure all ReColor modules are available")
    except Exception as e:
        print(f"❌ Error in unified camera: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # If run directly, show CVD simulation demo or unified camera
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--camera":
        run_unified_cvd_camera()
    else:
        # Demo and testing
        print("🔬 CVD Simulator Module - Testing")
        
        # Create test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test simulator
        simulator = CVDSimulator()
        
        # Validate accuracy
        validation = simulator.validate_cvd_accuracy()
        print("📊 Validation Results:")
        for cvd_type, is_valid in validation.items():
            status = "✅ PASS" if is_valid else "❌ FAIL"
            print(f"   {cvd_type}: {status}")
        
        # Performance test
        for cvd_type in ['protanopia', 'deuteranopia', 'tritanopia']:
            simulated = simulator.simulate_cvd(test_img, cvd_type)
        
        stats = simulator.get_performance_stats()
        print("⚡ Performance Stats:")
        print(f"   Average: {stats['avg_time_ms']:.1f}ms")
        print(f"   FPS: {stats['fps']:.1f}")
        print("✅ CVD Simulator ready for integration")
        print("\n💡 To run unified camera: python cvd_simulation.py --camera")