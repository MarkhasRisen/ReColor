"""
Optimized Real-Time Image Processor
===================================

High-performance real-time image processing with TensorFlow operations.
Includes performance monitoring, configuration management, and optimizations.

Author: AI Assistant
Date: September 2025
"""

import cv2
import tensorflow as tf
import numpy as np
import time
import os
from typing import Tuple, Optional
from realtime_config import (
    CAMERA_CONFIG, REALTIME_CONFIG, DISPLAY_CONFIG, 
    PERFORMANCE_CONFIG, CONTROLS_CONFIG, COLOR_MATRICES,
    get_optimized_tf_config, print_realtime_config
)


class OptimizedRealTimeProcessor:
    """
    Optimized real-time image processor with TensorFlow operations.
    Designed for maximum performance and minimal latency.
    """
    
    def __init__(self, camera_id: Optional[int] = None):
        """Initialize the optimized processor."""
        # Configure TensorFlow for optimal performance
        get_optimized_tf_config()
        
        # Camera settings
        self.camera_id = camera_id or CAMERA_CONFIG['default_camera_id']
        self.target_fps = CAMERA_CONFIG['target_fps']
        self.frame_delay = 1.0 / self.target_fps
        
        # Processing parameters
        self.current_k = REALTIME_CONFIG['default_k']
        self.current_deficiency = REALTIME_CONFIG['default_deficiency']
        self.k_options = REALTIME_CONFIG['k_options']
        self.deficiency_options = REALTIME_CONFIG['deficiency_options']
        
        # Check for environment variable to set initial deficiency
        import os
        env_deficiency = os.environ.get('REALTIME_DEFICIENCY')
        if env_deficiency and env_deficiency in self.deficiency_options:
            self.current_deficiency = env_deficiency
            print(f"Initial deficiency set from launcher: {env_deficiency}")
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.frame_times = []
        self.processing_times = []
        
        # Control flags
        self.is_running = False
        self.show_fps = DISPLAY_CONFIG['show_fps']
        self.show_info = DISPLAY_CONFIG['show_info_overlay']
        
        # Pre-compile TensorFlow matrices for performance
        self._compile_tf_matrices()
        
        print("Optimized Real-Time Processor initialized")
    
    def _compile_tf_matrices(self):
        """Pre-compile TensorFlow matrices for optimal performance."""
        # Convert color transformation matrices to TensorFlow constants
        self.rgb_to_lms_matrix = tf.constant(COLOR_MATRICES['rgb_to_lms'], dtype=tf.float32)
        self.lms_to_rgb_matrix = tf.constant(COLOR_MATRICES['lms_to_rgb'], dtype=tf.float32)
        
        # Compile deficiency simulation matrices
        self.sim_matrices = {}
        self.error_matrices = {}
        
        for deficiency in self.deficiency_options:
            self.sim_matrices[deficiency] = tf.constant(
                COLOR_MATRICES[f'{deficiency}_sim'], dtype=tf.float32
            )
            self.error_matrices[deficiency] = tf.constant(
                COLOR_MATRICES[f'{deficiency}_error'], dtype=tf.float32
            )
        
        print("TensorFlow matrices compiled for optimal performance")
    
    @tf.function
    def tf_normalize_frame(self, frame: tf.Tensor) -> tf.Tensor:
        """Fast frame normalization using tf.function compilation."""
        normalized = tf.cast(frame, tf.float32) / 255.0
        return tf.clip_by_value(normalized, 0.0, 1.0)
    
    @tf.function
    def tf_denormalize_frame(self, frame: tf.Tensor) -> tf.Tensor:
        """Fast frame denormalization using tf.function compilation."""
        clipped = tf.clip_by_value(frame, 0.0, 1.0)
        return tf.cast(clipped * 255.0, tf.uint8)
    
    def cv2_to_tf(self, cv2_frame: np.ndarray) -> tf.Tensor:
        """Optimized OpenCV to TensorFlow conversion."""
        # Convert BGR to RGB efficiently
        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        
        # Direct tensor conversion
        tf_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)
        
        # Normalize
        return self.tf_normalize_frame(tf_frame)
    
    def tf_to_cv2(self, tf_frame: tf.Tensor) -> np.ndarray:
        """Optimized TensorFlow to OpenCV conversion."""
        # Denormalize
        denormalized = self.tf_denormalize_frame(tf_frame)
        
        # Convert to numpy
        numpy_frame = denormalized.numpy()
        
        # Convert RGB to BGR
        return cv2.cvtColor(numpy_frame, cv2.COLOR_RGB2BGR)
    
    @tf.function
    def tf_kmeans_optimized(self, frame: tf.Tensor, k: tf.Tensor) -> tf.Tensor:
        """
        Optimized K-Means clustering for real-time performance.
        
        TODO: Implement efficient K-Means with TensorFlow operations.
        This version is optimized for speed over accuracy for real-time use.
        """
        # Current placeholder: Fast quantization
        # Replace with actual K-Means implementation
        k_float = tf.cast(k, tf.float32)
        quantized = tf.round(frame * (k_float - 1.0)) / (k_float - 1.0)
        return quantized
    
    @tf.function
    def tf_daltonize_optimized(self, frame: tf.Tensor, deficiency_index: tf.Tensor) -> tf.Tensor:
        """
        Optimized Daltonization using pre-compiled matrices.
        
        TODO: Implement actual Daltonization with matrix operations.
        This version provides the structure for efficient implementation.
        """
        # Placeholder implementation with conditional logic
        # Replace with actual Daltonization algorithm
        
        # Simple placeholder based on deficiency index
        if deficiency_index == 0:  # protanopia - complete red deficiency
            corrected = tf.stack([
                tf.clip_by_value(frame[:, :, 0] * 1.5, 0.0, 1.0),  # Strong red enhancement
                frame[:, :, 1],
                frame[:, :, 2]
            ], axis=-1)
        elif deficiency_index == 1:  # deuteranopia - complete green deficiency
            corrected = tf.stack([
                frame[:, :, 0],
                tf.clip_by_value(frame[:, :, 1] * 1.5, 0.0, 1.0),  # Strong green enhancement
                frame[:, :, 2]
            ], axis=-1)
        elif deficiency_index == 2:  # tritanopia - complete blue deficiency
            corrected = tf.stack([
                frame[:, :, 0],
                frame[:, :, 1],
                tf.clip_by_value(frame[:, :, 2] * 1.5, 0.0, 1.0)   # Strong blue enhancement
            ], axis=-1)
        elif deficiency_index == 3:  # protanomaly - reduced red sensitivity
            corrected = tf.stack([
                tf.clip_by_value(frame[:, :, 0] * 1.2, 0.0, 1.0),  # Moderate red enhancement
                frame[:, :, 1],
                frame[:, :, 2]
            ], axis=-1)
        elif deficiency_index == 4:  # deuteranomaly - reduced green sensitivity
            corrected = tf.stack([
                frame[:, :, 0],
                tf.clip_by_value(frame[:, :, 1] * 1.2, 0.0, 1.0),  # Moderate green enhancement
                frame[:, :, 2]
            ], axis=-1)
        else:  # fallback
            corrected = frame
        
        return corrected
    
    @tf.function
    def process_frame_optimized(self, frame: tf.Tensor, k: tf.Tensor, deficiency_index: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Optimized frame processing pipeline using tf.function."""
        # Keep original
        original = frame
        
        # Apply Daltonization
        daltonized = self.tf_daltonize_optimized(frame, deficiency_index)
        
        # Apply K-Means
        clustered = self.tf_kmeans_optimized(daltonized, k)
        
        return original, daltonized, clustered
    
    def update_performance_metrics(self, processing_time: float):
        """Update performance metrics."""
        self.fps_counter += 1
        self.processing_times.append(processing_time)
        
        # Keep only recent measurements
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        current_time = time.time()
        if current_time - self.fps_start_time >= PERFORMANCE_CONFIG['fps_update_interval']:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def add_performance_overlay(self, cv2_frame: np.ndarray, window_name: str) -> np.ndarray:
        """Add performance information overlay."""
        if not self.show_fps and not self.show_info:
            return cv2_frame
        
        y_offset = 30
        
        if self.show_fps:
            fps_text = f"{window_name}: {self.current_fps:.1f} FPS"
            cv2.putText(cv2_frame, fps_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG['font_scale'],
                       DISPLAY_CONFIG['text_color'], DISPLAY_CONFIG['font_thickness'])
            y_offset += 25
        
        if self.show_info and window_name == "Original":
            info_text = f"K={self.current_k}, Deficiency={self.current_deficiency}"
            cv2.putText(cv2_frame, info_text, (10, cv2_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG['font_scale'],
                       DISPLAY_CONFIG['info_color'], DISPLAY_CONFIG['font_thickness'])
            
            if self.processing_times:
                avg_time = sum(self.processing_times[-10:]) / min(len(self.processing_times), 10)
                time_text = f"Avg Process Time: {avg_time*1000:.1f}ms"
                cv2.putText(cv2_frame, time_text, (10, cv2_frame.shape[0] - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_CONFIG['font_scale'],
                           DISPLAY_CONFIG['info_color'], DISPLAY_CONFIG['font_thickness'])
        
        return cv2_frame
    
    def initialize_camera(self) -> cv2.VideoCapture:
        """Initialize camera with optimized settings."""
        print(f"Initializing camera {self.camera_id}...")
        
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['frame_height'])
        cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['target_fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_CONFIG['buffer_size'])
        
        # Verify settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps} FPS")
        
        return cap
    
    def setup_display_windows(self):
        """Setup optimized display windows."""
        windows = ['Original', 'Daltonized', 'Clustered']
        positions = list(DISPLAY_CONFIG['window_positions'].values())
        
        for window, pos in zip(windows, positions):
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(window, pos[0], pos[1])
        
        print("Display windows configured")
    
    def handle_controls(self, key: int) -> bool:
        """Handle keyboard controls."""
        if key == ord(CONTROLS_CONFIG['quit_key']):
            return False
        
        elif key == ord(CONTROLS_CONFIG['cycle_k_key']):
            current_index = self.k_options.index(self.current_k)
            self.current_k = self.k_options[(current_index + 1) % len(self.k_options)]
            print(f"K-Means clusters: {self.current_k}")
        
        elif key == ord(CONTROLS_CONFIG['cycle_deficiency_key']):
            current_index = self.deficiency_options.index(self.current_deficiency)
            self.current_deficiency = self.deficiency_options[(current_index + 1) % len(self.deficiency_options)]
            print(f"Deficiency type: {self.current_deficiency}")
        
        elif key == ord(CONTROLS_CONFIG['toggle_fps_key']):
            self.show_fps = not self.show_fps
            print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
        
        elif key == ord(CONTROLS_CONFIG.get('save_frame_key', 's')):
            print("Frame save requested (implement if needed)")
        
        elif key == ord(CONTROLS_CONFIG.get('reset_key', 'r')):
            self.current_k = REALTIME_CONFIG['default_k']
            self.current_deficiency = REALTIME_CONFIG['default_deficiency']
            print("Parameters reset to defaults")
        
        return True
    
    def run_optimized_processing(self):
        """Main optimized processing loop."""
        print_realtime_config()
        
        try:
            # Initialize camera
            cap = self.initialize_camera()
            
            # Setup display
            self.setup_display_windows()
            
            # Pre-compile TensorFlow constants for current parameters
            k_tensor = tf.constant(self.current_k, dtype=tf.int32)
            deficiency_index = tf.constant(
                self.deficiency_options.index(self.current_deficiency), 
                dtype=tf.int32
            )
            
            self.is_running = True
            frame_count = 0
            
            print("Starting optimized real-time processing...")
            print("Press 'h' for help, 'q' to quit")
            
            while self.is_running:
                loop_start = time.time()
                
                # Capture frame
                ret, cv2_frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Convert to TensorFlow
                processing_start = time.time()
                tf_frame = self.cv2_to_tf(cv2_frame)
                
                # Update tensors if parameters changed
                k_tensor = tf.constant(self.current_k, dtype=tf.int32)
                deficiency_index = tf.constant(
                    self.deficiency_options.index(self.current_deficiency),
                    dtype=tf.int32
                )
                
                # Process frame
                original, daltonized, clustered = self.process_frame_optimized(
                    tf_frame, k_tensor, deficiency_index
                )
                
                processing_time = time.time() - processing_start
                
                # Convert back to OpenCV
                original_cv2 = self.tf_to_cv2(original)
                daltonized_cv2 = self.tf_to_cv2(daltonized)
                clustered_cv2 = self.tf_to_cv2(clustered)
                
                # Add overlays
                original_cv2 = self.add_performance_overlay(original_cv2, "Original")
                daltonized_cv2 = self.add_performance_overlay(daltonized_cv2, "Daltonized")
                clustered_cv2 = self.add_performance_overlay(clustered_cv2, "Clustered")
                
                # Display
                cv2.imshow('Original', original_cv2)
                cv2.imshow('Daltonized', daltonized_cv2)
                cv2.imshow('Clustered', clustered_cv2)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_controls(key):
                    break
                
                # Update performance metrics
                self.update_performance_metrics(processing_time)
                frame_count += 1
                
                # Print status periodically
                if frame_count % PERFORMANCE_CONFIG['status_print_interval'] == 0:
                    avg_process_time = sum(self.processing_times[-10:]) / min(len(self.processing_times), 10)
                    print(f"Frame {frame_count}: {self.current_fps:.1f} FPS, "
                          f"Process time: {avg_process_time*1000:.1f}ms")
                
                # Frame rate control
                loop_time = time.time() - loop_start
                if loop_time < self.frame_delay:
                    time.sleep(self.frame_delay - loop_time)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("Cleaning up...")
            self.is_running = False
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                print(f"Average processing time: {avg_time*1000:.2f}ms")
                print(f"Final FPS: {self.current_fps:.1f}")


def main():
    """Main function for optimized real-time processing."""
    print("TensorFlow Optimized Real-Time Image Processor")
    print("=" * 60)
    
    try:
        processor = OptimizedRealTimeProcessor()
        processor.run_optimized_processing()
    
    except Exception as e:
        print(f"Failed to start processor: {e}")


if __name__ == "__main__":
    main()