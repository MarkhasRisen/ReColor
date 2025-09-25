"""
Real-Time Demo Script (No Camera Required)
==========================================

Demonstrates the real-time processing pipeline using synthetic video.
Perfect for testing when no camera is available.

Author: AI Assistant
Date: September 2025
"""

import cv2
import tensorflow as tf
import numpy as np
import time
import math
from typing import Tuple


class RealTimeDemo:
    """Demo class for real-time processing without camera."""
    
    def __init__(self):
        """Initialize demo processor."""
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        self.frame_delay = 1.0 / self.fps
        
        # Processing parameters
        self.current_k = 8
        self.current_deficiency = 'deuteranopia'
        self.k_options = [4, 8, 16, 32]
        self.deficiency_options = ['protanopia', 'deuteranopia', 'tritanopia', 'protanomaly', 'deuteranomaly']
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        self.is_running = False
        self.show_fps = True
        
        print("Real-Time Demo initialized (synthetic video)")
    
    @tf.function
    def tf_normalize_frame(self, frame: tf.Tensor) -> tf.Tensor:
        """Normalize frame to [0, 1] range."""
        return tf.clip_by_value(tf.cast(frame, tf.float32) / 255.0, 0.0, 1.0)
    
    @tf.function
    def tf_denormalize_frame(self, frame: tf.Tensor) -> tf.Tensor:
        """Convert normalized frame back to [0, 255] range."""
        clipped = tf.clip_by_value(frame, 0.0, 1.0)
        return tf.cast(clipped * 255.0, tf.uint8)
    
    def create_synthetic_frame(self, frame_number: int) -> tf.Tensor:
        """
        Create a synthetic animated frame for testing.
        
        Args:
            frame_number (int): Current frame number for animation
            
        Returns:
            tf.Tensor: Synthetic frame tensor [height, width, 3]
        """
        # Create coordinate grids
        x = tf.linspace(0.0, 1.0, self.frame_width)
        y = tf.linspace(0.0, 1.0, self.frame_height)
        X, Y = tf.meshgrid(x, y)
        
        # Animation parameter
        t = frame_number * 0.05
        
        # Create animated patterns
        # Ripple pattern
        distance = tf.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
        ripple = tf.sin(distance * 20.0 + t * 3.0) * 0.5 + 0.5
        
        # Wave patterns
        wave_x = tf.sin(X * 8.0 + t) * 0.3 + 0.5
        wave_y = tf.cos(Y * 6.0 + t * 0.8) * 0.3 + 0.5
        
        # Rotating spiral
        angle = tf.atan2(Y - 0.5, X - 0.5) + t
        spiral = tf.sin(angle * 4.0 + distance * 15.0) * 0.4 + 0.6
        
        # Combine patterns for RGB channels
        R = ripple * wave_x
        G = wave_y * spiral
        B = tf.sin((X + Y) * 10.0 + t * 2.0) * 0.4 + 0.6
        
        # Stack channels
        frame = tf.stack([R, G, B], axis=-1)
        
        # Add some noise for realism
        noise = tf.random.normal(tf.shape(frame), mean=0.0, stddev=0.02)
        frame = tf.clip_by_value(frame + noise, 0.0, 1.0)
        
        return frame
    
    @tf.function
    def tf_kmeans_demo(self, frame: tf.Tensor, k: int) -> tf.Tensor:
        """
        Demo K-Means clustering placeholder.
        
        TODO: Replace with actual K-Means implementation.
        """
        # Simple quantization for demo
        k_float = tf.cast(k, tf.float32)
        quantized = tf.round(frame * (k_float - 1.0)) / (k_float - 1.0)
        return quantized
    
    @tf.function
    def daltonize_demo(self, frame: tf.Tensor, deficiency: str = 'deuteranopia') -> tf.Tensor:
        """
        Demo Daltonization placeholder.
        
        TODO: Replace with actual Daltonization implementation.
        """
        # Simple color adjustment for demo
        if deficiency == 'deuteranopia':
            # Enhance green channel for complete green deficiency
            corrected = tf.stack([
                frame[:, :, 0],  # Red unchanged
                tf.clip_by_value(frame[:, :, 1] * 1.5, 0.0, 1.0),  # Strong green enhancement
                frame[:, :, 2]   # Blue unchanged
            ], axis=-1)
        elif deficiency == 'protanopia':
            # Enhance red channel for complete red deficiency
            corrected = tf.stack([
                tf.clip_by_value(frame[:, :, 0] * 1.5, 0.0, 1.0),  # Strong red enhancement
                frame[:, :, 1],  # Green unchanged
                frame[:, :, 2]   # Blue unchanged
            ], axis=-1)
        elif deficiency == 'tritanopia':
            # Enhance blue channel for complete blue deficiency
            corrected = tf.stack([
                frame[:, :, 0],  # Red unchanged
                frame[:, :, 1],  # Green unchanged
                tf.clip_by_value(frame[:, :, 2] * 1.5, 0.0, 1.0)   # Strong blue enhancement
            ], axis=-1)
        elif deficiency == 'protanomaly':
            # Moderate enhancement for reduced red sensitivity
            corrected = tf.stack([
                tf.clip_by_value(frame[:, :, 0] * 1.2, 0.0, 1.0),  # Moderate red enhancement
                frame[:, :, 1],  # Green unchanged
                frame[:, :, 2]   # Blue unchanged
            ], axis=-1)
        elif deficiency == 'deuteranomaly':
            # Moderate enhancement for reduced green sensitivity
            corrected = tf.stack([
                frame[:, :, 0],  # Red unchanged
                tf.clip_by_value(frame[:, :, 1] * 1.2, 0.0, 1.0),  # Moderate green enhancement
                frame[:, :, 2]   # Blue unchanged
            ], axis=-1)
        else:  # fallback
            corrected = frame
        
        return corrected
    
    def process_frame(self, frame: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Process frame through the complete pipeline."""
        # Keep original
        original = frame
        
        # Apply Daltonization
        daltonized = self.daltonize_demo(frame, self.current_deficiency)
        
        # Apply K-Means clustering
        clustered = self.tf_kmeans_demo(daltonized, self.current_k)
        
        return original, daltonized, clustered
    
    def tf_to_cv2(self, tf_frame: tf.Tensor) -> np.ndarray:
        """Convert TensorFlow tensor to OpenCV format."""
        # Denormalize
        denormalized = self.tf_denormalize_frame(tf_frame)
        
        # Convert to numpy
        numpy_frame = denormalized.numpy()
        
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(numpy_frame, cv2.COLOR_RGB2BGR)
        
        return bgr_frame
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def add_info_overlay(self, cv2_frame: np.ndarray, window_name: str) -> np.ndarray:
        """Add information overlay to frame."""
        if self.show_fps:
            fps_text = f"{window_name}: {self.current_fps:.1f} FPS"
            cv2.putText(cv2_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if window_name == "Original":
            info_text = f"K={self.current_k}, Deficiency={self.current_deficiency}"
            cv2.putText(cv2_frame, info_text, (10, cv2_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add demo indicator
            demo_text = "DEMO MODE (Synthetic Video)"
            cv2.putText(cv2_frame, demo_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return cv2_frame
    
    def setup_windows(self):
        """Setup display windows."""
        windows = ['Original', 'Daltonized', 'K-Means Clustered']
        positions = [(100, 100), (750, 100), (1400, 100)]
        
        for window, pos in zip(windows, positions):
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(window, pos[0], pos[1])
        
        print("Demo windows configured")
    
    def handle_controls(self, key: int) -> bool:
        """Handle keyboard controls."""
        if key == ord('q'):
            return False
        elif key == ord('k'):
            current_index = self.k_options.index(self.current_k)
            self.current_k = self.k_options[(current_index + 1) % len(self.k_options)]
            print(f"K-Means clusters: {self.current_k}")
        elif key == ord('d'):
            current_index = self.deficiency_options.index(self.current_deficiency)
            self.current_deficiency = self.deficiency_options[(current_index + 1) % len(self.deficiency_options)]
            print(f"Deficiency type: {self.current_deficiency}")
        elif key == ord('f'):
            self.show_fps = not self.show_fps
            print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
        elif key == ord('h'):
            self.print_help()
        
        return True
    
    def print_help(self):
        """Print help information."""
        print("\nDemo Controls:")
        print("  Q - Quit demo")
        print("  K - Cycle K-Means clusters (4, 8, 16, 32)")
        print("  D - Cycle deficiency types")
        print("  F - Toggle FPS display")
        print("  H - Show this help")
        print()
    
    def run_demo(self, duration_frames: int = 1000):
        """
        Run the real-time demo.
        
        Args:
            duration_frames (int): Number of frames to run (0 = infinite)
        """
        print("TensorFlow Real-Time Processing Demo")
        print("=" * 50)
        print("Synthetic video mode - no camera required")
        self.print_help()
        
        # Setup display
        self.setup_windows()
        
        self.is_running = True
        frame_count = 0
        
        try:
            while self.is_running and (duration_frames == 0 or frame_count < duration_frames):
                start_time = time.time()
                
                # Create synthetic frame
                tf_frame = self.create_synthetic_frame(frame_count)
                
                # Process frame
                original, daltonized, clustered = self.process_frame(tf_frame)
                
                # Convert to OpenCV format
                original_cv2 = self.tf_to_cv2(original)
                daltonized_cv2 = self.tf_to_cv2(daltonized)
                clustered_cv2 = self.tf_to_cv2(clustered)
                
                # Add overlays
                original_cv2 = self.add_info_overlay(original_cv2, "Original")
                daltonized_cv2 = self.add_info_overlay(daltonized_cv2, "Daltonized")
                clustered_cv2 = self.add_info_overlay(clustered_cv2, "Clustered")
                
                # Display frames
                cv2.imshow('Original', original_cv2)
                cv2.imshow('Daltonized', daltonized_cv2)
                cv2.imshow('K-Means Clustered', clustered_cv2)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_controls(key):
                    break
                
                # Update performance
                self.update_fps()
                frame_count += 1
                
                # Print status periodically
                if frame_count % 100 == 0:
                    print(f"Demo frame {frame_count}, FPS: {self.current_fps:.1f}")
                
                # Frame rate control
                processing_time = time.time() - start_time
                if processing_time < self.frame_delay:
                    time.sleep(self.frame_delay - processing_time)
        
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        except Exception as e:
            print(f"Demo error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("Cleaning up demo...")
            self.is_running = False
            cv2.destroyAllWindows()
            print(f"Demo completed. Processed {frame_count} frames.")


def main():
    """Main function for demo."""
    print("TensorFlow Real-Time Image Processing Demo")
    print("=" * 60)
    print("This demo runs without a camera using synthetic video.")
    print("Perfect for testing the processing pipeline!")
    print()
    
    try:
        demo = RealTimeDemo()
        demo.run_demo()  # Run indefinitely until quit
        
    except Exception as e:
        print(f"Failed to start demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()