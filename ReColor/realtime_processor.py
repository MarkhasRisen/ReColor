"""
Real-Time Image Processing Pipeline with TensorFlow Operations
=============================================================

A real-time webcam processing system using only TensorFlow operations
for image processing with placeholders for K-Means clustering and Daltonization.

Author: AI Assistant
Date: September 2025
Dependencies: tensorflow>=2.0, opencv-python, numpy (minimal usage for cv2 interfacing only)
"""

import cv2
import tensorflow as tf
import numpy as np
import time
from typing import Tuple, Optional
import threading
import queue


class RealTimeImageProcessor:
    """
    Real-time image processing pipeline using TensorFlow operations.
    Captures webcam frames and applies Daltonization and K-Means clustering.
    """
    
    def __init__(self, camera_id: int = 0, target_fps: int = 30):
        """
        Initialize the real-time processor.
        
        Args:
            camera_id (int): Camera device ID (usually 0 for default camera)
            target_fps (int): Target frames per second for processing
        """
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Processing flags
        self.is_running = False
        self.show_fps = True
        
        # Frame dimensions (will be set when camera initializes)
        self.frame_width = None
        self.frame_height = None
        
        # Default processing parameters
        self.default_k = 8
        self.default_deficiency = 'deuteranopia'
        self.deficiency_options = ['protanopia', 'deuteranopia', 'tritanopia', 'protanomaly', 'deuteranomaly']
        
        # Check for environment variable to set initial deficiency
        import os
        env_deficiency = os.environ.get('REALTIME_DEFICIENCY')
        if env_deficiency and env_deficiency in self.deficiency_options:
            self.default_deficiency = env_deficiency
            print(f"Initial deficiency set from launcher: {env_deficiency}")
        
        print("Real-Time Image Processor initialized")
        print(f"Target FPS: {target_fps}")
    
    def tf_normalize_frame(self, frame: tf.Tensor) -> tf.Tensor:
        """
        Normalize frame to [0, 1] range using TensorFlow operations.
        
        Args:
            frame (tf.Tensor): Input frame tensor with values in [0, 255]
            
        Returns:
            tf.Tensor: Normalized frame tensor with values in [0, 1]
        """
        # Convert to float32 and normalize
        normalized = tf.cast(frame, tf.float32) / 255.0
        
        # Ensure values are in valid range
        normalized = tf.clip_by_value(normalized, 0.0, 1.0)
        
        return normalized
    
    def tf_denormalize_frame(self, frame: tf.Tensor) -> tf.Tensor:
        """
        Convert normalized frame back to [0, 255] range for display.
        
        Args:
            frame (tf.Tensor): Normalized frame tensor with values in [0, 1]
            
        Returns:
            tf.Tensor: Frame tensor with values in [0, 255] as uint8
        """
        # Clip to ensure valid range
        clipped = tf.clip_by_value(frame, 0.0, 1.0)
        
        # Convert to [0, 255] range and uint8
        denormalized = tf.cast(clipped * 255.0, tf.uint8)
        
        return denormalized
    
    def cv2_to_tf(self, cv2_frame: np.ndarray) -> tf.Tensor:
        """
        Convert OpenCV frame to TensorFlow tensor.
        
        Args:
            cv2_frame (np.ndarray): OpenCV frame in BGR format
            
        Returns:
            tf.Tensor: TensorFlow tensor in RGB format, normalized to [0, 1]
        """
        # Convert BGR to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to TensorFlow tensor
        tf_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)
        
        # Normalize to [0, 1]
        normalized_frame = self.tf_normalize_frame(tf_frame)
        
        return normalized_frame
    
    def tf_to_cv2(self, tf_frame: tf.Tensor) -> np.ndarray:
        """
        Convert TensorFlow tensor back to OpenCV format for display.
        
        Args:
            tf_frame (tf.Tensor): TensorFlow tensor in RGB format, normalized to [0, 1]
            
        Returns:
            np.ndarray: OpenCV frame in BGR format, values in [0, 255]
        """
        # Denormalize to [0, 255]
        denormalized = self.tf_denormalize_frame(tf_frame)
        
        # Convert to numpy array
        numpy_frame = denormalized.numpy()
        
        # Convert RGB back to BGR for OpenCV display
        bgr_frame = cv2.cvtColor(numpy_frame, cv2.COLOR_RGB2BGR)
        
        return bgr_frame
    
    def tf_resize_frame(self, frame: tf.Tensor, target_size: Tuple[int, int]) -> tf.Tensor:
        """
        Resize frame using TensorFlow operations.
        
        Args:
            frame (tf.Tensor): Input frame tensor
            target_size (Tuple[int, int]): Target (height, width)
            
        Returns:
            tf.Tensor: Resized frame tensor
        """
        resized = tf.image.resize(frame, target_size, method='bilinear')
        return resized
    
    def tf_kmeans(self, frame: tf.Tensor, k: int, max_iters: int = 10, tol: float = 1e-4) -> tf.Tensor:
        """
        Simple K-Means clustering using pure TensorFlow operations.
        
        Args:
            frame (tf.Tensor): Input frame tensor [height, width, channels] with values 0-1
            k (int): Number of clusters  
            max_iters (int): Maximum iterations
            tol (float): Convergence tolerance (unused in simple version)
            
        Returns:
            tf.Tensor: Clustered frame with same shape as input
        """
        original_shape = tf.shape(frame)
        height, width, channels = original_shape[0], original_shape[1], original_shape[2]
        
        # Reshape to [num_pixels, channels] for clustering
        pixels = tf.reshape(frame, [-1, channels])
        num_pixels = tf.shape(pixels)[0]
        
        # Initialize centroids randomly
        centroids = tf.random.uniform([k, channels], minval=0.0, maxval=1.0, dtype=tf.float32)
        
        # K-Means iterations
        for _ in range(max_iters):
            # Compute distances from each pixel to each centroid
            expanded_pixels = tf.expand_dims(pixels, 1)  # [num_pixels, 1, channels]
            expanded_centroids = tf.expand_dims(centroids, 0)  # [1, k, channels]
            
            squared_distances = tf.reduce_sum(
                tf.square(expanded_pixels - expanded_centroids), axis=2
            )  # [num_pixels, k]
            
            # Assign each pixel to nearest centroid
            assignments = tf.argmin(squared_distances, axis=1)  # [num_pixels]
            
            # Update centroids using tf.boolean_mask and tf.reduce_mean
            new_centroids = []
            for cluster_id in range(k):
                # Find pixels assigned to this cluster
                mask = tf.equal(assignments, cluster_id)
                cluster_pixels = tf.boolean_mask(pixels, mask)
                
                # Compute new centroid (handle empty clusters)
                cluster_size = tf.shape(cluster_pixels)[0]
                new_centroid = tf.cond(
                    cluster_size > 0,
                    lambda: tf.reduce_mean(cluster_pixels, axis=0),
                    lambda: tf.random.uniform([channels], minval=0.0, maxval=1.0)
                )
                new_centroids.append(new_centroid)
            
            centroids = tf.stack(new_centroids)
        
        # Create final clustered image
        final_assignments = tf.argmin(squared_distances, axis=1)
        clustered_pixels = tf.gather(centroids, final_assignments)
        clustered_frame = tf.reshape(clustered_pixels, original_shape)
        
        return clustered_frame
    
    def daltonize(self, frame: tf.Tensor, deficiency: str = 'deuteranopia') -> tf.Tensor:
        """
        Simple daltonization to simulate and correct color vision deficiency using LMS color space.
        
        Args:
            frame (tf.Tensor): Input RGB frame tensor [height, width, 3] with values 0-1
            deficiency (str): Type of color vision deficiency
                            ('protanopia', 'deuteranopia', 'tritanopia')
                            
        Returns:
            tf.Tensor: Color-corrected frame tensor
        """
        # Validate deficiency type and set default
        valid_deficiencies = ['protanopia', 'deuteranopia', 'tritanopia', 'protanomaly', 'deuteranomaly']
        if deficiency not in valid_deficiencies:
            deficiency = 'deuteranopia'
            
        # Ensure RGB frame
        if frame.shape[-1] != 3:
            return frame
        
        original_shape = tf.shape(frame)
        
        # Flatten for matrix operations
        pixels = tf.reshape(frame, [-1, 3])
        
        # RGB to LMS transformation matrix (Hunt-Pointer-Estevez)
        rgb_to_lms = tf.constant([
            [0.31399022, 0.63951294, 0.04649755],
            [0.15537241, 0.75789446, 0.08670142],
            [0.01775239, 0.10944209, 0.87256922]
        ], dtype=tf.float32)
        
        # LMS to RGB transformation matrix
        lms_to_rgb = tf.constant([
            [ 5.47221206, -4.6419601,  0.16963708],
            [-1.1252419,   2.29317094, -0.1678952 ],
            [ 0.02980165, -0.19318073,  1.16364789]
        ], dtype=tf.float32)
        
        # Color vision deficiency simulation matrices
        deficiency_matrices = {
            'protanopia': tf.constant([
                [0.0, 1.05118294, -0.05116099],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=tf.float32),
            'deuteranopia': tf.constant([
                [1.0, 0.0, 0.0],
                [0.9513092, 0.0, 0.04866992],
                [0.0, 0.0, 1.0]
            ], dtype=tf.float32),
            'tritanopia': tf.constant([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-0.86744736, 1.86727089, 0.0]
            ], dtype=tf.float32),
            'protanomaly': tf.constant([
                [0.458064, 0.679578, -0.137642],
                [0.092785, 0.846313, 0.060902],
                [-0.007494, -0.016807, 1.024301]
            ], dtype=tf.float32),
            'deuteranomaly': tf.constant([
                [0.547494, 0.607765, -0.155259],
                [0.181692, 0.781742, 0.036566],
                [-0.010410, 0.027275, 0.983136]
            ], dtype=tf.float32)
        }
        
        # Convert RGB to LMS
        lms_pixels = tf.matmul(pixels, rgb_to_lms, transpose_b=True)
        
        # Apply deficiency simulation
        deficiency_matrix = deficiency_matrices.get(deficiency, deficiency_matrices['deuteranopia'])
        deficient_lms = tf.matmul(lms_pixels, deficiency_matrix, transpose_b=True)
        
        # Compute error
        error_lms = lms_pixels - deficient_lms
        
        # Apply correction matrix
        correction_matrix = tf.constant([
            [0.0, 0.0, 0.0],
            [0.7, 1.0, 0.0],
            [0.7, 0.0, 1.0]
        ], dtype=tf.float32)
        
        corrected_error = tf.matmul(error_lms, correction_matrix, transpose_b=True)
        daltonized_lms = deficient_lms + corrected_error
        
        # Convert back to RGB
        daltonized_rgb = tf.matmul(daltonized_lms, lms_to_rgb, transpose_b=True)
        daltonized_rgb = tf.clip_by_value(daltonized_rgb, 0.0, 1.0)
        
        # Reshape back to original shape
        daltonized_frame = tf.reshape(daltonized_rgb, original_shape)
        
        return daltonized_frame
    
    def process_frame(self, frame: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame (tf.Tensor): Input frame tensor [height, width, 3]
            
        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: (original, daltonized, clustered)
        """
        # Keep original for display
        original = frame
        
        # Apply Daltonization
        daltonized = self.daltonize(frame, deficiency=self.default_deficiency)
        
        # Apply K-Means clustering to daltonized frame
        clustered = self.tf_kmeans(daltonized, k=self.default_k)
        
        return original, daltonized, clustered
    
    def f1_score(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute F1 score using TensorFlow operations.
        
        Args:
            y_true (tf.Tensor): Ground truth binary labels [batch_size] (0 or 1)
            y_pred (tf.Tensor): Predicted binary labels [batch_size] (0 or 1)
            
        Returns:
            tf.Tensor: F1 score as a scalar tensor
        """
        # Convert to float32 for calculations
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calculate True Positives, False Positives, False Negatives
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-8)  # Add epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-8)
        
        # Calculate F1 score: F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return f1
    
    def update_fps(self):
        """Update FPS counter for performance monitoring."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def add_fps_overlay(self, cv2_frame: np.ndarray, window_name: str) -> np.ndarray:
        """
        Add FPS overlay to frame for display.
        
        Args:
            cv2_frame (np.ndarray): OpenCV frame
            window_name (str): Name of the window
            
        Returns:
            np.ndarray: Frame with FPS overlay
        """
        if not self.show_fps:
            return cv2_frame
            
        fps_text = f"{window_name}: {self.current_fps:.1f} FPS"
        cv2.putText(cv2_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return cv2_frame
    
    def initialize_camera(self) -> cv2.VideoCapture:
        """
        Initialize camera capture.
        
        Returns:
            cv2.VideoCapture: Initialized camera object
        """
        print(f"Initializing camera {self.camera_id}...")
        
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Get actual frame dimensions
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized: {self.frame_width}x{self.frame_height}")
        
        return cap
    
    def setup_windows(self):
        """Setup OpenCV display windows."""
        window_names = ['Original Feed', 'Daltonized', 'K-Means Clustered']
        
        for name in window_names:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
            
        # Arrange windows side by side
        cv2.moveWindow('Original Feed', 100, 100)
        cv2.moveWindow('Daltonized', 750, 100)
        cv2.moveWindow('K-Means Clustered', 1400, 100)
        
        print("Display windows initialized")
    
    def run_realtime_processing(self):
        """
        Main real-time processing loop.
        """
        print("Starting real-time image processing...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'k' - Cycle K values (4, 8, 16, 32)")
        print("  'd' - Cycle deficiency types")
        print("  'f' - Toggle FPS display")
        print("=" * 50)
        
        # Initialize camera
        cap = self.initialize_camera()
        
        # Setup display windows
        self.setup_windows()
        
        # K-Means values to cycle through
        k_values = [4, 8, 16, 32]
        k_index = 1  # Start with k=8
        
        # Deficiency types to cycle through
        deficiency_types = ['protanopia', 'deuteranopia', 'tritanopia', 'protanomaly', 'deuteranomaly']
        deficiency_index = 0
        
        self.is_running = True
        frame_count = 0
        
        try:
            while self.is_running:
                start_time = time.time()
                
                # Capture frame
                ret, cv2_frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Convert to TensorFlow tensor
                tf_frame = self.cv2_to_tf(cv2_frame)
                
                # Process frame
                original, daltonized, clustered = self.process_frame(tf_frame)
                
                # Convert back to OpenCV format for display
                original_cv2 = self.tf_to_cv2(original)
                daltonized_cv2 = self.tf_to_cv2(daltonized)
                clustered_cv2 = self.tf_to_cv2(clustered)
                
                # Add FPS overlay
                original_cv2 = self.add_fps_overlay(original_cv2, "Original")
                daltonized_cv2 = self.add_fps_overlay(daltonized_cv2, "Daltonized")
                clustered_cv2 = self.add_fps_overlay(clustered_cv2, "Clustered")
                
                # Add processing info overlay
                info_text = f"K={self.default_k}, Deficiency={self.default_deficiency}"
                cv2.putText(original_cv2, info_text, (10, self.frame_height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display frames
                cv2.imshow('Original Feed', original_cv2)
                cv2.imshow('Daltonized', daltonized_cv2)
                cv2.imshow('K-Means Clustered', clustered_cv2)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quit requested")
                    break
                elif key == ord('k'):
                    k_index = (k_index + 1) % len(k_values)
                    self.default_k = k_values[k_index]
                    print(f"K-Means clusters changed to: {self.default_k}")
                elif key == ord('d'):
                    deficiency_index = (deficiency_index + 1) % len(deficiency_types)
                    self.default_deficiency = deficiency_types[deficiency_index]
                    print(f"Deficiency type changed to: {self.default_deficiency}")
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                    print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
                
                # Update FPS counter
                self.update_fps()
                frame_count += 1
                
                # Control frame rate
                processing_time = time.time() - start_time
                if processing_time < self.frame_delay:
                    time.sleep(self.frame_delay - processing_time)
                
                # Print status every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames, Current FPS: {self.current_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            print("Cleaning up...")
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            print("Real-time processing stopped")
    
    def set_processing_parameters(self, k: int = None, deficiency: str = None):
        """
        Set processing parameters during runtime.
        
        Args:
            k (int, optional): Number of K-Means clusters
            deficiency (str, optional): Color vision deficiency type
        """
        if k is not None:
            self.default_k = k
            print(f"K-Means clusters set to: {k}")
            
        if deficiency is not None:
            valid_deficiencies = self.deficiency_options
            if deficiency in valid_deficiencies:
                self.default_deficiency = deficiency
                print(f"Deficiency type set to: {deficiency}")
            else:
                print(f"Invalid deficiency type. Valid options: {valid_deficiencies}")


def create_test_processor():
    """Create a test processor with synthetic video feed."""
    print("Creating test processor with synthetic video...")
    
    processor = RealTimeImageProcessor(camera_id=0)
    
    # Override the run method to use synthetic frames
    def run_test():
        processor.setup_windows()
        
        frame_count = 0
        while frame_count < 300:  # Run for 300 frames
            # Create synthetic frame
            height, width = 480, 640
            
            # Create colorful test pattern
            x = tf.linspace(0.0, 1.0, width)
            y = tf.linspace(0.0, 1.0, height)
            X, Y = tf.meshgrid(x, y)
            
            # Animated pattern
            t = frame_count * 0.1
            R = tf.sin(X * 6.28 + t) * 0.5 + 0.5
            G = tf.cos(Y * 6.28 + t) * 0.5 + 0.5
            B = tf.sin((X + Y) * 3.14 + t) * 0.5 + 0.5
            
            synthetic_frame = tf.stack([R, G, B], axis=-1)
            
            # Process frame
            original, daltonized, clustered = processor.process_frame(synthetic_frame)
            
            # Convert to OpenCV format
            original_cv2 = processor.tf_to_cv2(original)
            daltonized_cv2 = processor.tf_to_cv2(daltonized)
            clustered_cv2 = processor.tf_to_cv2(clustered)
            
            # Display
            cv2.imshow('Original Feed', original_cv2)
            cv2.imshow('Daltonized', daltonized_cv2)
            cv2.imshow('K-Means Clustered', clustered_cv2)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
                
            frame_count += 1
        
        cv2.destroyAllWindows()
    
    return run_test


def test_f1_score():
    """
    Test the F1 score function with dummy ground truth and predictions.
    """
    print("\n" + "=" * 60)
    print("Testing F1 Score Function")
    print("=" * 60)
    
    # Create a processor instance to use the F1 score method
    processor = RealTimeImageProcessor(camera_id=0, target_fps=30)
    
    # Create dummy ground truth and predictions
    print("Creating dummy test data...")
    
    # Test case 1: Perfect predictions
    y_true1 = tf.constant([1, 1, 0, 0, 1, 0, 1, 0, 1, 1], dtype=tf.float32)
    y_pred1 = tf.constant([1, 1, 0, 0, 1, 0, 1, 0, 1, 1], dtype=tf.float32)
    f1_perfect = processor.f1_score(y_true1, y_pred1)
    print(f"Test 1 - Perfect predictions: F1 = {f1_perfect.numpy():.4f} (Expected: 1.0000)")
    
    # Test case 2: Random predictions
    y_true2 = tf.constant([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=tf.float32)
    y_pred2 = tf.constant([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=tf.float32)
    f1_random = processor.f1_score(y_true2, y_pred2)
    print(f"Test 2 - Opposite predictions: F1 = {f1_random.numpy():.4f} (Expected: 0.0000)")
    
    # Test case 3: Partial accuracy
    y_true3 = tf.constant([1, 1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=tf.float32)
    y_pred3 = tf.constant([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=tf.float32)
    f1_partial = processor.f1_score(y_true3, y_pred3)
    print(f"Test 3 - Partial accuracy: F1 = {f1_partial.numpy():.4f}")
    
    # Test case 4: Larger random dataset
    tf.random.set_seed(42)  # For reproducibility
    y_true4 = tf.random.uniform([100], minval=0, maxval=2, dtype=tf.int32)
    y_pred4 = tf.random.uniform([100], minval=0, maxval=2, dtype=tf.int32)
    y_true4 = tf.cast(y_true4, tf.float32)
    y_pred4 = tf.cast(y_pred4, tf.float32)
    f1_large = processor.f1_score(y_true4, y_pred4)
    print(f"Test 4 - Random dataset (100 samples): F1 = {f1_large.numpy():.4f}")
    
    print("F1 Score testing completed!")
    print("=" * 60)


def main():
    """
    Main function to run the real-time image processor.
    """
    print("TensorFlow Real-Time Image Processing Pipeline")
    print("=" * 60)
    
    try:
        # Try to initialize with real camera
        processor = RealTimeImageProcessor(camera_id=0, target_fps=30)
        processor.run_realtime_processing()
        
        # After camera processing ends, run F1 score test
        test_f1_score()
        
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        print("Running with synthetic test video instead...")
        
        # Run test with synthetic video
        test_runner = create_test_processor()
        test_runner()
        
        # Run F1 score test after synthetic video
        test_f1_score()
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()