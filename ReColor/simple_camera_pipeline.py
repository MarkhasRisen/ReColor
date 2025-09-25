"""
Simple Camera Pipeline with TensorFlow Algorithms
================================================

A real-time webcam processing system with simple implementations of:
- Daltonization (LMS color space transform for color vision deficiency)
- K-Means clustering (pure TensorFlow implementation)
- F1 Score computation (precision/recall based)

Requirements: tensorflow>=2.0, opencv-python
Usage: python simple_camera_pipeline.py
"""

import cv2
import tensorflow as tf
import numpy as np
import time


def daltonize(image, deficiency='deuteranopia'):
    """
    Apply daltonization to simulate and correct color vision deficiency.
    
    Args:
        image: RGB image tensor [H, W, 3] with values 0-255
        deficiency: Type of color vision deficiency
                   ('protanopia', 'deuteranopia', 'tritanopia')
    
    Returns:
        Daltonized image tensor [H, W, 3]
    """
    # Convert to float [0,1]
    img_float = tf.cast(image, tf.float32) / 255.0
    original_shape = tf.shape(img_float)
    
    # Flatten for matrix operations
    pixels = tf.reshape(img_float, [-1, 3])
    
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
    
    # Reshape and convert back to [0,255]
    daltonized_image = tf.reshape(daltonized_rgb, original_shape)
    return tf.cast(daltonized_image * 255.0, tf.uint8)


def tf_kmeans(image, k=8, max_iters=10):
    """
    Apply K-Means clustering using pure TensorFlow operations.
    
    Args:
        image: RGB image tensor [H, W, 3] with values 0-255
        k: Number of clusters
        max_iters: Maximum iterations
    
    Returns:
        Clustered image tensor [H, W, 3]
    """
    original_shape = tf.shape(image)
    height, width, channels = original_shape[0], original_shape[1], original_shape[2]
    
    # Convert to float and reshape to [num_pixels, channels]
    frame = tf.cast(image, tf.float32) / 255.0
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
        
        # Update centroids using tf.math.unsorted_segment_sum
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
    
    return tf.cast(clustered_frame * 255.0, tf.uint8)


def f1_score(y_true, y_pred):
    """
    Compute F1 score using TensorFlow operations.
    
    Args:
        y_true: True binary labels tensor
        y_pred: Predicted binary labels tensor
    
    Returns:
        F1 score as scalar tensor
    """
    # Convert to float for calculations
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    # Calculate True Positives, False Positives, False Negatives
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    
    # Calculate precision and recall
    precision = tp / (tp + fp + 1e-8)  # Add epsilon to prevent division by zero
    recall = tp / (tp + fn + 1e-8)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1


def main():
    """
    Main camera processing pipeline.
    """
    print("🎥 Starting Simple Camera Pipeline with TensorFlow Algorithms")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  'k' - Cycle K values (4, 8, 16)")
    print("  'd' - Cycle deficiency types")
    print("=" * 60)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Processing parameters
    k_values = [4, 8, 16]
    k_index = 1  # Start with k=8
    
    deficiency_types = ['deuteranopia', 'protanopia', 'tritanopia']
    deficiency_index = 0  # Start with deuteranopia
    
    # Performance monitoring
    fps_counter = 0
    fps_start_time = time.time()
    
    print(f"🚀 Starting processing with K={k_values[k_index]}, deficiency={deficiency_types[deficiency_index]}")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Convert BGR to RGB for TensorFlow
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to TensorFlow tensor
            frame_tensor = tf.constant(frame_rgb, dtype=tf.uint8)
            
            # Apply algorithms
            start_time = time.time()
            
            # Daltonization
            daltonized_tensor = daltonize(frame_tensor, deficiency_types[deficiency_index])
            
            # K-Means clustering
            clustered_tensor = tf_kmeans(frame_tensor, k=k_values[k_index], max_iters=10)
            
            processing_time = time.time() - start_time
            
            # Convert back to numpy for display
            original_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            daltonized_display = cv2.cvtColor(daltonized_tensor.numpy(), cv2.COLOR_RGB2BGR)
            clustered_display = cv2.cvtColor(clustered_tensor.numpy(), cv2.COLOR_RGB2BGR)
            
            # Add text overlays
            cv2.putText(original_display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(daltonized_display, f"Daltonized ({deficiency_types[deficiency_index]})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(clustered_display, f"K-Means (k={k_values[k_index]})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # FPS calculation
            fps_counter += 1
            if fps_counter % 30 == 0:
                current_fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                print(f"📊 FPS: {current_fps:.1f} | Processing: {processing_time*1000:.1f}ms | "
                      f"K={k_values[k_index]} | Deficiency={deficiency_types[deficiency_index]}")
            
            # Display frames
            cv2.imshow('Original', original_display)
            cv2.imshow('Daltonized', daltonized_display)
            cv2.imshow('K-Means Clustered', clustered_display)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('k'):
                k_index = (k_index + 1) % len(k_values)
                print(f"🔄 Changed K to {k_values[k_index]}")
            elif key == ord('d'):
                deficiency_index = (deficiency_index + 1) % len(deficiency_types)
                print(f"🔄 Changed deficiency to {deficiency_types[deficiency_index]}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("📹 Camera released")
    
    # F1 Score test after camera loop
    print("\n🧪 Running F1 Score Test...")
    
    # Create dummy ground truth and predictions
    y_true = tf.constant([1, 1, 0, 1, 0, 0, 1, 0], dtype=tf.int32)
    y_pred = tf.constant([1, 0, 0, 1, 0, 1, 1, 0], dtype=tf.int32)
    
    # Compute F1 score
    f1 = f1_score(y_true, y_pred)
    
    print(f"✅ F1 Score Test Result: {f1.numpy():.4f}")
    print("   Ground Truth: [1, 1, 0, 1, 0, 0, 1, 0]")
    print("   Predictions:  [1, 0, 0, 1, 0, 1, 1, 0]")
    
    # Additional F1 tests
    print("\n📊 Additional F1 Score Tests:")
    
    # Perfect predictions
    y_perfect = tf.constant([1, 1, 0, 1, 0], dtype=tf.int32)
    f1_perfect = f1_score(y_perfect, y_perfect)
    print(f"   Perfect predictions: {f1_perfect.numpy():.4f}")
    
    # Completely wrong predictions
    y_wrong = tf.constant([0, 0, 1, 0, 1], dtype=tf.int32)
    f1_wrong = f1_score(y_perfect, y_wrong)
    print(f"   Wrong predictions: {f1_wrong.numpy():.4f}")
    
    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()