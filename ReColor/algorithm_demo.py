#!/usr/bin/env python3
"""
Algorithm Demonstration Script
=============================

Standalone demonstration of the implemented algorithms:
- K-Means clustering with TensorFlow
- Daltonization with LMS color space
- F1 Score computation

This script creates synthetic test images and shows the results.

Author: AI Assistant
Date: September 2025
"""

import tensorflow as tf
import cv2
import numpy as np
import time


def tf_kmeans(image: tf.Tensor, k: int = 8, max_iters: int = 10) -> tf.Tensor:
    """
    Simple K-Means clustering using pure TensorFlow operations.
    
    Args:
        image (tf.Tensor): Input image tensor [height, width, 3] with values 0-1
        k (int): Number of clusters
        max_iters (int): Maximum number of iterations
        
    Returns:
        tf.Tensor: Clustered image with same shape as input
    """
    original_shape = tf.shape(image)
    pixels = tf.reshape(image, [-1, 3])
    
    # Initialize centroids randomly
    centroids = tf.random.uniform([k, 3], minval=0.0, maxval=1.0, dtype=tf.float32)
    
    # K-Means iterations
    for _ in range(max_iters):
        # Compute distances and assign pixels to nearest centroids
        expanded_pixels = tf.expand_dims(pixels, 1)
        expanded_centroids = tf.expand_dims(centroids, 0)
        
        squared_distances = tf.reduce_sum(
            tf.square(expanded_pixels - expanded_centroids), axis=2
        )
        assignments = tf.argmin(squared_distances, axis=1)
        
        # Update centroids with better empty cluster handling
        new_centroids = []
        for cluster_id in range(k):
            mask = tf.equal(assignments, cluster_id)
            cluster_pixels = tf.boolean_mask(pixels, mask)
            
            # Handle empty clusters with random reinitialization
            cluster_size = tf.shape(cluster_pixels)[0]
            new_centroid = tf.cond(
                cluster_size > 0,
                lambda: tf.reduce_mean(cluster_pixels, axis=0),
                lambda: tf.random.uniform([3], minval=0.0, maxval=1.0)
            )
            new_centroids.append(new_centroid)
        
        centroids = tf.stack(new_centroids)
    
    # Assign final colors
    final_assignments = tf.argmin(squared_distances, axis=1)
    clustered_pixels = tf.gather(centroids, final_assignments)
    
    return tf.reshape(clustered_pixels, original_shape)


def daltonize(image: tf.Tensor, deficiency: str = 'deuteranopia') -> tf.Tensor:
    """
    Simple daltonization using LMS color space with Hunt-Pointer-Estevez matrices.
    
    Args:
        image (tf.Tensor): Input RGB image tensor [height, width, 3] with values 0-1
        deficiency (str): Type of color vision deficiency
        
    Returns:
        tf.Tensor: Color-corrected image tensor
    """
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
    
    deficiency_matrix = deficiency_matrices.get(deficiency, deficiency_matrices['deuteranopia'])
    
    # Transform image
    original_shape = tf.shape(image)
    pixels = tf.reshape(image, [-1, 3])
    
    # Convert RGB to LMS
    lms_pixels = tf.matmul(pixels, rgb_to_lms, transpose_b=True)
    
    # Apply deficiency simulation
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
    
    return tf.reshape(daltonized_rgb, original_shape)
    lms_pixels = tf.matmul(pixels, rgb_to_lms, transpose_b=True)
    deficient_lms = tf.matmul(lms_pixels, deficiency_matrix, transpose_b=True)
    deficient_rgb = tf.matmul(deficient_lms, lms_to_rgb, transpose_b=True)
    
    # Compute and apply error correction
    error = pixels - deficient_rgb
    corrected_rgb = pixels + error * 0.7
    corrected_rgb = tf.clip_by_value(corrected_rgb, 0.0, 1.0)
    
    return tf.reshape(corrected_rgb, original_shape)


def f1_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute F1 score using TensorFlow operations.
    
    Args:
        y_true (tf.Tensor): Ground truth binary labels
        y_pred (tf.Tensor): Predicted binary labels
        
    Returns:
        tf.Tensor: F1 score
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1


def create_test_image():
    """Create a colorful test image."""
    height, width = 240, 320
    
    # Create gradient pattern
    x = tf.linspace(0.0, 1.0, width)
    y = tf.linspace(0.0, 1.0, height)
    X, Y = tf.meshgrid(x, y)
    
    # RGB channels with different patterns
    R = tf.sin(X * 6.28) * 0.5 + 0.5
    G = tf.cos(Y * 6.28) * 0.5 + 0.5
    B = tf.sin((X + Y) * 3.14) * 0.5 + 0.5
    
    return tf.stack([R, G, B], axis=-1)


def main():
    """Main demonstration function."""
    print("Algorithm Demonstration")
    print("=" * 50)
    
    # Create test image
    print("Creating test image...")
    test_image = create_test_image()
    
    # Test K-Means clustering
    print("Testing K-Means clustering...")
    start_time = time.time()
    clustered = tf_kmeans(test_image, k=8, max_iters=10)
    kmeans_time = time.time() - start_time
    print(f"K-Means completed in {kmeans_time:.3f} seconds")
    
    # Test Daltonization
    print("Testing Daltonization...")
    deficiencies = ['protanopia', 'deuteranopia', 'tritanopia']
    daltonized_images = {}
    
    for deficiency in deficiencies:
        start_time = time.time()
        daltonized = daltonize(test_image, deficiency)
        dalton_time = time.time() - start_time
        daltonized_images[deficiency] = daltonized
        print(f"Daltonization ({deficiency}) completed in {dalton_time:.3f} seconds")
    
    # Test F1 Score
    print("Testing F1 Score...")
    
    # Test cases
    test_cases = [
        ([1, 1, 0, 0, 1], [1, 1, 0, 0, 1], "Perfect"),
        ([1, 0, 1, 0, 1], [0, 1, 0, 1, 0], "Opposite"), 
        ([1, 1, 1, 0, 0], [1, 0, 1, 0, 1], "Partial")
    ]
    
    for y_true, y_pred, name in test_cases:
        y_true_tensor = tf.constant(y_true, dtype=tf.float32)
        y_pred_tensor = tf.constant(y_pred, dtype=tf.float32)
        f1 = f1_score(y_true_tensor, y_pred_tensor)
        print(f"F1 Score ({name}): {f1.numpy():.4f}")
    
    # Convert images for display
    print("Preparing images for display...")
    
    def tensor_to_cv2(tensor):
        """Convert TensorFlow tensor to OpenCV format."""
        array = tensor.numpy()
        array = (array * 255).astype(np.uint8)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    
    # Convert all images
    original_cv2 = tensor_to_cv2(test_image)
    clustered_cv2 = tensor_to_cv2(clustered)
    
    # Display images
    print("Displaying results (press any key to continue)...")
    
    cv2.imshow('Original', original_cv2)
    cv2.imshow('K-Means Clustered', clustered_cv2)
    
    for deficiency, daltonized in daltonized_images.items():
        daltonized_cv2 = tensor_to_cv2(daltonized)
        cv2.imshow(f'Daltonized ({deficiency})', daltonized_cv2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Demonstration completed!")


if __name__ == "__main__":
    main()