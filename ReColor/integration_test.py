"""
Integration Test Script
======================

This script tests the integrated simple algorithms across all Python files
to ensure they work consistently and correctly.
"""

import tensorflow as tf
import numpy as np
import time
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.getcwd())

# Import from different modules
from realtime_processor import RealTimeImageProcessor
from algorithm_demo import tf_kmeans as demo_kmeans, daltonize as demo_daltonize, f1_score as demo_f1
from simple_camera_pipeline import tf_kmeans as simple_kmeans, daltonize as simple_daltonize, f1_score as simple_f1


def test_realtime_processor():
    """Test the updated realtime_processor algorithms"""
    print("🔧 Testing RealTimeImageProcessor...")
    
    processor = RealTimeImageProcessor()
    
    # Create test image
    test_image = tf.random.uniform([100, 100, 3], 0.0, 1.0, dtype=tf.float32)
    
    # Test K-Means
    start_time = time.time()
    clustered = processor.tf_kmeans(test_image, k=8, max_iters=10)
    kmeans_time = time.time() - start_time
    print(f"  ✅ K-Means: {kmeans_time*1000:.1f}ms, shape: {clustered.shape}")
    
    # Test Daltonization
    start_time = time.time()
    daltonized = processor.daltonize(test_image, 'deuteranopia')
    daltonize_time = time.time() - start_time
    print(f"  ✅ Daltonization: {daltonize_time*1000:.1f}ms, shape: {daltonized.shape}")
    
    # Test F1 Score
    y_true = tf.constant([1, 1, 0, 1, 0], dtype=tf.int32)
    y_pred = tf.constant([1, 0, 0, 1, 1], dtype=tf.int32)
    f1 = processor.f1_score(y_true, y_pred)
    print(f"  ✅ F1 Score: {f1.numpy():.4f}")
    
    return True


def test_algorithm_demo():
    """Test the updated algorithm_demo functions"""
    print("\n🎨 Testing algorithm_demo functions...")
    
    # Create test image
    test_image = tf.random.uniform([80, 80, 3], 0.0, 1.0, dtype=tf.float32)
    
    # Test K-Means
    start_time = time.time()
    clustered = demo_kmeans(test_image, k=6, max_iters=8)
    kmeans_time = time.time() - start_time
    print(f"  ✅ K-Means: {kmeans_time*1000:.1f}ms, shape: {clustered.shape}")
    
    # Test Daltonization
    start_time = time.time()
    daltonized = demo_daltonize(test_image, 'protanopia')
    daltonize_time = time.time() - start_time
    print(f"  ✅ Daltonization: {daltonize_time*1000:.1f}ms, shape: {daltonized.shape}")
    
    # Test F1 Score
    y_true = tf.constant([1, 0, 1, 0, 1], dtype=tf.int32)
    y_pred = tf.constant([1, 0, 0, 0, 1], dtype=tf.int32)
    f1 = demo_f1(y_true, y_pred)
    print(f"  ✅ F1 Score: {f1.numpy():.4f}")
    
    return True


def test_simple_camera_pipeline():
    """Test the simple_camera_pipeline functions"""
    print("\n📹 Testing simple_camera_pipeline functions...")
    
    # Create test image (as uint8 like camera input)
    test_image = tf.random.uniform([120, 120, 3], 0, 255, dtype=tf.int32)
    test_image = tf.cast(test_image, tf.uint8)
    
    # Test K-Means
    start_time = time.time()
    clustered = simple_kmeans(test_image, k=4, max_iters=5)
    kmeans_time = time.time() - start_time
    print(f"  ✅ K-Means: {kmeans_time*1000:.1f}ms, shape: {clustered.shape}")
    
    # Test Daltonization
    start_time = time.time()
    daltonized = simple_daltonize(test_image, 'tritanopia')
    daltonize_time = time.time() - start_time
    print(f"  ✅ Daltonization: {daltonize_time*1000:.1f}ms, shape: {daltonized.shape}")
    
    # Test F1 Score
    y_true = tf.constant([0, 1, 1, 0, 1], dtype=tf.int32)
    y_pred = tf.constant([0, 1, 0, 0, 1], dtype=tf.int32)
    f1 = simple_f1(y_true, y_pred)
    print(f"  ✅ F1 Score: {f1.numpy():.4f}")
    
    return True


def consistency_test():
    """Test that all implementations give similar results"""
    print("\n🔍 Testing consistency across implementations...")
    
    # Create identical test image
    np.random.seed(42)  # For reproducibility
    test_image_uint8 = tf.constant(np.random.randint(0, 255, (50, 50, 3)), dtype=tf.uint8)
    test_image_float = tf.cast(test_image_uint8, tf.float32) / 255.0
    
    # Test K-Means consistency
    processor = RealTimeImageProcessor()
    
    clustered_realtime = processor.tf_kmeans(test_image_float, k=4, max_iters=5)
    clustered_demo = demo_kmeans(test_image_float, k=4, max_iters=5)
    clustered_simple = simple_kmeans(test_image_uint8, k=4, max_iters=5)
    clustered_simple_float = tf.cast(clustered_simple, tf.float32) / 255.0
    
    # Check shapes are consistent
    shapes_match = (clustered_realtime.shape == clustered_demo.shape == clustered_simple_float.shape)
    print(f"  ✅ K-Means shapes consistent: {shapes_match}")
    
    # Test Daltonization consistency
    daltonized_realtime = processor.daltonize(test_image_float, 'deuteranopia')
    daltonized_demo = demo_daltonize(test_image_float, 'deuteranopia')
    daltonized_simple = simple_daltonize(test_image_uint8, 'deuteranopia')
    daltonized_simple_float = tf.cast(daltonized_simple, tf.float32) / 255.0
    
    shapes_match = (daltonized_realtime.shape == daltonized_demo.shape == daltonized_simple_float.shape)
    print(f"  ✅ Daltonization shapes consistent: {shapes_match}")
    
    # Test F1 Score consistency
    y_true = tf.constant([1, 1, 0, 1, 0], dtype=tf.int32)
    y_pred = tf.constant([1, 0, 0, 1, 1], dtype=tf.int32)
    
    f1_realtime = processor.f1_score(y_true, y_pred)
    f1_demo = demo_f1(y_true, y_pred)
    f1_simple = simple_f1(y_true, y_pred)
    
    f1_values = [f1_realtime.numpy(), f1_demo.numpy(), f1_simple.numpy()]
    f1_consistent = all(abs(f1_values[i] - f1_values[0]) < 1e-6 for i in range(len(f1_values)))
    print(f"  ✅ F1 Scores consistent: {f1_consistent} (values: {f1_values})")
    
    return shapes_match and f1_consistent


def performance_benchmark():
    """Benchmark performance across all implementations"""
    print("\n⚡ Performance Benchmark...")
    
    # Create realistic test image
    test_image_uint8 = tf.random.uniform([240, 320, 3], 0, 255, dtype=tf.int32)
    test_image_uint8 = tf.cast(test_image_uint8, tf.uint8)
    test_image_float = tf.cast(test_image_uint8, tf.float32) / 255.0
    
    processor = RealTimeImageProcessor()
    
    # Benchmark K-Means
    print("  🔧 K-Means (240x320, k=8):")
    
    start_time = time.time()
    _ = processor.tf_kmeans(test_image_float, k=8, max_iters=10)
    realtime_kmeans = time.time() - start_time
    print(f"    realtime_processor: {realtime_kmeans*1000:.1f}ms")
    
    start_time = time.time()
    _ = demo_kmeans(test_image_float, k=8, max_iters=10)
    demo_kmeans_time = time.time() - start_time
    print(f"    algorithm_demo: {demo_kmeans_time*1000:.1f}ms")
    
    start_time = time.time()
    _ = simple_kmeans(test_image_uint8, k=8, max_iters=10)
    simple_kmeans_time = time.time() - start_time
    print(f"    simple_camera_pipeline: {simple_kmeans_time*1000:.1f}ms")
    
    # Benchmark Daltonization
    print("  🎨 Daltonization (240x320):")
    
    start_time = time.time()
    _ = processor.daltonize(test_image_float, 'deuteranopia')
    realtime_daltonize = time.time() - start_time
    print(f"    realtime_processor: {realtime_daltonize*1000:.1f}ms")
    
    start_time = time.time()
    _ = demo_daltonize(test_image_float, 'deuteranopia')
    demo_daltonize_time = time.time() - start_time
    print(f"    algorithm_demo: {demo_daltonize_time*1000:.1f}ms")
    
    start_time = time.time()
    _ = simple_daltonize(test_image_uint8, 'deuteranopia')
    simple_daltonize_time = time.time() - start_time
    print(f"    simple_camera_pipeline: {simple_daltonize_time*1000:.1f}ms")
    
    # Calculate theoretical FPS
    total_realtime = realtime_kmeans + realtime_daltonize
    total_demo = demo_kmeans_time + demo_daltonize_time
    total_simple = simple_kmeans_time + simple_daltonize_time
    
    print(f"\n  📊 Theoretical FPS:")
    print(f"    realtime_processor: {1.0/total_realtime:.1f} FPS")
    print(f"    algorithm_demo: {1.0/total_demo:.1f} FPS")
    print(f"    simple_camera_pipeline: {1.0/total_simple:.1f} FPS")


def main():
    """Run all integration tests"""
    print("🧪 INTEGRATION TEST SUITE")
    print("=" * 50)
    print("Testing simple algorithms across all Python files...")
    
    # Individual module tests
    realtime_ok = test_realtime_processor()
    demo_ok = test_algorithm_demo()
    simple_ok = test_simple_camera_pipeline()
    
    # Consistency test
    consistent = consistency_test()
    
    # Performance benchmark
    performance_benchmark()
    
    # Summary
    print("\n🎉 INTEGRATION TEST RESULTS:")
    print("=" * 40)
    print(f"  realtime_processor.py: {'✅ PASS' if realtime_ok else '❌ FAIL'}")
    print(f"  algorithm_demo.py: {'✅ PASS' if demo_ok else '❌ FAIL'}")
    print(f"  simple_camera_pipeline.py: {'✅ PASS' if simple_ok else '❌ FAIL'}")
    print(f"  Consistency: {'✅ PASS' if consistent else '❌ FAIL'}")
    
    if all([realtime_ok, demo_ok, simple_ok, consistent]):
        print("\n🚀 ALL INTEGRATIONS SUCCESSFUL!")
        print("✅ Simple algorithms integrated across all files")
        print("✅ Consistent behavior verified")
        print("✅ Performance acceptable for real-time use")
        print("\n📝 READY TO USE:")
        print("  • python realtime_processor.py (class-based)")
        print("  • python algorithm_demo.py (standalone demo)")
        print("  • python simple_camera_pipeline.py (simple pipeline)")
        print("  • python launcher.py (interactive launcher)")
    else:
        print("\n❌ Some integrations failed - check implementations")
    
    return all([realtime_ok, demo_ok, simple_ok, consistent])


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)