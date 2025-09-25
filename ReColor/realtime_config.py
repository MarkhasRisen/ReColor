"""
Real-Time Processing Configuration
=================================

Configuration parameters for the real-time image processing pipeline.
Optimized for performance and real-time operation.

Author: AI Assistant
Date: September 2025
"""

# Camera settings
CAMERA_CONFIG = {
    'default_camera_id': 0,
    'frame_width': 640,
    'frame_height': 480,
    'target_fps': 30,
    'buffer_size': 1,  # Minimize buffer for low latency
}

# Real-time processing parameters
REALTIME_CONFIG = {
    # K-Means settings optimized for real-time
    'default_k': 8,
    'max_iterations': 10,  # Reduced for real-time performance
    'tolerance': 1e-3,     # Relaxed tolerance for faster convergence
    'k_options': [4, 8, 16, 32],  # Available K values for cycling
    
    # Daltonization settings
    'default_deficiency': 'deuteranopia',
    'deficiency_options': ['protanopia', 'deuteranopia', 'tritanopia', 'protanomaly', 'deuteranomaly'],
    'correction_strength': 0.8,
    
    # Performance settings
    'enable_gpu': True,
    'memory_growth': True,
    'threading': False,  # Disable for simplicity, enable if needed
}

# Display settings
DISPLAY_CONFIG = {
    'window_positions': {
        'original': (100, 100),
        'daltonized': (750, 100),
        'clustered': (1400, 100)
    },
    'show_fps': True,
    'show_info_overlay': True,
    'font_scale': 0.7,
    'font_thickness': 2,
    'text_color': (0, 255, 0),  # Green
    'info_color': (255, 255, 0),  # Yellow
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    'fps_update_interval': 1.0,  # Update FPS every second
    'status_print_interval': 100,  # Print status every N frames
    'enable_profiling': False,  # Enable for performance analysis
}

# Keyboard controls
CONTROLS_CONFIG = {
    'quit_key': 'q',
    'cycle_k_key': 'k',
    'cycle_deficiency_key': 'd',
    'toggle_fps_key': 'f',
    'save_frame_key': 's',
    'reset_key': 'r',
}

# TensorFlow optimization settings
TF_CONFIG = {
    'enable_xla': False,  # May improve performance on some systems
    'enable_mixed_precision': False,  # For advanced optimization
    'log_device_placement': False,
    'allow_soft_placement': True,
}

# Color space conversion matrices for Daltonization
# (Optimized as TensorFlow constants for real-time processing)
COLOR_MATRICES = {
    # Hunt-Pointer-Estevez RGB to LMS conversion
    'rgb_to_lms': [
        [0.4002, 0.7075, -0.0807],
        [-0.2280, 1.1500, 0.0612],
        [0.0000, 0.0000, 0.9184]
    ],
    
    # LMS to RGB conversion
    'lms_to_rgb': [
        [1.8599, -1.1293, 0.2198],
        [0.3611, 0.6388, -0.0000],
        [0.0000, 0.0000, 1.0888]
    ],
    
    # Dichromacy simulation matrices
    'protanopia_sim': [
        [0.0, 2.02344, -2.52581],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    
    'deuteranopia_sim': [
        [1.0, 0.0, 0.0],
        [0.494207, 0.0, 1.24827],
        [0.0, 0.0, 1.0]
    ],
    
    'tritanopia_sim': [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-0.395913, 0.801109, 0.0]
    ],
    
    # Anomaly simulation matrices (partial color deficiency)
    'protanomaly_sim': [
        [0.817, 0.183, 0.0],
        [0.333, 0.667, 0.0],
        [0.0, 0.125, 0.875]
    ],
    
    'deuteranomaly_sim': [
        [0.8, 0.2, 0.0],
        [0.258, 0.742, 0.0],
        [0.0, 0.142, 0.858]
    ],
    
    # Error correction matrices
    'protanopia_error': [
        [0.0, 0.0, 0.0],
        [0.7, 1.0, 0.0],
        [0.7, 0.0, 1.0]
    ],
    
    'deuteranopia_error': [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.7, 1.0]
    ],
    
    'tritanopia_error': [
        [1.0, 0.0, 0.7],
        [0.0, 1.0, 0.7],
        [0.0, 0.0, 0.0]
    ],
    
    # Error correction matrices for anomalies (milder correction)
    'protanomaly_error': [
        [0.0, 0.0, 0.0],
        [0.3, 1.0, 0.0],
        [0.3, 0.0, 1.0]
    ],
    
    'deuteranomaly_error': [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.3, 1.0]
    ]
}

# Frame processing options
FRAME_CONFIG = {
    'enable_resize': False,
    'target_size': (480, 640),  # (height, width)
    'enable_blur': False,
    'blur_sigma': 0.5,
    'enable_noise_reduction': False,
}

# Debug and testing options
DEBUG_CONFIG = {
    'save_processed_frames': False,
    'save_directory': 'realtime_output',
    'save_interval': 30,  # Save every N frames
    'synthetic_mode': False,  # Use synthetic video for testing
    'verbose_logging': False,
}


def get_optimized_tf_config():
    """
    Get TensorFlow configuration optimized for real-time processing.
    
    Returns:
        dict: TensorFlow configuration settings
    """
    import tensorflow as tf
    
    # Configure GPU memory growth to avoid allocation issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus and REALTIME_CONFIG['enable_gpu']:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Enable XLA compilation if requested
    if TF_CONFIG['enable_xla']:
        tf.config.optimizer.set_jit(True)
        print("XLA JIT compilation enabled")
    
    return TF_CONFIG


def print_realtime_config():
    """Print current real-time configuration."""
    print("Real-Time Processing Configuration")
    print("=" * 50)
    print(f"Camera: {CAMERA_CONFIG['frame_width']}x{CAMERA_CONFIG['frame_height']} @ {CAMERA_CONFIG['target_fps']} FPS")
    print(f"K-Means: K={REALTIME_CONFIG['default_k']}, Max Iterations={REALTIME_CONFIG['max_iterations']}")
    print(f"Daltonization: {REALTIME_CONFIG['default_deficiency']}")
    print(f"GPU Enabled: {REALTIME_CONFIG['enable_gpu']}")
    print(f"Show FPS: {DISPLAY_CONFIG['show_fps']}")
    print("\nControls:")
    for action, key in CONTROLS_CONFIG.items():
        print(f"  {key.upper()}: {action.replace('_', ' ').title()}")
    print("=" * 50)


if __name__ == "__main__":
    print_realtime_config()