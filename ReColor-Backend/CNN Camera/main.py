"""
Main entry point for ReColor TensorFlow colorblind detection system.
Provides command-line interface and application startup.
"""

import argparse
import sys
import os
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recolor_app import ReColorApp


def create_argument_parser():
    """
    Create and configure argument parser for command-line interface.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="ReColor - TensorFlow GPU-Optimized Real-time Colorblind Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default camera (ID 0)
  python main.py --camera 0
  
  # Run with specific resolution and FPS
  python main.py --camera 0 --width 800 --height 600 --fps 30
  
  # Load pre-trained model weights
  python main.py --camera 0 --model models/my_model.h5
  
  # Custom training parameters
  python main.py --camera 0 --train-samples 1000 --train-epochs 50
  
  # Disable auto-training (use untrained model)
  python main.py --camera 0 --no-auto-train
  
  # Custom log directory
  python main.py --camera 0 --log-dir my_logs
        """
    )
    
    # Camera settings
    parser.add_argument("--camera", 
                       type=int, 
                       default=0,
                       help="Camera device ID (default: 0)")
    
    parser.add_argument("--width", 
                       type=int, 
                       default=640,
                       help="Camera frame width (default: 640)")
    
    parser.add_argument("--height", 
                       type=int, 
                       default=480,
                       help="Camera frame height (default: 480)")
    
    parser.add_argument("--fps", 
                       type=int, 
                       default=30,
                       help="Target FPS (default: 30)")
    
    # Model settings
    parser.add_argument("--model", 
                       type=str, 
                       default=None,
                       help="Path to pre-trained model weights file")
    
    parser.add_argument("--no-auto-train", 
                       action="store_true",
                       help="Disable automatic model training on synthetic data")
    
    parser.add_argument("--train-samples", 
                       type=int, 
                       default=500,
                       help="Training samples per color class (default: 500)")
    
    parser.add_argument("--train-epochs", 
                       type=int, 
                       default=20,
                       help="Training epochs (default: 20)")
    
    # Logging settings
    parser.add_argument("--log-dir", 
                       type=str, 
                       default="logs",
                       help="Directory for log files (default: logs)")
    
    parser.add_argument("--verbose", 
                       action="store_true",
                       help="Enable verbose logging")
    
    parser.add_argument("--debug", 
                       action="store_true",
                       help="Enable debug logging")
    
    # Enhanced features
    parser.add_argument("--kmeans-clusters", 
                       type=int, 
                       default=5,
                       help="Number of K-means color clusters for analysis (default: 5)")
    
    parser.add_argument("--simplification-clusters", 
                       type=int, 
                       default=8,
                       help="Number of K-means clusters for image simplification (default: 8)")
    
    parser.add_argument("--daltonization-strength", 
                       type=float, 
                       default=1.5,
                       help="Daltonization enhancement strength (default: 1.5)")
    
    parser.add_argument("--enable-kmeans", 
                       action="store_true",
                       help="Enable K-means analysis display on startup")
    
    parser.add_argument("--enable-simplification", 
                       action="store_true",
                       help="Enable K-means image simplification on startup")
    
    parser.add_argument("--enable-daltonization", 
                       action="store_true",
                       help="Enable daltonization on startup")
    
    parser.add_argument("--enable-realistic-cvd", 
                       action="store_true",
                       help="Enable realistic CVD simulation on startup")
    
    parser.add_argument("--enable-unified-pipeline", 
                       action="store_true",
                       help="Enable unified K-means + CNN + Daltonization pipeline on startup")
    
    # Application info
    parser.add_argument("--version", 
                       action="version",
                       version="ReColor TensorFlow v1.0.0")
    
    parser.add_argument("--system-info", 
                       action="store_true",
                       help="Show system information and exit")
    
    return parser


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
        debug: Enable debug logging
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def show_system_info() -> None:
    """Display system information."""
    try:
        from utils import setup_gpu
        import tensorflow as tf
        import cv2
        import numpy as np
        
        print("=" * 60)
        print("🖥️  RECOLOR SYSTEM INFORMATION")
        print("=" * 60)
        
        # Python version
        print(f"Python Version: {sys.version}")
        
        # TensorFlow version and GPU info
        print(f"TensorFlow Version: {tf.__version__}")
        gpu_info = setup_gpu()
        
        if gpu_info['gpu_available']:
            print(f"✅ GPU: {gpu_info['gpu_name']}")
            print(f"GPU Memory: {gpu_info['gpu_memory']}")
            print(f"CUDA Device: {gpu_info['device']}")
        else:
            print("⚠️ GPU: Not available (using CPU)")
        
        # OpenCV version
        print(f"OpenCV Version: {cv2.__version__}")
        
        # NumPy version
        print(f"NumPy Version: {np.__version__}")
        
        # Available cameras
        print("\n📷 Available Cameras:")
        available_cameras = []
        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                available_cameras.append(f"  Camera {i}: {width}x{height} @ {fps:.1f} FPS")
                cap.release()
        
        if available_cameras:
            for cam_info in available_cameras:
                print(cam_info)
        else:
            print("  No cameras detected")
        
        # Directory information
        print(f"\n📁 Working Directory: {os.getcwd()}")
        print(f"Script Location: {os.path.abspath(__file__)}")
        
        # Model directory
        models_dir = os.path.join(os.getcwd(), 'models')
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith(('.h5', '.weights'))]
            if model_files:
                print(f"\n🧠 Available Model Files:")
                for model_file in model_files:
                    print(f"  {os.path.join(models_dir, model_file)}")
            else:
                print(f"\n🧠 Models Directory: {models_dir} (empty)")
        else:
            print(f"\n🧠 Models Directory: {models_dir} (not found)")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error displaying system information: {e}")


def validate_arguments(args) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        True if arguments are valid, False otherwise
    """
    # Validate camera ID
    if args.camera < 0:
        print(f"❌ Error: Camera ID must be non-negative (got {args.camera})")
        return False
    
    # Validate resolution
    if args.width <= 0 or args.height <= 0:
        print(f"❌ Error: Resolution must be positive (got {args.width}x{args.height})")
        return False
    
    # Validate FPS
    if args.fps <= 0:
        print(f"❌ Error: FPS must be positive (got {args.fps})")
        return False
    
    # Validate training parameters
    if args.train_samples <= 0:
        print(f"❌ Error: Training samples must be positive (got {args.train_samples})")
        return False
    
    if args.train_epochs <= 0:
        print(f"❌ Error: Training epochs must be positive (got {args.train_epochs})")
        return False
    
    # Validate model file if provided
    if args.model and not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return False
    
    # Validate log directory (create if doesn't exist)
    try:
        os.makedirs(args.log_dir, exist_ok=True)
    except Exception as e:
        print(f"❌ Error: Cannot create log directory {args.log_dir}: {e}")
        return False
    
    return True


def main():
    """Main entry point function."""
    try:
        # Parse command-line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Setup logging
        setup_logging(verbose=args.verbose, debug=args.debug)
        
        # Show system info and exit if requested
        if args.system_info:
            show_system_info()
            return 0
        
        # Validate arguments
        if not validate_arguments(args):
            return 1
        
        # Create application configuration from arguments
        config = {
            'camera_id': args.camera,
            'frame_width': args.width,
            'frame_height': args.height,
            'fps_target': args.fps,
            'model_path': args.model,
            'log_directory': args.log_dir,
            'auto_train_model': not args.no_auto_train,
            'training_samples_per_class': args.train_samples,
            'training_epochs': args.train_epochs,
            # Enhanced features
            'kmeans_clusters': args.kmeans_clusters,
            'simplification_clusters': args.simplification_clusters,
            'daltonization_strength': args.daltonization_strength,
            'enable_kmeans': args.enable_kmeans,
            'enable_simplification': args.enable_simplification,
            'enable_daltonization': args.enable_daltonization,
            'enable_realistic_cvd': args.enable_realistic_cvd,
            'enable_unified_pipeline': args.enable_unified_pipeline
        }
        
        # Print startup banner
        print("\n" + "=" * 60)
        print("🎯 RECOLOR - TENSORFLOW COLORBLIND DETECTION")
        print("=" * 60)
        print(f"Camera: {args.camera} | Resolution: {args.width}x{args.height} | FPS: {args.fps}")
        print(f"Auto-train: {'Yes' if not args.no_auto_train else 'No'} | Log dir: {args.log_dir}")
        if args.model:
            print(f"Model: {args.model}")
        print("=" * 60)
        
        # Create and initialize application
        app = ReColorApp()
        
        print("🔄 Initializing application components...")
        if not app.initialize(config):
            print("❌ Failed to initialize application")
            return 1
        
        print("✅ Application initialized successfully!")
        
        # Show enhanced controls
        print("\n📋 ENHANCED KEYBOARD CONTROLS:")
        print("=" * 50)
        print("🎯 Basic Controls:")
        print("  C - Capture color and save to CSV")
        print("  N - Cycle CVD types (Normal → Protanopia → Deuteranopia → Tritanopia)")
        print("  Q - Quit application")
        print("  P - Pause/Resume camera")
        print("  S - Toggle side-by-side display")
        print("  I - Toggle color info overlay")
        print("  F - Toggle FPS display")
        print("\n🚀 Enhanced Features:")
        print("  K - Toggle K-means color analysis")
        print("  M - Toggle K-means image simplification (posterization)")
        print("  D - Toggle daltonization (color enhancement for colorblind)")
        print("  R - Toggle realistic CVD simulation (Ishihara-style)")
        print("  U - Toggle unified pipeline (K-means → CNN → Daltonization)")
        print("\n⚙️  Adjustments:")
        print("  1/2 - Decrease/Increase K-means clusters")
        print("  3/4 - Decrease/Increase daltonization strength")
        print("=" * 50)
        print("\n💡 TIP: Make sure your camera is connected and not in use by other applications")
        print("Press any key when ready to start...")
        
        # Wait for user input before starting camera
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("\nStartup cancelled by user")
            return 0
        
        # Run main application
        app.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
        return 0
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if args.debug if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())