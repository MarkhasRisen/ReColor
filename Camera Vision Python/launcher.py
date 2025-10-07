"""
TensorFlow Real-Time Image Pro    print("2. 🎯 Modular ReColor System")
    print("   - Full modular pipeline with all options")
    print("   - Command-line configuration")
    print("   - Advanced parameter control")ing - Launcher
===============================================

Interactive launcher to help users choose the appropriate processing mode.

Author: AI Assistant
Date: September 2025
"""

import os
import sys
import subprocess


def print_banner():
    """Print application banner."""
    print("🎥" + "=" * 48 + "🎥")
    print("        ReColor CVD Camera System")
    print("   Real-time Color Vision Processing")
    print("🎥" + "=" * 48 + "🎥")
    print("  Unified camera interface for CVD assistance")
    print("  Features: CVD simulation, Daltonization, K-Means clustering")
    print("="*50)


def print_menu():
    """Print main menu options."""
    print("\nSelect processing mode:")
    print()
    print("1. � Unified CVD Camera (RECOMMENDED)")
    print("   - Single streamlined camera interface")
    print("   - Real-time CVD simulation & correction")
    print("   - Interactive controls and multiple modes")
    print("   - GPU-accelerated processing")
    print()
    print("2. � Hybrid CVD Processing")
    print("   - TensorFlow + PyTorch hybrid engine")
    print("   - Advanced neural network enhancement")
    print("   - Real-time framework switching")
    print()
    print("3. 🎯 Ishihara Test Optimization")
    print("   - Specialized Ishihara color blindness test optimization")
    print("   - Interactive test plate generation and optimization")
    print("   - Before/after comparison for all CVD types")
    print()
    print("4. 🧪 GPU Performance Test")
    print("   - Test RTX 4050 GPU acceleration")
    print("   - PyTorch CUDA and TensorFlow benchmarks")
    print("   - Performance optimization")
    print()
    print("5. ⚙️  System Configuration")
    print("   - View current settings")
    print("   - Performance parameters")
    print("   - Framework status")
    print()
    print("6. 🚪 Exit")
    print()


def select_deficiency_type():
    """Allow user to select a color vision deficiency type."""
    deficiency_options = [
        ('protanopia', '🔴 Protanopia (Red color blindness)'),
        ('deuteranopia', '🟢 Deuteranopia (Green color blindness)'),
        ('tritanopia', '🔵 Tritanopia (Blue color blindness)'),
        ('protanomaly', '🟠 Protanomaly (Reduced red sensitivity)'),
        ('deuteranomaly', '🟡 Deuteranomaly (Reduced green sensitivity)')
    ]
    
    print("\nSelect Color Vision Deficiency Type:")
    print("=" * 40)
    
    for i, (deficiency, description) in enumerate(deficiency_options, 1):
        print(f"{i}. {description}")
    
    print("6. 🔄 Use default (deuteranopia)")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '6':
                return 'deuteranopia'
            
            choice_num = int(choice)
            if 1 <= choice_num <= 5:
                selected_deficiency = deficiency_options[choice_num - 1][0]
                print(f"Selected: {deficiency_options[choice_num - 1][1]}")
                return selected_deficiency
            else:
                print("Please enter a number between 1 and 6.")
                
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def run_unified_camera():
    """Run unified CVD camera with integrated processing."""
    print("\n� Starting Unified CVD Camera...")
    print("Single streamlined interface combining all CVD processing.")
    print("This is the most optimized version for RTX 4050 GPU performance.")
    print("\nUnified Features:")
    print("  🎯 GPU-accelerated K-means clustering (PyTorch)")
    print("  👁️  Real-time CVD simulation (Protanopia/Deuteranopia/Tritanopia)")
    print("  🔧 Daltonization enhancement for better visibility")
    print("  🔄 Four modes: Unified, Simulation, Correction, Clustering")
    print("  ⚡ Single window interface with all controls")
    print("\nProcessing Pipeline:")
    print("  1. K-means color segmentation")
    print("  2. CVD simulation application")
    print("  3. Daltonization correction")
    print("  4. Unified blending and display")
    print("\nInteractive Controls:")
    print("  • M: Switch modes (Unified/Simulation/Correction/Clustering)")
    print("  • D: Cycle CVD types (Protanopia/Deuteranopia/Tritanopia)")
    print("  • K: Change K-means clusters (3-16)")
    print("  • +/-: Adjust daltonization strength")
    print("  • O: Toggle original view | P: Toggle palette")
    print("  • R: Reset defaults | Space: Pause | H: Help | Q: Quit")
    
    print()
    choice = input("Start unified CVD camera? (y/N): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        return
    
    try:
        # Import and run unified camera directly
        from cvd_simulation import run_unified_cvd_camera
        run_unified_cvd_camera()
    except ImportError:
        print("Error: CVD simulation module not found or unified camera not available")
        print("Falling back to separate script...")
        try:
            subprocess.run([sys.executable, "unified_camera.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running unified camera: {e}")
            print("Make sure all dependencies are installed in your virtual environment")
    except KeyboardInterrupt:
        print("\nUnified camera stopped by user")


def run_modular_system():
    """Run full modular ReColor system."""
    print("\n🎯 Starting Modular ReColor System...")
    print("Full modular pipeline with all processing options.")
    print("This provides complete control over all parameters.")
    print("\nModular Features:")
    print("  🔧 Complete CVD simulation and correction pipeline")
    print("  🧠 Adaptive K-means clustering with mobile optimization")
    print("  ⚡ Command-line parameter control")
    print("  🎮 TensorFlow and PyTorch GPU acceleration")
    print("  🔄 All CVD types and processing modes")
    print("\nOptimizations enabled:")
    print("  • Real-time performance monitoring")
    print("  • Adaptive parameter adjustment")
    print("  • Mobile deployment ready")
    print("  • Scientific accuracy with LMS color space")
    
    # CVD type selection
    cvd_type = select_deficiency_type()
    if not cvd_type:
        return
    
    print()
    choice = input("Start modular system? (y/N): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        return
    
    try:
        subprocess.run([sys.executable, "main.py", "--cvd-type", cvd_type, "--clusters", "8", "--optimization", "balanced"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running modular system: {e}")
        print("Make sure all dependencies are installed in your virtual environment")
    except KeyboardInterrupt:
        print("Modular system stopped by user")


def run_ishihara_demo():
    """Run Ishihara test optimization demonstration."""
    print("\n🎯 Starting Ishihara Test Optimization...")
    print("Specialized optimization for Ishihara color blindness tests.")
    print("This demonstrates enhanced visibility for color vision deficiency tests.")
    print("\nIshihara Features:")
    print("  🔬 Scientifically accurate test plate generation")
    print("  🎨 Red-green color discrimination enhancement")
    print("  📊 Before/after comparison for all CVD types")
    print("  ⚡ Real-time interactive testing")
    print("  📈 Performance analysis and optimization")
    print("\nDemo Components:")
    print("  • Generate test plates with numbers 8, 3, 5, 2")
    print("  • Show CVD simulation vs Ishihara optimization")
    print("  • Interactive testing with adjustable parameters")
    print("  • Performance benchmarking")
    
    print()
    choice = input("Start Ishihara test demonstration? (y/N): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        return
    
    try:
        subprocess.run([sys.executable, "ishihara_demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Ishihara demo: {e}")
        print("Make sure Ishihara optimizer is properly installed:")
        print("  All dependencies should be in your virtual environment")
    except KeyboardInterrupt:
        print("Ishihara demo cancelled by user")

def run_gpu_test():
    """Run RTX 4050 GPU performance test."""
    print("\n🧪 Starting RTX 4050 GPU Performance Test...")
    print("Comprehensive testing of your laptop GPU acceleration.")
    print("This will benchmark both PyTorch CUDA and TensorFlow GPU performance.")
    print("\nGPU Test Features:")
    print("  🔧 PyTorch CUDA detection and benchmark")
    print("  👁️  TensorFlow GPU acceleration test")
    print("  ⚡ Real-time camera processing performance")
    print("  🎮 Memory usage and optimization analysis")
    print("  🔄 FPS measurement and latency testing")
    print("\nTest Components:")
    print("  • GPU device detection and capabilities")
    print("  • CUDA memory allocation testing")
    print("  • Real-time image processing benchmarks")
    print("  • CVD processing pipeline performance")
    
    print()
    choice = input("Start GPU performance test? (y/N): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        return
    
    try:
        subprocess.run([sys.executable, "gpu_test.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running GPU test: {e}")
        print("Make sure PyTorch and TensorFlow are installed:")
        print("  pip install torch torchvision tensorflow")
    except KeyboardInterrupt:
        print("GPU test cancelled by user")


def show_configuration():
    """Show system configuration and framework status."""
    print("\n⚙️ ReColor System Configuration")
    print("=" * 45)
    
    # Check framework availability
    print("🧪 Framework Detection:")
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow: {tf.__version__}")
        
        # Check GPU support
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"     🎮 GPU Support: {len(gpus)} GPU(s) detected")
            for gpu in gpus:
                print(f"        - {gpu.name.split('/')[-1]}")
        else:
            print(f"     ⚠️ GPU Support: No GPUs detected")
    except ImportError:
        print(f"  ❌ TensorFlow: Not installed")
    
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"     🎮 CUDA Support: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"        - {torch.cuda.get_device_name(i)}")
        else:
            print(f"     ⚠️ CUDA Support: Not available")
    except ImportError:
        print(f"  ❌ PyTorch: Not installed")
    
    try:
        import cv2
        print(f"  ✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print(f"  ❌ OpenCV: Not installed")
    
    try:
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")
    except ImportError:
        print(f"  ❌ NumPy: Not installed")
    
    # Show project files
    print(f"\n📁 Project Files:")
    files = [f for f in os.listdir('.') if f.endswith('.py')]
    for file in sorted(files):
        if os.path.exists(file):
            print(f"  📄 {file}")
    
    # System requirements
    print(f"\n💻 System Requirements:")
    print(f"  • Python 3.8+ (for TensorFlow)")
    print(f"  • Python 3.7+ (for PyTorch)")
    print(f"  • NVIDIA GPU with CUDA support (recommended)")
    print(f"  • 8GB+ RAM for optimal performance")
    print(f"  • Windows 10/11 with latest GPU drivers")
    
    # Installation guide
    print(f"\n📦 Installation Commands:")
    print(f"  pip install -r requirements.txt")
    print(f"  # OR install individually:")
    print(f"  pip install tensorflow[and-cuda] torch torchvision opencv-python")
    
    input(f"\nPress Enter to continue...")


def check_dependencies():
    """Check if required dependencies are installed."""
    # Map package names to their import names
    required_packages = {
        'tensorflow': 'tensorflow',
        'opencv-python': 'cv2',
        'numpy': 'numpy'
    }
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"  ❌ {package}")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True


def main():
    """Main launcher function."""
    print_banner()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies before continuing.")
        input("Press Enter to exit...")
        return
    
    while True:
        try:
            print_menu()
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                run_unified_camera()
            elif choice == '2':
                run_modular_system()
            elif choice == '3':
                run_ishihara_demo()
            elif choice == '4':
                run_gpu_test()
            elif choice == '5':
                show_configuration()
            elif choice == '6':
                print("\n👋 Thank you for using ReColor CVD Processing Pipeline!")
                print("Happy coding! 🚀")
                break
            else:
                print("\n❌ Invalid choice. Please enter 1-6.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()