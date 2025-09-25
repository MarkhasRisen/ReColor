"""
TensorFlow Real-Time Image Processing - Launcher
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
    print("=" * 70)
    print("  TensorFlow Real-Time Image Processing Pipeline")
    print("=" * 70)
    print("  Pure TensorFlow implementation for real-time video processing")
    print("  Features: K-Means clustering, Daltonization, webcam capture")
    print("=" * 70)


def print_menu():
    """Print main menu options."""
    print("\nSelect processing mode:")
    print()
    print("1. 🎮 Demo Mode (No camera required)")
    print("   - Synthetic animated video")
    print("   - Perfect for testing algorithms")
    print("   - Safe to run anywhere")
    print()
    print("2. 📹 Basic Real-Time (Camera required)")
    print("   - Standard real-time processing")
    print("   - Good for development and testing")
    print("   - Shows all processing steps")
    print()
    print("3. ⚡ Optimized Real-Time (Camera required)")
    print("   - Performance-optimized version")
    print("   - Best for production use")
    print("   - Advanced performance monitoring")
    print()
    print("4. ⚙️  Configuration Info")
    print("   - View current settings")
    print("   - Performance parameters")
    print("   - System requirements")
    print()
    print("5. 📚 Documentation")
    print("   - Open README file")
    print("   - Implementation guide")
    print("   - Usage instructions")
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


def run_demo_mode():
    """Run demo mode."""
    print("\n🎮 Starting Demo Mode...")
    print("This will show synthetic animated video with real-time processing.")
    print("Perfect for testing when no camera is available.")
    print("\nControls:")
    print("  Q - Quit")
    print("  K - Cycle K-Means clusters")
    print("  D - Cycle deficiency types")
    print("  F - Toggle FPS display")
    print()
    
    input("Press Enter to start demo, or Ctrl+C to cancel...")
    
    try:
        subprocess.run([sys.executable, "realtime_demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
    except KeyboardInterrupt:
        print("Demo cancelled by user")


def run_basic_realtime():
    """Run basic real-time processing with deficiency type selection."""
    print("\n📹 Starting Basic Real-Time Processing...")
    print("This will use your camera for real-time video processing.")
    print("Make sure your camera is connected and not in use by other apps.")
    print("\nThe system will show three windows:")
    print("  1. Original camera feed")
    print("  2. Daltonized output")
    print("  3. K-Means clustered output")
    
    # Select deficiency type
    selected_deficiency = select_deficiency_type()
    if selected_deficiency is None:
        return
    
    print()
    choice = input("Continue? (y/N): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        return
    
    # Set environment variable for the selected deficiency
    import os
    os.environ['REALTIME_DEFICIENCY'] = selected_deficiency
    print(f"Starting with deficiency type: {selected_deficiency}")
    
    try:
        subprocess.run([sys.executable, "realtime_processor.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running real-time processor: {e}")
        print("Try demo mode if camera is not available.")
    except KeyboardInterrupt:
        print("Real-time processing cancelled by user")
    finally:
        # Clean up environment variable
        if 'REALTIME_DEFICIENCY' in os.environ:
            del os.environ['REALTIME_DEFICIENCY']


def run_optimized_realtime():
    """Run optimized real-time processing with deficiency type selection."""
    print("\n⚡ Starting Optimized Real-Time Processing...")
    print("This is the performance-optimized version with advanced monitoring.")
    print("Best for production use and performance testing.")
    
    # Select deficiency type
    selected_deficiency = select_deficiency_type()
    if selected_deficiency is None:
        return
    
    print()
    choice = input("Continue? (y/N): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        return
    
    # Set environment variable for the selected deficiency
    import os
    os.environ['REALTIME_DEFICIENCY'] = selected_deficiency
    print(f"Starting with deficiency type: {selected_deficiency}")
    
    try:
        subprocess.run([sys.executable, "optimized_realtime.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running optimized processor: {e}")
        print("Try basic mode or demo mode instead.")
    except KeyboardInterrupt:
        print("Optimized processing cancelled by user")
    finally:
        # Clean up environment variable
        if 'REALTIME_DEFICIENCY' in os.environ:
            del os.environ['REALTIME_DEFICIENCY']


def show_configuration():
    """Show configuration information."""
    print("\n⚙️  Configuration Information")
    print("=" * 50)
    
    try:
        # Import and display configuration
        from realtime_config import print_realtime_config
        print_realtime_config()
        
        print("\nFiles in project:")
        files = [f for f in os.listdir('.') if f.endswith(('.py', '.md', '.txt'))]
        for file in sorted(files):
            print(f"  📄 {file}")
        
    except ImportError:
        print("Configuration module not found.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
    
    print("\nSystem Requirements:")
    print("  • Python 3.8+")
    print("  • TensorFlow 2.10+")
    print("  • OpenCV 4.5+")
    print("  • 4GB+ RAM recommended")
    print("  • GPU optional but recommended")
    
    input("\nPress Enter to continue...")


def show_documentation():
    """Show documentation."""
    print("\n📚 Documentation")
    print("=" * 30)
    
    docs = [
        ("REALTIME_README.md", "Main documentation for real-time processing"),
        ("realtime_config.py", "Configuration parameters and settings"),
        ("requirements.txt", "Required Python packages"),
    ]
    
    print("Available documentation:")
    for doc, desc in docs:
        if os.path.exists(doc):
            print(f"  ✅ {doc} - {desc}")
        else:
            print(f"  ❌ {doc} - {desc} (Missing)")
    
    print("\nImplementation Status:")
    print("  🔧 tf_kmeans() - Placeholder (needs implementation)")
    print("  🔧 daltonize() - Placeholder (needs implementation)")
    print("  ✅ Real-time pipeline - Complete")
    print("  ✅ Camera capture - Complete")
    print("  ✅ Performance monitoring - Complete")
    
    print("\nQuick Start:")
    print("  1. Run demo mode to test without camera")
    print("  2. Implement tf_kmeans() function")
    print("  3. Implement daltonize() function")
    print("  4. Test with real camera")
    
    input("\nPress Enter to continue...")


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
                run_demo_mode()
            elif choice == '2':
                run_basic_realtime()
            elif choice == '3':
                run_optimized_realtime()
            elif choice == '4':
                show_configuration()
            elif choice == '5':
                show_documentation()
            elif choice == '6':
                print("\n👋 Thank you for using TensorFlow Real-Time Image Processing!")
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