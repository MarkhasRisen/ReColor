#!/usr/bin/env python3
"""
ReColor Backend Quick Start
Uses Python 3.13 with TensorFlow 2.20.0
"""

import sys
import os
from pathlib import Path

def main():
    """Start the ReColor backend server."""
    print("🎨 ReColor Backend - Starting with Python 3.13")
    print("=" * 50)
    
    try:
        # Test package availability first
        print("🔍 Checking required packages...")
        
        import tensorflow as tf
        print(f"   ✅ TensorFlow: {tf.__version__}")
        
        import cv2
        print(f"   ✅ OpenCV: {cv2.__version__}")
        
        import flask
        print(f"   ✅ Flask: {flask.__version__}")
        
        import numpy as np
        print(f"   ✅ NumPy: {np.__version__}")
        
        print("\n🚀 All packages loaded successfully!")
        print("🌐 Starting ReColor API server...")
        print("📡 Available at: http://localhost:8000")
        print("🎯 Features:")
        print("   ✓ CNN color detection (TensorFlow 2.20.0)")
        print("   ✓ Primary colors: Red, Green, Blue")
        print("   ✓ Secondary colors: Orange, Purple, Pink, Brown, Yellow, White")
        print("   ✓ CVD simulation: Protanopia, Deuteranopia, Tritanopia")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        # Import and start the API server
        from api_server import api_instance
        api_instance.run(host='0.0.0.0', port=8000, debug=False)
        
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\n💡 Solution:")
        print("   Make sure you're in the Recolor-Backend directory (note: lowercase 'c')")
        print("   The packages are installed there with Python 3.13")
        print("   Run: cd ../Recolor-Backend")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down ReColor Backend...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()