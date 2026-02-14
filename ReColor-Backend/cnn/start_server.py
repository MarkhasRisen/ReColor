#!/usr/bin/env python3
"""
ReColor Backend Startup Script
Quick startup for the unified color detection API server
"""

import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Start the ReColor API server."""
    print("🎨 ReColor Backend - Unified Color Detection System")
    print("=" * 50)
    
    try:
        # Import and start the API server
        from api_server import api_instance
        
        print("🚀 Starting ReColor API server...")
        print("📡 Endpoints available:")
        print("   - GET  /health                  (Health check)")
        print("   - POST /api/detect-color       (Color detection with CVD simulation)")
        print("   - GET  /api/cvd-types          (Available CVD types)")
        print("   - GET  /api/color-categories   (Primary vs Secondary colors)")
        print("   - POST /api/simulate-cvd       (CVD simulation only)")
        print()
        print("🔬 Features:")
        print("   ✓ CNN-based color classification (Primary: RGB, Secondary: Others)")
        print("   ✓ CVD simulation (Protanopia, Deuteranopia, Tritanopia)")
        print("   ✓ Unified color pipeline")
        print("   ✓ Real-time processing optimized for React Native")
        print()
        print("🌐 Server starting on http://localhost:8000")
        print("📱 Make sure your React Native app points to this URL")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the server
        api_instance.run(host='0.0.0.0', port=8000, debug=False)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down ReColor Backend...")
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install flask flask-cors tensorflow opencv-python pillow numpy")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()