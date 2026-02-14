#!/usr/bin/env python3
"""
ReColor Backend Server - Improved Startup Script
Ensures proper directory navigation and model loading
"""

import os
import sys
from pathlib import Path
import logging

def setup_backend_environment():
    """Setup the backend environment and change to correct directory"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    backend_dir = script_dir
    
    # Change to the backend directory
    original_cwd = os.getcwd()
    os.chdir(backend_dir)
    
    print(f"Starting ReColor Backend API")
    print(f"Script directory: {script_dir}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Original directory: {original_cwd}")
    
    # Add backend directory to Python path
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    return backend_dir

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'flask',
        'flask_cors', 
        'tensorflow',
        'numpy',
        'opencv-python',
        'pillow',
        'scikit-learn'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'opencv-python':
                import cv2
            elif module == 'flask_cors':
                from flask_cors import CORS
            elif module == 'pillow':
                from PIL import Image
            elif module == 'scikit-learn':
                import sklearn
            else:
                __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Missing required modules: {', '.join(missing_modules)}")
        print("Install them with: pip install " + " ".join(missing_modules))
        return False
    
    print("All required dependencies are available")
    return True

def check_model_files():
    """Check for model files and create directory if needed"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Check for existing model weights
    weight_files = list(models_dir.glob('*.h5'))
    
    if weight_files:
        print(f"Found model weights: {[f.name for f in weight_files]}")
        return True
    else:
        print("No pre-trained model weights found. Model will train on startup.")
        return False

def main():
    """Main startup function"""
    try:
        print("ReColor Backend - Starting up...")
        print("=" * 50)
        
        # Setup environment
        backend_dir = setup_backend_environment()
        
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # Check model files
        check_model_files()
        
        # Import and start the API server
        print("\nImporting API server...")
        
        try:
            from api_server import app
            print("API server module loaded successfully")
        except ImportError as e:
            print(f"Failed to import API server: {e}")
            print("Make sure api_server.py exists in the current directory")
            return 1
        
        print(f"\nStarting ReColor API server...")
        print(f"Backend will be available at: http://localhost:8000")
        print(f"Health check: http://localhost:8000/health")
        print(f"API endpoints: http://localhost:8000/api/")
        print("=" * 50)
        
        # Configure logging for Flask
        logging.getLogger('werkzeug').setLevel(logging.INFO)
        
        # Start the Flask application
        app.run(
            host='0.0.0.0',
            port=8000,
            debug=False,
            use_reloader=False,  # Disable reloader to prevent double startup
            threaded=True       # Enable threading for better performance
        )
        
    except KeyboardInterrupt:
        print("\nServer shutdown requested by user")
        return 0
    except Exception as e:
        print(f"Fatal error starting backend: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())