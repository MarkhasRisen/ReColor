#!/usr/bin/env python3
"""
ReColor Backend Server Startup Script
This script ensures the server runs from the correct directory to find all dependencies.
"""

import os
import sys
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the backend directory
    backend_dir = script_dir
    os.chdir(backend_dir)
    
    print(f"Starting ReColor API server from: {backend_dir}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Import and run the API server
    try:
        from api_server import app
        print("✅ API server module loaded successfully")
        
        # Start the server
        app.run(
            host='0.0.0.0',
            port=8000,
            debug=False,
            use_reloader=False  # Disable reloader to prevent double startup
        )
        
    except ImportError as e:
        print(f"❌ Failed to import API server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()