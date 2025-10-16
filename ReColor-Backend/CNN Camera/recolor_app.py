"""
ReColorApp - Main application controller for TensorFlow colorblind detection system.
Orchestrates all components including color model, CVD simulator, camera handler, and logger.
"""

import logging
import signal
import sys
import os
import time
from typing import Dict, Optional

from color_model import ColorModel
from colorblind_detector import ColorBlindnessSimulator, CVDType
from camera_handler import CameraHandler
from color_logger import ColorLogger
from utils import setup_gpu


class ReColorApp:
    """
    Main application controller that orchestrates all ReColor components.
    Provides initialization, startup, and shutdown management for the colorblind detection system.
    """
    
    def __init__(self):
        """Initialize ReColorApp with default settings."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Application components
        self.color_model = None
        self.cvd_simulator = None
        self.camera_handler = None
        self.color_logger = None
        
        # Application state
        self.is_running = False
        self.gpu_info = None
        
        # Configuration
        self.config = {
            'camera_id': 0,
            'frame_width': 640,
            'frame_height': 480,
            'fps_target': 30,
            'model_path': None,
            'log_directory': 'logs',
            'auto_train_model': True,
            'training_samples_per_class': 500,
            'training_epochs': 20,
            # Enhanced features
            'kmeans_clusters': 5,
            'simplification_clusters': 8,
            'daltonization_strength': 1.5,
            'enable_kmeans': False,
            'enable_simplification': False,
            'enable_daltonization': False,
            'enable_realistic_cvd': False,
            'enable_unified_pipeline': False
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("ReColorApp initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self, config: Optional[Dict] = None) -> bool:
        """
        Initialize all application components.
        
        Args:
            config: Optional configuration dictionary to override defaults
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Update configuration if provided
            if config:
                self.config.update(config)
            
            self.logger.info("Initializing ReColor application...")
            
            # Setup GPU and get information
            self.gpu_info = setup_gpu()
            self._log_system_info()
            
            # Initialize color model
            if not self._initialize_color_model():
                return False
            
            # Initialize CVD simulator
            if not self._initialize_cvd_simulator():
                return False
            
            # Initialize color logger
            if not self._initialize_color_logger():
                return False
            
            # Initialize camera handler
            if not self._initialize_camera_handler():
                return False
            
            self.logger.info("ReColor application initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            return False
    
    def _log_system_info(self) -> None:
        """Log system and GPU information."""
        self.logger.info("=== System Information ===")
        
        if self.gpu_info['gpu_available']:
            self.logger.info(f"✅ GPU: {self.gpu_info['gpu_name']}")
            self.logger.info(f"GPU Memory: {self.gpu_info['gpu_memory']}")
            self.logger.info(f"Device: {self.gpu_info['device']}")
        else:
            self.logger.warning("⚠️ No GPU detected. Using CPU for inference.")
            self.logger.info(f"Device: {self.gpu_info['device']}")
        
        self.logger.info(f"Camera ID: {self.config['camera_id']}")
        self.logger.info(f"Target Resolution: {self.config['frame_width']}x{self.config['frame_height']}")
        self.logger.info(f"Target FPS: {self.config['fps_target']}")
        self.logger.info("=" * 30)
    
    def _initialize_color_model(self) -> bool:
        """Initialize the color recognition model."""
        try:
            self.logger.info("Initializing color model...")
            
            # Create color model
            self.color_model = ColorModel(num_classes=9, input_shape=(64, 64, 3))
            
            # Load pre-trained weights if available
            model_path = self.config.get('model_path')
            if model_path and os.path.exists(model_path):
                if self.color_model.load_weights(model_path):
                    self.logger.info(f"Loaded pre-trained weights from {model_path}")
                else:
                    self.logger.warning("Failed to load pre-trained weights")
            
            # Train model if not already trained and auto-training is enabled
            if not self.color_model.is_trained and self.config.get('auto_train_model', True):
                self.logger.info("Training color model on synthetic data...")
                self.logger.info("This may take a few minutes...")
                
                history = self.color_model.train_on_synthetic_data(
                    samples_per_class=self.config.get('training_samples_per_class', 500),
                    epochs=self.config.get('training_epochs', 20),
                    validation_split=0.2
                )
                
                if history:
                    final_acc = history.get('accuracy', [0])[-1]
                    final_val_acc = history.get('val_accuracy', [0])[-1]
                    self.logger.info(f"Training completed! Accuracy: {final_acc:.3f}, Val Accuracy: {final_val_acc:.3f}")
                    
                    # Save trained model
                    save_path = os.path.join('models', 'color_model_weights.h5')
                    if self.color_model.save_weights(save_path):
                        self.logger.info(f"Model weights saved to {save_path}")
                else:
                    self.logger.error("Model training failed")
                    return False
            
            # Test model with evaluation
            if self.color_model.is_trained:
                metrics = self.color_model.evaluate_on_test_data(200)
                self.logger.info(f"Model evaluation - Accuracy: {metrics.get('test_accuracy', 0):.2f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing color model: {e}")
            return False
    
    def _initialize_cvd_simulator(self) -> bool:
        """Initialize the color vision deficiency simulator."""
        try:
            self.logger.info("Initializing CVD simulator...")
            
            self.cvd_simulator = ColorBlindnessSimulator()
            
            # Test CVD simulator
            import numpy as np
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            for cvd_type in [CVDType.PROTANOPIA, CVDType.DEUTERANOPIA, CVDType.TRITANOPIA]:
                simulated = self.cvd_simulator.simulate(test_image, cvd_type)
                if simulated is not None:
                    self.logger.info(f"✅ {cvd_type.value.capitalize()} simulation ready")
                else:
                    self.logger.warning(f"⚠️ {cvd_type.value.capitalize()} simulation failed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing CVD simulator: {e}")
            return False
    
    def _initialize_color_logger(self) -> bool:
        """Initialize the color logging system."""
        try:
            self.logger.info("Initializing color logger...")
            
            log_directory = self.config.get('log_directory', 'logs')
            self.color_logger = ColorLogger(
                log_directory=log_directory,
                auto_create_session=True
            )
            
            self.logger.info(f"Color logger ready. Log directory: {log_directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing color logger: {e}")
            return False
    
    def _initialize_camera_handler(self) -> bool:
        """Initialize the camera handler."""
        try:
            self.logger.info("Initializing camera handler...")
            
            self.camera_handler = CameraHandler(
                color_model=self.color_model,
                cvd_simulator=self.cvd_simulator,
                camera_id=self.config['camera_id'],
                frame_width=self.config['frame_width'],
                frame_height=self.config['frame_height'],
                fps_target=self.config['fps_target']
            )
            
            # Configure enhanced features
            self.camera_handler.kmeans_k = self.config['kmeans_clusters']
            self.camera_handler.kmeans_simplification_k = self.config['simplification_clusters']
            self.camera_handler.daltonization_strength = self.config['daltonization_strength']
            self.camera_handler.show_kmeans_analysis = self.config['enable_kmeans']
            self.camera_handler.show_simplified = self.config['enable_simplification']
            self.camera_handler.show_daltonization = self.config['enable_daltonization']
            self.camera_handler.use_realistic_cvd = self.config['enable_realistic_cvd']
            self.camera_handler.use_unified_pipeline = self.config['enable_unified_pipeline']
            
            # Set up event callbacks
            self.camera_handler.set_frame_capture_callback(self._on_frame_captured)
            self.camera_handler.set_quit_callback(self._on_quit_requested)
            
            self.logger.info("Camera handler ready")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera handler: {e}")
            return False
    
    def _on_frame_captured(self, color_info: Dict) -> None:
        """
        Handle frame capture events from camera handler.
        
        Args:
            color_info: Color information dictionary
        """
        try:
            # Get current CVD type
            current_cvd = self.cvd_simulator.get_current_cvd_type()
            
            # Log the capture
            success = self.color_logger.log_color_capture(color_info, current_cvd)
            
            if success:
                rgb = color_info.get('dominant_rgb', (0, 0, 0))
                predicted = color_info.get('predicted_color', 'Unknown')
                confidence = color_info.get('confidence', 0.0)
                
                self.logger.info(f"📸 Captured: {predicted} (conf: {confidence:.2f}) - RGB{rgb} - CVD: {current_cvd.value}")
            else:
                self.logger.error("Failed to log color capture")
                
        except Exception as e:
            self.logger.error(f"Error handling frame capture: {e}")
    
    def _on_quit_requested(self) -> None:
        """Handle quit request from camera handler."""
        self.logger.info("Quit requested by user")
        self.shutdown()
    
    def run(self) -> None:
        """Start the main application loop."""
        try:
            if not self.color_model or not self.cvd_simulator or not self.camera_handler:
                self.logger.error("Application not properly initialized. Call initialize() first.")
                return
            
            self.is_running = True
            
            self.logger.info("🚀 Starting ReColor application...")
            self.logger.info("=" * 50)
            
            # Print usage instructions
            self._print_usage_instructions()
            
            # Start camera loop (this will block until user quits)
            self.camera_handler.start_camera_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during application run: {e}")
        finally:
            self.shutdown()
    
    def _print_usage_instructions(self) -> None:
        """Print usage instructions for the user."""
        instructions = [
            "🎯 ReColor - TensorFlow Colorblind Detection System",
            "",
            "CONTROLS:",
            "  C - Capture current color and save to CSV",
            "  N - Cycle through CVD types (Normal → Protanopia → Deuteranopia → Tritanopia)",
            "  P - Pause/Resume camera feed",
            "  S - Toggle side-by-side display (Normal vs CVD simulation)",
            "  I - Toggle color information overlay",
            "  F - Toggle FPS display",
            "  Q - Quit application",
            "",
            "FEATURES:",
            "  • Real-time color detection using TensorFlow CNN",
            "  • Scientifically accurate colorblind vision simulation",
            "  • Side-by-side comparison of normal vs CVD vision",
            "  • Color information display (RGB, HEX, AI prediction)",
            "  • CSV logging of all captured colors with timestamps",
            "  • GPU acceleration (when available)",
            "",
            "📊 All captured data is automatically saved to CSV files in the logs/ directory",
            "=" * 50
        ]
        
        for line in instructions:
            print(line)
        
        print()  # Extra line for spacing
    
    def shutdown(self) -> None:
        """Shutdown application and cleanup resources."""
        if not self.is_running:
            return
        
        self.logger.info("🔄 Shutting down ReColor application...")
        
        self.is_running = False
        
        # Stop camera handler
        if self.camera_handler:
            try:
                self.camera_handler.stop_camera()
                self.logger.info("Camera handler stopped")
            except Exception as e:
                self.logger.error(f"Error stopping camera handler: {e}")
        
        # Generate final session summary
        if self.color_logger:
            try:
                summary = self.color_logger.get_session_summary()
                self._print_session_summary(summary)
                
                # Export session data
                export_path = self.color_logger.export_session_data("json")
                if export_path:
                    self.logger.info(f"Session data exported to: {export_path}")
                
                # Close session
                self.color_logger.close_session()
                
            except Exception as e:
                self.logger.error(f"Error during logger shutdown: {e}")
        
        self.logger.info("✅ ReColor application shutdown complete")
    
    def _print_session_summary(self, summary: Dict) -> None:
        """Print session summary to console."""
        try:
            print("\n" + "=" * 50)
            print("📊 SESSION SUMMARY")
            print("=" * 50)
            
            if summary.get('total_captures', 0) > 0:
                print(f"Session ID: {summary.get('session_id', 'N/A')}")
                print(f"Duration: {summary.get('session_duration_seconds', 0):.1f} seconds")
                print(f"Total Captures: {summary.get('total_captures', 0)}")
                print(f"Unique Colors: {summary.get('unique_colors', 0)}")
                print(f"Most Common Color: {summary.get('most_common_color', 'N/A')}")
                print(f"Average Confidence: {summary.get('avg_confidence', 0):.3f}")
                print(f"Log File: {summary.get('log_file', 'N/A')}")
                
                # Color distribution
                color_dist = summary.get('color_distribution', {})
                if color_dist:
                    print(f"\nColor Distribution:")
                    for color, count in sorted(color_dist.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {color}: {count} captures")
                
                # CVD type distribution
                cvd_dist = summary.get('cvd_type_distribution', {})
                if cvd_dist:
                    print(f"\nCVD Type Distribution:")
                    for cvd_type, count in cvd_dist.items():
                        print(f"  {cvd_type.title()}: {count} captures")
                        
            else:
                print("No captures were made during this session.")
            
            print("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Error printing session summary: {e}")
    
    def get_system_status(self) -> Dict:
        """
        Get current system status.
        
        Returns:
            Dictionary with system status information
        """
        status = {
            'app_running': self.is_running,
            'gpu_info': self.gpu_info,
            'config': self.config.copy()
        }
        
        if self.color_model:
            status['model_trained'] = self.color_model.is_trained
            status['model_device'] = self.color_model.device
        
        if self.camera_handler:
            status['camera_info'] = self.camera_handler.get_camera_info()
        
        if self.color_logger:
            status['session_summary'] = self.color_logger.get_session_summary()
        
        return status


# Test the ReColorApp (basic initialization test)
if __name__ == "__main__":
    print("Testing ReColorApp initialization...")
    
    try:
        # Create app instance
        app = ReColorApp()
        
        # Test initialization with minimal training for speed
        test_config = {
            'camera_id': 0,  # May fail if no camera
            'auto_train_model': True,
            'training_samples_per_class': 50,  # Reduced for testing
            'training_epochs': 3  # Reduced for testing
        }
        
        print("Initializing application components...")
        success = app.initialize(test_config)
        
        if success:
            print("✅ ReColorApp initialization successful!")
            
            # Get system status
            status = app.get_system_status()
            print(f"GPU Available: {status['gpu_info']['gpu_available']}")
            print(f"Model Trained: {status.get('model_trained', False)}")
            
            # Note: We don't run the main loop in testing to avoid requiring a camera
            print("📝 Note: Main application loop not started in test mode")
            print("To run full application with camera, use: python main.py --camera 0")
            
        else:
            print("❌ ReColorApp initialization failed")
        
        # Cleanup
        app.shutdown()
        
        print("ReColorApp test completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()