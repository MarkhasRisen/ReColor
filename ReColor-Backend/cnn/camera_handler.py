"""
CameraHandler for ReColor TensorFlow colorblind detection system.
Manages webcam capture, real-time display, and user interactions with keyboard controls.
"""

import cv2
import numpy as np
import logging
import time
from typing import Tuple, Dict, Optional, Callable
import threading
from queue import Queue

from color_model import ColorModel
from colorblind_detector import ColorBlindnessSimulator, CVDType
from utils import (get_dominant_color, rgb_to_hex, get_color_name, create_color_swatch,
                   extract_dominant_colors_kmeans, simplify_image_kmeans, analyze_color_distribution)
from unified_color_pipeline import UnifiedColorPipeline
from error_handler import (
    AnomalyDetector, RecoveryManager, HealthMonitor, 
    error_context, AnomalyType, AnomalyEvent
)
from improved_color_detection import (
    ImprovedColorDetector, get_improved_color_info, analyze_color_consistency
)


class CameraHandler:
    """
    Handles webcam capture, real-time color detection, and CVD simulation display.
    Provides interactive controls for capturing frames and cycling through CVD types.
    """
    
    def __init__(self, 
                 color_model: ColorModel,
                 cvd_simulator: ColorBlindnessSimulator,
                 camera_id: int = 0,
                 frame_width: int = 640,
                 frame_height: int = 480,
                 fps_target: int = 30):
        """
        Initialize CameraHandler.
        
        Args:
            color_model: Trained ColorModel instance for color prediction
            cvd_simulator: ColorBlindnessSimulator for CVD simulation
            camera_id: Camera device ID (0 for default camera)
            frame_width: Target frame width
            frame_height: Target frame height
            fps_target: Target FPS for display
        """
        self.color_model = color_model
        self.cvd_simulator = cvd_simulator
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps_target = fps_target
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Enhanced error handling components
        self.anomaly_detector = AnomalyDetector(self.logger)
        self.recovery_manager = RecoveryManager(self.logger)
        self.health_monitor = HealthMonitor(self.logger)
        
        # Improved color detection system
        self.improved_color_detector = ImprovedColorDetector()
        
        # Camera state
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.camera_recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        # Display settings
        self.show_side_by_side = True
        self.show_color_info = True
        self.show_fps = True
        self.show_kmeans_analysis = False
        self.show_daltonization = False
        self.use_realistic_cvd = False
        
        # K-means settings
        self.kmeans_k = 5  # Number of color clusters
        self.kmeans_simplification_k = 8  # For image simplification
        self.show_simplified = False
        
        # Daltonization settings
        self.daltonization_strength = 1.5
        
        # Unified pipeline settings
        self.use_unified_pipeline = False
        self.unified_pipeline = None
        self.pipeline_results = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.frame_times = []
        
        # Color analysis
        self.current_color_info = None
        self.current_kmeans_analysis = None
        self.roi_size = 50  # Size of region of interest for color detection
        
        # Event callbacks
        self.on_frame_captured = None
        self.on_quit = None
        
        # Threading for smooth display
        self.frame_queue = Queue(maxsize=2)
        self.processing_thread = None
        self.capture_thread = None
        
        # Initialize unified pipeline
        self._initialize_unified_pipeline()
        
        self.logger.info(f"CameraHandler initialized for camera {camera_id}")
        self.logger.info(f"Target resolution: {frame_width}x{frame_height}")
        self.logger.info(f"Target FPS: {fps_target}")
    
    def _initialize_unified_pipeline(self) -> None:
        """Initialize the unified color processing pipeline."""
        try:
            self.unified_pipeline = UnifiedColorPipeline(
                color_model=self.color_model,
                cvd_simulator=self.cvd_simulator,
                kmeans_clusters=self.kmeans_k,
                cnn_patch_size=32,
                enable_cpu_optimization=True
            )
            self.logger.info("Unified color pipeline initialized")
        except Exception as e:
            self.logger.error(f"Error initializing unified pipeline: {e}")
            self.unified_pipeline = None
    
    def initialize_camera(self) -> bool:
        """
        Initialize camera with enhanced error handling and recovery.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        with error_context("camera_initialization", self.logger, 
                          self.anomaly_detector, self.recovery_manager):
            
            # Clean up existing camera connection
            if self.cap and self.cap.isOpened():
                self.cap.release()
                time.sleep(0.5)  # Allow time for resource cleanup
            
            # Try different camera backends if default fails
            backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
            
            for backend in [None] + backends_to_try:  # Try default first
                try:
                    # Open camera with specific backend if specified
                    if backend is None:
                        self.cap = cv2.VideoCapture(self.camera_id)
                    else:
                        self.cap = cv2.VideoCapture(self.camera_id, backend)
                    
                    if not self.cap.isOpened():
                        if backend is not None:
                            self.logger.warning(f"Camera {self.camera_id} failed with backend {backend}")
                        continue
                    
                    # Test camera by reading a frame
                    ret, test_frame = self.cap.read()
                    if not ret or test_frame is None:
                        self.logger.warning(f"Camera {self.camera_id} opened but cannot read frames")
                        self.cap.release()
                        continue
                    
                    # Check for frame anomalies
                    frame_anomalies = self.anomaly_detector.detect_frame_anomalies(test_frame)
                    if frame_anomalies:
                        severe_anomalies = [a for a in frame_anomalies if a.severity in ['high', 'critical']]
                        if severe_anomalies:
                            self.logger.warning(f"Camera {self.camera_id} has severe frame anomalies")
                            for anomaly in severe_anomalies:
                                self.anomaly_detector.log_anomaly(anomaly)
                            self.cap.release()
                            continue
                    
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
                    
                    # Set additional properties for better performance
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
                    
                    # Get actual camera properties
                    actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    
                    # Validate properties
                    if actual_width <= 0 or actual_height <= 0:
                        raise ValueError(f"Invalid camera resolution: {actual_width}x{actual_height}")
                    
                    self.logger.info(f"Camera {self.camera_id} initialized successfully")
                    self.logger.info(f"Actual resolution: {actual_width}x{actual_height}")
                    self.logger.info(f"Actual FPS: {actual_fps}")
                    if backend is not None:
                        self.logger.info(f"Using backend: {backend}")
                    
                    self.camera_recovery_attempts = 0  # Reset recovery counter on success
                    return True
                    
                except cv2.error as e:
                    self.logger.error(f"OpenCV error with camera {self.camera_id}: {e}")
                    if self.cap:
                        self.cap.release()
                    
                    # Log camera failure anomaly
                    anomaly = AnomalyEvent(
                        timestamp=time.time(),
                        anomaly_type=AnomalyType.CAMERA_FAILURE,
                        severity='high',
                        component='camera',
                        description=f'Camera initialization failed: {str(e)}',
                        context={
                            'camera_id': self.camera_id,
                            'backend': backend,
                            'error_type': 'cv2_error'
                        }
                    )
                    self.anomaly_detector.log_anomaly(anomaly)
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error initializing camera {self.camera_id}: {e}")
                    if self.cap:
                        self.cap.release()
            
            # All backends failed
            self.logger.error(f"Failed to initialize camera {self.camera_id} with all available backends")
            
            # Log critical camera failure
            anomaly = AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.CAMERA_FAILURE,
                severity='critical',
                component='camera',
                description='Camera initialization completely failed',
                context={'camera_id': self.camera_id}
            )
            self.anomaly_detector.log_anomaly(anomaly)
            
            return False
    
    def start_camera_loop(self) -> None:
        """Start the main camera loop with real-time display."""
        if not self.initialize_camera():
            self.logger.error("Failed to initialize camera. Exiting.")
            return
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Camera loop started")
        self.logger.info("Controls:")
        self.logger.info("  C - Capture current frame and color")
        self.logger.info("  N - Cycle through CVD simulation types")
        self.logger.info("  P - Toggle pause")
        self.logger.info("  S - Toggle side-by-side display")
        self.logger.info("  I - Toggle color info display")
        self.logger.info("  F - Toggle FPS display")
        self.logger.info("  Q - Quit")
        
        try:
            # Main display loop
            while self.is_running:
                # Get processed frame from queue
                if not self.frame_queue.empty():
                    display_frame = self.frame_queue.get()
                    
                    # Display frame
                    cv2.imshow('ReColor - Colorblind Detection', display_frame)
                    
                    # Update FPS
                    self._update_fps()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard(key):
                    break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(1 / (self.fps_target * 2))
            
        except KeyboardInterrupt:
            self.logger.info("Camera loop interrupted by user")
        finally:
            self.stop_camera()
    
    def _processing_loop(self) -> None:
        """Background processing loop for frame capture and analysis."""
        while self.is_running:
            frame_start_time = time.time()
            
            with error_context("frame_processing", self.logger, 
                              self.anomaly_detector, self.recovery_manager):
                
                # Check camera connection
                if self.cap is None or not self.cap.isOpened():
                    self.logger.error("Camera connection lost during processing")
                    
                    # Attempt camera recovery
                    if self.camera_recovery_attempts < self.max_recovery_attempts:
                        self.camera_recovery_attempts += 1
                        self.logger.info(f"Attempting camera recovery ({self.camera_recovery_attempts}/{self.max_recovery_attempts})")
                        
                        if self.recovery_manager.attempt_recovery(
                            AnomalyEvent(
                                timestamp=time.time(),
                                anomaly_type=AnomalyType.CAMERA_FAILURE,
                                severity='high',
                                component='processing_loop',
                                description='Camera connection lost',
                                context={'recovery_attempt': self.camera_recovery_attempts}
                            ),
                            {'camera': self}
                        ):
                            continue
                    
                    break
                
                # Capture frame with timeout protection
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.logger.warning("Failed to capture frame")
                    
                    # Log frame capture failure
                    anomaly = AnomalyEvent(
                        timestamp=time.time(),
                        anomaly_type=AnomalyType.FRAME_CORRUPTION,
                        severity='medium',
                        component='frame_capture',
                        description='Frame capture returned no data',
                        context={'ret': ret, 'frame_none': frame is None}
                    )
                    self.anomaly_detector.log_anomaly(anomaly)
                    continue
                
                # Detect frame anomalies
                frame_anomalies = self.anomaly_detector.detect_frame_anomalies(frame)
                for anomaly in frame_anomalies:
                    self.anomaly_detector.log_anomaly(anomaly)
                    
                    # Skip severely corrupted frames
                    if anomaly.severity == 'critical':
                        continue
                
                # Skip processing if paused
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Convert BGR to RGB for processing
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except cv2.error as e:
                    self.logger.error(f"Color conversion failed: {e}")
                    
                    anomaly = AnomalyEvent(
                        timestamp=time.time(),
                        anomaly_type=AnomalyType.FRAME_CORRUPTION,
                        severity='high',
                        component='color_conversion',
                        description=f'Frame color conversion failed: {str(e)}',
                        context={'frame_shape': frame.shape if frame is not None else None}
                    )
                    self.anomaly_detector.log_anomaly(anomaly)
                    continue
                
                # Analyze color in center region
                self.current_color_info = self._analyze_frame_color(rgb_frame)
                
                # Create display frame
                display_frame = self._create_display_frame(rgb_frame)
                
                # Add to display queue with error handling
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put(display_frame)
                    else:
                        # Replace old frame
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(display_frame)
                        except:
                            pass
                except Exception as e:
                    self.logger.warning(f"Error updating display queue: {e}")
                
                # Update performance metrics
                processing_time = time.time() - frame_start_time
                self.health_monitor.update_metrics(processing_time)
                
                # Check for processing time anomalies
                timing_anomalies = self.anomaly_detector.detect_processing_anomalies(processing_time)
                for anomaly in timing_anomalies:
                    self.anomaly_detector.log_anomaly(anomaly)
                
                # Log periodic health reports
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    self.health_monitor.log_health_report()
                
            time.sleep(0.001)  # Small delay
    
    def _analyze_frame_color(self, rgb_frame: np.ndarray) -> Dict:
        """
        Analyze color in the center region of the frame.
        
        Args:
            rgb_frame: Input frame in RGB format
            
        Returns:
            Dictionary with color analysis results
        """
        try:
            # Get frame center
            h, w = rgb_frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Extract ROI (region of interest) around center
            roi_half = self.roi_size // 2
            x1 = max(0, center_x - roi_half)
            y1 = max(0, center_y - roi_half)
            x2 = min(w, center_x + roi_half)
            y2 = min(h, center_y + roi_half)
            
            roi = rgb_frame[y1:y2, x1:x2]
            
            # Improved color detection (primary - fast and accurate)
            improved_color_info = get_improved_color_info(roi)
            
            # Get traditional dominant color for comparison
            dominant_rgb = get_dominant_color(roi, method="mean")
            dominant_hex = rgb_to_hex(dominant_rgb)
            
            # Predict color using CNN model
            predicted_color, confidence, probabilities = self.color_model.predict_color(roi)
            
            # Get traditional color name from RGB
            rgb_color_name = get_color_name(dominant_rgb)
            
            # Analyze color consistency across multiple regions
            color_consistency_analysis = analyze_color_consistency(rgb_frame, self.roi_size)
            
            # K-means analysis for dominant colors and simplification
            kmeans_analysis = None
            if self.show_kmeans_analysis:
                try:
                    kmeans_analysis = analyze_color_distribution(roi, self.kmeans_k)
                except Exception as e:
                    self.logger.warning(f"K-means analysis error: {e}")
            
            # Unified pipeline analysis if enabled
            unified_analysis = None
            if self.use_unified_pipeline and self.unified_pipeline is not None:
                try:
                    current_cvd = self.cvd_simulator.get_current_cvd_type()
                    self.pipeline_results = self.unified_pipeline.process_frame(
                        rgb_frame, 
                        current_cvd, 
                        return_intermediate=True
                    )
                    unified_analysis = {
                        'color_families': len(self.pipeline_results['color_families']['cluster_centers']) if self.pipeline_results['color_families'] else 0,
                        'cnn_patches': self.pipeline_results['cnn_classifications']['patches_processed'] if self.pipeline_results['cnn_classifications'] else 0,
                        'pipeline_fps': self.pipeline_results['pipeline_stats']['fps'],
                        'processing_time_ms': self.pipeline_results['pipeline_stats']['total_time_ms']
                    }
                except Exception as e:
                    self.logger.warning(f"Unified pipeline analysis error: {e}")
                    self.pipeline_results = None
            
            return {
                'roi_coords': (x1, y1, x2, y2),
                # Traditional color detection
                'dominant_rgb': dominant_rgb,
                'dominant_hex': dominant_hex,
                'rgb_color_name': rgb_color_name,
                # CNN model prediction
                'predicted_color': predicted_color,
                'confidence': confidence,
                'probabilities': probabilities,
                # Improved color detection results (primary)
                'improved_color_info': improved_color_info,
                'color_consistency_analysis': color_consistency_analysis,
                # Additional analysis
                'kmeans_analysis': kmeans_analysis,
                'unified_analysis': unified_analysis,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing frame color: {e}")
            return {
                'dominant_rgb': (128, 128, 128),
                'dominant_hex': '#808080',
                'rgb_color_name': 'Gray',
                'predicted_color': 'Unknown',
                'confidence': 0.0,
                'probabilities': np.zeros(9),
                'timestamp': time.time()
            }
    
    def _create_display_frame(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Create display frame with normal and CVD simulation views.
        
        Args:
            rgb_frame: Input frame in RGB format
            
        Returns:
            Display frame ready for showing
        """
        try:
            # Use unified pipeline results if available
            if self.use_unified_pipeline and self.pipeline_results is not None:
                if self.show_side_by_side:
                    # Show original vs unified pipeline result
                    normal_frame = self.pipeline_results['processed_frame']
                    enhanced_frame = self.pipeline_results['daltonized_frame']
                    combined_frame = np.hstack([normal_frame, enhanced_frame])
                    display_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
                else:
                    # Show only unified pipeline result
                    display_frame = cv2.cvtColor(self.pipeline_results['daltonized_frame'], cv2.COLOR_RGB2BGR)
            else:
                # Use traditional processing
                if self.show_side_by_side:
                    # Create side-by-side display
                    display_frame = self._create_side_by_side_display(rgb_frame)
                else:
                    # Show only current CVD simulation
                    simulated_frame = self.cvd_simulator.simulate_current(rgb_frame)
                    display_frame = cv2.cvtColor(simulated_frame, cv2.COLOR_RGB2BGR)
            
            # Add overlays
            if self.show_color_info and self.current_color_info:
                display_frame = self._add_color_info_overlay(display_frame)
            
            if self.show_fps:
                display_frame = self._add_fps_overlay(display_frame)
            
            # Add ROI indicator
            display_frame = self._add_roi_indicator(display_frame)
            
            # Add CVD type indicator
            display_frame = self._add_cvd_type_overlay(display_frame)
            
            return display_frame
            
        except Exception as e:
            self.logger.error(f"Error creating display frame: {e}")
            # Return original frame as fallback
            return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    def _create_side_by_side_display(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Create side-by-side display with normal and CVD simulation.
        
        Args:
            rgb_frame: Input frame in RGB format
            
        Returns:
            Side-by-side display frame in BGR format
        """
        # Apply K-means simplification if enabled
        if self.show_simplified:
            frame_to_process = simplify_image_kmeans(rgb_frame, self.kmeans_simplification_k)
        else:
            frame_to_process = rgb_frame.copy()
        
        # Normal vision (left side)
        if self.show_daltonization and self.cvd_simulator.get_current_cvd_type() != CVDType.NORMAL:
            # Show daltonized version for comparison
            normal_frame = self.cvd_simulator.daltonize(
                frame_to_process, 
                self.cvd_simulator.get_current_cvd_type(),
                self.daltonization_strength
            )
        else:
            normal_frame = frame_to_process.copy()
        
        # CVD simulation (right side)
        if self.use_realistic_cvd:
            simulated_frame = self.cvd_simulator.simulate_realistic_confusion(
                frame_to_process, 
                self.cvd_simulator.get_current_cvd_type()
            )
        else:
            simulated_frame = self.cvd_simulator.simulate_current(frame_to_process)
        
        # Combine side by side
        combined_frame = np.hstack([normal_frame, simulated_frame])
        
        # Convert to BGR for display
        return cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
    
    def _add_color_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Add color information overlay to frame.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Frame with color info overlay
        """
        try:
            if not self.current_color_info:
                return frame
            
            info = self.current_color_info
            
            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (255, 255, 255)  # White text
            
            # Background settings
            bg_color = (0, 0, 0)  # Black background
            padding = 10
            
            # Prepare text lines with improved color information
            improved_info = info.get('improved_color_info', {})
            consistency_info = info.get('color_consistency_analysis', {})
            
            text_lines = [
                # Improved detection results (primary - fast and accurate)
                f"🎯 Improved: {improved_info.get('name', 'Unknown')} ({improved_info.get('confidence', 0.0):.1%})",
                f"RGB: {improved_info.get('rgb', info['dominant_rgb'])}",
                f"HEX: {improved_info.get('hex', info['dominant_hex'])}",
                "",
                # Traditional detection (for comparison)
                f"Traditional: {info['rgb_color_name']}",
                f"CNN Model: {info['predicted_color']} ({info['confidence']:.2f})",
                "",
                # Color properties from improved detection
                f"Temperature: {improved_info.get('color_temperature', 'neutral').title()}",
                f"Saturation: {improved_info.get('saturation_level', 'low').title()}",
                f"Brightness: {improved_info.get('brightness_level', 'medium').title()}"
            ]
            
            # Add alternatives if available
            alternatives = improved_info.get('alternatives', [])
            if alternatives:
                text_lines.extend([
                    "",
                    f"Alt: {', '.join(alternatives[:2])}"  # Show first 2 alternatives
                ])
            
            # Add consistency analysis
            if consistency_info.get('color_consistency', False):
                consensus = consistency_info.get('consensus_color', '')
                strength = consistency_info.get('consensus_strength', 0.0)
                text_lines.extend([
                    "",
                    f"✅ Consistent: {consensus} ({strength:.1%})"
                ])
            else:
                variation = consistency_info.get('color_variation', 1)
                text_lines.extend([
                    "",
                    f"⚠️ Variation: {variation} colors detected"
                ])
            
            # Add K-means analysis if available
            if self.show_kmeans_analysis and info.get('kmeans_analysis'):
                kmeans = info['kmeans_analysis']
                text_lines.extend([
                    "",  # Empty line for spacing
                    f"K-means ({self.kmeans_k} clusters):",
                    f"Primary: {kmeans['primary_color_name']} ({kmeans['most_dominant_percentage']:.1f}%)",
                    f"Diversity: {kmeans['color_diversity_score']:.1f}%"
                ])
            
            # Add unified pipeline information
            if self.use_unified_pipeline and info.get('unified_analysis'):
                unified = info['unified_analysis']
                text_lines.extend([
                    "",  # Empty line for spacing
                    f"🚀 Unified Pipeline:",
                    f"Families: {unified['color_families']} | Patches: {unified['cnn_patches']}",
                    f"Pipeline FPS: {unified['pipeline_fps']:.1f} ({unified['processing_time_ms']:.1f}ms)"
                ])
            
            # Add mode indicators
            mode_indicators = []
            if self.use_unified_pipeline:
                mode_indicators.append("🚀 Unified Pipeline")
            if self.show_simplified:
                mode_indicators.append(f"Simplified ({self.kmeans_simplification_k}K)")
            if self.show_daltonization:
                mode_indicators.append(f"Daltonized ({self.daltonization_strength:.1f}x)")
            if self.use_realistic_cvd:
                mode_indicators.append("Realistic CVD")
            
            if mode_indicators:
                text_lines.extend([""] + mode_indicators)
            
            # Calculate text dimensions
            text_height = 0
            max_width = 0
            for line in text_lines:
                (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                text_height += h + 5
                max_width = max(max_width, w)
            
            # Position overlay at top-left
            overlay_x = padding
            overlay_y = padding
            overlay_width = max_width + 2 * padding
            overlay_height = text_height + 2 * padding
            
            # Draw background rectangle
            cv2.rectangle(frame, 
                         (overlay_x, overlay_y), 
                         (overlay_x + overlay_width, overlay_y + overlay_height),
                         bg_color, -1)
            
            # Draw text lines
            y_offset = overlay_y + padding + 20
            for line in text_lines:
                cv2.putText(frame, line, (overlay_x + padding, y_offset),
                           font, font_scale, color, thickness)
                y_offset += 25
            
            # Add color swatch
            swatch_size = 60
            swatch_x = overlay_x + overlay_width + padding
            swatch_y = overlay_y
            
            swatch = create_color_swatch(info['dominant_rgb'], (swatch_size, swatch_size))
            swatch_bgr = cv2.cvtColor(swatch, cv2.COLOR_RGB2BGR)
            
            # Ensure swatch fits in frame
            frame_h, frame_w = frame.shape[:2]
            if swatch_x + swatch_size < frame_w and swatch_y + swatch_size < frame_h:
                frame[swatch_y:swatch_y + swatch_size, 
                      swatch_x:swatch_x + swatch_size] = swatch_bgr
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error adding color info overlay: {e}")
            return frame
    
    def _add_fps_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add FPS overlay to frame."""
        try:
            fps_text = f"FPS: {self.current_fps:.1f}"
            
            # Position at top-right
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (0, 255, 0)  # Green text
            
            (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
            
            frame_h, frame_w = frame.shape[:2]
            x = frame_w - text_width - 10
            y = text_height + 10
            
            # Add background rectangle
            cv2.rectangle(frame, (x-5, y-text_height-5), (x+text_width+5, y+5), (0, 0, 0), -1)
            
            # Add text
            cv2.putText(frame, fps_text, (x, y), font, font_scale, color, thickness)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error adding FPS overlay: {e}")
            return frame
    
    def _add_roi_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Add ROI (region of interest) indicator as a simple dot crosshair."""
        try:
            if not self.current_color_info or 'roi_coords' not in self.current_color_info:
                return frame
            
            x1, y1, x2, y2 = self.current_color_info['roi_coords']
            
            # Calculate center point of ROI
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Adjust coordinates for side-by-side display
            if self.show_side_by_side:
                # Draw dot on both sides
                frame_h, frame_w = frame.shape[:2]
                half_width = frame_w // 2
                
                # Left side (normal) - simple white dot with black outline
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 0), -1)  # Black filled circle
                cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)  # White inner circle
                
                # Right side (CVD simulation)
                cv2.circle(frame, (center_x + half_width, center_y), 4, (0, 0, 0), -1)  # Black filled circle
                cv2.circle(frame, (center_x + half_width, center_y), 3, (255, 255, 255), -1)  # White inner circle
            else:
                # Single view - simple white dot with black outline
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 0), -1)  # Black filled circle
                cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)  # White inner circle
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error adding ROI indicator: {e}")
            return frame
    
    def _add_cvd_type_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add CVD type indicator overlay with enhanced information."""
        try:
            cvd_type = self.cvd_simulator.get_current_cvd_type()
            
            # Build informative text
            cvd_text = f"CVD: {cvd_type.value.upper()}"
            
            # Add mode indicators
            if self.use_realistic_cvd:
                cvd_text += " (Realistic)"
            if self.show_daltonization:
                cvd_text += " + Daltonized"
            
            # Check Ishihara visibility for current CVD type
            if cvd_type != CVDType.NORMAL:
                # Test with common Ishihara numbers
                test_numbers = [12, 8, 5, 3]
                invisible_count = sum(1 for num in test_numbers 
                                    if not self.cvd_simulator.check_ishihara_visibility(num, cvd_type))
                if invisible_count > 0:
                    cvd_text += f" (Ishihara: {invisible_count}/{len(test_numbers)} invisible)"
            
            # Position at bottom-left
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (255, 255, 0)  # Yellow text
            
            # Split long text into multiple lines if needed
            max_width = frame.shape[1] - 20
            (text_width, text_height), _ = cv2.getTextSize(cvd_text, font, font_scale, thickness)
            
            frame_h, frame_w = frame.shape[:2]
            x = 10
            y = frame_h - 40  # Leave more space for potential multi-line text
            
            if text_width > max_width:
                # Split text into two lines
                parts = cvd_text.split(' ')
                mid = len(parts) // 2
                line1 = ' '.join(parts[:mid])
                line2 = ' '.join(parts[mid:])
                
                # Calculate dimensions for both lines
                (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
                (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
                
                # Background rectangle for both lines
                total_width = max(w1, w2)
                total_height = h1 + h2 + 10
                cv2.rectangle(frame, (x-5, y-h1-h2-15), (x+total_width+5, y+5), (0, 0, 0), -1)
                
                # Draw both lines
                cv2.putText(frame, line1, (x, y-h2-5), font, font_scale, color, thickness)
                cv2.putText(frame, line2, (x, y), font, font_scale, color, thickness)
            else:
                # Single line
                cv2.rectangle(frame, (x-5, y-text_height-5), (x+text_width+5, y+5), (0, 0, 0), -1)
                cv2.putText(frame, cvd_text, (x, y), font, font_scale, color, thickness)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error adding CVD type overlay: {e}")
            return frame
    
    def _update_fps(self) -> None:
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _handle_keyboard(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            False if should quit, True otherwise
        """
        if key == ord('q') or key == ord('Q'):
            # Quit
            self.logger.info("Quit requested")
            if self.on_quit:
                self.on_quit()
            return False
        
        elif key == ord('c') or key == ord('C'):
            # Capture current frame and color
            if self.current_color_info and self.on_frame_captured:
                self.logger.info("Frame capture requested")
                self.on_frame_captured(self.current_color_info.copy())
        
        elif key == ord('n') or key == ord('N'):
            # Cycle CVD type
            new_cvd = self.cvd_simulator.cycle_cvd_type()
            self.logger.info(f"CVD type changed to: {new_cvd.value}")
        
        elif key == ord('p') or key == ord('P'):
            # Toggle pause
            self.is_paused = not self.is_paused
            self.logger.info(f"Camera {'paused' if self.is_paused else 'resumed'}")
        
        elif key == ord('s') or key == ord('S'):
            # Toggle side-by-side display
            self.show_side_by_side = not self.show_side_by_side
            self.logger.info(f"Side-by-side display {'enabled' if self.show_side_by_side else 'disabled'}")
        
        elif key == ord('i') or key == ord('I'):
            # Toggle color info display
            self.show_color_info = not self.show_color_info
            self.logger.info(f"Color info display {'enabled' if self.show_color_info else 'disabled'}")
        
        elif key == ord('f') or key == ord('F'):
            # Toggle FPS display
            self.show_fps = not self.show_fps
            self.logger.info(f"FPS display {'enabled' if self.show_fps else 'disabled'}")
        
        elif key == ord('k') or key == ord('K'):
            # Toggle K-means analysis display
            self.show_kmeans_analysis = not self.show_kmeans_analysis
            self.logger.info(f"K-means analysis {'enabled' if self.show_kmeans_analysis else 'disabled'}")
        
        elif key == ord('m') or key == ord('M'):
            # Toggle simplified image (K-means posterization)
            self.show_simplified = not self.show_simplified
            self.logger.info(f"K-means simplification {'enabled' if self.show_simplified else 'disabled'}")
        
        elif key == ord('d') or key == ord('D'):
            # Toggle daltonization
            self.show_daltonization = not self.show_daltonization
            self.logger.info(f"Daltonization {'enabled' if self.show_daltonization else 'disabled'}")
        
        elif key == ord('r') or key == ord('R'):
            # Toggle realistic CVD simulation
            self.use_realistic_cvd = not self.use_realistic_cvd
            self.logger.info(f"Realistic CVD simulation {'enabled' if self.use_realistic_cvd else 'disabled'}")
        
        elif key == ord('1'):
            # Adjust K-means clusters (decrease)
            self.kmeans_k = max(2, self.kmeans_k - 1)
            self.logger.info(f"K-means clusters: {self.kmeans_k}")
        
        elif key == ord('2'):
            # Adjust K-means clusters (increase)
            self.kmeans_k = min(15, self.kmeans_k + 1)
            self.logger.info(f"K-means clusters: {self.kmeans_k}")
        
        elif key == ord('3'):
            # Adjust daltonization strength (decrease)
            self.daltonization_strength = max(0.5, self.daltonization_strength - 0.2)
            self.logger.info(f"Daltonization strength: {self.daltonization_strength:.1f}")
        
        elif key == ord('4'):
            # Adjust daltonization strength (increase)
            self.daltonization_strength = min(3.0, self.daltonization_strength + 0.2)
            self.logger.info(f"Daltonization strength: {self.daltonization_strength:.1f}")
        
        elif key == ord('u') or key == ord('U'):
            # Toggle unified pipeline
            self.use_unified_pipeline = not self.use_unified_pipeline
            if self.use_unified_pipeline and self.unified_pipeline is None:
                self._initialize_unified_pipeline()
            self.logger.info(f"Unified pipeline {'enabled' if self.use_unified_pipeline else 'disabled'}")
        
        return True
    
    def stop_camera(self) -> None:
        """Stop camera and cleanup resources."""
        self.logger.info("Stopping camera...")
        
        self.is_running = False
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        self.logger.info("Camera stopped")
    
    def set_frame_capture_callback(self, callback: Callable[[Dict], None]) -> None:
        """Set callback function for frame capture events."""
        self.on_frame_captured = callback
    
    def set_quit_callback(self, callback: Callable[[], None]) -> None:
        """Set callback function for quit events."""
        self.on_quit = callback
    
    def get_camera_info(self) -> Dict:
        """
        Get camera information.
        
        Returns:
            Dictionary with camera information
        """
        info = {
            'camera_id': self.camera_id,
            'target_resolution': (self.frame_width, self.frame_height),
            'target_fps': self.fps_target,
            'current_fps': self.current_fps,
            'is_running': self.is_running,
            'is_paused': self.is_paused
        }
        
        if self.cap and self.cap.isOpened():
            info['actual_resolution'] = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            info['actual_fps'] = self.cap.get(cv2.CAP_PROP_FPS)
        
        return info


# Test the CameraHandler (requires connected camera)
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing CameraHandler...")
    print("Note: This test requires a connected camera")
    
    try:
        # Import required components
        from color_model import ColorModel
        from colorblind_detector import ColorBlindnessSimulator
        
        # Create components
        color_model = ColorModel()
        cvd_simulator = ColorBlindnessSimulator()
        
        # Train model quickly for testing
        print("Training model for testing...")
        color_model.train_on_synthetic_data(samples_per_class=50, epochs=3)
        
        # Create camera handler
        camera_handler = CameraHandler(color_model, cvd_simulator, camera_id=0)
        
        # Set up callbacks
        def on_capture(color_info):
            print(f"Captured: {color_info['predicted_color']} - RGB{color_info['dominant_rgb']}")
        
        def on_quit():
            print("Quit callback triggered")
        
        camera_handler.set_frame_capture_callback(on_capture)
        camera_handler.set_quit_callback(on_quit)
        
        # Get camera info
        info = camera_handler.get_camera_info()
        print(f"Camera info: {info}")
        
        # Start camera (comment out if no camera available)
        # camera_handler.start_camera_loop()
        
        print("CameraHandler test completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()