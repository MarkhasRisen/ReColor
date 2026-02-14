"""
Enhanced error handling and anomaly detection for ReColor Backend.

This module provides comprehensive error handling, anomaly detection,
and recovery mechanisms for robust application operation.
"""

import logging
import traceback
import threading
import time
import cv2
import numpy as np
from typing import Optional, Any, Callable, Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager


class AnomalyType(Enum):
    """Types of anomalies that can occur in the system."""
    CAMERA_FAILURE = "camera_failure"
    FRAME_CORRUPTION = "frame_corruption" 
    MODEL_ERROR = "model_error"
    MEMORY_OVERFLOW = "memory_overflow"
    PROCESSING_TIMEOUT = "processing_timeout"
    INVALID_INPUT = "invalid_input"
    HARDWARE_ISSUE = "hardware_issue"
    NETWORK_ERROR = "network_error"
    FILE_IO_ERROR = "file_io_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class AnomalyEvent:
    """Record of an anomaly occurrence."""
    timestamp: float
    anomaly_type: AnomalyType
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str
    description: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False


class AnomalyDetector:
    """Detects various types of anomalies in real-time processing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.anomaly_history: List[AnomalyEvent] = []
        self.performance_metrics = {
            'frame_processing_times': [],
            'memory_usage': [],
            'error_count': 0,
            'recovery_count': 0
        }
        self._lock = threading.Lock()
        
    def detect_frame_anomalies(self, frame: np.ndarray) -> List[AnomalyEvent]:
        """Detect anomalies in captured frames."""
        anomalies = []
        
        try:
            if frame is None:
                anomalies.append(AnomalyEvent(
                    timestamp=time.time(),
                    anomaly_type=AnomalyType.FRAME_CORRUPTION,
                    severity='high',
                    component='camera',
                    description='Captured frame is None',
                    context={'frame_shape': None}
                ))
                return anomalies
            
            # Check frame dimensions
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                anomalies.append(AnomalyEvent(
                    timestamp=time.time(),
                    anomaly_type=AnomalyType.FRAME_CORRUPTION,
                    severity='high',
                    component='camera',
                    description='Invalid frame dimensions',
                    context={'frame_shape': frame.shape}
                ))
            
            # Check if frame is completely black
            if np.mean(frame) < 5:
                anomalies.append(AnomalyEvent(
                    timestamp=time.time(),
                    anomaly_type=AnomalyType.FRAME_CORRUPTION,
                    severity='medium',
                    component='camera',
                    description='Frame appears completely black',
                    context={'mean_brightness': np.mean(frame)}
                ))
            
            # Check for unusual brightness patterns
            brightness_std = np.std(frame)
            if brightness_std < 1:
                anomalies.append(AnomalyEvent(
                    timestamp=time.time(),
                    anomaly_type=AnomalyType.FRAME_CORRUPTION,
                    severity='low',
                    component='camera',
                    description='Very low brightness variation detected',
                    context={'brightness_std': brightness_std}
                ))
                
        except Exception as e:
            anomalies.append(AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.PROCESSING_TIMEOUT,
                severity='high',
                component='anomaly_detector',
                description=f'Error during frame anomaly detection: {str(e)}',
                context={'error_type': type(e).__name__}
            ))
        
        return anomalies
    
    def detect_processing_anomalies(self, processing_time: float, 
                                  expected_max_time: float = 0.1) -> List[AnomalyEvent]:
        """Detect processing time anomalies."""
        anomalies = []
        
        with self._lock:
            self.performance_metrics['frame_processing_times'].append(processing_time)
            
            # Keep only last 100 measurements
            if len(self.performance_metrics['frame_processing_times']) > 100:
                self.performance_metrics['frame_processing_times'].pop(0)
        
        if processing_time > expected_max_time * 3:
            severity = 'critical' if processing_time > expected_max_time * 10 else 'high'
            anomalies.append(AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.PROCESSING_TIMEOUT,
                severity=severity,
                component='processing',
                description=f'Processing time exceeded threshold: {processing_time:.3f}s',
                context={
                    'processing_time': processing_time,
                    'threshold': expected_max_time,
                    'ratio': processing_time / expected_max_time
                }
            ))
        
        return anomalies
    
    def log_anomaly(self, anomaly: AnomalyEvent) -> None:
        """Log an anomaly event."""
        with self._lock:
            self.anomaly_history.append(anomaly)
            self.performance_metrics['error_count'] += 1
            
            # Keep only last 1000 anomalies
            if len(self.anomaly_history) > 1000:
                self.anomaly_history.pop(0)
        
        severity_to_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        level = severity_to_level.get(anomaly.severity, logging.WARNING)
        self.logger.log(level, 
            f"ANOMALY [{anomaly.severity.upper()}] {anomaly.component}: "
            f"{anomaly.description} | Context: {anomaly.context}")
    
    def get_recent_anomalies(self, minutes: int = 5) -> List[AnomalyEvent]:
        """Get anomalies from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        with self._lock:
            return [a for a in self.anomaly_history if a.timestamp > cutoff_time]


class RecoveryManager:
    """Manages recovery strategies for different types of anomalies."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.recovery_strategies = {
            AnomalyType.CAMERA_FAILURE: self._recover_camera,
            AnomalyType.FRAME_CORRUPTION: self._recover_frame_corruption,
            AnomalyType.MODEL_ERROR: self._recover_model_error,
            AnomalyType.MEMORY_OVERFLOW: self._recover_memory_overflow,
            AnomalyType.PROCESSING_TIMEOUT: self._recover_processing_timeout,
        }
        self.recovery_counts = {anomaly_type: 0 for anomaly_type in AnomalyType}
        
    def attempt_recovery(self, anomaly: AnomalyEvent, context: Dict[str, Any]) -> bool:
        """Attempt to recover from an anomaly."""
        strategy = self.recovery_strategies.get(anomaly.anomaly_type)
        
        if not strategy:
            self.logger.warning(f"No recovery strategy for {anomaly.anomaly_type}")
            return False
        
        try:
            self.logger.info(f"Attempting recovery for {anomaly.anomaly_type}")
            success = strategy(anomaly, context)
            
            if success:
                self.recovery_counts[anomaly.anomaly_type] += 1
                self.logger.info(f"Recovery successful for {anomaly.anomaly_type}")
            else:
                self.logger.error(f"Recovery failed for {anomaly.anomaly_type}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed with exception: {e}")
            return False
    
    def _recover_camera(self, anomaly: AnomalyEvent, context: Dict[str, Any]) -> bool:
        """Attempt to recover camera connection."""
        camera = context.get('camera')
        if not camera:
            return False
        
        try:
            # Release and reinitialize camera
            if hasattr(camera, 'cap') and camera.cap:
                camera.cap.release()
                time.sleep(1)  # Brief pause
            
            # Attempt to reinitialize
            return camera.initialize_camera()
            
        except Exception as e:
            self.logger.error(f"Camera recovery failed: {e}")
            return False
    
    def _recover_frame_corruption(self, anomaly: AnomalyEvent, context: Dict[str, Any]) -> bool:
        """Handle frame corruption by skipping frames."""
        # For frame corruption, we typically just skip and try the next frame
        return True
    
    def _recover_model_error(self, anomaly: AnomalyEvent, context: Dict[str, Any]) -> bool:
        """Attempt to recover from model prediction errors."""
        model = context.get('model')
        if not model:
            return False
        
        try:
            # Try to reload the model or reset its state
            # This is a placeholder - actual implementation depends on model architecture
            self.logger.info("Attempting model state reset")
            return True
            
        except Exception as e:
            self.logger.error(f"Model recovery failed: {e}")
            return False
    
    def _recover_memory_overflow(self, anomaly: AnomalyEvent, context: Dict[str, Any]) -> bool:
        """Attempt to free memory."""
        try:
            import gc
            gc.collect()
            
            # Clear any cached data if available
            cache = context.get('cache')
            if cache and hasattr(cache, 'clear'):
                cache.clear()
            
            return True
            
        except Exception:
            return False
    
    def _recover_processing_timeout(self, anomaly: AnomalyEvent, context: Dict[str, Any]) -> bool:
        """Handle processing timeouts by adjusting parameters."""
        # This could involve reducing processing quality or skipping frames
        return True


@contextmanager
def error_context(component: str, logger: logging.Logger, 
                 anomaly_detector: Optional[AnomalyDetector] = None,
                 recovery_manager: Optional[RecoveryManager] = None):
    """Context manager for enhanced error handling."""
    start_time = time.time()
    
    try:
        yield
        
    except cv2.error as e:
        error_msg = f"OpenCV error in {component}: {str(e)}"
        logger.error(error_msg)
        
        if anomaly_detector:
            anomaly = AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.HARDWARE_ISSUE,
                severity='high',
                component=component,
                description=error_msg,
                context={'error_type': 'cv2_error', 'opencv_error': str(e)}
            )
            anomaly_detector.log_anomaly(anomaly)
        
        raise
        
    except MemoryError as e:
        error_msg = f"Memory error in {component}: {str(e)}"
        logger.critical(error_msg)
        
        if anomaly_detector:
            anomaly = AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.MEMORY_OVERFLOW,
                severity='critical',
                component=component,
                description=error_msg,
                context={'error_type': 'memory_error'}
            )
            anomaly_detector.log_anomaly(anomaly)
            
            if recovery_manager:
                recovery_manager.attempt_recovery(anomaly, {})
        
        raise
        
    except Exception as e:
        error_msg = f"Unexpected error in {component}: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        if anomaly_detector:
            anomaly = AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.PROCESSING_TIMEOUT,
                severity='medium',
                component=component,
                description=error_msg,
                context={
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
            )
            anomaly_detector.log_anomaly(anomaly)
        
        raise
        
    finally:
        processing_time = time.time() - start_time
        
        if anomaly_detector and processing_time > 0.1:  # 100ms threshold
            anomalies = anomaly_detector.detect_processing_anomalies(processing_time)
            for anomaly in anomalies:
                anomaly_detector.log_anomaly(anomaly)


def setup_enhanced_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up enhanced logging with anomaly tracking."""
    logger = logging.getLogger("ReColor")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with enhanced formatting
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler for persistent logging
    try:
        file_handler = logging.FileHandler('logs/recolor_enhanced.log')
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")
    
    return logger


class HealthMonitor:
    """Monitor system health and performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {
            'frames_processed': 0,
            'errors_occurred': 0,
            'recoveries_successful': 0,
            'average_processing_time': 0.0,
            'start_time': time.time(),
            'last_health_check': time.time()
        }
        self._lock = threading.Lock()
    
    def update_metrics(self, processing_time: float, error_occurred: bool = False,
                      recovery_successful: bool = False) -> None:
        """Update performance metrics."""
        with self._lock:
            self.metrics['frames_processed'] += 1
            
            if error_occurred:
                self.metrics['errors_occurred'] += 1
            
            if recovery_successful:
                self.metrics['recoveries_successful'] += 1
            
            # Update rolling average processing time
            frames = self.metrics['frames_processed']
            current_avg = self.metrics['average_processing_time']
            self.metrics['average_processing_time'] = (
                (current_avg * (frames - 1) + processing_time) / frames
            )
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get current system health report."""
        with self._lock:
            runtime = time.time() - self.metrics['start_time']
            fps = self.metrics['frames_processed'] / runtime if runtime > 0 else 0
            error_rate = (self.metrics['errors_occurred'] / 
                         max(1, self.metrics['frames_processed']))
            
            return {
                'runtime_seconds': runtime,
                'frames_processed': self.metrics['frames_processed'],
                'average_fps': fps,
                'error_rate': error_rate,
                'errors_total': self.metrics['errors_occurred'],
                'recoveries_successful': self.metrics['recoveries_successful'],
                'average_processing_time': self.metrics['average_processing_time'],
                'health_status': self._determine_health_status(fps, error_rate)
            }
    
    def _determine_health_status(self, fps: float, error_rate: float) -> str:
        """Determine overall system health status."""
        if error_rate > 0.1:  # More than 10% error rate
            return 'CRITICAL'
        elif error_rate > 0.05:  # More than 5% error rate
            return 'WARNING'
        elif fps < 10:  # Less than 10 FPS
            return 'DEGRADED'
        else:
            return 'HEALTHY'
    
    def log_health_report(self) -> None:
        """Log periodic health report."""
        report = self.get_health_report()
        self.logger.info(f"Health Report: {report['health_status']} | "
                        f"FPS: {report['average_fps']:.1f} | "
                        f"Errors: {report['errors_total']} | "
                        f"Error Rate: {report['error_rate']:.2%}")


if __name__ == "__main__":
    # Test the error handling system
    logger = setup_enhanced_logging("DEBUG")
    detector = AnomalyDetector(logger)
    recovery = RecoveryManager(logger)
    monitor = HealthMonitor(logger)
    
    logger.info("Enhanced error handling system initialized successfully")
    
    # Simulate some test scenarios
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    anomalies = detector.detect_frame_anomalies(test_frame)
    
    for anomaly in anomalies:
        detector.log_anomaly(anomaly)
    
    health = monitor.get_health_report()
    logger.info(f"Initial health status: {health}")