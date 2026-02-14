"""
ColorLogger for ReColor TensorFlow colorblind detection system.
Handles CSV logging of captured color data with timestamps and CVD simulation information.
"""

import csv
import os
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from utils import format_timestamp, ensure_directory_exists
from colorblind_detector import CVDType


class ColorLogger:
    """
    Handles logging of color capture data to CSV files with comprehensive metadata.
    Supports session tracking, data export, and statistical analysis.
    """
    
    def __init__(self, 
                 log_directory: str = "logs",
                 log_filename: str = None,
                 auto_create_session: bool = True):
        """
        Initialize ColorLogger.
        
        Args:
            log_directory: Directory to store log files
            log_filename: Custom log filename (None for auto-generated)
            auto_create_session: Whether to automatically create a new session
        """
        self.log_directory = log_directory
        self.log_filename = log_filename
        self.auto_create_session = auto_create_session
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Session tracking
        self.session_id = None
        self.session_start_time = None
        self.session_data = []
        
        # CSV headers
        self.csv_headers = [
            'timestamp',
            'session_id',
            'color_name_rgb',
            'color_name_ai',
            'rgb_r',
            'rgb_g',
            'rgb_b',
            'hex_color',
            'ai_confidence',
            'cvd_type',
            'cvd_description',
            'roi_x1',
            'roi_y1',
            'roi_x2',
            'roi_y2',
            'capture_time_ms'
        ]
        
        # Statistics
        self.total_captures = 0
        self.captures_by_color = {}
        self.captures_by_cvd_type = {}
        
        # Initialize
        self._initialize_logger()
        
        if self.auto_create_session:
            self.start_new_session()
    
    def _initialize_logger(self) -> None:
        """Initialize logger and ensure directories exist."""
        try:
            # Ensure log directory exists
            if not ensure_directory_exists(self.log_directory):
                self.logger.error(f"Failed to create log directory: {self.log_directory}")
                return
            
            # Generate filename if not provided
            if self.log_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_filename = f"recolor_colors_{timestamp}.csv"
            
            # Full path to log file
            self.log_filepath = os.path.join(self.log_directory, self.log_filename)
            
            self.logger.info(f"ColorLogger initialized")
            self.logger.info(f"Log file: {self.log_filepath}")
            
        except Exception as e:
            self.logger.error(f"Error initializing ColorLogger: {e}")
    
    def start_new_session(self) -> str:
        """
        Start a new logging session.
        
        Returns:
            Session ID string
        """
        try:
            # Generate session ID
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_start_time = time.time()
            self.session_data = []
            
            # Reset statistics
            self.total_captures = 0
            self.captures_by_color = {}
            self.captures_by_cvd_type = {}
            
            self.logger.info(f"New session started: {self.session_id}")
            
            # Create CSV file with headers if it doesn't exist
            self._ensure_csv_file()
            
            return self.session_id
            
        except Exception as e:
            self.logger.error(f"Error starting new session: {e}")
            return None
    
    def _ensure_csv_file(self) -> None:
        """Ensure CSV file exists with proper headers."""
        try:
            if not os.path.exists(self.log_filepath):
                with open(self.log_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.csv_headers)
                self.logger.info(f"Created new CSV file: {self.log_filepath}")
            
        except Exception as e:
            self.logger.error(f"Error ensuring CSV file: {e}")
    
    def log_color_capture(self, 
                         color_info: Dict,
                         cvd_type: CVDType = CVDType.NORMAL,
                         additional_data: Optional[Dict] = None) -> bool:
        """
        Log a color capture event to CSV.
        
        Args:
            color_info: Color information dictionary from camera handler
            cvd_type: Current CVD simulation type
            additional_data: Optional additional data to include
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            if self.session_id is None:
                self.logger.warning("No active session. Starting new session.")
                self.start_new_session()
            
            # Extract color information
            timestamp = color_info.get('timestamp', time.time())
            rgb = color_info.get('dominant_rgb', (0, 0, 0))
            hex_color = color_info.get('dominant_hex', '#000000')
            rgb_color_name = color_info.get('rgb_color_name', 'Unknown')
            ai_color_name = color_info.get('predicted_color', 'Unknown')
            ai_confidence = color_info.get('confidence', 0.0)
            roi_coords = color_info.get('roi_coords', (0, 0, 0, 0))
            
            # Calculate capture time
            capture_time_ms = int((timestamp - self.session_start_time) * 1000) if self.session_start_time else 0
            
            # Prepare CSV row
            csv_row = [
                format_timestamp(timestamp),
                self.session_id,
                rgb_color_name,
                ai_color_name,
                rgb[0],
                rgb[1],
                rgb[2],
                hex_color,
                f"{ai_confidence:.4f}",
                cvd_type.value,
                self._get_cvd_description(cvd_type),
                roi_coords[0],
                roi_coords[1],
                roi_coords[2],
                roi_coords[3],
                capture_time_ms
            ]
            
            # Write to CSV file
            with open(self.log_filepath, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_row)
            
            # Add to session data
            session_entry = {
                'timestamp': timestamp,
                'rgb': rgb,
                'hex': hex_color,
                'rgb_color_name': rgb_color_name,
                'ai_color_name': ai_color_name,
                'ai_confidence': ai_confidence,
                'cvd_type': cvd_type.value,
                'roi_coords': roi_coords,
                'capture_time_ms': capture_time_ms
            }
            
            if additional_data:
                session_entry.update(additional_data)
            
            self.session_data.append(session_entry)
            
            # Update statistics
            self._update_statistics(rgb_color_name, ai_color_name, cvd_type)
            
            self.logger.info(f"Color capture logged: {ai_color_name} ({rgb_color_name}) - RGB{rgb} - CVD: {cvd_type.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging color capture: {e}")
            return False
    
    def _get_cvd_description(self, cvd_type: CVDType) -> str:
        """Get description for CVD type."""
        descriptions = {
            CVDType.NORMAL: "Normal vision",
            CVDType.PROTANOPIA: "Red-blind (missing L cones)",
            CVDType.DEUTERANOPIA: "Green-blind (missing M cones)",
            CVDType.TRITANOPIA: "Blue-blind (missing S cones)"
        }
        return descriptions.get(cvd_type, "Unknown CVD type")
    
    def _update_statistics(self, rgb_color_name: str, ai_color_name: str, cvd_type: CVDType) -> None:
        """Update capture statistics."""
        self.total_captures += 1
        
        # Update color statistics
        if ai_color_name not in self.captures_by_color:
            self.captures_by_color[ai_color_name] = 0
        self.captures_by_color[ai_color_name] += 1
        
        # Update CVD type statistics
        cvd_name = cvd_type.value
        if cvd_name not in self.captures_by_cvd_type:
            self.captures_by_cvd_type[cvd_name] = 0
        self.captures_by_cvd_type[cvd_name] += 1
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current session.
        
        Returns:
            Dictionary with session statistics
        """
        try:
            if not self.session_data:
                return {
                    'session_id': self.session_id,
                    'total_captures': 0,
                    'session_duration_seconds': 0,
                    'message': 'No captures in current session'
                }
            
            # Calculate session duration
            current_time = time.time()
            session_duration = current_time - self.session_start_time if self.session_start_time else 0
            
            # Calculate confidence statistics
            confidences = [entry['ai_confidence'] for entry in self.session_data]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            max_confidence = max(confidences) if confidences else 0
            min_confidence = min(confidences) if confidences else 0
            
            # Most common colors
            color_counts = {}
            for entry in self.session_data:
                color = entry['ai_color_name']
                color_counts[color] = color_counts.get(color, 0) + 1
            
            most_common_color = max(color_counts, key=color_counts.get) if color_counts else "None"
            
            # CVD type distribution
            cvd_counts = {}
            for entry in self.session_data:
                cvd = entry['cvd_type']
                cvd_counts[cvd] = cvd_counts.get(cvd, 0) + 1
            
            return {
                'session_id': self.session_id,
                'session_start': format_timestamp(self.session_start_time) if self.session_start_time else None,
                'session_duration_seconds': round(session_duration, 2),
                'total_captures': self.total_captures,
                'unique_colors': len(color_counts),
                'most_common_color': most_common_color,
                'avg_confidence': round(avg_confidence, 4),
                'max_confidence': round(max_confidence, 4),
                'min_confidence': round(min_confidence, 4),
                'color_distribution': color_counts,
                'cvd_type_distribution': cvd_counts,
                'log_file': self.log_filepath
            }
            
        except Exception as e:
            self.logger.error(f"Error generating session summary: {e}")
            return {'error': str(e)}
    
    def export_session_data(self, 
                           format: str = "json",
                           filename: str = None) -> Optional[str]:
        """
        Export session data to file.
        
        Args:
            format: Export format ("json", "csv")
            filename: Custom filename (None for auto-generated)
            
        Returns:
            Path to exported file, or None if failed
        """
        try:
            if not self.session_data:
                self.logger.warning("No session data to export")
                return None
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recolor_session_{self.session_id}_{timestamp}.{format}"
            
            export_path = os.path.join(self.log_directory, filename)
            
            if format.lower() == "json":
                # Export as JSON
                export_data = {
                    'session_info': {
                        'session_id': self.session_id,
                        'session_start': format_timestamp(self.session_start_time) if self.session_start_time else None,
                        'total_captures': len(self.session_data),
                        'export_timestamp': format_timestamp(time.time())
                    },
                    'captures': self.session_data,
                    'summary': self.get_session_summary()
                }
                
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
            elif format.lower() == "csv":
                # Export as CSV
                with open(export_path, 'w', newline='', encoding='utf-8') as csvfile:
                    if self.session_data:
                        fieldnames = self.session_data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(self.session_data)
            
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None
            
            self.logger.info(f"Session data exported to: {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Error exporting session data: {e}")
            return None
    
    def load_session_data(self, filepath: str) -> bool:
        """
        Load session data from exported file.
        
        Args:
            filepath: Path to exported session file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"Session file not found: {filepath}")
                return False
            
            if filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if 'captures' in data:
                    self.session_data = data['captures']
                    if 'session_info' in data:
                        self.session_id = data['session_info'].get('session_id')
                    
                    self.logger.info(f"Loaded {len(self.session_data)} captures from {filepath}")
                    return True
                    
            elif filepath.endswith('.csv'):
                self.session_data = []
                with open(filepath, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.session_data.append(dict(row))
                
                self.logger.info(f"Loaded {len(self.session_data)} captures from {filepath}")
                return True
            
            else:
                self.logger.error(f"Unsupported file format: {filepath}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading session data: {e}")
            return False
    
    def get_color_statistics(self) -> Dict[str, Any]:
        """
        Get detailed color statistics from session data.
        
        Returns:
            Dictionary with color analysis statistics
        """
        try:
            if not self.session_data:
                return {'error': 'No session data available'}
            
            # Color frequency analysis
            ai_colors = [entry['ai_color_name'] for entry in self.session_data]
            rgb_colors = [entry['rgb_color_name'] for entry in self.session_data]
            
            ai_color_counts = {}
            rgb_color_counts = {}
            
            for color in ai_colors:
                ai_color_counts[color] = ai_color_counts.get(color, 0) + 1
            
            for color in rgb_colors:
                rgb_color_counts[color] = rgb_color_counts.get(color, 0) + 1
            
            # Confidence analysis by color
            confidence_by_color = {}
            for entry in self.session_data:
                color = entry['ai_color_name']
                confidence = entry['ai_confidence']
                
                if color not in confidence_by_color:
                    confidence_by_color[color] = []
                confidence_by_color[color].append(confidence)
            
            # Calculate average confidence per color
            avg_confidence_by_color = {}
            for color, confidences in confidence_by_color.items():
                avg_confidence_by_color[color] = sum(confidences) / len(confidences)
            
            return {
                'total_captures': len(self.session_data),
                'ai_color_frequency': ai_color_counts,
                'rgb_color_frequency': rgb_color_counts,
                'unique_ai_colors': len(ai_color_counts),
                'unique_rgb_colors': len(rgb_color_counts),
                'average_confidence_by_color': avg_confidence_by_color,
                'most_frequent_ai_color': max(ai_color_counts, key=ai_color_counts.get) if ai_color_counts else None,
                'most_frequent_rgb_color': max(rgb_color_counts, key=rgb_color_counts.get) if rgb_color_counts else None
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating color statistics: {e}")
            return {'error': str(e)}
    
    def close_session(self) -> Dict[str, Any]:
        """
        Close current session and return summary.
        
        Returns:
            Session summary dictionary
        """
        try:
            summary = self.get_session_summary()
            
            self.logger.info(f"Session {self.session_id} closed")
            self.logger.info(f"Total captures: {summary.get('total_captures', 0)}")
            
            # Reset session
            self.session_id = None
            self.session_start_time = None
            self.session_data = []
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error closing session: {e}")
            return {'error': str(e)}


# Test the ColorLogger
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ColorLogger...")
    
    try:
        # Import needed for test
        from utils import rgb_to_hex
        
        # Create logger
        logger = ColorLogger(log_directory="test_logs")
        
        # Test color capture logging
        test_color_info = {
            'timestamp': time.time(),
            'dominant_rgb': (255, 128, 64),
            'dominant_hex': '#FF8040',
            'rgb_color_name': 'Orange',
            'predicted_color': 'Orange',
            'confidence': 0.87,
            'roi_coords': (100, 100, 150, 150)
        }
        
        # Log multiple captures
        for i in range(5):
            # Modify color slightly for each capture
            test_color_info['dominant_rgb'] = (255 - i*10, 128 + i*5, 64 + i*3)
            test_color_info['dominant_hex'] = rgb_to_hex(test_color_info['dominant_rgb'])
            test_color_info['confidence'] = 0.8 + i * 0.02
            test_color_info['timestamp'] = time.time() + i
            
            success = logger.log_color_capture(test_color_info, CVDType.PROTANOPIA)
            print(f"Capture {i+1} logged: {success}")
            
            time.sleep(0.1)  # Small delay
        
        # Get session summary
        summary = logger.get_session_summary()
        print(f"\nSession Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Get color statistics
        stats = logger.get_color_statistics()
        print(f"\nColor Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict) and len(value) < 10:  # Don't print large dicts
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        # Export session data
        export_path = logger.export_session_data("json")
        print(f"\nSession exported to: {export_path}")
        
        # Close session
        final_summary = logger.close_session()
        print(f"\nSession closed. Final captures: {final_summary.get('total_captures', 0)}")
        
        print("\nColorLogger test completed successfully!")
        
        # Cleanup test files
        import shutil
        if os.path.exists("test_logs"):
            shutil.rmtree("test_logs")
            print("Test files cleaned up.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()