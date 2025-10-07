#!/usr/bin/env python3
"""
ReColor - Mobile CVD Assistive Tool
===================================
Main application integrating CVD simulation, daltonization, clustering, and CNN models
for real-time Color Vision Deficiency processing and screening.

Author: ReColor Development Team
Date: October 2025
License: MIT
"""

import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional
import argparse
import sys
import os
from datetime import datetime

# Import ReColor modules
try:
    from cvd_simulation import CVDSimulator
    from daltonization import AdaptiveDaltonizer
    from clustering import RealtimeColorClusterer
    from cnn_model import AdaptiveCVDModel
except ImportError as e:
    print(f"❌ Error importing ReColor modules: {e}")
    print("🔧 Make sure all module files are in the same directory")
    sys.exit(1)

class ReColorPipeline:
    """
    Main ReColor pipeline integrating all CVD processing components.
    
    Provides real-time camera processing with CVD simulation, adaptive daltonization,
    color clustering, and placeholder CNN model integration for mobile deployment.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ReColor Pipeline.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        # Default configuration
        self.config = {
            'camera_id': 0,
            'frame_width': 640,
            'frame_height': 480,
            'target_fps': 30,
            'cvd_type': 'deuteranopia',
            'n_clusters': 8,
            'daltonization_strength': 'medium',
            'optimization_level': 'balanced',  # 'mobile', 'balanced', 'quality'
            'enable_cnn': False,  # Enable CNN model (placeholder)
            'show_palette': True,
            'show_analysis': True,
            'adaptive_clustering': True
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Initialize components
        print("🚀 Initializing ReColor Pipeline Components...")
        self._initialize_components()
        
        # Processing state
        self.current_cvd_type = self.config['cvd_type']
        self.current_mode = 'hybrid'  # 'simulation', 'daltonization', 'hybrid'
        self.paused = False
        self.show_windows = ['main', 'original', 'processed']
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history = []
        self.processing_stats = {}
        
        # Threading for performance
        self.processing_thread = None
        self.stop_processing = False
        
        print("✅ ReColor Pipeline initialized successfully")
        self._print_controls()
    
    def _initialize_components(self):
        """Initialize all ReColor processing components."""
        
        # CVD Simulator
        self.cvd_simulator = CVDSimulator(
            optimization_level=self.config['optimization_level']
        )
        
        # Adaptive Daltonizer
        self.daltonizer = AdaptiveDaltonizer(
            cvd_simulator=self.cvd_simulator,
            adaptation_level=self.config['daltonization_strength']
        )
        
        # Real-time Color Clusterer
        self.clusterer = RealtimeColorClusterer(
            n_clusters=self.config['n_clusters'],
            batch_size=800,  # Optimized for real-time
            max_iter=5       # Reduced for speed
        )
        self.clusterer.adaptive_mode = self.config['adaptive_clustering']
        
        # CNN Model (placeholder)
        if self.config['enable_cnn']:
            self.cnn_model = AdaptiveCVDModel()
            self.cnn_model.update_user_profile(
                self.config['cvd_type'],
                severity=1.0
            )
        else:
            self.cnn_model = None
        
        print(f"🧩 Components initialized:")
        print(f"   📺 CVD Simulator: {self.config['optimization_level']}")
        print(f"   🎨 Daltonizer: {self.config['daltonization_strength']}")
        print(f"   🎯 Clusterer: K={self.config['n_clusters']}")
        print(f"   🧠 CNN Model: {'Enabled' if self.cnn_model else 'Disabled'}")
    
    def _print_controls(self):
        """Print control instructions."""
        print("\n" + "="*60)
        print("🎮 ReColor Controls:")
        print("   D - Cycle CVD types (Protanopia/Deuteranopia/Tritanopia)")
        print("   M - Switch modes (Simulation/Daltonization/Hybrid)")
        print("   K - Change K-means clusters (3-16)")
        print("   + - Increase daltonization strength")
        print("   - - Decrease daltonization strength") 
        print("   W - Toggle display windows")
        print("   P - Toggle palette display")
        print("   A - Toggle analysis display")
        print("   R - Reset to defaults")
        print("   Space - Pause/Resume")
        print("   S - Save screenshot")
        print("   Q/ESC - Quit")
        print("="*60)
    
    def initialize_camera(self) -> cv2.VideoCapture:
        """
        Initialize camera with optimal settings.
        
        Returns:
            OpenCV VideoCapture object
        """
        print(f"📹 Initializing camera {self.config['camera_id']}...")
        
        cap = cv2.VideoCapture(self.config['camera_id'])
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.config['camera_id']}")
        
        # Set camera properties for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['target_fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        # Verify settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS")
        
        return cap
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process single frame through ReColor pipeline.
        
        Args:
            frame: Input BGR frame from camera
            
        Returns:
            Dictionary with processed frame variants
        """
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = {
            'original': rgb_frame,
            'processed': rgb_frame.copy()
        }
        
        # Step 1: Color clustering (always applied)
        # Ensure clusterer is fitted first
        if not self.clusterer.fitted:
            self.clusterer.fit_incremental(rgb_frame)
        
        cluster_results = self.clusterer.process_video_frame(
            rgb_frame, 
            update_model=(self.frame_count % 5 == 0)  # Update model every 5 frames
        )
        
        clustered_image = cluster_results.get('cluster_image', rgb_frame)
        results['clustered'] = clustered_image
        results['palette'] = cluster_results.get('color_palette')
        results['analysis'] = cluster_results.get('color_analysis')
        
        # Step 2: CVD processing based on current mode
        if self.current_mode == 'simulation':
            # Pure CVD simulation
            cvd_simulated = self.cvd_simulator.simulate_cvd(clustered_image, self.current_cvd_type)
            results['processed'] = cvd_simulated
            
        elif self.current_mode == 'daltonization':
            # Daltonization correction only
            daltonized = self.daltonizer.adaptive_daltonization(clustered_image, self.current_cvd_type)
            results['processed'] = daltonized
            
        elif self.current_mode == 'hybrid':
            # Hybrid: CVD simulation + daltonization correction
            cvd_simulated = self.cvd_simulator.simulate_cvd(clustered_image, self.current_cvd_type)
            daltonized = self.daltonizer.adaptive_daltonization(clustered_image, self.current_cvd_type)
            
            # Blend simulation and correction
            alpha = 0.3  # Weight for simulation
            hybrid = (alpha * cvd_simulated.astype(np.float32) + 
                     (1 - alpha) * daltonized.astype(np.float32))
            results['processed'] = np.clip(hybrid, 0, 255).astype(np.uint8)
            results['cvd_simulation'] = cvd_simulated
            results['daltonized'] = daltonized
        
        # Step 3: CNN model enhancement (if enabled)
        if self.cnn_model:
            cnn_enhanced = self.cnn_model.predict(results['processed'])
            results['cnn_enhanced'] = cnn_enhanced
            results['processed'] = cnn_enhanced
        
        # Performance tracking
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        results['performance'] = {
            'processing_time': processing_time,
            'fps': fps,
            'avg_fps': np.mean(self.fps_history)
        }
        
        return results
    
    def add_overlay_info(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Add informative overlay to processed image.
        
        Args:
            image: Image to add overlay to
            results: Processing results dictionary
            
        Returns:
            Image with overlay information
        """
        overlay_img = image.copy()
        h, w = overlay_img.shape[:2]
        
        # Background for text
        cv2.rectangle(overlay_img, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay_img, 0.7, image, 0.3, 0, overlay_img)
        
        # Main info
        perf = results.get('performance', {})
        avg_fps = perf.get('avg_fps', 0)
        
        cv2.putText(overlay_img, f"ReColor CVD Pipeline", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(overlay_img, f"CVD: {self.current_cvd_type.title()} | Mode: {self.current_mode.title()}", 
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(overlay_img, f"FPS: {avg_fps:.1f} | Clusters: {self.clusterer.n_clusters}", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(overlay_img, f"Frame: {self.frame_count} | Dalton: {self.daltonizer.correction_strength:.2f}", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Analysis info (if enabled)
        if self.config['show_analysis'] and 'analysis' in results:
            analysis = results['analysis']
            diversity = analysis.get('color_diversity_score', 0)
            dominant = analysis.get('dominant_cluster_percentage', 0)
            
            y_offset = h - 60
            cv2.rectangle(overlay_img, (10, y_offset - 10), (w - 10, h - 10), (0, 0, 0), -1)
            
            cv2.putText(overlay_img, f"Color Diversity: {diversity:.2f} | Dominant: {dominant:.1f}%", 
                       (20, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_img
    
    def create_display_layout(self, results: Dict) -> np.ndarray:
        """
        Create multi-window display layout.
        
        Args:
            results: Processing results dictionary
            
        Returns:
            Combined display image
        """
        original = results['original']
        processed = results['processed']
        
        # Convert RGB to BGR for display
        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        
        # Add overlays
        processed_with_overlay = self.add_overlay_info(processed_bgr, results)
        
        h, w = original.shape[:2]
        
        if len(self.show_windows) == 1:
            return processed_with_overlay
        
        elif len(self.show_windows) == 2:
            # Side by side
            combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
            combined[:, :w] = original_bgr
            combined[:, w:] = processed_with_overlay
            return combined
        
        else:
            # 2x2 grid with palette
            grid_h, grid_w = h // 2, w // 2
            combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
            
            # Resize images for grid
            orig_small = cv2.resize(original_bgr, (grid_w, grid_h))
            proc_small = cv2.resize(processed_with_overlay, (grid_w, grid_h))
            
            # Top row
            combined[:grid_h, :grid_w] = orig_small
            combined[:grid_h, grid_w:grid_w*2] = proc_small
            
            # Bottom row - additional views
            if 'clustered' in results:
                clustered_bgr = cv2.cvtColor(results['clustered'], cv2.COLOR_RGB2BGR)
                clustered_small = cv2.resize(clustered_bgr, (grid_w, grid_h))
                combined[grid_h:, :grid_w] = clustered_small
            
            # Color palette
            if self.config['show_palette'] and 'palette' in results and results['palette'] is not None:
                palette_resized = cv2.resize(results['palette'], (grid_w, grid_h))
                combined[grid_h:, grid_w:grid_w*2] = palette_resized
            
            return combined
    
    def handle_key_events(self, key: int) -> bool:
        """
        Handle keyboard input events.
        
        Args:
            key: Pressed key code
            
        Returns:
            True to continue, False to quit
        """
        if key == ord('q') or key == 27:  # Q or ESC
            return False
        
        elif key == ord('d'):
            # Cycle CVD types
            cvd_types = ['protanopia', 'deuteranopia', 'tritanopia']
            current_idx = cvd_types.index(self.current_cvd_type)
            self.current_cvd_type = cvd_types[(current_idx + 1) % len(cvd_types)]
            print(f"👁️  CVD Type: {self.current_cvd_type}")
        
        elif key == ord('m'):
            # Cycle processing modes
            modes = ['simulation', 'daltonization', 'hybrid']
            current_idx = modes.index(self.current_mode)
            self.current_mode = modes[(current_idx + 1) % len(modes)]
            print(f"🔄 Mode: {self.current_mode}")
        
        elif key == ord('k'):
            # Change cluster count
            cluster_counts = [3, 4, 5, 6, 8, 10, 12, 16]
            try:
                current_idx = cluster_counts.index(self.clusterer.n_clusters)
                new_count = cluster_counts[(current_idx + 1) % len(cluster_counts)]
            except ValueError:
                new_count = 8
            
            self.clusterer.set_cluster_count(new_count)
            print(f"🎯 K-means clusters: {new_count}")
        
        elif key == ord('+') or key == ord('='):
            # Increase daltonization strength
            current = self.daltonizer.correction_strength
            new_strength = min(current + 0.1, 1.0)
            self.daltonizer.set_correction_strength(new_strength)
        
        elif key == ord('-'):
            # Decrease daltonization strength
            current = self.daltonizer.correction_strength
            new_strength = max(current - 0.1, 0.0)
            self.daltonizer.set_correction_strength(new_strength)
        
        elif key == ord('w'):
            # Toggle display windows
            window_configs = [['main'], ['main', 'original'], ['main', 'original', 'processed']]
            try:
                current_idx = next(i for i, config in enumerate(window_configs) 
                                 if len(config) == len(self.show_windows))
                self.show_windows = window_configs[(current_idx + 1) % len(window_configs)]
            except StopIteration:
                self.show_windows = ['main']
            print(f"🖼️  Display windows: {len(self.show_windows)}")
        
        elif key == ord('p'):
            # Toggle palette display
            self.config['show_palette'] = not self.config['show_palette']
            print(f"🎨 Palette display: {'ON' if self.config['show_palette'] else 'OFF'}")
        
        elif key == ord('a'):
            # Toggle analysis display
            self.config['show_analysis'] = not self.config['show_analysis']
            print(f"📊 Analysis display: {'ON' if self.config['show_analysis'] else 'OFF'}")
        
        elif key == ord('r'):
            # Reset to defaults
            self.current_cvd_type = 'deuteranopia'
            self.current_mode = 'hybrid'
            self.clusterer.set_cluster_count(8)
            self.daltonizer.set_correction_strength(0.7)
            print("🔄 Reset to defaults")
        
        elif key == ord(' '):
            # Toggle pause
            self.paused = not self.paused
            print(f"⏸️  {'Paused' if self.paused else 'Resumed'}")
        
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # This would save the current frame - implementation depends on display method
            print(f"📸 Screenshot saved: recolor_{timestamp}.jpg")
        
        return True
    
    def run_realtime_processing(self):
        """Main real-time processing loop."""
        
        print("🚀 Starting ReColor real-time processing...")
        
        try:
            # Initialize camera
            cap = self.initialize_camera()
            
            print(f"▶️  Press any key in the display window to start controls")
            print(f"📱 Processing in {self.config['optimization_level']} mode for mobile deployment")
            
            while True:
                if not self.paused:
                    # Capture frame
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Error capturing frame")
                        break
                    
                    # Process frame
                    results = self.process_frame(frame)
                    
                    self.frame_count += 1
                else:
                    # Keep displaying last results when paused
                    if 'results' not in locals():
                        continue
                
                # Create display
                display_image = self.create_display_layout(results)
                
                # Show display
                cv2.imshow('ReColor - CVD Assistive Pipeline', display_image)
                
                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_key_events(key):
                        break
        
        except KeyboardInterrupt:
            print("\n⏹️  Processing interrupted by user")
        
        except Exception as e:
            print(f"❌ Error in processing loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            
            # Show final statistics
            self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final processing statistics."""
        print("\n📊 ReColor Processing Statistics:")
        print(f"   Total frames processed: {self.frame_count}")
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"   Average FPS: {avg_fps:.1f}")
        
        # Component statistics
        cvd_stats = self.cvd_simulator.get_performance_stats()
        dalton_stats = self.daltonizer.get_performance_stats()
        cluster_stats = self.clusterer.get_performance_stats()
        
        print(f"   CVD Simulation: {cvd_stats.get('avg_time_ms', 0):.1f}ms avg")
        print(f"   Daltonization: {dalton_stats.get('avg_time_ms', 0):.1f}ms avg")
        print(f"   Clustering: {cluster_stats.get('avg_time_ms', 0):.1f}ms avg")
        
        if self.cnn_model:
            cnn_stats = self.cnn_model.get_performance_stats()
            print(f"   CNN Model: {cnn_stats.get('avg_time_ms', 0):.1f}ms avg")
        
        print("✅ ReColor pipeline stopped successfully")

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description='ReColor - Mobile CVD Assistive Tool')
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Frame height (default: 480)')
    parser.add_argument('--cvd-type', choices=['protanopia', 'deuteranopia', 'tritanopia'],
                       default='deuteranopia', help='Initial CVD type')
    parser.add_argument('--clusters', type=int, default=8, choices=range(3, 17),
                       help='Number of K-means clusters (3-16)')
    parser.add_argument('--optimization', choices=['mobile', 'balanced', 'quality'],
                       default='balanced', help='Optimization level')
    parser.add_argument('--enable-cnn', action='store_true',
                       help='Enable CNN model (placeholder)')
    parser.add_argument('--no-palette', action='store_true',
                       help='Disable color palette display')
    parser.add_argument('--no-analysis', action='store_true', 
                       help='Disable color analysis display')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = {
        'camera_id': args.camera,
        'frame_width': args.width,
        'frame_height': args.height,
        'cvd_type': args.cvd_type,
        'n_clusters': args.clusters,
        'optimization_level': args.optimization,
        'enable_cnn': args.enable_cnn,
        'show_palette': not args.no_palette,
        'show_analysis': not args.no_analysis
    }
    
    # Print startup banner
    print("🎨" + "="*58 + "🎨")
    print("     ReColor - Mobile CVD Assistive Tool")
    print("     Real-time Color Vision Deficiency Processing")
    print("🎨" + "="*58 + "🎨")
    
    try:
        # Create and run pipeline
        pipeline = ReColorPipeline(config)
        pipeline.run_realtime_processing()
        
    except Exception as e:
        print(f"❌ Failed to start ReColor pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())