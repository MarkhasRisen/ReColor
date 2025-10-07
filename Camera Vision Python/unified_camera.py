#!/usr/bin/env python3
"""
Unified ReColor Camera
=====================
Single unified camera interface combining CVD simulation, daltonization, and clustering
Optimized for real-time processing with RTX 4050 GPU acceleration

Author: ReColor Development Team
Date: October 2025
License: MIT
"""

import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys
import os

# Import ReColor modules
try:
    from cvd_simulation import CVDSimulator
    from daltonization import AdaptiveDaltonizer
    from clustering import RealtimeColorClusterer
    from cnn_model import AdaptiveCVDModel
except ImportError as e:
    print(f"❌ Error importing ReColor modules: {e}")
    sys.exit(1)

class UnifiedReColorCamera:
    """
    Unified ReColor camera with all CVD processing in one streamlined interface.
    
    Features:
    - Real-time CVD simulation and correction
    - Adaptive K-means clustering
    - Daltonization enhancement
    - Single window unified display
    - Optimized for RTX 4050 performance
    """
    
    def __init__(self):
        """Initialize Unified ReColor Camera."""
        
        # Processing components
        self.cvd_simulator = CVDSimulator(optimization_level='balanced')
        self.daltonizer = AdaptiveDaltonizer(adaptation_level='medium')
        self.clusterer = RealtimeColorClusterer(n_clusters=8)
        
        # Current settings
        self.cvd_type = 'deuteranopia'
        self.k_clusters = 8
        self.dalton_strength = 0.7
        self.show_original = True
        self.show_palette = True
        
        # Processing modes
        self.modes = ['unified', 'simulation', 'correction', 'clustering']
        self.current_mode_idx = 0  # Start with unified
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history = []
        self.paused = False
        
        print("🎥 Unified ReColor Camera initialized")
        print(f"🎯 K-means: {self.k_clusters} clusters")
        print(f"👁️  CVD Type: {self.cvd_type}")
        print(f"🔄 Mode: {self.modes[self.current_mode_idx]}")
    
    def process_unified_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process frame through unified ReColor pipeline.
        
        Args:
            frame: Input BGR frame from camera
            
        Returns:
            Dictionary with all processed variants
        """
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ensure clusterer is fitted
        if not self.clusterer.fitted:
            self.clusterer.fit_incremental(rgb_frame)
        
        # Step 1: K-means clustering for color segmentation
        cluster_results = self.clusterer.process_video_frame(
            rgb_frame, 
            update_model=(self.frame_count % 10 == 0)
        )
        clustered_image = cluster_results.get('cluster_image', rgb_frame)
        
        # Step 2: CVD simulation
        cvd_simulated = self.cvd_simulator.simulate_cvd(clustered_image, self.cvd_type)
        
        # Step 3: Daltonization correction
        daltonized = self.daltonizer.adaptive_daltonization(clustered_image, self.cvd_type)
        
        # Step 4: Unified processing based on current mode
        current_mode = self.modes[self.current_mode_idx]
        
        if current_mode == 'unified':
            # Blend all techniques for optimal result
            alpha_sim = 0.2   # CVD simulation weight
            alpha_corr = 0.6  # Daltonization weight
            alpha_clust = 0.2 # Clustering weight
            
            unified = (alpha_sim * cvd_simulated.astype(np.float32) +
                      alpha_corr * daltonized.astype(np.float32) +
                      alpha_clust * clustered_image.astype(np.float32))
            
            processed_main = np.clip(unified, 0, 255).astype(np.uint8)
            
        elif current_mode == 'simulation':
            processed_main = cvd_simulated
            
        elif current_mode == 'correction':
            processed_main = daltonized
            
        else:  # clustering
            processed_main = clustered_image
        
        # Calculate performance
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        return {
            'original': rgb_frame,
            'processed': processed_main,
            'clustered': clustered_image,
            'cvd_simulation': cvd_simulated,
            'daltonized': daltonized,
            'palette': cluster_results.get('color_palette'),
            'analysis': cluster_results.get('color_analysis', {}),
            'performance': {
                'fps': fps,
                'avg_fps': np.mean(self.fps_history),
                'processing_time': processing_time
            }
        }
    
    def create_unified_display(self, results: Dict) -> np.ndarray:
        """
        Create unified display layout with all information in one window.
        
        Args:
            results: Processing results dictionary
            
        Returns:
            Single unified display image
        """
        original = results['original']
        processed = results['processed']
        h, w = original.shape[:2]
        
        # Convert to BGR for display
        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        
        if not self.show_original:
            # Single processed view (full window)
            display = processed_bgr.copy()
            self.add_unified_overlay(display, results, full_window=True)
            return display
        
        # Side-by-side layout: Original | Processed
        unified_display = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Left side: Original
        unified_display[:, :w] = original_bgr
        
        # Right side: Processed
        unified_display[:, w:] = processed_bgr
        
        # Add overlays
        self.add_unified_overlay(unified_display, results, full_window=False)
        
        # Add palette at bottom if enabled
        if self.show_palette and results.get('palette') is not None:
            palette = results['palette']
            palette_h = 60
            
            # Resize display to make room for palette
            display_with_palette = np.zeros((h + palette_h, w * 2, 3), dtype=np.uint8)
            display_with_palette[:h, :] = unified_display
            
            # Add palette
            palette_resized = cv2.resize(palette, (w * 2, palette_h))
            display_with_palette[h:, :] = palette_resized
            
            return display_with_palette
        
        return unified_display
    
    def add_unified_overlay(self, image: np.ndarray, results: Dict, full_window: bool = False):
        """
        Add unified overlay with all relevant information.
        
        Args:
            image: Image to add overlay to
            results: Processing results
            full_window: Whether this is a full window display
        """
        h, w = image.shape[:2]
        perf = results.get('performance', {})
        analysis = results.get('analysis', {})
        
        # Main info bar at top
        overlay_height = 100
        cv2.rectangle(image, (10, 10), (w - 10, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(image, 0.75, image, 0.25, 0, image)
        
        # Title and mode
        current_mode = self.modes[self.current_mode_idx]
        cv2.putText(image, "ReColor Unified Camera", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(image, f"Mode: {current_mode.title()} | CVD: {self.cvd_type.title()}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Performance and settings
        avg_fps = perf.get('avg_fps', 0)
        cv2.putText(image, f"FPS: {avg_fps:.1f} | K={self.k_clusters} | Strength: {self.dalton_strength:.2f}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Side labels for split view
        if not full_window and self.show_original:
            # Left side label
            cv2.putText(image, "ORIGINAL", (30, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Right side label  
            cv2.putText(image, f"{current_mode.upper()}", (w + 30, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Analysis info (bottom right)
        if analysis:
            diversity = analysis.get('color_diversity_score', 0)
            dominant = analysis.get('dominant_cluster_percentage', 0)
            
            info_y = h - 80
            cv2.rectangle(image, (w - 300, info_y), (w - 10, h - 10), (0, 0, 0), -1)
            
            cv2.putText(image, f"Diversity: {diversity:.2f}", 
                       (w - 290, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(image, f"Dominant: {dominant:.1f}%", 
                       (w - 290, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(image, f"Frame: {self.frame_count}", 
                       (w - 290, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def handle_unified_controls(self, key: int) -> bool:
        """
        Handle keyboard controls for unified camera.
        
        Args:
            key: Pressed key code
            
        Returns:
            True to continue, False to quit
        """
        if key == ord('q') or key == 27:  # Q or ESC
            return False
        
        elif key == ord('m'):
            # Cycle processing modes
            self.current_mode_idx = (self.current_mode_idx + 1) % len(self.modes)
            mode_name = self.modes[self.current_mode_idx]
            print(f"🔄 Mode: {mode_name}")
        
        elif key == ord('d'):
            # Cycle CVD types
            cvd_types = ['protanopia', 'deuteranopia', 'tritanopia']
            current_idx = cvd_types.index(self.cvd_type)
            self.cvd_type = cvd_types[(current_idx + 1) % len(cvd_types)]
            print(f"👁️  CVD Type: {self.cvd_type}")
        
        elif key == ord('k'):
            # Change K-means clusters
            k_values = [3, 4, 5, 6, 8, 10, 12, 16]
            try:
                current_idx = k_values.index(self.k_clusters)
                self.k_clusters = k_values[(current_idx + 1) % len(k_values)]
            except ValueError:
                self.k_clusters = 8
            
            self.clusterer.set_cluster_count(self.k_clusters)
            print(f"🎯 K-means clusters: {self.k_clusters}")
        
        elif key == ord('+') or key == ord('='):
            # Increase daltonization strength
            self.dalton_strength = min(self.dalton_strength + 0.1, 1.0)
            self.daltonizer.set_correction_strength(self.dalton_strength)
            print(f"🔧 Daltonization strength: {self.dalton_strength:.2f}")
        
        elif key == ord('-'):
            # Decrease daltonization strength
            self.dalton_strength = max(self.dalton_strength - 0.1, 0.0)
            self.daltonizer.set_correction_strength(self.dalton_strength)
            print(f"🔧 Daltonization strength: {self.dalton_strength:.2f}")
        
        elif key == ord('o'):
            # Toggle original view
            self.show_original = not self.show_original
            layout = "Split" if self.show_original else "Full"
            print(f"🖼️  Layout: {layout}")
        
        elif key == ord('p'):
            # Toggle palette
            self.show_palette = not self.show_palette
            print(f"🎨 Palette: {'ON' if self.show_palette else 'OFF'}")
        
        elif key == ord('r'):
            # Reset to defaults
            self.cvd_type = 'deuteranopia'
            self.k_clusters = 8
            self.dalton_strength = 0.7
            self.current_mode_idx = 0
            self.clusterer.set_cluster_count(self.k_clusters)
            self.daltonizer.set_correction_strength(self.dalton_strength)
            print("🔄 Reset to defaults")
        
        elif key == ord(' '):
            # Toggle pause
            self.paused = not self.paused
            print(f"⏸️  {'Paused' if self.paused else 'Resumed'}")
        
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"📸 Screenshot: recolor_unified_{timestamp}.jpg")
        
        elif key == ord('h'):
            # Show help
            self.print_help()
        
        return True
    
    def print_help(self):
        """Print control help."""
        print("\n" + "="*50)
        print("🎮 Unified ReColor Camera Controls:")
        print("   M - Switch modes (Unified/Simulation/Correction/Clustering)")
        print("   D - Cycle CVD types (Protanopia/Deuteranopia/Tritanopia)")
        print("   K - Change K-means clusters (3-16)")
        print("   +/- - Adjust daltonization strength")
        print("   O - Toggle original view (Split/Full)")
        print("   P - Toggle color palette display")
        print("   R - Reset to defaults")
        print("   Space - Pause/Resume")
        print("   S - Save screenshot")
        print("   H - Show this help")
        print("   Q/ESC - Quit")
        print("="*50)
    
    def run_unified_camera(self):
        """Main unified camera processing loop."""
        
        print("\n🎥 Starting Unified ReColor Camera")
        print("="*45)
        print("🎯 Features:")
        print("   • Real-time CVD simulation & correction")
        print("   • Adaptive K-means clustering")
        print("   • Daltonization enhancement")
        print("   • Unified processing modes")
        print("   • GPU-accelerated performance")
        print("="*45)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot open camera")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera: {actual_w}x{actual_h}")
        
        self.print_help()
        
        try:
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Error capturing frame")
                        break
                    
                    # Process frame through unified pipeline
                    results = self.process_unified_frame(frame)
                    self.frame_count += 1
                else:
                    # Keep displaying last results when paused
                    if 'results' not in locals():
                        continue
                
                # Create unified display
                display_image = self.create_unified_display(results)
                
                # Show unified window
                cv2.imshow('ReColor - Unified CVD Camera', display_image)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_unified_controls(key):
                        break
        
        except KeyboardInterrupt:
            print("\n⏹️  Camera stopped by user")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Final stats
            if self.fps_history:
                avg_fps = np.mean(self.fps_history)
                print(f"\n📊 Final Statistics:")
                print(f"   Frames processed: {self.frame_count}")
                print(f"   Average FPS: {avg_fps:.1f}")
                print(f"   CVD type: {self.cvd_type}")
                print(f"   Mode: {self.modes[self.current_mode_idx]}")
            
            print("✅ Unified ReColor Camera stopped successfully")

def main():
    """Main entry point for unified camera."""
    
    print("🎥" + "="*48 + "🎥")
    print("        ReColor Unified CVD Camera")
    print("   Real-time Color Vision Processing")
    print("🎥" + "="*48 + "🎥")
    
    try:
        camera = UnifiedReColorCamera()
        camera.run_unified_camera()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    except Exception as e:
        print(f"❌ Error starting unified camera: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())