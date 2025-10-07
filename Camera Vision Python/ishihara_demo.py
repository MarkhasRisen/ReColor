#!/usr/bin/env python3
"""
Ishihara Test Demo
================
Demonstration of Ishihara color blindness test optimization.
Shows before/after comparison for different CVD types.

Author: ReColor Development Team
Date: October 2025
License: MIT
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvd_simulation import CVDSimulator
from ishihara_optimizer import IshiharaOptimizer

def create_ishihara_demo():
    """Create comprehensive Ishihara test demonstration."""
    
    print("🎯 Ishihara Test Optimization Demo")
    print("="*40)
    
    # Initialize components
    simulator = CVDSimulator()
    optimizer = IshiharaOptimizer()
    
    # Create test Ishihara plates
    test_numbers = ["8", "3", "5", "2"]
    cvd_types = ['protanopia', 'deuteranopia', 'tritanopia']
    
    # Create comparison grid for each CVD type
    for cvd_type in cvd_types:
        print(f"\n📋 Testing {cvd_type.title()}...")
        
        fig, axes = plt.subplots(len(test_numbers), 4, figsize=(16, 12))
        fig.suptitle(f'Ishihara Test Optimization - {cvd_type.title()}', fontsize=16)
        
        for i, number in enumerate(test_numbers):
            # Create original Ishihara plate
            original_plate = optimizer.create_ishihara_test_simulation(400, 400, number)
            
            # Simulate CVD
            cvd_simulated = simulator.simulate_cvd(original_plate, cvd_type)
            
            # Apply Ishihara optimization
            ishihara_results = optimizer.optimize_ishihara_visibility(original_plate, cvd_type, 1.5)
            
            # Get CVD simulation with Ishihara optimization
            optimized_results = simulator.simulate_cvd_ishihara_optimized(original_plate, cvd_type, 1.5)
            
            # Display results
            axes[i, 0].imshow(original_plate)
            axes[i, 0].set_title(f'Original (Number {number})')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(cvd_simulated)
            axes[i, 1].set_title(f'{cvd_type.title()} Simulation')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(ishihara_results['final_optimized'])
            axes[i, 2].set_title('Ishihara Optimized')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(optimized_results['ishihara_optimized'])
            axes[i, 3].set_title('Combined Optimization')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'ishihara_demo_{cvd_type}.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: ishihara_demo_{cvd_type}.png")
        plt.close()
    
    # Performance comparison
    print(f"\n📊 Performance Analysis:")
    
    test_plate = optimizer.create_ishihara_test_simulation(640, 480, "8")
    
    # Standard CVD simulation
    import time
    times_standard = []
    for _ in range(10):
        start = time.time()
        simulator.simulate_cvd(test_plate, 'deuteranopia')
        times_standard.append(time.time() - start)
    
    # Ishihara-optimized simulation
    times_optimized = []
    for _ in range(10):
        start = time.time()
        simulator.simulate_cvd_ishihara_optimized(test_plate, 'deuteranopia', 1.5)
        times_optimized.append(time.time() - start)
    
    avg_standard = np.mean(times_standard) * 1000
    avg_optimized = np.mean(times_optimized) * 1000
    
    print(f"   Standard CVD simulation: {avg_standard:.1f}ms")
    print(f"   Ishihara optimization: {avg_optimized:.1f}ms")
    print(f"   Overhead: {avg_optimized - avg_standard:.1f}ms")
    print(f"   FPS (Standard): {1000/avg_standard:.1f}")
    print(f"   FPS (Optimized): {1000/avg_optimized:.1f}")

def interactive_ishihara_test():
    """Interactive Ishihara test with real-time optimization."""
    
    print("\n🎮 Interactive Ishihara Test")
    print("Press keys to control:")
    print("  1-4: Switch test numbers")
    print("  P/D/T: Switch CVD type (Protanopia/Deuteranopia/Tritanopia)")
    print("  +/-: Adjust optimization strength")
    print("  O: Toggle optimization on/off")
    print("  Q: Quit")
    
    # Initialize
    simulator = CVDSimulator()
    optimizer = IshiharaOptimizer()
    
    # Settings
    current_number_idx = 0
    numbers = ["8", "3", "5", "2", "6", "9"]
    current_cvd_idx = 1  # Start with deuteranopia
    cvd_types = ['protanopia', 'deuteranopia', 'tritanopia']
    optimization_strength = 1.5
    optimization_enabled = True
    
    while True:
        # Create test plate
        current_number = numbers[current_number_idx]
        current_cvd = cvd_types[current_cvd_idx]
        
        original_plate = optimizer.create_ishihara_test_simulation(500, 500, current_number)
        
        # Process based on settings
        if optimization_enabled:
            results = simulator.simulate_cvd_ishihara_optimized(original_plate, current_cvd, optimization_strength)
            processed_plate = results['ishihara_optimized']
            title = f"Ishihara Optimized - {current_cvd.title()} - Number {current_number}"
        else:
            processed_plate = simulator.simulate_cvd(original_plate, current_cvd)
            title = f"Standard CVD - {current_cvd.title()} - Number {current_number}"
        
        # Display
        display = np.hstack([original_plate, processed_plate])
        
        # Add title
        cv2.putText(display, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Strength: {optimization_strength:.1f} | Optimization: {'ON' if optimization_enabled else 'OFF'}", 
                   (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Interactive Ishihara Test', display)
        
        # Handle input
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key >= ord('1') and key <= ord('6'):
            current_number_idx = key - ord('1')
            current_number_idx = min(current_number_idx, len(numbers) - 1)
        elif key == ord('p'):
            current_cvd_idx = 0  # Protanopia
        elif key == ord('d'):
            current_cvd_idx = 1  # Deuteranopia
        elif key == ord('t'):
            current_cvd_idx = 2  # Tritanopia
        elif key == ord('+') or key == ord('='):
            optimization_strength = min(optimization_strength + 0.1, 3.0)
        elif key == ord('-'):
            optimization_strength = max(optimization_strength - 0.1, 0.5)
        elif key == ord('o'):
            optimization_enabled = not optimization_enabled
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🎯 Ishihara Test Optimization System")
    print("="*50)
    
    try:
        # Create demonstration images
        create_ishihara_demo()
        
        # Ask for interactive test
        choice = input("\n🎮 Run interactive test? (y/N): ").lower().strip()
        if choice == 'y':
            interactive_ishihara_test()
        
        print("\n✅ Ishihara demo complete!")
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()