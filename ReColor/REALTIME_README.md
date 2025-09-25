# TensorFlow Real-Time Image Processing Pipeline

A high-performance real-time image processing system using TensorFlow operations for webcam capture, K-Means clustering, and Daltonization color correction.

## 🎯 Overview

This system provides a complete real-time image processing pipeline that:
- Captures video frames from webcam using OpenCV
- Processes frames with pure TensorFlow operations (no Keras/scikit-learn)
- Applies Daltonization and K-Means clustering in real-time
- Displays original, daltonized, and clustered video streams simultaneously
- Achieves 25-30 FPS performance on modern hardware

## 🚀 Features

- **Real-Time Processing**: Optimized for 30 FPS video processing
- **Pure TensorFlow**: All image processing uses TensorFlow operations only
- **Multi-Window Display**: Shows original, daltonized, and clustered streams
- **Interactive Controls**: Change parameters during runtime
- **Performance Monitoring**: Real-time FPS and processing time display
- **No-Camera Demo**: Synthetic video mode for testing without hardware
- **Modular Design**: Clean separation for easy algorithm implementation

## 📁 File Structure

```
realtime_processor.py     # Main real-time processor with camera support
optimized_realtime.py     # Performance-optimized version
realtime_demo.py         # Demo mode (no camera required)
realtime_config.py       # Configuration parameters
requirements.txt         # Dependencies
```

## 🛠 Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python realtime_demo.py
   ```

## 🎮 Usage

### Quick Start (No Camera Required)

```bash
python realtime_demo.py
```

This runs a demo with synthetic animated video to test the processing pipeline.

### Real Camera Processing

```bash
python realtime_processor.py
```

or for optimized performance:

```bash
python optimized_realtime.py
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `K` | Cycle K-Means clusters (4, 8, 16, 32) |
| `D` | Cycle deficiency types (protanopia, deuteranopia, tritanopia) |
| `F` | Toggle FPS display |
| `H` | Show help (demo mode) |
| `S` | Save current frame |
| `R` | Reset to default parameters |

## 🔧 Core Functions to Implement

### 1. Real-Time K-Means Clustering

**Location**: `realtime_processor.py`, `tf_kmeans()` function

**Current Status**: Fast quantization placeholder

**TODO**: Implement efficient K-Means for real-time performance:

```python
@tf.function
def tf_kmeans_optimized(self, frame: tf.Tensor, k: tf.Tensor) -> tf.Tensor:
    # Your real-time K-Means implementation here:
    # 1. Limit iterations for real-time performance (5-10 max)
    # 2. Use frame-to-frame centroid caching for temporal consistency
    # 3. Consider spatial subsampling for speed
    # 4. Implement early convergence detection
    # 5. Use vectorized operations for GPU acceleration
```

**Performance Requirements**:
- Process 640x480 frames at 30 FPS
- Maximum 10 iterations per frame
- Target: <10ms processing time per frame

### 2. Real-Time Daltonization

**Location**: `realtime_processor.py`, `daltonize()` function

**Current Status**: Simple color channel adjustment placeholder

**TODO**: Implement efficient Daltonization:

```python
@tf.function
def tf_daltonize_optimized(self, frame: tf.Tensor, deficiency_index: tf.Tensor) -> tf.Tensor:
    # Your real-time Daltonization implementation here:
    # 1. Use pre-compiled TensorFlow matrices (self.sim_matrices)
    # 2. Batch matrix operations for efficiency
    # 3. Minimize memory allocations
    # 4. Consider lookup tables for very fast processing
```

**Performance Requirements**:
- Process frames in <5ms
- Use pre-compiled transformation matrices
- Support all three deficiency types

## ⚙️ Configuration

Modify `realtime_config.py` for different settings:

```python
# Camera settings
CAMERA_CONFIG = {
    'frame_width': 640,
    'frame_height': 480,
    'target_fps': 30,
}

# Processing parameters
REALTIME_CONFIG = {
    'default_k': 8,
    'max_iterations': 10,  # Reduced for real-time
    'default_deficiency': 'deuteranopia',
}
```

## 📊 Performance Optimization

### TensorFlow Optimizations

The system includes several performance optimizations:

1. **Function Compilation**: Uses `@tf.function` for critical functions
2. **Pre-compiled Matrices**: Color transformation matrices as `tf.constant`
3. **Memory Growth**: GPU memory growth enabled
4. **Efficient Conversions**: Optimized OpenCV ↔ TensorFlow conversions

### Real-Time Considerations

- **Reduced Iterations**: K-Means limited to 10 iterations max
- **Frame Rate Control**: Maintains consistent 30 FPS
- **Memory Management**: Efficient tensor operations
- **GPU Acceleration**: Automatic GPU usage when available

## 🎥 Demo Modes

### 1. Synthetic Video Demo (`realtime_demo.py`)

- Animated patterns and colors
- No camera required
- Perfect for testing algorithms
- Runs at 30 FPS with performance monitoring

### 2. Real Camera Processing

- Live webcam feed
- Three simultaneous windows
- Real-time parameter adjustment
- Performance monitoring overlay

## 🔍 Example Workflow

```python
from realtime_processor import RealTimeImageProcessor

# Initialize processor
processor = RealTimeImageProcessor(camera_id=0, target_fps=30)

# Start real-time processing
processor.run_realtime_processing()

# The system will:
# 1. Capture frames from camera
# 2. Convert to TensorFlow tensors
# 3. Apply Daltonization
# 4. Apply K-Means clustering  
# 5. Display all three streams
# 6. Monitor performance
```

## 🚧 Implementation Guidelines

### For K-Means Implementation:

1. **Use Vectorized Operations**: Leverage TensorFlow's batch operations
2. **Limit Iterations**: Target 5-10 iterations maximum for real-time
3. **Cache Centroids**: Use previous frame centroids as initialization
4. **Early Termination**: Stop when convergence is reached
5. **Spatial Subsampling**: Consider processing every Nth pixel for speed

### For Daltonization Implementation:

1. **Matrix Pre-computation**: Use `tf.constant` for transformation matrices
2. **Batch Processing**: Process entire frames with matrix multiplication
3. **Memory Efficiency**: Minimize intermediate tensor creation
4. **Color Space Conversion**: Implement efficient RGB ↔ LMS conversion

### Performance Targets:

- **Overall Frame Rate**: 25-30 FPS
- **K-Means Processing**: <10ms per frame
- **Daltonization Processing**: <5ms per frame
- **Total Pipeline Latency**: <50ms end-to-end

## 🎛 Window Layout

The system displays three windows side-by-side:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Original  │  │ Daltonized  │  │ K-Means     │
│   Feed      │  │   Output    │  │ Clustered   │
│             │  │             │  │             │
│ [Camera]    │  │ [Enhanced]  │  │ [Reduced    │
│             │  │             │  │  Colors]    │
└─────────────┘  └─────────────┘  └─────────────┘
```

Each window shows:
- Current FPS
- Processing parameters
- Performance metrics

## 🔬 Testing Your Implementation

1. **Start with Demo Mode**:
   ```bash
   python realtime_demo.py
   ```

2. **Test Algorithm Changes**:
   - Modify `tf_kmeans()` or `daltonize()` functions
   - Run demo to see immediate results
   - Monitor FPS to ensure real-time performance

3. **Performance Profiling**:
   ```python
   # Enable profiling in config
   PERFORMANCE_CONFIG['enable_profiling'] = True
   ```

4. **Validate with Real Camera**:
   ```bash
   python optimized_realtime.py
   ```

## 🎯 Success Criteria

Your implementation is successful when:

- [ ] Demo runs at 25+ FPS consistently
- [ ] Real camera processing maintains 25+ FPS
- [ ] K-Means produces visually distinct color clusters
- [ ] Daltonization creates noticeable color corrections
- [ ] Interactive controls work smoothly
- [ ] No frame drops or processing delays
- [ ] Memory usage remains stable over time

## 🐛 Troubleshooting

### Low FPS Issues:
- Reduce K-Means iterations (`max_iterations` in config)
- Lower camera resolution
- Disable FPS overlay temporarily
- Check GPU utilization

### Camera Issues:
- Try different camera IDs (0, 1, 2)
- Check camera permissions
- Use demo mode for testing algorithms

### Memory Issues:
- Enable GPU memory growth
- Reduce frame buffer size
- Monitor TensorFlow memory usage

## 📚 Implementation References

- **K-Means**: Lloyd's Algorithm with TensorFlow operations
- **Daltonization**: Fidaner et al. (2005) color correction method
- **Real-time Processing**: TensorFlow performance optimization guide
- **Color Vision**: Hunt-Pointer-Estevez color space transformations

## 🎉 Ready for Real-Time Implementation!

The framework provides:
- ✅ Complete real-time pipeline structure
- ✅ Performance optimization foundation
- ✅ Interactive testing environment
- ✅ Demo mode for algorithm development
- ✅ Configuration management system

Focus on implementing the core algorithms while the framework handles all real-time infrastructure!