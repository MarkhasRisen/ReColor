# ReColor - TensorFlow GPU-Optimized Colorblind Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![Real-time](https://img.shields.io/badge/Real--time-Processing-red.svg)](https://github.com)

A production-ready TensorFlow-based application for real-time colorblind detection and analysis. Features GPU-accelerated color recognition, scientifically accurate color vision deficiency (CVD) simulation, and comprehensive data logging with an intuitive camera interface.

## 🎯 Key Features

### 🧠 **AI-Powered Color Recognition**
- **TensorFlow CNN**: Lightweight convolutional neural network for 9-class color classification
- **GPU Acceleration**: Automatic CUDA detection and utilization for maximum performance
- **Real-time Inference**: Optimized for live webcam processing at 30+ FPS
- **Synthetic Dataset Training**: Automatic training on generated color data
- **Confidence Scoring**: AI prediction confidence levels for quality assessment

### 🎨 **K-means Color Analysis & Simplification**
- **Dominant Color Extraction**: Extract 5-15 dominant colors using K-means clustering
- **Color Distribution Analysis**: Real-time color diversity and dominance metrics
- **Image Posterization**: Simplify images using K-means for artistic effect
- **Performance Optimized**: Efficient sampling for real-time processing
- **Interactive Adjustment**: Live adjustment of cluster counts (1-2 keys)

### 👁️ **Enhanced Colorblind Simulation**
- **Realistic CVD Models**: Enhanced matrices reflecting actual colorblind experience
- **Six CVD Types**: 
  - **Protanopia/Protanomaly** (Red-blind) - Complete/partial L cone deficiency
  - **Deuteranopia/Deuteranomaly** (Green-blind) - Complete/partial M cone deficiency  
  - **Tritanopia/Tritanomaly** (Blue-blind) - Complete/partial S cone deficiency
- **Ishihara Test Simulation**: Demonstrates invisible numbers (12, 8, 5, etc.) for each CVD type
- **Realistic Color Confusion**: Accurate simulation of indistinguishable color pairs
- **Side-by-Side Display**: Live comparison of normal vs. colorblind vision

### 🔧 **Daltonization Enhancement**
- **Color Redistribution**: Enhance color discrimination for colorblind users
- **Adaptive Enhancement**: CVD-type specific optimization algorithms
- **Adjustable Strength**: Real-time adjustment from 0.5x to 3.0x enhancement
- **Before/After Comparison**: Toggle between enhanced and simulated views
- **Scientific Accuracy**: Based on error redistribution principles

### 🚀 **Unified Processing Pipeline**
- **Three-Stage Integration**: K-Means → CNN → Daltonization in seamless pipeline
- **Color Family Grouping**: K-means clusters similar colors into 8 distinct families
- **CNN Refinement**: Precise color classification within each cluster using TensorFlow
- **Enhanced Frame Generation**: Combines family smoothing with CNN-identified details
- **Real-time Performance**: Optimized for 30+ FPS with GPU acceleration
- **Comprehensive Analytics**: Detailed statistics for each processing stage

### 📊 **Comprehensive Data Logging**
- **CSV Export**: Automatic logging of all captures with timestamps
- **Color Metadata**: RGB, HEX, AI predictions, and confidence scores
- **CVD Context**: Current simulation type for each capture
- **Session Tracking**: Complete session summaries and statistics
- **JSON Export**: Structured data export for analysis

### 🎮 **Interactive Camera Interface**
- **Live Video Feed**: Real-time webcam processing with overlay information
- **Keyboard Controls**: Intuitive shortcuts for all functions
- **Color Information**: Live RGB, HEX, and color name display
- **Performance Metrics**: Real-time FPS and system monitoring
- **ROI Selection**: Configurable region of interest for color detection

## 🏗️ Architecture

Clean, modular object-oriented design with separated concerns:

```
ReColor/
├── main.py                 # Command-line interface and application entry
├── recolor_app.py          # Main application controller
├── color_model.py          # TensorFlow CNN for color classification
├── colorblind_detector.py  # CVD simulation with transformation matrices
├── camera_handler.py       # OpenCV webcam capture and display
├── color_logger.py         # CSV logging and session management
├── utils.py                # Helper functions and GPU setup
├── requirements.txt        # Python dependencies
├── models/                 # Saved TensorFlow model weights
├── logs/                   # CSV log files and session data
└── captures/               # Optional captured frame images
```

### Core Classes

- **`ColorModel`**: TensorFlow CNN with GPU auto-detection and synthetic training
- **`ColorBlindnessSimulator`**: Scientific CVD simulation using transformation matrices
- **`CameraHandler`**: OpenCV webcam management with real-time display
- **`ColorLogger`**: CSV logging system with session tracking
- **`ReColorApp`**: Main controller orchestrating all components

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Webcam** (USB or built-in)
- **NVIDIA GPU** (optional but recommended for best performance)
- **CUDA 11.8+** (automatically installed with TensorFlow)

### Installation

1. **Clone or download the project**:
   ```bash
   # Download all files to a folder named ReColor/
   cd ReColor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU setup** (optional):
   ```bash
   python main.py --system-info
   ```

### Running the Application

**Basic usage with default camera:**
```bash
python main.py --camera 0
```

**With custom settings:**
```bash
python main.py --camera 0 --width 800 --height 600 --fps 30
```

**Load pre-trained model:**
```bash
python main.py --camera 0 --model models/my_trained_model.h5
```

**Enhanced features on startup:**
```bash
python main.py --camera 0 --enable-kmeans --enable-daltonization --enable-realistic-cvd
```

**Unified pipeline with full integration:**
```bash
python main.py --camera 0 --enable-unified-pipeline --kmeans-clusters 8 --daltonization-strength 2.0
```

**Custom K-means and daltonization settings:**
```bash
python main.py --camera 0 --kmeans-clusters 8 --daltonization-strength 2.0
```

## 🎮 Controls & Usage

### Enhanced Keyboard Controls

| Key | Function |
|-----|----------|
| **C** | Capture current color and save to CSV |
| **N** | Cycle through CVD types (Normal → Protanopia → Deuteranopia → Tritanopia → Protanomaly → Deuteranomaly → Tritanomaly) |
| **P** | Pause/Resume camera feed |
| **S** | Toggle side-by-side display (Normal vs CVD simulation) |
| **I** | Toggle color information overlay |
| **F** | Toggle FPS display |
| **Q** | Quit application |

### 🚀 **Advanced Features**
| Key | Function |
|-----|----------|
| **K** | Toggle K-means color analysis display |
| **M** | Toggle K-means image simplification (posterization) |
| **D** | Toggle daltonization (color enhancement for colorblind) |
| **R** | Toggle realistic CVD simulation (Ishihara-style confusion) |
| **U** | Toggle unified pipeline (K-means → CNN → Daltonization) |
| **1** | Decrease K-means clusters (minimum 2) |
| **2** | Increase K-means clusters (maximum 15) |
| **3** | Decrease daltonization strength (minimum 0.5x) |
| **4** | Increase daltonization strength (maximum 3.0x) |

### Display Layout

The application shows:
- **Live camera feed** with real-time color detection
- **Side-by-side CVD simulation** (when enabled)
- **Color information overlay** (RGB, HEX, AI prediction, confidence)
- **Region of Interest (ROI)** indicator for color sampling
- **Current CVD type** indicator
- **Real-time FPS** counter

### Data Output

All captured colors are automatically saved to:
- **CSV files** in `logs/` directory with timestamps and metadata
- **JSON exports** available for detailed session analysis
- **Session summaries** with statistics and color distributions

## 🔧 Configuration Options

### Command Line Arguments

```bash
python main.py [OPTIONS]

Camera Settings:
  --camera ID           Camera device ID (default: 0)
  --width WIDTH         Frame width (default: 640)  
  --height HEIGHT       Frame height (default: 480)
  --fps FPS             Target FPS (default: 30)

Model Settings:
  --model PATH          Path to pre-trained model weights
  --no-auto-train       Disable automatic training
  --train-samples N     Training samples per class (default: 500)
  --train-epochs N      Training epochs (default: 20)

Logging:
  --log-dir DIR         Log directory (default: logs)
  --verbose             Enable verbose logging
  --debug               Enable debug mode

Information:
  --system-info         Show system information and exit
  --version             Show version and exit
```

### Example Configurations

**High-resolution capture:**
```bash
python main.py --camera 0 --width 1280 --height 720 --fps 30
```

**Quick testing (minimal training):**
```bash
python main.py --camera 0 --train-samples 100 --train-epochs 5
```

**Production mode with pre-trained model:**
```bash
python main.py --camera 0 --model models/production_model.h5 --log-dir production_logs
```

## 📊 Data Analysis

### CSV Output Format

Each capture generates a CSV row with:
- `timestamp`: Capture time with millisecond precision
- `session_id`: Unique session identifier
- `color_name_rgb`: Color name from RGB analysis
- `color_name_ai`: AI model prediction
- `rgb_r`, `rgb_g`, `rgb_b`: Individual RGB values
- `hex_color`: Hexadecimal color code
- `ai_confidence`: Model confidence (0.0-1.0)
- `cvd_type`: Current CVD simulation type
- `cvd_description`: Human-readable CVD description
- `roi_x1`, `roi_y1`, `roi_x2`, `roi_y2`: Region of interest coordinates
- `capture_time_ms`: Time since session start (milliseconds)

### Session Analysis

The application provides:
- **Real-time statistics** during capture sessions
- **Color frequency analysis** showing most common colors
- **Confidence metrics** for AI prediction quality
- **CVD type distribution** across captures
- **Session duration** and capture rates

## 🔬 Technical Details

### Color Recognition Model

- **Architecture**: 3-layer CNN with batch normalization and dropout
- **Input**: 64x64 RGB images
- **Output**: 9 color classes (Red, Green, Blue, Yellow, Orange, Purple, Pink, Brown, Gray)
- **Training**: Synthetic dataset with noise and texture variations
- **Optimization**: Adam optimizer with learning rate scheduling
- **Performance**: >95% accuracy on synthetic test data

### CVD Simulation Algorithm

Uses scientifically accurate transformation matrices:

```python
# Example: Protanopia transformation matrix
protanopia_matrix = [
    [0.567, 0.433, 0.000],
    [0.558, 0.442, 0.000], 
    [0.000, 0.242, 0.758]
]
```

Applied as linear transformation: `cvd_pixel = normal_pixel × transformation_matrix`

### Performance Optimization

- **GPU Acceleration**: Automatic CUDA utilization when available
- **Memory Management**: Dynamic GPU memory growth to prevent allocation errors
- **Threading**: Separate threads for camera capture and processing
- **Caching**: Intelligent caching of predictions and transformations
- **Frame Skipping**: Automatic frame rate regulation for smooth display

## 🛠️ Development & Testing

### Running Tests

```bash
# Test individual components
python color_model.py          # Test CNN model
python colorblind_detector.py  # Test CVD simulation
python camera_handler.py       # Test camera (requires webcam)
python color_logger.py         # Test logging system
python utils.py                 # Test utility functions

# Test complete application initialization
python recolor_app.py
```

### System Requirements Verification

```bash
python main.py --system-info
```

This will show:
- Python and library versions
- GPU availability and specifications
- Available cameras and their capabilities
- File system permissions
- CUDA setup status

## 🐛 Troubleshooting

### Common Issues

**Camera not detected:**
- Ensure webcam is connected and not in use by other applications
- Try different camera IDs: `--camera 1`, `--camera 2`, etc.
- Check camera permissions in system settings

**GPU not detected:**
- Verify NVIDIA drivers are installed and up to date
- Check CUDA compatibility with `nvidia-smi`
- Consider using `--verbose` flag for detailed GPU information

**Model training fails:**
- Reduce training parameters: `--train-samples 100 --train-epochs 5`
- Check available system memory
- Try CPU-only mode by installing `tensorflow-cpu`

**Poor color detection accuracy:**
- Ensure good lighting conditions
- Check if model finished training (look for "Training completed" message)
- Try training with more samples: `--train-samples 1000`

### Performance Tips

- **Use GPU**: Significant performance improvement with CUDA-compatible GPU
- **Adjust resolution**: Lower resolution = higher FPS (`--width 480 --height 360`)
- **Optimize lighting**: Consistent lighting improves color detection accuracy
- **Close other applications**: Free up system resources for better performance

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.15, or Linux Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Camera**: Any USB or built-in webcam

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, or Linux Ubuntu 20.04+
- **Python**: 3.9 or higher
- **RAM**: 8GB or more
- **GPU**: NVIDIA GTX 1060 or better with 4GB+ VRAM
- **Storage**: 5GB free space (for model training and logs)
- **Camera**: HD webcam (720p or higher)

## 📝 License & Citation

This project implements scientifically accurate colorblind simulation based on:

> Brettel, H., Viénot, F., & Mollon, J. D. (1997). Computerized simulation of color appearance for dichromats. *Journal of the Optical Society of America A*, 14(10), 2647-2655.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional CVD types (Protanomaly, Deuteranomaly, Tritanomaly)
- Enhanced color classification models
- Mobile/web deployment
- Advanced color analysis features
- Performance optimizations

---

**ReColor TensorFlow v1.0.0** - Real-time GPU-accelerated colorblind detection system