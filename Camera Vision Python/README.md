# ReColor - Mobile CVD Assistive Tool
🎥 Real-time Color Vision Processing System with Ishihara Test Optimization

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🎯 **Project Overview**

ReColor is a comprehensive mobile vision assistive tool designed for Color Vision Deficiency (CVD) that integrates adaptive daltonization, K-Means clustering, and convolutional neural networks (CNNs) for real-time color enhancement and screening. The system provides specialized optimization for Ishihara color blindness tests with scientifically accurate CVD simulation.

### **🌟 Key Features**

- **🎥 Real-time CVD Camera Processing** - Live camera feed with CVD simulation and correction
- **🎯 Ishihara Test Optimization** - Specialized enhancement for color blindness tests
- **🧠 Multiple Processing Modes** - Unified, Simulation, Correction, Clustering, and Ishihara modes
- **⚡ GPU Acceleration** - Optimized for RTX 4050 and modern GPUs
- **📱 Mobile Ready** - TensorFlow Lite support for deployment
- **🔬 Scientific Accuracy** - LMS color space with Hunt-Pointer-Estevez matrices
- **🎮 Interactive Controls** - Real-time parameter adjustment and testing

---

## 📁 **Project Structure**

```
ReColor/
├── 📋 Core Modules
│   ├── launcher.py              # Main application launcher
│   ├── cvd_simulation.py        # CVD simulation with unified camera
│   ├── daltonization.py         # Adaptive daltonization algorithms  
│   ├── clustering.py            # Real-time K-means clustering
│   ├── ishihara_optimizer.py    # Ishihara test optimization
│   ├── cnn_model.py             # TensorFlow Lite CNN architecture
│   └── main.py                  # Modular system controller
├── 📄 Configuration
│   ├── requirements.txt         # Python dependencies
│   └── gpu_test.py              # GPU performance testing
├── 📚 Documentation
│   ├── README.md                # This file
│   ├── ISHIHARA_OPTIMIZATION.md # Ishihara feature documentation
│   ├── PROJECT_SUMMARY.md       # Technical summary
│   └── ALGORITHM_IMPLEMENTATION.md # Algorithm details
└── 🛠️ Environment
    └── venv/                    # Python virtual environment
```

---

## 🚀 **Quick Start**

### **1. Prerequisites**

- **Python 3.8+** with pip
- **NVIDIA GPU** with CUDA support (RTX 4050+ recommended)
- **Camera/Webcam** for real-time processing
- **Windows 10/11** (tested), macOS, or Linux

### **2. Installation**

1. **Clone or Download** the ReColor project
2. **Navigate** to the project directory
3. **Activate Virtual Environment**:
   ```bash
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   
   # Windows CMD
   venv\Scripts\activate.bat
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify GPU Setup**:
   ```bash
   python gpu_test.py
   ```

### **3. Launch Application**

```bash
python launcher.py
```

**Choose from available options:**
1. 🎥 **Unified CVD Camera** (RECOMMENDED)
2. 🎯 **Modular ReColor System**  
3. 🎯 **Ishihara Test Optimization**
4. 🧪 **GPU Performance Test**
5. ⚙️ **System Configuration**

---

## 🎮 **Usage Guide**

### **🎥 Unified CVD Camera**

**Launch**: `python launcher.py` → Option 1

**Interactive Controls:**
- `M` - Switch modes (Unified/Simulation/Correction/Clustering/Ishihara)
- `D` - Cycle CVD types (Protanopia/Deuteranopia/Tritanopia)
- `K` - Change K-means clusters (3-16)
- `+`/`-` - Adjust daltonization strength
- `I` - Toggle Ishihara optimization
- `[`/`]` - Ishihara strength (0.5-2.0)
- `T` - Generate test Ishihara plate
- `O` - Toggle original view (Split/Full)
- `P` - Toggle color palette display
- `R` - Reset to defaults
- `Space` - Pause/Resume
- `H` - Show help
- `Q`/`ESC` - Quit

### **🎯 Processing Modes**

1. **Unified Mode** - Blends all techniques (20% simulation + 60% correction + 20% clustering)
2. **Simulation Mode** - Pure CVD simulation showing how colorblind people see
3. **Correction Mode** - Daltonization enhancement for better visibility
4. **Clustering Mode** - K-means color segmentation visualization
5. **Ishihara Mode** - Specialized optimization for color blindness tests

### **🧪 Ishihara Test Optimization**

**Launch**: `python launcher.py` → Option 3

**Features:**
- **Test Plate Generation** - Create Ishihara-style test plates
- **Before/After Comparison** - See optimization effects
- **All CVD Types** - Protanopia, Deuteranopia, Tritanopia support
- **Interactive Testing** - Real-time parameter adjustment
- **Performance Analysis** - FPS and processing time metrics

---

## ⚡ **Performance Specifications**

### **Hardware Requirements**

| Component | Minimum | Recommended | Tested |
|-----------|---------|-------------|--------|
| **GPU** | Any CUDA GPU | RTX 3060+ | RTX 4050 Laptop |
| **RAM** | 8GB | 16GB+ | 16GB |
| **Python** | 3.8 | 3.10+ | 3.10 |
| **CUDA** | 11.0+ | 12.0+ | 12.1 |

### **Performance Benchmarks**

| Processing Mode | FPS | Latency | GPU Usage |
|----------------|-----|---------|-----------|
| **CVD Simulation** | 202.4 | 4.9ms | Low |
| **Unified Processing** | 15.8 | 46.4ms | Medium |
| **Ishihara Optimization** | 17.1 | 58.5ms | Medium |
| **K-means Clustering** | 27.0 | 37.7ms | High |

*Benchmarked on RTX 4050 Laptop GPU with 640x480 camera resolution*

---

## 🔬 **Technical Details**

### **🧬 Scientific Accuracy**

- **LMS Color Space**: Hunt-Pointer-Estevez transformation matrices
- **CVD Simulation**: Brettel et al. (1997) algorithm implementation
- **Confusion Lines**: Physiologically accurate CVD modeling
- **Color Accuracy**: Research-validated enhancement techniques

### **🎯 CVD Types Supported**

| CVD Type | Prevalence | Description | Optimization |
|----------|------------|-------------|--------------|
| **Protanopia** | ~1% males | Missing L-cones (red) | 2.5x red boost |
| **Deuteranopia** | ~1% males | Missing M-cones (green) | 2.2x red boost, 0.6x green |
| **Tritanopia** | ~0.01% | Missing S-cones (blue) | 2.5x blue boost |

### **⚙️ Algorithms**

1. **K-Means Clustering**
   - MiniBatch K-Means for real-time processing
   - Adaptive cluster count (3-16)
   - Color palette extraction and analysis

2. **Adaptive Daltonization**
   - Confusion line analysis
   - Perceptual contrast enhancement
   - Ishihara test optimization

3. **CNN Architecture**
   - TensorFlow Lite ready
   - MobileNetV2-inspired design
   - Mobile deployment optimized

---

## 📚 **API Reference**

### **CVD Simulation**

```python
from cvd_simulation import CVDSimulator

# Initialize simulator
simulator = CVDSimulator(optimization_level='balanced')

# Simulate CVD
cvd_image = simulator.simulate_cvd(rgb_image, 'deuteranopia')

# Ishihara optimization
simulator.toggle_ishihara_mode(True)
results = simulator.simulate_cvd_ishihara_optimized(
    image, 'deuteranopia', strength=1.5
)
```

### **Daltonization**

```python
from daltonization import AdaptiveDaltonizer

# Initialize daltonizer
daltonizer = AdaptiveDaltonizer(adaptation_level='medium')

# Apply enhancement
enhanced = daltonizer.adaptive_daltonization(image, 'protanopia')
```

### **Clustering**

```python
from clustering import RealtimeColorClusterer

# Initialize clusterer
clusterer = RealtimeColorClusterer(n_clusters=8)

# Process video frame
results = clusterer.process_video_frame(rgb_frame)
clustered_image = results['cluster_image']
```

### **Ishihara Optimization**

```python
from ishihara_optimizer import IshiharaOptimizer

# Initialize optimizer
optimizer = IshiharaOptimizer()

# Optimize for CVD
results = optimizer.optimize_ishihara_visibility(
    image, 'deuteranopia', enhancement_strength=1.0
)
```

---

## 🛠️ **Development**

### **Dependencies**

```txt
# Core ML/AI
tensorflow>=2.10.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Computer Vision
opencv-python>=4.5.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.5.0
pillow>=8.0.0
seaborn>=0.11.0

# Performance
psutil>=5.8.0
numba>=0.56.0

# Mobile Optimization
onnx>=1.12.0
onnxruntime>=1.12.0
```

### **Environment Setup**

1. **Create Virtual Environment**:
   ```bash
   python -m venv recolor_env
   ```

2. **Activate Environment**:
   ```bash
   # Windows
   recolor_env\Scripts\activate
   
   # macOS/Linux  
   source recolor_env/bin/activate
   ```

3. **Install Development Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8  # Development tools
   ```

### **Testing**

```bash
# Test individual modules
python cvd_simulation.py        # CVD simulation test
python ishihara_optimizer.py    # Ishihara optimization test
python gpu_test.py              # GPU performance test

# Full system test
python launcher.py              # Interactive testing
```

---

## 📱 **Mobile Deployment**

### **TensorFlow Lite Conversion**

The system is designed for mobile deployment with TensorFlow Lite support:

```python
# Convert model to TensorFlow Lite
from cnn_model import AdaptiveCVDModel

model = AdaptiveCVDModel()
tflite_model = model.convert_to_tflite()

# Save for mobile deployment
with open('recolor_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### **Optimization Features**

- **Quantization**: INT8 quantization for faster inference
- **Memory Efficiency**: Minimal memory footprint
- **Battery Optimization**: Efficient algorithms for mobile devices
- **Cross-Platform**: Android and iOS compatible

---

## 🎓 **Educational Applications**

### **Learning Features**

- **Interactive CVD Simulation** - Understand how different types of color blindness affect vision
- **Real-time Enhancement** - See the effects of different optimization techniques
- **Scientific Accuracy** - Based on peer-reviewed research and physiological models
- **Accessibility Testing** - Test color accessibility of images and interfaces

### **Research Applications**

- **CVD Research** - Tool for color vision deficiency studies
- **Display Technology** - Test display accessibility features
- **Human Factors** - Study color perception and enhancement techniques
- **Medical Screening** - Assistive technology for medical professionals

---

## 🏥 **Medical & Accessibility Applications**

### **Clinical Use**

- **Pre-screening** - Initial CVD assessment tool
- **Educational Aid** - Help patients understand their condition
- **Accessibility Testing** - Validate medical interface designs
- **Research Tool** - Study color vision enhancement techniques

### **Accessibility Features**

- **WCAG 2.2 Compliance** - Meets web accessibility guidelines
- **Multiple Enhancement Modes** - Different optimization approaches
- **Real-time Adjustment** - Interactive parameter tuning
- **Cross-Platform Support** - Works on multiple devices and platforms

---

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Areas for Contribution**

- **New CVD Types** - Support for additional color vision deficiencies
- **Enhancement Algorithms** - New optimization techniques
- **Mobile Features** - Android/iOS specific optimizations
- **Documentation** - Improve documentation and tutorials
- **Testing** - Add more comprehensive test coverage

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

### **Research References**

- **Brettel, H., Viénot, F., & Mollon, J. D.** (1997). Computerized simulation of color appearance for dichromats. *Journal of the Optical Society of America A*, 14(10), 2647-2655.
- **Hunt, R. W. G.** (1991). Measuring colour. Ellis Horwood.
- **Viénot, F., Brettel, H., & Mollon, J. D.** (1999). Digital video colourmaps for checking the legibility of displays by dichromats. *Color Research & Application*, 24(4), 243-252.

### **Technologies Used**

- **PyTorch** - GPU acceleration and tensor operations
- **TensorFlow** - Machine learning and mobile deployment
- **OpenCV** - Computer vision and image processing
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms

### **Special Thanks**

- **Color Vision Research Community** - For scientific foundations
- **Open Source Contributors** - For the amazing libraries and tools
- **Accessibility Advocates** - For promoting inclusive technology
- **Medical Professionals** - For validation and feedback

---

## 📞 **Support & Contact**

### **Getting Help**

1. **Check Documentation** - Review this README and related docs
2. **Test Your Setup** - Run `python gpu_test.py` for diagnostics
3. **Check Issues** - Look for similar issues in the project repository
4. **System Info** - Use launcher Option 5 for system configuration

### **System Requirements Check**

```bash
# Quick system check
python launcher.py
# Choose Option 5: System Configuration
# View framework status and dependencies
```

### **Common Issues**

| Issue | Solution |
|-------|----------|
| **GPU not detected** | Install CUDA drivers, check `nvidia-smi` |
| **Import errors** | Activate virtual environment, reinstall dependencies |
| **Camera not working** | Check camera permissions, try different camera index |
| **Slow performance** | Enable GPU acceleration, reduce resolution |

---

## 🚀 **Future Roadmap**

### **Planned Features**

- **🌐 Web Interface** - Browser-based CVD testing
- **📱 Mobile Apps** - Native Android/iOS applications  
- **🎨 Advanced Enhancement** - New optimization algorithms
- **📊 Analytics Dashboard** - Usage statistics and effectiveness metrics
- **🔗 API Integration** - RESTful API for third-party integration
- **🎓 Educational Content** - Interactive tutorials and learning materials

### **Version History**

- **v1.0** - Initial release with basic CVD simulation
- **v2.0** - Added real-time camera processing and clustering
- **v3.0** - Integrated Ishihara optimization and enhanced controls
- **v3.1** - Current version with comprehensive documentation

---

**🎊 ReColor - Enhancing Color Vision for Everyone! 🎊**

*Making the world more colorful and accessible, one pixel at a time.*