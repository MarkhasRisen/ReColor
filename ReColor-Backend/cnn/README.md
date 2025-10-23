# ReColor - AI-Powered Colorblind Detection System

A real-time application that helps people understand and work with color vision differences using artificial intelligence.

## 🎯 What Does This Do?

**ReColor** is an AI-powered camera application that:
- **Detects colors** in real-time using your webcam
- **Simulates colorblindness** so you can see how colorblind people see the world
- **Enhances colors** to help colorblind users distinguish between difficult colors
- **Analyzes color patterns** using smart algorithms
- **Saves data** about colors you capture for later analysis

## 🌟 Key Features

### 🤖 **Smart Color Detection**
- Uses AI (TensorFlow) to recognize 9 different colors
- Works in different lighting conditions
- Gives confidence scores for color predictions

### 👁️ **Colorblind Simulation**
- Shows 6 types of colorblindness:
  - **Protanopia** (red-blind)
  - **Deuteranopia** (green-blind) 
  - **Tritanopia** (blue-blind)
  - **Protanomaly** (red-weak)
  - **Deuteranomaly** (green-weak)
  - **Tritanomaly** (blue-weak)

### 🎨 **Color Enhancement**
- **Daltonization**: Redistributes colors to help colorblind users see differences
- **K-means Analysis**: Groups similar colors together
- **Color Simplification**: Reduces complex images to main colors

### 📊 **Data Logging**
- Automatically saves all color captures to CSV files
- Tracks timestamps, RGB values, and AI predictions
- Exports session summaries in JSON format

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Check Your System
```bash
python main.py --system-info
```

### 3. Start the Application
```bash
python main.py --camera 0
```

## 🎮 How to Use

### Basic Controls
- **C** - Capture the current color and save it
- **N** - Switch between different types of colorblindness
- **Q** - Quit the application
- **P** - Pause/Resume the camera
- **S** - Toggle side-by-side view (normal vs colorblind)

### Advanced Features
- **K** - Toggle color analysis mode
- **D** - Toggle color enhancement for colorblind users
- **U** - Toggle unified AI processing pipeline
- **F** - Show/hide FPS counter
- **I** - Show/hide color information overlay

### Keyboard Shortcuts for Adjustments
- **1/2** - Decrease/Increase number of color groups
- **3/4** - Decrease/Increase enhancement strength

## 📁 Project Structure

```
ReColor-Backend/
├── main.py                    # Start here - main application
├── recolor_app.py            # Main application logic
├── camera_handler.py         # Camera and video processing
├── color_model.py            # AI color recognition
├── colorblind_detector.py    # Colorblind simulation
├── color_logger.py           # Data saving and logging
├── unified_color_pipeline.py # Advanced processing pipeline
├── utils.py                  # Helper functions
├── requirements.txt          # Required packages
├── logs/                     # Saved color data
└── models/                   # AI model files
```

## 🔧 System Requirements

- **Python**: 3.13.9 or newer
- **Camera**: Any USB webcam or built-in camera
- **CPU**: Multi-core processor (uses all available cores)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for models and logs

## 💡 Use Cases

### For Designers & Developers
- Test website/app accessibility for colorblind users
- Understand how your designs look to people with color vision differences
- Create more inclusive color schemes

### For Educators
- Demonstrate colorblindness to students
- Create awareness about color accessibility
- Generate data for research projects

### For Individuals
- Understand your own color vision
- Test if you might have color vision differences
- Learn about different types of colorblindness

## 🧪 Example Commands

```bash
# Basic usage with default camera
python main.py --camera 0

# Higher resolution and frame rate
python main.py --camera 0 --width 1280 --height 720 --fps 60

# Start with all features enabled
python main.py --camera 0 --enable-kmeans --enable-daltonization --enable-unified-pipeline

# Custom settings
python main.py --camera 0 --kmeans-clusters 8 --daltonization-strength 2.0
```

## 📊 Understanding the Output

### Color Information Display
- **RGB Values**: Red, Green, Blue numbers (0-255)
- **HEX Code**: Web-friendly color code (e.g., #FF0000)
- **Color Name**: AI prediction of the color name
- **Confidence**: How sure the AI is about the color (0-100%)

### Data Files
- **CSV Files**: Spreadsheet format with all captured colors
- **JSON Files**: Detailed session information for analysis
- **Log Files**: System performance and error information

## 🔬 Technical Details

### AI Model
- **Type**: Convolutional Neural Network (CNN)
- **Colors**: 9 classes (Red, Green, Blue, Yellow, Orange, Purple, Pink, Brown, Gray)
- **Training**: Synthetic dataset with various lighting conditions
- **Performance**: Real-time inference on CPU

### Colorblind Simulation
- **Method**: Scientifically accurate transformation matrices
- **Validation**: Based on peer-reviewed research
- **Accuracy**: Matches real colorblind user experience

### Performance
- **Frame Rate**: 30+ FPS on modern computers
- **Processing**: Multi-threaded CPU optimization
- **Memory**: Efficient algorithms for real-time performance

## 🤝 Contributing

This project is designed to help people understand color vision differences. Feel free to:
- Report bugs or issues
- Suggest new features
- Contribute improvements
- Share your use cases

## 📜 License

This project is for educational and accessibility purposes. Please use responsibly to help create a more inclusive world.

---

**Made with ❤️ to help people see the world differently**