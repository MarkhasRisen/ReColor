# Training Pipeline for Adaptive Color Correction

This directory contains scripts and resources for training the adaptive color correction system, including:
- **Dataset generation** for K-Means clustering
- **Centroid precomputation** for profile-specific initialization  
- **CNN training** (coming soon) for adaptive feature extraction

## 📁 Directory Structure

```
training/
├── scripts/
│   ├── generate_dataset.py      # Create synthetic color datasets
│   ├── compute_centroids.py     # Precompute K-Means centroids
│   └── train_cnn.py              # Train TFLite CNN models (TODO)
├── notebooks/
│   ├── kmeans.ipynb             # Original K-Means experiments
│   ├── main.ipynb               # Integration examples
│   └── Dataset.ipynb            # Dataset generation experiments
├── datasets/
│   └── color_varied/            # Generated training data
│       ├── red/
│       ├── orange/
│       ├── yellow/
│       ├── green/
│       ├── cyan/
│       ├── blue/
│       ├── indigo/
│       └── violet/
├── models/
│   └── centroids/               # Precomputed K-Means centroids
│       ├── protan_centroids_lab.npy
│       ├── deutan_centroids_lab.npy
│       ├── tritan_centroids_lab.npy
│       └── normal_centroids_lab.npy
└── requirements.txt             # Training dependencies

```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Create virtual environment (if not already done)
cd training
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Generate Dataset

Creates synthetic color images with controlled variations for K-Means training:

```powershell
python scripts/generate_dataset.py
```

**Output:** `datasets/color_varied/` with 8000 images (1000 per color family)

**Parameters (edit in script):**
- `SAMPLES_PER_FAMILY = 1000` - Images per color
- `IMG_SIZE = (64, 64)` - Image dimensions
- `HUE_JITTER = 6.0` - Hue variation in degrees

### 3. Compute Centroids

Trains K-Means on the dataset in LAB color space and saves centroids for each deficiency type:

```powershell
python scripts/compute_centroids.py
```

**Output:** `models/centroids/` with:
- `{deficiency}_centroids_lab.npy` - NumPy LAB centroids
- `{deficiency}_centroids_rgb.npy` - RGB conversion for visualization
- `{deficiency}_centroids.json` - Human-readable JSON
- `{deficiency}_kmeans_model.pkl` - Scikit-learn model

**Deficiency Types:**
- `protan` - Red-green (red weak) - Priority: red, orange, green
- `deutan` - Green-red (green weak) - Priority: green, red, yellow
- `tritan` - Blue-yellow - Priority: blue, yellow, cyan
- `normal` - Full spectrum

## 🧪 How It Works

### LAB Color Space Clustering

The original ReColor-CJ implementation used LAB color space for K-Means clustering, which we've integrated:

**Why LAB?**
- **Perceptually uniform**: Equal distances = equal perceived color differences
- **Better discrimination**: Separates luminance (L) from chromaticity (a, b)
- **Color blindness friendly**: Confusion lines align better with a-b axes

**Pipeline Flow:**
1. **RGB → LAB**: Convert normalized RGB [0,1] to LAB
2. **K-Means clustering**: Group pixels by perceptual similarity
3. **LAB → RGB**: Convert centroids back for display

### Profile-Specific Initialization

Instead of random initialization, we bias K-Means with precomputed centroids:

```python
# Load precomputed centroids for protan deficiency
centroids_lab = load_centroids("protan")

# Initialize K-Means with bias
segmenter.fit_predict(pixels, centroid_bias=centroids_lab)
```

**Benefits:**
- ⚡ **Faster convergence**: Fewer iterations needed
- 🎯 **Better accuracy**: Focuses on confusion-prone colors
- 🔄 **Consistent results**: Reproducible across frames

## 📊 Dataset Statistics

### Color Families

| Family  | Hue (°) | Samples | Use Case                    |
|---------|---------|---------|------------------------------|
| Red     | 0       | 1000    | Protan/Deutan confusion     |
| Orange  | 30      | 1000    | Red-green transitions       |
| Yellow  | 60      | 1000    | Deutan/Tritan confusion     |
| Green   | 120     | 1000    | Protan/Deutan confusion     |
| Cyan    | 180     | 1000    | Tritan confusion            |
| Blue    | 240     | 1000    | Tritan confusion            |
| Indigo  | 275     | 1000    | Blue-violet transitions     |
| Violet  | 300     | 1000    | Tritan edge cases           |

### Generation Techniques

Each image uses one of four generators for texture variation:

1. **Gradient**: Linear hue/saturation transitions
2. **Stripes**: Vertical bands with color shifts
3. **Blobs**: Soft elliptical regions with Gaussian blur
4. **Texture**: Layered noise with color tinting

## 🔬 Integration with Backend

The backend automatically loads centroids when processing images:

```python
from backend.app.utils.centroid_loader import get_centroid_bias_for_profile

# Get bias for moderate protan deficiency (severity=0.6)
bias = get_centroid_bias_for_profile("protan", severity=0.6)
# Returns: np.ndarray (9, 3) in LAB space

# For low severity (< 0.3), returns None (random init)
bias = get_centroid_bias_for_profile("protan", severity=0.2)
# Returns: None
```

## 🎯 Next Steps

### TODO: CNN Training

Train TensorFlow Lite models for adaptive feature extraction:

```powershell
python scripts/train_cnn.py --deficiency protan --epochs 50
```

**Architecture** (planned):
- Encoder-decoder with skip connections
- Perceptual loss (CIEDE2000 in LAB space)
- Quantization-aware training for TFLite
- Profile-specific models per deficiency type

### TODO: Centroid Caching in Firebase

Upload precomputed centroids to Firebase Storage for cross-device synchronization:

```python
# In backend/app/services/firebase.py
def upload_centroids(deficiency: str, centroids: np.ndarray):
    bucket = storage.bucket()
    blob = bucket.blob(f"centroids/{deficiency}_lab.npy")
    blob.upload_from_string(centroids.tobytes())
```

## 📚 References

- **Original Implementation**: ReColor-CJ (attached notebooks)
- **LAB Color Space**: CIE 1976 (L\*a\*b\*) perceptually uniform
- **K-Means**: sklearn.cluster.KMeans with custom initialization
- **Daltonization**: Brettel et al. (1997) confusion line model

## 🐛 Troubleshooting

### "Dataset not found" error
```
Run generate_dataset.py first to create training data
```

### Slow centroid computation
```
Reduce SAMPLES_PER_FAMILY in generate_dataset.py (e.g., 500 instead of 1000)
```

### Out of memory during K-Means
```
Use smaller image sizes or batch processing in compute_centroids.py
```
