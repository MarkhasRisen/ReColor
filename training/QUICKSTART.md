# Quick Start: Training Pipeline

## Run Training Pipeline

Generate dataset and compute centroids in one command:

```powershell
cd training
pip install -r requirements.txt
python run_training_pipeline.py
```

This will:
1. Generate 8000 synthetic color images → `datasets/color_varied/`
2. Train K-Means in LAB space for each deficiency type
3. Save centroids → `models/centroids/`

**Time:** ~5-10 minutes depending on hardware

## Copy Centroids to Backend

After training completes:

```powershell
# From training directory
New-Item -ItemType Directory -Path "..\backend\models\centroids" -Force
Copy-Item -Path "models\centroids\*" -Destination "..\backend\models\centroids\" -Force
```

Or manually copy files from:
- **Source:** `training/models/centroids/`
- **Destination:** `backend/models/centroids/`

## Verify Integration

Test the backend with precomputed centroids:

```powershell
cd ..
.\.venv\Scripts\python.exe test_lab_integration.py
```

Expected output:
```
✅ LAB integration test passed!
```

## Start Backend Server

```powershell
cd backend
..\. venv\Scripts\python.exe -m flask --app app.main run --host 0.0.0.0 --port 8000
```

## Test Image Processing

```powershell
# Test with calibration + image processing
$calibration = @{
    user_id = "test-user"
    responses = @{
        p1 = "incorrect"
        p2 = "incorrect"
        d1 = "correct"
    }
} | ConvertTo-Json

Invoke-RestMethod -Method Post http://127.0.0.1:8000/calibration/ `
    -ContentType 'application/json' `
    -Body $calibration

# Process an image (base64 encoded)
$image = [Convert]::ToBase64String([IO.File]::ReadAllBytes("test_image.png"))
$process = @{
    user_id = "test-user"
    image_base64 = $image
} | ConvertTo-Json

Invoke-RestMethod -Method Post http://127.0.0.1:8000/process/ `
    -ContentType 'application/json' `
    -Body $process
```

## Generated Files

After running the pipeline:

```
training/
├── datasets/
│   └── color_varied/
│       ├── red/           # 1000 images
│       ├── orange/        # 1000 images
│       ├── yellow/        # 1000 images
│       ├── green/         # 1000 images
│       ├── cyan/          # 1000 images
│       ├── blue/          # 1000 images
│       ├── indigo/        # 1000 images
│       └── violet/        # 1000 images
└── models/
    └── centroids/
        ├── protan_centroids_lab.npy       # LAB centroids
        ├── protan_centroids_rgb.npy       # RGB conversion
        ├── protan_centroids.json          # Human-readable
        ├── protan_kmeans_model.pkl        # Scikit-learn model
        ├── deutan_centroids_lab.npy
        ├── deutan_centroids_rgb.npy
        ├── deutan_centroids.json
        ├── deutan_kmeans_model.pkl
        ├── tritan_centroids_lab.npy
        ├── tritan_centroids_rgb.npy
        ├── tritan_centroids.json
        ├── tritan_kmeans_model.pkl
        ├── normal_centroids_lab.npy
        ├── normal_centroids_rgb.npy
        ├── normal_centroids.json
        └── normal_kmeans_model.pkl
```

## Individual Scripts

### Generate Dataset Only

```powershell
python scripts/generate_dataset.py
```

Customize in script:
- `SAMPLES_PER_FAMILY = 1000` - Images per color
- `IMG_SIZE = (64, 64)` - Image size
- `HUE_JITTER = 6.0` - Color variation

### Compute Centroids Only

```powershell
python scripts/compute_centroids.py
```

Requires existing dataset in `datasets/color_varied/`

## Troubleshooting

### Missing tqdm
```powershell
pip install tqdm
```

### Out of memory
Reduce `SAMPLES_PER_FAMILY` in `scripts/generate_dataset.py`

### Dataset not found
Run `generate_dataset.py` first

### Import errors
```powershell
pip install -r requirements.txt
```
