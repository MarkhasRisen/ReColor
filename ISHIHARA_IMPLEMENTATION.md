# Ishihara Color Vision Test - Implementation Guide

## Overview

The Ishihara module implements a clinical-grade color vision screening test using DaltonLens methodology (BSD-2-Clause license). It supports both quick (14 plates) and comprehensive (38 plates) testing modes with standard clinical scoring thresholds.

## Features

### ✅ Two Testing Modes

1. **Quick Test (14 plates)**
   - Duration: 2-3 minutes
   - Standard screening
   - High sensitivity for red-green CVD
   - Suitable for routine screening

2. **Comprehensive Test (38 plates)**
   - Duration: 5-7 minutes  
   - Complete diagnostic assessment
   - Includes tracing plates
   - Clinical-grade diagnosis with protan/deutan classification

### ✅ Clinical Standards

- **Normal threshold**: ≥86% correct (≥12/14 or ≥33/38)
- **Mild CVD**: 57-86% correct
- **Moderate CVD**: 29-57% correct  
- **Strong CVD**: <29% correct
- **Control plate validation**: Must pass for valid test

### ✅ Plate Types

1. **Control plates** - Everyone should see (plates 1-2)
2. **Transformation plates** - Different answers for normal vs CVD
3. **Vanishing plates** - Only visible to normal vision
4. **Hidden digit plates** - Only visible to CVD
5. **Classification plates** - Distinguish protan from deutan
6. **Tracing plates** - Follow colored paths (comprehensive only)

### ✅ CVD Types Detected

- **Protan** (Protanomaly/Protanopia) - Red deficiency (1% males)
- **Deutan** (Deuteranomaly/Deuteranopia) - Green deficiency (5% males)
- **Normal** - No color vision deficiency

## API Endpoints

### 1. GET /ishihara/plates

Get list of test plates for a specific mode.

**Query Parameters:**
- `mode`: `"quick"` (default) or `"comprehensive"`

**Response:**
```json
{
  "mode": "quick",
  "total_plates": 14,
  "plates": [
    {
      "plate_number": 1,
      "image_url": "/static/ishihara/plate_01.png",
      "is_control": true
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/ishihara/plates?mode=quick
```

### 2. POST /ishihara/evaluate

Evaluate test responses and provide diagnosis.

**Request Body:**
```json
{
  "user_id": "optional_user_id",
  "mode": "quick",
  "responses": {
    "1": "12",
    "3": "6",
    "4": "29",
    "5": "57"
  },
  "save_profile": true
}
```

**Response:**
```json
{
  "result": {
    "total_plates": 14,
    "correct_normal": 12,
    "correct_protan": 2,
    "correct_deutan": 1,
    "incorrect": 0,
    "control_failed": 0,
    "classification_score": {
      "protan": 2,
      "deutan": 0
    }
  },
  "diagnosis": {
    "cvd_type": "protan",
    "severity": 0.6,
    "confidence": 0.85,
    "interpretation": "Moderate Protanomaly/Protanopia (red deficiency) detected (confidence: 85%). Clinical confirmation recommended."
  },
  "profile_saved": true
}
```

**Example:**
```python
import requests

response = requests.post('http://localhost:8000/ishihara/evaluate', json={
    "user_id": "user123",
    "mode": "quick",
    "responses": {
        "1": "12", "3": "6", "4": "29", "5": "57",
        "6": "5", "7": "3", "8": "15", "9": "74",
        "10": "2", "11": "6", "16": "",
        "18": "26", "19": "42", "20": "35"
    },
    "save_profile": true
})

result = response.json()
print(f"Diagnosis: {result['diagnosis']['cvd_type']}")
print(f"Severity: {result['diagnosis']['severity']}")
print(f"Interpretation: {result['diagnosis']['interpretation']}")
```

### 3. GET /ishihara/info

Get information about the Ishihara test.

**Response:**
```json
{
  "name": "Ishihara Color Vision Test",
  "version": "DaltonLens compatible implementation",
  "license": "BSD-2-Clause",
  "modes": {
    "quick": {
      "plates": 14,
      "duration": "2-3 minutes",
      "description": "Standard screening test...",
      "accuracy": "High sensitivity for detecting red-green CVD"
    },
    "comprehensive": {
      "plates": 38,
      "duration": "5-7 minutes",
      "description": "Complete test...",
      "accuracy": "Clinical-grade diagnosis"
    }
  },
  "instructions": [...],
  "clinical_standards": {...},
  "cvd_types": {...}
}
```

## Usage Examples

### Python Client

```python
from app.ishihara.test import IshiharaTest, CVDType

# Initialize test (quick or comprehensive)
test = IshiharaTest(use_comprehensive=False)  # Quick test

# Collect responses (from user or simulated)
responses = {
    1: "12",   # Plate 1 answer
    3: "6",    # Plate 3 answer
    4: "29",   # etc...
    # ... all plate responses
}

# Evaluate test
result = test.evaluate_test(responses)

# Check results
print(f"CVD Type: {result.cvd_type.value}")
print(f"Severity: {result.severity}")  # 0.0-1.0
print(f"Confidence: {result.confidence}")  # 0.0-1.0
print(f"Interpretation: {result.interpretation}")

# Access detailed scores
print(f"Correct (Normal): {result.correct_normal}/{result.total_plates}")
print(f"Protan Score: {result.classification_score['protan']}")
print(f"Deutan Score: {result.classification_score['deutan']}")
```

### REST API Client

```bash
# Get plates list
curl http://localhost:8000/ishihara/plates?mode=quick

# Submit test results
curl -X POST http://localhost:8000/ishihara/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "mode": "quick",
    "responses": {
      "1": "12", "3": "6", "4": "29", "5": "57",
      "6": "5", "7": "3", "8": "15", "9": "74",
      "10": "2", "11": "6", "16": "",
      "18": "26", "19": "42", "20": "35"
    },
    "save_profile": true
  }'

# Get test information
curl http://localhost:8000/ishihara/info
```

## Test Administration Guidelines

### Proper Testing Conditions

1. **Lighting**: Natural daylight or daylight-equivalent LED (D65 standard)
2. **Distance**: Hold plates at comfortable reading distance (~75cm)
3. **Time**: 3-5 seconds maximum per plate
4. **Viewing angle**: Plates should be perpendicular to line of sight
5. **Corrective lenses**: Keep on if normally worn

### Response Guidelines

- State what you see immediately
- Don't guess or spend too long analyzing
- If nothing is visible, respond with empty string or "nothing"
- For tracing plates, indicate if you can follow the path

### Interpretation

#### Control Plates (1-2)
- Must be answered correctly
- If failed: retest under proper lighting

#### Transformation Plates
- Normal: Sees specific number
- CVD: Sees different number or nothing

#### Hidden Digit Plates
- Normal: Sees nothing or very faint
- CVD: Clearly sees hidden number

#### Classification Plates (18-21)
- Protan: Sees specific digit pattern
- Deutan: Sees different digit pattern
- Used to distinguish red vs green deficiency

## Scoring Algorithm

### Normal Vision Detection
```
If correct_normal >= 86% of total_plates:
    Diagnosis: NORMAL
    Severity: 0.0
    Confidence: 0.95
```

### CVD Classification
```
If correct_normal < 86%:
    # Determine type from classification plates
    If classification_protan > classification_deutan:
        Type: PROTAN
    Else:
        Type: DEUTAN
    
    # Calculate severity
    If correct_normal > 57%:
        Severity: 0.3 (Mild)
    Elif correct_normal > 29%:
        Severity: 0.6 (Moderate)
    Else:
        Severity: 0.9 (Strong)
    
    # Confidence from classification consistency
    Confidence: 0.6 to 0.95 based on classification_ratio
```

## Testing Examples

### Test 1: Normal Vision
```python
responses = {
    1: "12", 3: "6", 4: "29", 5: "57", 6: "5",
    7: "3", 8: "15", 9: "74", 10: "2", 11: "6",
    16: "", 18: "26", 19: "42", 20: "35"
}
# Expected: CVDType.NORMAL, severity=0.0
```

### Test 2: Protanopia
```python
responses = {
    1: "12", 3: "5", 4: "70", 5: "35", 6: "2",
    7: "5", 8: "17", 9: "21", 10: "", 11: "",
    16: "45", 18: "6", 19: "2", 20: "5"
}
# Expected: CVDType.PROTAN, severity=0.9
```

### Test 3: Deuteranopia
```python
responses = {
    1: "12", 3: "5", 4: "70", 5: "35", 6: "2",
    7: "5", 8: "17", 9: "21", 10: "", 11: "",
    16: "45", 18: "2", 19: "4", 20: "3"
}
# Expected: CVDType.DEUTAN, severity=0.9
```

## Integration with ReColor System

### Calibration Flow
```python
from app.ishihara.test import IshiharaTest
from app.services import firebase
from app.pipeline.profile import VisionProfile

# 1. Administer test
test = IshiharaTest(use_comprehensive=False)
result = test.evaluate_test(user_responses)

# 2. Save profile if valid
if result.control_failed == 0:
    profile = VisionProfile(
        deficiency=result.cvd_type.value,
        severity=result.severity,
        confidence=result.confidence
    )
    firebase.save_profile(user_id, profile, metadata)

# 3. Use for color correction
# The saved profile automatically applies to /process endpoint
```

### Evaluation Metrics

Use with the evaluation module:

```python
from app.evaluation.metrics import IshiharaEvaluator

# Clinical validation data
clinical_diagnoses = [...]  # From ophthalmologist
system_predictions = [...]  # From Ishihara test

# Compute metrics
evaluator = IshiharaEvaluator()
metrics = evaluator.compute_metrics(system_predictions, clinical_diagnoses)

print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"Sensitivity: {metrics.sensitivity:.2%}")
print(f"Specificity: {metrics.specificity:.2%}")
print(f"Cohen's Kappa: {metrics.cohens_kappa:.3f}")
```

## Files Structure

```
backend/app/ishihara/
├── __init__.py          # Package exports
└── test.py              # Core implementation
    ├── IshiharaPlate    # Plate data structure
    ├── IshiharaTest     # Test administration
    ├── TestResult       # Result data structure
    ├── CVDType          # Enum for CVD types
    ├── PlateType        # Enum for plate types
    ├── COMPREHENSIVE_PLATES  # 38-plate configuration
    └── QUICK_TEST_PLATES     # 14-plate configuration

backend/app/routes/
└── ishihara.py          # REST API endpoints

static/ishihara/
└── plate_XX.png         # Plate images (01-38)

test_ishihara.py         # Comprehensive test suite
```

## License & Attribution

- **Implementation**: BSD-2-Clause (compatible with DaltonLens)
- **Methodology**: Based on standard Ishihara Color Vision Test
- **Clinical Standards**: Follows established optometric guidelines

**Note**: This is a screening tool. Clinical diagnosis should always be confirmed by a qualified optometrist or ophthalmologist.

## References

- Ishihara, S. (1917). Tests for Color-Blindness
- Birch, J. (2001). Diagnosis of Defective Colour Vision (2nd ed.)
- DaltonLens: https://daltonlens.org/ (BSD-2-Clause)
- Clinical validation standards from American Optometric Association

## Testing

Run the comprehensive test suite:

```bash
python test_ishihara.py
```

**Expected output:**
- ✅ Normal vision detection
- ✅ Protanopia classification
- ✅ Deuteranopia classification  
- ✅ Severity grading
- ✅ Control plate validation
- ✅ Comprehensive mode (38 plates)
