# Adaptive Daltonization Pipeline

This repository contains an end-to-end adaptive color correction system tailored for color-vision deficiencies. The platform combines perceptual modeling, clustering, and neural color transforms to deliver real-time corrections across web/mobile clients.

## Subsystems

- **backend/** – Flask API serving TensorFlow Lite models, Daltonization utilities, and calibration workflows.
- **training/** – Notebooks and scripts for dataset preparation, K-Means centroid caching, daltonization calibration, and CNN export to TFLite.
- **mobile/** – React Native client for real-time capture, on-device inference, and Firebase-backed profile synchronization.
- **docs/** – Architecture notes, research references, and design records.

## High-Level Flow

1. Users complete an Ishihara-based calibration module through the mobile client.
2. Calibration responses are posted to the backend, generating individualized deficiency profiles stored in Firebase.
3. The backend selects or composes a TensorFlow Lite color transform pipeline that applies:
   - Pixel grouping via K-Means clustering to limit per-frame compute cost.
   - Daltonization adjustments along confusion lines derived from the user profile.
   - CNN-driven adaptive corrections to restore perceptual contrast while preserving luminance cues.
4. The corrected imagery is delivered back to the client or executed locally when on-device models are available.

## Next Steps

1. Flesh out pipeline modules (clustering, daltonization, CNN inference) under `backend/app/pipeline/`.
2. Implement calibration and processing routes in the Flask API.
3. Set up Firebase integration for profile storage and secure auth (see `docs/firebase_setup.md`).
4. Configure optional offload inference by setting `OFFLOAD_ENDPOINT` and `OFFLOAD_TIMEOUT` if remote CNN execution is available.
5. Prepare datasets and training pipelines for CNN export under `training/`.
6. Scaffold the React Native client with live preview, calibration workflow, and Firebase Auth/Firestore integration.
