# Training Toolkit

This folder hosts tooling for preparing datasets, fitting the adaptive CNN, and exporting TensorFlow Lite models.

## Components

- `datasets/` – Raw and preprocessed image collections paired with simulated deficiency labels.
- `notebooks/` – Research and prototyping notebooks for loss tuning and augmentation design.
- `scripts/` – Automated pipelines for centroid caching, model training, evaluation, and TFLite conversion.

## Workflow Outline

1. **Dataset Assembly**
   - Download or generate color-critical imagery.
   - Apply color-vision deficiency simulators to create paired ground truth targets.
   - Store metadata describing scene context and deficiency parameters.

2. **Centroid Precomputation**
   - Run adaptive K-Means over the corpus to precompute centroid priors per deficiency type.
   - Persist centroids in `artifacts/centroids/{profile}.npy` for reuse in inference.

3. **CNN Training**
   - Train the adaptive color transform network using TensorFlow.
   - Optimize for perceptual difference metrics while constraining luminance drift.
   - Track experiments with the tool of your choice (Weights & Biases, MLflow, etc.).

4. **TFLite Export**
   - Convert the frozen model to TensorFlow Lite with post-training quantization.
   - Attach profile metadata in the TFLite metadata buffer.
   - Publish the model to Firebase Storage and update Firestore with the new version.

5. **Regression Testing**
   - Run `pytest` suites from `backend/tests/` to ensure compatibility.
   - Execute notebook-based visual comparisons to validate perceptual gains.
