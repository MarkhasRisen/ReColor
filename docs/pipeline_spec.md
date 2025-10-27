# Adaptive Color Correction Pipeline Specification

## Objectives

- Reduce visual confusion for individuals with common color-vision deficiencies (protan, deutan, tritan) while preserving overall scene fidelity.
- Support both cloud-assisted and on-device inference paths with transparent fallback.
- Continuously refine personalization parameters using calibration history and in-app feedback loops.

## Pipeline Stages

1. **Acquisition & Preprocessing**
   - Accept RGB frames or still images.
   - Optionally downsample or convert to linear color space for consistent processing.
   - Normalize pixel data and reshape batches for clustering and CNN execution.

2. **Adaptive K-Means Branch (On-Device Path)**
   - Cluster pixels per frame to limit the number of color corrections.
   - Use user profile weights to bias centroid initialization toward hues prone to confusion.
   - Cache centroids for successive frames when motion is limited to avoid recomputation.

3. **CNN Feature Branch (On-Device or Offloaded)**
   - Default to executing the TensorFlow Lite CNN locally for real-time residual correction.
   - Invoke a resource monitor to detect thermal, memory, or latency thresholds; offload inference to the backend when limits are crossed, reusing the same data contract.
   - Normalize and cache CNN outputs so they can be merged deterministically with the clustered path.
   - Resource monitors can be latency-budget windows or device telemetry adapters surfaced to the pipeline via `PipelineConfig.resource_monitor`.

4. **Merge Junction & Daltonization Core**
   - Blend the K-Means reconstruction and CNN feature maps using a weighted merge before daltonization (P4).
   - Compute confusion lines and neutral axes from the user profile (severity coefficients derived from Ishihara calibration).
   - Apply shift vectors to the merged frame, remapping ambiguous colors to more distinguishable hues while respecting luminance constraints.
   - Blend corrected output with the original frame to prevent oversaturation or structural distortions.

5. **Post-Processing & Output**
   - Reconstruct the full-resolution image/video frame.
   - Apply optional tone-mapping or sharpening tailored to the display context.
   - Stream corrected frames to the client or return still image blobs.

## Personalization Inputs

- **Ishihara Calibration Results**: Plate-level responses converted to severity indices per cone type.
- **User Feedback**: Explicit user ratings or toggled comparisons logged via Firebase.
- **Usage Context**: Device GPU/CPU capability, network availability, and lighting conditions (if provided).

## Deployment Targets

- **Flask API**: Hosts the pipeline for server-side correction, integrates Firebase Admin SDK for authentication and profile retrieval.
- **React Native Client**: Handles capture, displays before/after views, and optionally runs TFLite inference locally via on-device delegates.
- **TensorFlow Lite Models**: Stored in Firebase Storage with metadata maintained in Firestore for dynamic delivery.

## Metrics & Evaluation

- Perceptual difference scores (CIEDE2000) between original and corrected outputs under simulated deficiency.
- User satisfaction metrics collected after calibration and usage sessions.
- Latency benchmarks for both cloud and device execution paths.

## Open Questions

- What baseline dataset will be used to train the CNN (e.g., color-annotated natural scenes, synthetic augmentations)?
- How frequently should profiles be re-calibrated or re-fitted using new Ishihara results?
- Do we need to support other color-blindness tests (e.g., Farnsworth D-15) for enhanced severity estimation?
