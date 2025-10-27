# API Reference

## POST /calibration

Submit Ishihara calibration responses and receive an updated vision profile.

### Request Body

```json
{
  "user_id": "abc123",
  "responses": {
    "p1": "incorrect",
    "p2": "correct",
    "t1": "skipped"
  }
}
```

### Response Body

```json
{
  "deficiency": "deutan",
  "severity": 0.62,
  "confidence": 0.85
}
```

## POST /process

Process an RGB image using the adaptive color correction pipeline.

### Request Body

```json
{
  "user_id": "abc123",
  "image_base64": "...",
  "profile_version": "v1"
}
```

### Response Body

```json
{
  "content_type": "image/png",
  "data": "...base64 encoded PNG..."
}
```

### Notes

- When running on-device, provide the encoded frame captured from the React Native camera module.
- The backend attempts to load a per-user TensorFlow Lite model from `TFLITE_MODEL_DIR`; if unavailable, the daltonization-only path is returned.
- The service responds with a PNG to simplify client rendering via `Image` components on React Native.
