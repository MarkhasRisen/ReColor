# Calibration Data Model

## Firestore Layout

```
visionProfiles/{userId}
  profile:
    deficiency: "deutan"
    severity: 0.67
    confidence: 0.81
  metadata:
    calibratedAt: "2025-10-26T00:00:00Z"
    source: "ishihara"
    ishiharaVersion: "v1"
    platform: "mobile"
  history:
    - calibratedAt: "2025-09-12T15:04:51Z"
      deficiency: "deutan"
      severity: 0.52
      confidence: 0.74
```

`history` is optional and can be appended using batched writes or Cloud Functions.

## Ishihara Payload

```json
{
  "user_id": "abc123",
  "responses": {
    "p1": "incorrect",
    "p2": "correct",
    "d1": "skipped"
  }
}
```

## Derived Profile Fields

- **deficiency** – one of `protan`, `deutan`, `tritan`, or `normal`.
- **severity** – float in `[0, 1]`, computed as the normalized error ratio for the dominant deficiency class.
- **confidence** – float in `[0, 1]`, decreasing when responses are sparse or contradictory.

## Model Selection Strategy

1. Read user profile document.
2. If `profileVersion` is requested by the client, attempt to load `{deficiency}_{profileVersion}.tflite` from the model directory.
3. Fall back to latest known version for the deficiency type.
4. If no personalized model exists, use the generic daltonization-only pipeline with severity weighting.
