# Firebase Setup Guide

Follow these steps to enable calibration storage and pipeline personalization.

## 1. Create a Firebase Project

1. Visit [https://console.firebase.google.com](https://console.firebase.google.com) and create a new project (or reuse an existing one).
2. Enable the following products in the project:
   - **Firestore** (in native mode).
   - **Cloud Storage** (for TensorFlow Lite models and artifacts).
   - **Authentication** (Email/Password or another provider; mobile app will link against it later).

## 2. Generate Service Account Credentials

1. In the Google Cloud Console, open *IAM & Admin → Service Accounts*.
2. Create a service account with the `Firebase Admin` role (or narrower custom roles that include Firestore/Storage access).
3. Generate a JSON key for the service account. Save the file locally, e.g. `C:\secrets\firebase-admin.json`.

## 3. Populate `.env`

Create `backend/.env` with the following entries:

```
FIREBASE_CREDENTIAL_PATH=C:\secrets\firebase-admin.json
FIREBASE_PROJECT_ID=<your-project-id>
TFLITE_MODEL_DIR=models
PIPELINE_CACHE_TTL=60
OFFLOAD_ENDPOINT=
OFFLOAD_TIMEOUT=5
```

> Use double backslashes (`C:\\secrets\\firebase-admin.json`) if you prefer. `OFFLOAD_ENDPOINT` is optional for now.

## 4. Install Dependencies

Run inside the repository (venv active):

```
python -m pip install -e backend
```

`python-dotenv` is bundled so the Flask app reads `backend/.env` automatically.

## 5. Verify Calibration Storage

1. Start the backend from `backend/`:
   ```
   python -m flask --app app.main run --host 0.0.0.0 --port 8000
   ```
2. POST calibration data:
   ```powershell
   Invoke-RestMethod -Method Post http://127.0.0.1:8000/calibration/ `
       -ContentType 'application/json' `
       -Body (@{user_id='demo-user'; responses=@{p1='incorrect'; p2='correct'; d1='skipped'}} | ConvertTo-Json)
   ```
3. Confirm a document appears under `visionProfiles/demo-user` in Firestore.

## 6. Confirm `/process/` Uses Stored Profiles

1. Upload (or drop) a TensorFlow Lite model under `backend/models/<deficiency>_v1.tflite` for the desired profile. If none exists, the pipeline will still run with daltonization-only corrections.
2. Send an image for processing:
   ```powershell
   $img = [Convert]::ToBase64String([IO.File]::ReadAllBytes('C:\path\to\image.png'))
   Invoke-RestMethod -Method Post http://127.0.0.1:8000/process/ `
       -ContentType 'application/json' `
       -Body (@{user_id='demo-user'; image_base64=$img} | ConvertTo-Json)
   ```
3. Observe that the response uses the stored deficiency/severity rather than the normal fallback.

## 7. Optional: Secure Access

- Restrict the service account key to development environments.
- Add authentication middleware (Firebase ID token verification) before exposing the API beyond local testing.
- Store the service account JSON in a secure location; avoid committing it to the repository.
