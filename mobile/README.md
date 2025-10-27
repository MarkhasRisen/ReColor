# React Native Client

This package provides the mobile front end for the adaptive daltonization platform.

## Key Responsibilities

- Capture live camera frames and still images for correction.
- Run Ishihara-based calibration flows and submit results to the Flask backend.
- Execute TensorFlow Lite models on-device when available, falling back to the backend API when latency budgets permit.
- Sync user profiles, calibration history, and feedback via Firebase Auth and Firestore.

## Folder Structure

- `src/components/` – Reusable UI components (calibration controls, preview overlays).
- `src/screens/` – Top-level navigation screens (Calibration, LivePreview, Settings).
- `src/services/` – API clients for Flask endpoints, Firebase integration, and TensorFlow Lite inference wrappers.

## Getting Started

1. Initialize the project with `npx react-native init DaltonizationMobile --template react-native-template-typescript`.
2. Copy the source folders in this directory into the generated project.
3. Configure Firebase via the React Native Firebase SDK or Expo Firebase wrappers.
4. Install the TensorFlow Lite React Native package that matches your architecture (e.g., `npm install @tensorflow/tfjs-react-native`).
