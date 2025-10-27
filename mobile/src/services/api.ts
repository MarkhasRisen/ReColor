import { Platform } from "react-native";

// Use your local machine's IP address so physical devices can connect
// Android emulator uses 10.0.2.2 to reach host machine's localhost
const API_BASE_URL = Platform.select({
  ios: "http://192.168.1.9:8000",
  android: "http://10.0.2.2:8000", // For emulator; use http://192.168.1.9:8000 for physical device
  default: "http://192.168.1.9:8000",
});

export type CalibrationPayload = {
  userId: string;
  responses: Record<string, "correct" | "incorrect" | "skipped">;
};

export async function submitCalibration(payload: CalibrationPayload) {
  const response = await fetch(`${API_BASE_URL}/calibration/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: payload.userId, responses: payload.responses }),
  });

  if (!response.ok) {
    throw new Error(`Calibration failed: ${response.status}`);
  }
  return response.json();
}

export async function submitImage(userId: string, imageBase64: string) {
  const response = await fetch(`${API_BASE_URL}/process/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, image_base64: imageBase64 }),
  });

  if (!response.ok) {
    throw new Error(`Processing failed: ${response.status}`);
  }
  return response.json();
}
