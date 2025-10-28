import { Platform } from "react-native";

// Use your local machine's IP address so physical devices can connect
// Android emulator uses 10.0.2.2 to reach host machine's localhost
// TODO: Update to production URL after backend deployment
const API_BASE_URL = Platform.select({
  ios: "http://192.168.1.9:8000",
  android: "http://10.0.2.2:8000", // For emulator; use http://192.168.1.9:8000 for physical device
  default: "http://192.168.1.9:8000",
});

// For production deployment:
// const API_BASE_URL = "https://recolor-api.herokuapp.com";

export type CalibrationPayload = {
  userId: string;
  responses: Record<string, "correct" | "incorrect" | "skipped">;
};

export type IshiharaPlate = {
  plate_number: number;
  image_url: string;
  is_control: boolean;
};

export type IshiharaResponse = {
  [plateNumber: number]: string;
};

export type IshiharaDiagnosis = {
  cvd_type: "normal" | "protan" | "deutan" | "tritan" | "total";
  severity: number;
  confidence: number;
  interpretation: string;
};

// Get Ishihara test plates
export async function getIshiharaPlates(mode: "quick" | "comprehensive" = "quick") {
  const response = await fetch(`${API_BASE_URL}/ishihara/plates?mode=${mode}`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch Ishihara plates: ${response.status}`);
  }
  return response.json();
}

// Evaluate Ishihara test responses
export async function evaluateIshiharaTest(
  userId: string,
  responses: IshiharaResponse,
  mode: "quick" | "comprehensive" = "quick",
  saveProfile: boolean = true
) {
  const response = await fetch(`${API_BASE_URL}/ishihara/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      mode,
      responses,
      save_profile: saveProfile,
    }),
  });

  if (!response.ok) {
    throw new Error(`Ishihara evaluation failed: ${response.status}`);
  }
  return response.json();
}

// Legacy calibration endpoint (deprecated - use Ishihara instead)
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
