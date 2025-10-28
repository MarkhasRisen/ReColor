import { Platform } from "react-native";

// API Configuration
// Development: Use local backend server
// Production: Use Heroku deployment (update after deployment)
const DEV_API_URL = Platform.select({
  ios: "http://192.168.1.9:8000",
  android: "http://10.0.2.2:8000", // For emulator; use http://192.168.1.9:8000 for physical device
  default: "http://192.168.1.9:8000",
});

// TODO: Update this after deploying backend to Heroku
const PROD_API_URL = "https://recolor-api.herokuapp.com";

// Set to false for production
const USE_DEV_SERVER = true;

const API_BASE_URL = USE_DEV_SERVER ? DEV_API_URL : PROD_API_URL;

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
  try {
    const response = await fetch(`${API_BASE_URL}/api/ishihara/plates?mode=${mode}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to fetch plates (${response.status}): ${errorText}`);
    }
    
    return await response.json();
  } catch (error: any) {
    console.error('getIshiharaPlates error:', error);
    throw new Error(error.message || 'Network error - could not connect to server');
  }
}

// Evaluate Ishihara test responses
export async function evaluateIshiharaTest(
  userId: string,
  responses: IshiharaResponse,
  mode: "quick" | "comprehensive" = "quick",
  saveProfile: boolean = true
) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/ishihara/evaluate`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify({
        user_id: userId,
        mode,
        responses,
        save_profile: saveProfile,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Evaluation failed (${response.status}): ${errorText}`);
    }
    
    return await response.json();
  } catch (error: any) {
    console.error('evaluateIshiharaTest error:', error);
    throw new Error(error.message || 'Network error - could not evaluate test');
  }
}

// Legacy calibration endpoint (deprecated - use Ishihara instead)
export async function submitCalibration(payload: CalibrationPayload) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/calibration/`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify({ 
        user_id: payload.userId, 
        responses: payload.responses 
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Calibration failed (${response.status}): ${errorText}`);
    }
    
    return await response.json();
  } catch (error: any) {
    console.error('submitCalibration error:', error);
    throw new Error(error.message || 'Network error - could not submit calibration');
  }
}

export async function submitImage(userId: string, imageBase64: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/process/`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify({ 
        user_id: userId, 
        image_base64: imageBase64 
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Processing failed (${response.status}): ${errorText}`);
    }
    
    return await response.json();
  } catch (error: any) {
    console.error('submitImage error:', error);
    throw new Error(error.message || 'Network error - could not process image');
  }
}
