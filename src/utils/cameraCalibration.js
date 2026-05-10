/**
 * cameraCalibration.js
 *
 * Cross-device color consistency for the three cameras (Color Identifier,
 * CVD Simulation, Camera Enhancement).
 *
 * Two layers:
 *   1. Manual calibration (preferred when available) — user shows the camera
 *      a white reference once via Settings → Camera Calibration. We compute
 *      per-channel scalars (rScale, gScale, bScale) that make the captured
 *      "white" actually neutral. Stored in Firestore and AsyncStorage.
 *
 *   2. Auto white-balance fallback (when no manual calibration exists) —
 *      gray-world reflectance assumption (Buchsbaum 1980).
 */

import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  deleteDoc,
  doc,
  getDoc,
  serverTimestamp,
  setDoc,
} from "firebase/firestore";
import { auth, db } from "../../firebaseConfig";

const STORAGE_KEY = "@recolor_camera_calibration";

// Sane bounds on the per-channel scalars
const SCALE_MIN = 0.5;
const SCALE_MAX = 2.0;

const IDENTITY = { rScale: 1, gScale: 1, bScale: 1 };

// ─────────────────────────────────────────────────────────────
// Storage
// ─────────────────────────────────────────────────────────────

/**
 * @returns {Promise<{rScale, gScale, bScale, calibratedAt} | null>}
 */
export async function loadCalibration() {
  try {
    const user = auth.currentUser;
    if (user) {
      const docRef = doc(db, "users", user.uid, "settings", "calibration");
      const snap = await getDoc(docRef);
      if (snap.exists()) {
        const data = snap.data();
        if (typeof data.rScale === "number") {
          // Cache locally for offline/fast access
          await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(data));
          return data;
        }
      }
    }
    // Fallback to local storage
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (typeof parsed.rScale !== "number") return null;
    return parsed;
  } catch {
    return null;
  }
}

export async function saveCalibration(rScale, gScale, bScale) {
  const payload = {
    rScale: clampScale(rScale),
    gScale: clampScale(gScale),
    bScale: clampScale(bScale),
  };

  try {
    const user = auth.currentUser;
    if (user) {
      const docRef = doc(db, "users", user.uid, "settings", "calibration");
      await setDoc(
        docRef,
        {
          ...payload,
          calibratedAt: serverTimestamp(),
        },
        { merge: true },
      );
    }
    // Save locally as well
    await AsyncStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        ...payload,
        calibratedAt: new Date().toISOString(),
      }),
    );
  } catch (e) {
    console.warn("Failed to save calibration to cloud", e);
  }
  return payload;
}

export async function clearCalibration() {
  try {
    const user = auth.currentUser;
    if (user) {
      const docRef = doc(db, "users", user.uid, "settings", "calibration");
      await deleteDoc(docRef);
    }
    await AsyncStorage.removeItem(STORAGE_KEY);
  } catch (e) {
    console.warn("Failed to clear calibration", e);
  }
}

function clampScale(s) {
  if (!Number.isFinite(s) || s <= 0) return 1;
  return Math.max(SCALE_MIN, Math.min(SCALE_MAX, s));
}

// ─────────────────────────────────────────────────────────────
// Manual calibration: derive scalars from a white reference patch
// ─────────────────────────────────────────────────────────────

export function deriveScalarsFromWhitePatch(avgR, avgG, avgB) {
  const target = Math.max(avgR, avgG, avgB, 240);
  return {
    rScale: clampScale(target / Math.max(avgR, 1)),
    gScale: clampScale(target / Math.max(avgG, 1)),
    bScale: clampScale(target / Math.max(avgB, 1)),
  };
}

// ─────────────────────────────────────────────────────────────
// Auto white-balance fallback: gray-world from a pixel buffer
// ─────────────────────────────────────────────────────────────

export function grayWorldScalars(rgba, stride = 16) {
  let sumR = 0,
    sumG = 0,
    sumB = 0,
    count = 0;
  for (let i = 0; i < rgba.length; i += 4 * stride) {
    sumR += rgba[i];
    sumG += rgba[i + 1];
    sumB += rgba[i + 2];
    count++;
  }
  if (count === 0) return IDENTITY;
  const meanR = sumR / count;
  const meanG = sumG / count;
  const meanB = sumB / count;
  const target = (meanR + meanG + meanB) / 3;
  if (target <= 0) return IDENTITY;
  return {
    rScale: clampScale(target / Math.max(meanR, 1)),
    gScale: clampScale(target / Math.max(meanG, 1)),
    bScale: clampScale(target / Math.max(meanB, 1)),
  };
}

// ─────────────────────────────────────────────────────────────
// Apply calibration
// ─────────────────────────────────────────────────────────────

export function applyCalibrationToRGB(r, g, b, calib) {
  if (!calib) return [r, g, b];
  return [
    clamp255(Math.round(r * calib.rScale)),
    clamp255(Math.round(g * calib.gScale)),
    clamp255(Math.round(b * calib.bScale)),
  ];
}

export function applyCalibrationToBuffer(rgba, calib) {
  if (!calib) return rgba;
  const r = calib.rScale,
    g = calib.gScale,
    b = calib.bScale;
  if (r === 1 && g === 1 && b === 1) return rgba;
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = clamp255(Math.round(rgba[i] * r));
    rgba[i + 1] = clamp255(Math.round(rgba[i + 1] * g));
    rgba[i + 2] = clamp255(Math.round(rgba[i + 2] * b));
  }
  return rgba;
}

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// ─────────────────────────────────────────────────────────────
// Public resolver
// ─────────────────────────────────────────────────────────────

export async function resolveCalibration(rgba = null) {
  const manual = await loadCalibration();
  if (manual) {
    return {
      rScale: manual.rScale,
      gScale: manual.gScale,
      bScale: manual.bScale,
      source: "manual",
    };
  }
  if (rgba && rgba.length > 0) {
    return { ...grayWorldScalars(rgba), source: "auto" };
  }
  return { ...IDENTITY, source: "identity" };
}
