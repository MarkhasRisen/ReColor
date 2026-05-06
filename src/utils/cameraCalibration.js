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
 *      "white" actually neutral. Stored in AsyncStorage and applied to every
 *      capture afterwards.
 *
 *   2. Auto white-balance fallback (when no manual calibration exists) —
 *      gray-world reflectance assumption (Buchsbaum 1980): the average of a
 *      natural scene's pixels is roughly neutral. Compute mean(R), mean(G),
 *      mean(B) → derive scalars that equalise them. Robust enough for casual
 *      use without any setup.
 *
 * The result of either layer is a {rScale, gScale, bScale} triple that the
 * three camera screens apply to RGB pixels before passing them to
 * identifyColor / applyDaltonization / applyHueRotation.
 */

import AsyncStorage from "@react-native-async-storage/async-storage";

const STORAGE_KEY = "@recolor_camera_calibration";

// Sane bounds on the per-channel scalars — anything beyond suggests bad input
// (e.g., user calibrated against a saturated color instead of white). We clamp
// rather than reject so the worst-case behaviour is "weak correction" not
// "broken colors".
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
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (
      typeof parsed.rScale !== "number" ||
      typeof parsed.gScale !== "number" ||
      typeof parsed.bScale !== "number"
    ) {
      return null;
    }
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
    calibratedAt: new Date().toISOString(),
  };
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  return payload;
}

export async function clearCalibration() {
  await AsyncStorage.removeItem(STORAGE_KEY);
}

function clampScale(s) {
  if (!Number.isFinite(s) || s <= 0) return 1;
  return Math.max(SCALE_MIN, Math.min(SCALE_MAX, s));
}

// ─────────────────────────────────────────────────────────────
// Manual calibration: derive scalars from a white reference patch
// ─────────────────────────────────────────────────────────────

/**
 * Given the average RGB of a region the user pointed at a white reference,
 * compute the per-channel scalars that would make it neutral white (255,255,255).
 *
 * Uses the brighter of (max channel, 240) as the target so that a sample of
 * a slightly off-white surface (e.g., paper at 230,225,220) is still treated
 * as white instead of inflating all channels.
 *
 * @param {number} avgR  Captured average red   [0..255]
 * @param {number} avgG  Captured average green [0..255]
 * @param {number} avgB  Captured average blue  [0..255]
 * @returns {{rScale, gScale, bScale}}
 */
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

/**
 * Gray-world auto white-balance: assume the average of the scene is neutral.
 * Compute mean RGB across all sampled pixels, then scale each channel so the
 * means are equal.
 *
 * Robust for natural scenes; degrades gracefully when one color dominates
 * (the over-correction is bounded by clampScale).
 *
 * @param {Uint8Array} rgba  RGBA pixel buffer
 * @param {number} stride    Sample every Nth pixel for speed (default 16)
 * @returns {{rScale, gScale, bScale}}
 */
export function grayWorldScalars(rgba, stride = 16) {
  let sumR = 0, sumG = 0, sumB = 0, count = 0;
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
  // Use the luminance-weighted mean as the neutral target, then scale each
  // channel so its mean matches that target.
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

/**
 * Apply per-channel scalars to a single RGB triple.
 * @returns {[number, number, number]}  Calibrated R, G, B (0..255 ints)
 */
export function applyCalibrationToRGB(r, g, b, calib) {
  if (!calib) return [r, g, b];
  return [
    clamp255(Math.round(r * calib.rScale)),
    clamp255(Math.round(g * calib.gScale)),
    clamp255(Math.round(b * calib.bScale)),
  ];
}

/**
 * Apply per-channel scalars to an RGBA pixel buffer in place.
 * Used by Camera Enhancement before daltonization / hue rotation so the
 * algorithm sees calibrated input.
 */
export function applyCalibrationToBuffer(rgba, calib) {
  if (!calib) return rgba;
  const r = calib.rScale, g = calib.gScale, b = calib.bScale;
  if (r === 1 && g === 1 && b === 1) return rgba;
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i]     = clamp255(Math.round(rgba[i]     * r));
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

/**
 * Resolve the calibration to use for a given capture. Manual calibration wins
 * when present; otherwise we fall back to gray-world from the buffer (or
 * identity if no buffer is provided).
 *
 * @param {Uint8Array | null} rgba  Optional pixel buffer for fallback
 * @returns {Promise<{rScale, gScale, bScale, source: 'manual'|'auto'|'identity'}>}
 */
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
