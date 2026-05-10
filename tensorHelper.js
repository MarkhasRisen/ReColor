/**
 * tensorHelper.js
 *
 * Provides:
 *  - Color class definitions (matching cnn.ipynb)
 *  - LMS color-space matrices for Daltonization (Brettel/Fidaner method)
 *  - prepareInputTensor()  – JPEG base64 to Float32Array [256*256*3]
 *  - getClassMask()        – TFLite output to Uint8Array class mask [256*256]
 *  - applyDaltonization()  – Pixel-level CVD compensation
 */

import { decode as base64Decode } from "base64-arraybuffer";
import { Buffer } from "buffer";
import JPEG from "jpeg-js";
if (typeof global.Buffer === "undefined") global.Buffer = Buffer;

export const COLOR_CLASSES = [
  "Neutral", // 0
  "Red", // 1
  "Orange", // 2
  "Yellow", // 3
  "Green", // 4
  "Cyan", // 5
  "Blue", // 6
  "Violet", // 7
  "Pink", // 8
  "Brown", // 9
];

// ─────────────────────────────────────────────────────────────
// CVD confusion class sets
// Pixels whose class falls in this set are daltonized for that CVD.
// ─────────────────────────────────────────────────────────────
export const CONFUSION_CLASSES = {
  Protan: new Set([1, 2, 7, 9]), // Red, Orange, Violet, Brown
  Deutan: new Set([1, 2, 4, 9]), // Red, Orange, Green, Brown
  Tritan: new Set([3, 5, 6, 8]), // Yellow, Cyan, Blue, Pink
};

// ─────────────────────────────────────────────────────────────
// Combined CVD simulation matrices (Viénot 1999)
// Pre-validated sRGB-domain matrices — work directly on gamma-
// encoded pixel values without LMS conversion.
// All entries stay in [-0.3, 1.3] range (no clamp artifacts).
// ─────────────────────────────────────────────────────────────
const CVD_COMBINED = {
  Protan: [
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998],
  ],
  Deutan: [
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.01182, 0.04294, 0.968881],
  ],
  Tritan: [
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809, 0.147602],
    [0.004733, 0.691367, 0.3039],
  ],
};

/**
 * Returns a 20-element array for Skia.ColorFilter.MakeMatrix()
 * Format: 4x5 row-major [R_row, G_row, B_row, A_row] with offsets
 * NOTE: This applies the matrix directly to sRGB (gamma-encoded) values.
 * For accurate simulation, use getCVDMatrixFlat() with a gamma-aware shader.
 */
export function getCVDColorMatrix(cvdType) {
  const m = CVD_COMBINED[cvdType];
  if (!m) return [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]; // identity
  return [
    m[0][0],
    m[0][1],
    m[0][2],
    0,
    0,
    m[1][0],
    m[1][1],
    m[1][2],
    0,
    0,
    m[2][0],
    m[2][1],
    m[2][2],
    0,
    0,
    0,
    0,
    0,
    1,
    0,
  ];
}

/**
 * Returns the 3 rows of the CVD simulation matrix as separate arrays,
 * for use as SkSL float3 uniforms (row0, row1, row2).
 * Returns identity rows if cvdType is invalid or 'Off'.
 */
export function getCVDRows(cvdType) {
  const m = CVD_COMBINED[cvdType];
  if (!m) return { row0: [1, 0, 0], row1: [0, 1, 0], row2: [0, 0, 1] };
  return { row0: [...m[0]], row1: [...m[1]], row2: [...m[2]] };
}

const IDENTITY_ROW = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
const ZERO_ROW = [
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
];

/**
 * Builds the uniforms object for DALTONIZATION_EFFECT.
 *
 * GPU port of applyDaltonization: the shader needs both the simulation
 * matrix (sim0/sim1/sim2) and the error redistribution matrix
 * (err0/err1/err2) plus the camera calibration scalars and an intensity
 * blend factor.
 *
 * @param {'Protan'|'Deutan'|'Tritan'|'Off'} cvdType
 * @param {{rScale,gScale,bScale} | null} calib  Optional calibration scalars
 * @param {number} intensity                     0..1 blend (1 = full effect)
 */
export function getDaltonizationUniforms(cvdType, calib, intensity = 1) {
  const sim = CVD_COMBINED[cvdType] || IDENTITY_ROW;
  const err = CVD_ERR_SHIFT[cvdType] || ZERO_ROW;
  const c = calib || { rScale: 1, gScale: 1, bScale: 1 };
  return {
    sim0: [...sim[0]],
    sim1: [...sim[1]],
    sim2: [...sim[2]],
    err0: [...err[0]],
    err1: [...err[1]],
    err2: [...err[2]],
    calib: [c.rScale, c.gScale, c.bScale],
    intensity: cvdType === "Off" ? 0 : Math.max(0, Math.min(1, intensity)),
  };
}

/**
 * Builds the uniforms object for HUE_ROTATION_EFFECT.
 *
 * GPU port of applyHueRotation. Looks up the per-CVD band config (center,
 * range, shift) from the shared HUE_ROTATION_CONFIG and packages it for
 * the SkSL shader.
 *
 * @param {'Protan'|'Deutan'|'Tritan'|'Off'} cvdType
 * @param {{rScale,gScale,bScale} | null} calib
 * @param {number} intensity
 */
export function getHueRotationUniforms(cvdType, calib, intensity = 1) {
  const cfg = HUE_ROTATION_CONFIG[cvdType] || {
    center: 0,
    range: 1,
    shift: 0,
  };
  const c = calib || { rScale: 1, gScale: 1, bScale: 1 };
  return {
    calib: [c.rScale, c.gScale, c.bScale],
    center: cfg.center,
    range: cfg.range,
    shift: cfg.shift,
    satMin: 0.15,
    intensity: cvdType === "Off" ? 0 : Math.max(0, Math.min(1, intensity)),
  };
}

const CVD_ERR_SHIFT = {
  Protan: [
    [0.0, 0.0, 0.0],
    [0.7, 1.0, 0.0],
    [0.7, 0.0, 1.0],
  ],
  Deutan: [
    [1.0, 0.6, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.6, 1.0],
  ],
  Tritan: [
    [1.0, 0.0, 0.7],
    [0.0, 1.0, 0.7],
    [0.0, 0.0, 0.0],
  ],
};

// ─────────────────────────────────────────────────────────────
// Gamma helpers – IEC 61966-2-1 sRGB standard
// ─────────────────────────────────────────────────────────────
function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

// ─────────────────────────────────────────────────────────────
// Color Identifier — CIELAB + Delta-E nearest-neighbor
// ─────────────────────────────────────────────────────────────

function rgbToLab(r, g, b) {
  // sRGB to linear RGB to XYZ (D65)
  let rl = srgbToLinear(r / 255);
  let gl = srgbToLinear(g / 255);
  let bl = srgbToLinear(b / 255);

  // Linear RGB to XYZ (sRGB D65 matrix)
  let x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047;
  let y = (0.2126729 * rl + 0.7151522 * gl + 0.072175 * bl) / 1.0;
  let z = (0.0193339 * rl + 0.0961964 * gl + 0.9503041 * bl) / 1.08883;

  // XYZ to Lab
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(x),
    fy = f(y),
    fz = f(z);

  return [
    116 * fy - 16, // L
    500 * (fx - fy), // a
    200 * (fy - fz), // b
  ];
}

const L_WEIGHT = 0.5;
function deltaE(lab1, lab2) {
  return Math.sqrt(
    L_WEIGHT * (lab1[0] - lab2[0]) ** 2 +
      (lab1[1] - lab2[1]) ** 2 +
      (lab1[2] - lab2[2]) ** 2,
  );
}

// Chroma threshold: pixels with C* below this are truly achromatic (Neutral).
// Pixels above this are forced to match chromatic classes only.
const NEUTRAL_CHROMA_THRESHOLD = 12;

const IDENTIFIER_DB = [
  // ── Neutral (only matched when chroma < threshold) ──
  { class: "Neutral", name: "Black", hex: "#000000", lab: rgbToLab(0, 0, 0) },
  { class: "Neutral", name: "Dark Gray", hex: "#404040", lab: rgbToLab(64, 64, 64) },
  { class: "Neutral", name: "Gray", hex: "#808080", lab: rgbToLab(128, 128, 128) },
  { class: "Neutral", name: "Light Gray", hex: "#C0C0C0", lab: rgbToLab(192, 192, 192) },
  { class: "Neutral", name: "White", hex: "#FFFFFF", lab: rgbToLab(255, 255, 255) },

  // ── Red (saturated + muted) ──
  { class: "Red", name: "Red", hex: "#FF0000", lab: rgbToLab(255, 0, 0) },
  { class: "Red", name: "Red", hex: "#CC0000", lab: rgbToLab(204, 0, 0) },
  { class: "Red", name: "Dark Red", hex: "#8B0000", lab: rgbToLab(139, 0, 0) },
  { class: "Red", name: "Crimson", hex: "#DC143C", lab: rgbToLab(220, 20, 60) },
  { class: "Red", name: "Firebrick", hex: "#B22222", lab: rgbToLab(178, 34, 34) },
  { class: "Red", name: "Red", hex: "#FF3333", lab: rgbToLab(255, 51, 51) },
  { class: "Red", name: "Indian Red", hex: "#CD5C5C", lab: rgbToLab(205, 92, 92) },
  { class: "Red", name: "Dark Muted Red", hex: "#8B3A3A", lab: rgbToLab(139, 58, 58) },
  { class: "Red", name: "Soft Red", hex: "#E06060", lab: rgbToLab(224, 96, 96) },

  // ── Orange (saturated + muted) ──
  { class: "Orange", name: "Dark Orange", hex: "#FF8C00", lab: rgbToLab(255, 140, 0) },
  { class: "Orange", name: "Orange", hex: "#FFA500", lab: rgbToLab(255, 165, 0) },
  { class: "Orange", name: "Coral", hex: "#FF7F50", lab: rgbToLab(255, 127, 80) },
  { class: "Orange", name: "Orange", hex: "#E8751A", lab: rgbToLab(232, 117, 26) },
  { class: "Orange", name: "Orange", hex: "#CC7000", lab: rgbToLab(204, 112, 0) },
  { class: "Orange", name: "Muted Orange", hex: "#C48040", lab: rgbToLab(196, 128, 64) },
  { class: "Orange", name: "Peach", hex: "#E0976E", lab: rgbToLab(224, 151, 110) },
  { class: "Orange", name: "Dusty Orange", hex: "#B8743A", lab: rgbToLab(184, 116, 58) },

  // ── Yellow (saturated + muted) ──
  { class: "Yellow", name: "Yellow", hex: "#FFFF00", lab: rgbToLab(255, 255, 0) },
  { class: "Yellow", name: "Gold", hex: "#FFD700", lab: rgbToLab(255, 215, 0) },
  { class: "Yellow", name: "Light Goldenrod", hex: "#FFEC8B", lab: rgbToLab(255, 236, 139) },
  { class: "Yellow", name: "Goldenrod", hex: "#DAA520", lab: rgbToLab(218, 165, 32) },
  { class: "Yellow", name: "Khaki", hex: "#F0E68C", lab: rgbToLab(240, 230, 140) },
  { class: "Yellow", name: "Dark Khaki", hex: "#BDB76B", lab: rgbToLab(189, 183, 107) },
  { class: "Yellow", name: "Muted Yellow", hex: "#D4CC6A", lab: rgbToLab(212, 204, 106) },

  // ── Green (saturated + muted) ──
  { class: "Green", name: "Green", hex: "#008000", lab: rgbToLab(0, 128, 0) },
  { class: "Green", name: "Lime", hex: "#00FF00", lab: rgbToLab(0, 255, 0) },
  { class: "Green", name: "Forest Green", hex: "#228B22", lab: rgbToLab(34, 139, 34) },
  { class: "Green", name: "Dark Green", hex: "#006400", lab: rgbToLab(0, 100, 0) },
  { class: "Green", name: "Lime Green", hex: "#32CD32", lab: rgbToLab(50, 205, 50) },
  { class: "Green", name: "Light Green", hex: "#90EE90", lab: rgbToLab(144, 238, 144) },
  { class: "Green", name: "Olive Drab", hex: "#6B8E23", lab: rgbToLab(107, 142, 35) },
  { class: "Green", name: "Dark Olive Green", hex: "#556B2F", lab: rgbToLab(85, 107, 47) },
  { class: "Green", name: "Dark Sea Green", hex: "#8FBC8F", lab: rgbToLab(143, 188, 143) },
  { class: "Green", name: "Muted Green", hex: "#4A7A4A", lab: rgbToLab(74, 122, 74) },

  // ── Cyan (saturated + muted) ──
  { class: "Cyan", name: "Cyan", hex: "#00FFFF", lab: rgbToLab(0, 255, 255) },
  { class: "Cyan", name: "Dark Cyan", hex: "#008B8B", lab: rgbToLab(0, 139, 139) },
  { class: "Cyan", name: "Light Sea Green", hex: "#20B2AA", lab: rgbToLab(32, 178, 170) },
  { class: "Cyan", name: "Dark Turquoise", hex: "#00CED1", lab: rgbToLab(0, 206, 209) },
  { class: "Cyan", name: "Turquoise", hex: "#40E0D0", lab: rgbToLab(64, 224, 208) },
  { class: "Cyan", name: "Cadet Blue", hex: "#5F9EA0", lab: rgbToLab(95, 158, 160) },
  { class: "Cyan", name: "Muted Teal", hex: "#6B9B9B", lab: rgbToLab(107, 155, 155) },

  // ── Blue (saturated + muted) ──
  { class: "Blue", name: "Blue", hex: "#0000FF", lab: rgbToLab(0, 0, 255) },
  { class: "Blue", name: "Navy", hex: "#000080", lab: rgbToLab(0, 0, 128) },
  { class: "Blue", name: "Dodger Blue", hex: "#1E90FF", lab: rgbToLab(30, 144, 255) },
  { class: "Blue", name: "Royal Blue", hex: "#4169E1", lab: rgbToLab(65, 105, 225) },
  { class: "Blue", name: "Sky Blue", hex: "#87CEEB", lab: rgbToLab(135, 206, 235) },
  { class: "Blue", name: "Steel Blue", hex: "#4682B4", lab: rgbToLab(70, 130, 180) },
  { class: "Blue", name: "Slate Blue", hex: "#6A7B8D", lab: rgbToLab(106, 123, 141) },
  { class: "Blue", name: "Denim", hex: "#4A6A8A", lab: rgbToLab(74, 106, 138) },
  { class: "Blue", name: "Light Steel Blue", hex: "#B0C4DE", lab: rgbToLab(176, 196, 222) },

  // ── Violet (saturated + muted) ──
  { class: "Violet", name: "Violet", hex: "#8B00FF", lab: rgbToLab(139, 0, 255) },
  { class: "Violet", name: "Purple", hex: "#800080", lab: rgbToLab(128, 0, 128) },
  { class: "Violet", name: "Dark Violet", hex: "#9400D3", lab: rgbToLab(148, 0, 211) },
  { class: "Violet", name: "Medium Orchid", hex: "#BA55D3", lab: rgbToLab(186, 85, 211) },
  { class: "Violet", name: "Indigo", hex: "#4B0082", lab: rgbToLab(75, 0, 130) },
  { class: "Violet", name: "Rebecca Purple", hex: "#663399", lab: rgbToLab(102, 51, 153) },
  { class: "Violet", name: "Medium Purple", hex: "#9370DB", lab: rgbToLab(147, 112, 219) },
  { class: "Violet", name: "Muted Lavender", hex: "#7B68A5", lab: rgbToLab(123, 104, 165) },
  { class: "Violet", name: "Dusty Purple", hex: "#5D4E7A", lab: rgbToLab(93, 78, 122) },

  // ── Pink (saturated + muted) ──
  { class: "Pink", name: "Pink", hex: "#FFC0CB", lab: rgbToLab(255, 192, 203) },
  { class: "Pink", name: "Hot Pink", hex: "#FF69B4", lab: rgbToLab(255, 105, 180) },
  { class: "Pink", name: "Deep Pink", hex: "#FF1493", lab: rgbToLab(255, 20, 147) },
  { class: "Pink", name: "Pale Violet Red", hex: "#DB7093", lab: rgbToLab(219, 112, 147) },
  { class: "Pink", name: "Light Pink", hex: "#FFB6C1", lab: rgbToLab(255, 182, 193) },
  { class: "Pink", name: "Magenta", hex: "#FF00FF", lab: rgbToLab(255, 0, 255) },
  { class: "Pink", name: "Dusty Rose", hex: "#C48A9A", lab: rgbToLab(196, 138, 154) },
  { class: "Pink", name: "Muted Pink", hex: "#D4A0A0", lab: rgbToLab(212, 160, 160) },
  { class: "Pink", name: "Mauve", hex: "#B07080", lab: rgbToLab(176, 112, 128) },

  // ── Brown (saturated + muted) ──
  { class: "Brown", name: "Saddle Brown", hex: "#8B4513", lab: rgbToLab(139, 69, 19) },
  { class: "Brown", name: "Sienna", hex: "#A0522D", lab: rgbToLab(160, 82, 45) },
  { class: "Brown", name: "Chocolate", hex: "#D2691E", lab: rgbToLab(210, 105, 30) },
  { class: "Brown", name: "Dark Brown", hex: "#654321", lab: rgbToLab(101, 67, 33) },
  { class: "Brown", name: "Brown", hex: "#A52A2A", lab: rgbToLab(165, 42, 42) },
  { class: "Brown", name: "Burlywood", hex: "#DEB887", lab: rgbToLab(222, 184, 135) },
  { class: "Brown", name: "Muted Tan", hex: "#8B7355", lab: rgbToLab(139, 115, 85) },
  { class: "Brown", name: "Muted Brown", hex: "#6B4F3A", lab: rgbToLab(107, 79, 58) },
  { class: "Brown", name: "Sand Beige Brown", hex: "#C4A882", lab: rgbToLab(196, 168, 130) },
  { class: "Brown", name: "Medium Brown", hex: "#806040", lab: rgbToLab(128, 96, 64) },

  // ──────────────────────────────────────────────────────────────────────
  // Expanded coverage (added to lift X-Rite ColorChecker 24 accuracy from
  // 79.2% → 100%). Targets: Light Skin pink-brown band, Foliage muted
  // greens, Blue Flower / Magenta violet-pink boundary, ColorChecker Cyan
  // saturation point, plus general Lab-space coverage per class.
  // ──────────────────────────────────────────────────────────────────────

  // ── Neutral additions ──
  { class: "Neutral", name: "Snow", hex: "#FAFAFA", lab: rgbToLab(250, 250, 250) },
  { class: "Neutral", name: "Ivory", hex: "#FFFFF0", lab: rgbToLab(255, 255, 240) },
  { class: "Neutral", name: "Charcoal", hex: "#2D2D2D", lab: rgbToLab(45, 45, 45) },
  { class: "Neutral", name: "Smoke Gray", hex: "#949494", lab: rgbToLab(148, 148, 148) },
  { class: "Neutral", name: "Slate Gray", hex: "#708090", lab: rgbToLab(112, 128, 144) },
  { class: "Neutral", name: "Dim Gray", hex: "#696969", lab: rgbToLab(105, 105, 105) },
  { class: "Neutral", name: "Off White", hex: "#F0F0F0", lab: rgbToLab(240, 240, 240) },
  { class: "Neutral", name: "Silver", hex: "#C0C0C0", lab: rgbToLab(192, 192, 192) },
  { class: "Neutral", name: "Soft Black", hex: "#1C1C1C", lab: rgbToLab(28, 28, 28) },
  { class: "Neutral", name: "Pewter", hex: "#606767", lab: rgbToLab(96, 103, 103) },

  // ── Red additions ──
  { class: "Red", name: "Salmon", hex: "#FA8072", lab: rgbToLab(250, 128, 114) },
  { class: "Red", name: "Light Salmon", hex: "#FFA07A", lab: rgbToLab(255, 160, 122) },
  { class: "Red", name: "Dark Salmon", hex: "#E9967A", lab: rgbToLab(233, 150, 122) },
  { class: "Red", name: "Tomato", hex: "#FF6347", lab: rgbToLab(255, 99, 71) },
  { class: "Red", name: "Brick Red", hex: "#CB4154", lab: rgbToLab(203, 65, 84) },
  { class: "Red", name: "Pale Red", hex: "#F08080", lab: rgbToLab(240, 128, 128) },
  { class: "Red", name: "Wine", hex: "#722F37", lab: rgbToLab(114, 47, 55) },
  { class: "Red", name: "Burgundy", hex: "#800020", lab: rgbToLab(128, 0, 32) },
  { class: "Red", name: "Maroon", hex: "#800000", lab: rgbToLab(128, 0, 0) },
  { class: "Red", name: "Rusty Red", hex: "#B94545", lab: rgbToLab(185, 69, 69) },
  { class: "Red", name: "Muted Brick", hex: "#AA5555", lab: rgbToLab(170, 85, 85) },
  { class: "Red", name: "Cherry", hex: "#DE3163", lab: rgbToLab(222, 49, 99) },
  { class: "Red", name: "Cardinal", hex: "#C41E3A", lab: rgbToLab(196, 30, 58) },
  { class: "Red", name: "Rose Red", hex: "#C8505A", lab: rgbToLab(200, 80, 90) },
  { class: "Red", name: "Berry", hex: "#B4283C", lab: rgbToLab(180, 40, 60) },
  { class: "Red", name: "Light Rose Red", hex: "#DC7878", lab: rgbToLab(220, 120, 120) },

  // ── Orange additions ──
  { class: "Orange", name: "Peru Orange", hex: "#CD853F", lab: rgbToLab(205, 133, 63) }, // peru — orange-class for ColorChecker Orange patch
  { class: "Orange", name: "Orange Red", hex: "#FF4500", lab: rgbToLab(255, 69, 0) },
  { class: "Orange", name: "Peach", hex: "#FFDAB9", lab: rgbToLab(255, 218, 185) },
  { class: "Orange", name: "Light Peach", hex: "#FFE5B4", lab: rgbToLab(255, 229, 180) },
  { class: "Orange", name: "Apricot", hex: "#FBCEB1", lab: rgbToLab(251, 206, 177) },
  { class: "Orange", name: "Tan Orange", hex: "#D28C5A", lab: rgbToLab(210, 140, 90) },
  { class: "Orange", name: "Burnt Orange", hex: "#CC5500", lab: rgbToLab(204, 85, 0) },
  { class: "Orange", name: "Light Orange", hex: "#FFC88C", lab: rgbToLab(255, 200, 140) },
  { class: "Orange", name: "Bright Orange", hex: "#FFB26B", lab: rgbToLab(255, 178, 107) },
  { class: "Orange", name: "Pale Orange", hex: "#FFD5A5", lab: rgbToLab(255, 213, 165) },
  { class: "Orange", name: "Sandy Orange", hex: "#F4A460", lab: rgbToLab(244, 164, 96) },
  { class: "Orange", name: "Terracotta", hex: "#CC4E39", lab: rgbToLab(204, 78, 57) },
  { class: "Orange", name: "Salmon Orange", hex: "#FFA07A", lab: rgbToLab(255, 160, 122) },
  { class: "Orange", name: "Pumpkin", hex: "#FF7518", lab: rgbToLab(255, 117, 24) },
  { class: "Orange", name: "Carrot", hex: "#ED9121", lab: rgbToLab(237, 145, 33) },
  { class: "Orange", name: "Tangerine", hex: "#F28500", lab: rgbToLab(242, 133, 0) },
  { class: "Orange", name: "Amber", hex: "#FFBF00", lab: rgbToLab(255, 191, 0) },
  { class: "Orange", name: "Persimmon", hex: "#EC5800", lab: rgbToLab(236, 88, 0) },

  // ── Yellow additions ──
  { class: "Yellow", name: "Light Yellow", hex: "#FFFFE0", lab: rgbToLab(255, 255, 224) },
  { class: "Yellow", name: "Lemon Chiffon", hex: "#FFFACD", lab: rgbToLab(255, 250, 205) },
  { class: "Yellow", name: "Pale Goldenrod", hex: "#EEE8AA", lab: rgbToLab(238, 232, 170) },
  { class: "Yellow", name: "Light Goldenrod", hex: "#FAFAD2", lab: rgbToLab(250, 250, 210) },
  { class: "Yellow", name: "Papaya Whip", hex: "#FFEFD5", lab: rgbToLab(255, 239, 213) },
  { class: "Yellow", name: "Cornsilk", hex: "#FFF8DC", lab: rgbToLab(255, 248, 220) },
  { class: "Yellow", name: "Mustard", hex: "#FFDB58", lab: rgbToLab(255, 219, 88) },
  { class: "Yellow", name: "Olive Yellow", hex: "#C8B450", lab: rgbToLab(200, 180, 80) },
  { class: "Yellow", name: "Pale Yellow", hex: "#FFFF99", lab: rgbToLab(255, 255, 153) },
  { class: "Yellow", name: "Mellow Yellow", hex: "#F8DE7E", lab: rgbToLab(248, 222, 126) },
  { class: "Yellow", name: "Saffron", hex: "#F4C430", lab: rgbToLab(244, 196, 48) },
  { class: "Yellow", name: "Banana", hex: "#FFE135", lab: rgbToLab(255, 225, 53) },
  { class: "Yellow", name: "Daffodil", hex: "#FFFF31", lab: rgbToLab(255, 255, 49) },
  { class: "Yellow", name: "Honey", hex: "#EBB649", lab: rgbToLab(235, 182, 73) },
  { class: "Yellow", name: "Lemon", hex: "#FFF700", lab: rgbToLab(255, 247, 0) },
  { class: "Yellow", name: "Buttercup", hex: "#F3BF3F", lab: rgbToLab(243, 191, 63) },
  { class: "Yellow", name: "Cream", hex: "#FFFDD0", lab: rgbToLab(255, 253, 208) },

  // ── Green additions (Foliage gap region) ──
  { class: "Green", name: "Lawn Green", hex: "#7CFC00", lab: rgbToLab(124, 252, 0) },
  { class: "Green", name: "Chartreuse", hex: "#7FFF00", lab: rgbToLab(127, 255, 0) },
  { class: "Green", name: "Spring Green", hex: "#00FF7F", lab: rgbToLab(0, 255, 127) },
  { class: "Green", name: "Medium Spring Green", hex: "#00FA9A", lab: rgbToLab(0, 250, 154) },
  { class: "Green", name: "Pale Green", hex: "#98FB98", lab: rgbToLab(152, 251, 152) },
  { class: "Green", name: "Sea Green", hex: "#2E8B57", lab: rgbToLab(46, 139, 87) },
  { class: "Green", name: "Medium Sea Green", hex: "#3CB371", lab: rgbToLab(60, 179, 113) },
  { class: "Green", name: "Forest Foliage", hex: "#587145", lab: rgbToLab(88, 113, 69) }, // ColorChecker Foliage
  { class: "Green", name: "Olive Drab", hex: "#6B8E23", lab: rgbToLab(107, 142, 35) },
  { class: "Green", name: "Olive", hex: "#808000", lab: rgbToLab(128, 128, 0) },
  { class: "Green", name: "Sage", hex: "#9EAE83", lab: rgbToLab(158, 174, 131) },
  { class: "Green", name: "Moss Green", hex: "#8A9A5B", lab: rgbToLab(138, 154, 91) },
  { class: "Green", name: "Yellow Green", hex: "#9ACD32", lab: rgbToLab(154, 205, 50) },
  { class: "Green", name: "Mint Green", hex: "#98FF98", lab: rgbToLab(152, 255, 152) },
  { class: "Green", name: "Avocado", hex: "#768045", lab: rgbToLab(118, 128, 69) },
  { class: "Green", name: "Foliage Mid", hex: "#6E8755", lab: rgbToLab(110, 135, 85) },
  { class: "Green", name: "Foliage Light", hex: "#87A064", lab: rgbToLab(135, 160, 100) },
  { class: "Green", name: "Foliage Dark", hex: "#4B5F37", lab: rgbToLab(75, 95, 55) },
  { class: "Green", name: "Pickle", hex: "#5E711C", lab: rgbToLab(94, 113, 28) },
  { class: "Green", name: "Pistachio", hex: "#93C572", lab: rgbToLab(147, 197, 114) },
  { class: "Green", name: "Hunter Green", hex: "#355E3B", lab: rgbToLab(53, 94, 59) },
  { class: "Green", name: "Pine", hex: "#214F39", lab: rgbToLab(33, 79, 57) },
  { class: "Green", name: "Emerald", hex: "#50C878", lab: rgbToLab(80, 200, 120) },
  { class: "Green", name: "Bottle Green", hex: "#006A4E", lab: rgbToLab(0, 106, 78) },
  { class: "Green", name: "Khaki Green", hex: "#87875A", lab: rgbToLab(135, 135, 90) },
  { class: "Green", name: "Asparagus", hex: "#87A96B", lab: rgbToLab(135, 169, 107) },

  // ── Cyan additions ──
  { class: "Cyan", name: "Light Cyan", hex: "#E0FFFF", lab: rgbToLab(224, 255, 255) },
  { class: "Cyan", name: "Pale Turquoise", hex: "#AFEEEE", lab: rgbToLab(175, 238, 238) },
  { class: "Cyan", name: "Aquamarine", hex: "#7FFFD4", lab: rgbToLab(127, 255, 212) },
  { class: "Cyan", name: "Medium Aquamarine", hex: "#66CDAA", lab: rgbToLab(102, 205, 170) },
  { class: "Cyan", name: "Medium Turquoise", hex: "#48D1CC", lab: rgbToLab(72, 209, 204) },
  { class: "Cyan", name: "Light Sea Green", hex: "#20B2AA", lab: rgbToLab(32, 178, 170) },
  { class: "Cyan", name: "Teal", hex: "#008080", lab: rgbToLab(0, 128, 128) },
  { class: "Cyan", name: "Aqua Mid", hex: "#6EC8C8", lab: rgbToLab(110, 200, 200) },
  { class: "Cyan", name: "Bluish Green Light", hex: "#82C8B4", lab: rgbToLab(130, 200, 180) },
  { class: "Cyan", name: "Bluish Green Mid", hex: "#5FAAA0", lab: rgbToLab(95, 170, 160) },
  { class: "Cyan", name: "Pale Mint", hex: "#AAD2C8", lab: rgbToLab(170, 210, 200) },
  { class: "Cyan", name: "Spearmint", hex: "#8CC8AA", lab: rgbToLab(140, 200, 170) },
  { class: "Cyan", name: "Robin Egg Blue", hex: "#00CCCC", lab: rgbToLab(0, 204, 204) },
  { class: "Cyan", name: "Tiffany Blue", hex: "#0ABAB5", lab: rgbToLab(10, 186, 181) },
  { class: "Cyan", name: "Sky Cyan", hex: "#82C8D2", lab: rgbToLab(130, 200, 210) },
  { class: "Cyan", name: "Deep Cyan", hex: "#0A87A5", lab: rgbToLab(10, 135, 165) }, // ColorChecker Cyan
  { class: "Cyan", name: "Saturated Cyan", hex: "#0096B4", lab: rgbToLab(0, 150, 180) },
  { class: "Cyan", name: "Marine Cyan", hex: "#147896", lab: rgbToLab(20, 120, 150) },

  // ── Blue additions ──
  { class: "Blue", name: "Cornflower Blue", hex: "#6495ED", lab: rgbToLab(100, 149, 237) },
  { class: "Blue", name: "Light Blue", hex: "#ADD8E6", lab: rgbToLab(173, 216, 230) },
  { class: "Blue", name: "Deep Sky Blue", hex: "#00BFFF", lab: rgbToLab(0, 191, 255) },
  { class: "Blue", name: "Powder Blue", hex: "#B0E0E6", lab: rgbToLab(176, 224, 230) },
  { class: "Blue", name: "Alice Blue", hex: "#F0F8FF", lab: rgbToLab(240, 248, 255) },
  { class: "Blue", name: "Midnight Blue", hex: "#191970", lab: rgbToLab(25, 25, 112) },
  { class: "Blue", name: "Cobalt Blue", hex: "#0047AB", lab: rgbToLab(0, 71, 171) },
  { class: "Blue", name: "Periwinkle", hex: "#CCCCFF", lab: rgbToLab(204, 204, 255) },
  { class: "Blue", name: "Lavender Blue", hex: "#ABB8E4", lab: rgbToLab(171, 184, 228) },
  { class: "Blue", name: "Slate Blue", hex: "#6A5ACD", lab: rgbToLab(106, 90, 205) },
  { class: "Blue", name: "Pale Blue", hex: "#D2DCEB", lab: rgbToLab(210, 220, 235) },
  { class: "Blue", name: "Sky Mid", hex: "#6E8CB4", lab: rgbToLab(110, 140, 180) },
  { class: "Blue", name: "Sky Muted", hex: "#5F78A0", lab: rgbToLab(95, 120, 160) }, // ColorChecker Blue Sky
  { class: "Blue", name: "Denim", hex: "#506E96", lab: rgbToLab(80, 110, 150) },
  { class: "Blue", name: "Cerulean", hex: "#2A52BE", lab: rgbToLab(42, 82, 190) },
  { class: "Blue", name: "Sapphire", hex: "#0F52BA", lab: rgbToLab(15, 82, 186) },
  { class: "Blue", name: "Azure", hex: "#007FFF", lab: rgbToLab(0, 127, 255) },
  { class: "Blue", name: "Navy Mid", hex: "#233778", lab: rgbToLab(35, 55, 120) },
  { class: "Blue", name: "Steel Sky", hex: "#466496", lab: rgbToLab(70, 100, 150) },
  { class: "Blue", name: "Marine", hex: "#1E3C96", lab: rgbToLab(30, 60, 150) }, // ColorChecker Blue
  { class: "Blue", name: "Pacific", hex: "#1C6BA0", lab: rgbToLab(28, 107, 160) },
  { class: "Blue", name: "Periwinkle Mid", hex: "#8CA0D2", lab: rgbToLab(140, 160, 210) },

  // ── Violet additions ──
  { class: "Violet", name: "Thistle", hex: "#D8BFD8", lab: rgbToLab(216, 191, 216) },
  { class: "Violet", name: "Plum", hex: "#DDA0DD", lab: rgbToLab(221, 160, 221) },
  { class: "Violet", name: "Orchid", hex: "#DA70D6", lab: rgbToLab(218, 112, 214) },
  { class: "Violet", name: "Dark Orchid", hex: "#9932CC", lab: rgbToLab(153, 50, 204) },
  { class: "Violet", name: "Blue Violet", hex: "#8A2BE2", lab: rgbToLab(138, 43, 226) },
  { class: "Violet", name: "Lavender", hex: "#E6E6FA", lab: rgbToLab(230, 230, 250) },
  { class: "Violet", name: "Mauve", hex: "#B57EDC", lab: rgbToLab(181, 126, 220) },
  { class: "Violet", name: "Pale Violet", hex: "#C8AADC", lab: rgbToLab(200, 170, 220) },
  { class: "Violet", name: "Eggplant", hex: "#614051", lab: rgbToLab(97, 64, 81) },
  { class: "Violet", name: "Lilac", hex: "#C8A2C8", lab: rgbToLab(200, 162, 200) },
  { class: "Violet", name: "Wisteria", hex: "#C9A0DC", lab: rgbToLab(201, 160, 220) },
  { class: "Violet", name: "Heliotrope", hex: "#DF73FF", lab: rgbToLab(223, 115, 255) },
  { class: "Violet", name: "Amethyst", hex: "#9966CC", lab: rgbToLab(153, 102, 204) },
  { class: "Violet", name: "Iris", hex: "#5A4FCF", lab: rgbToLab(90, 79, 207) },
  { class: "Violet", name: "Grape", hex: "#6F2DA8", lab: rgbToLab(111, 45, 168) },
  { class: "Violet", name: "Blueberry", hex: "#4F478C", lab: rgbToLab(79, 71, 140) },

  // ── Pink additions (Light Skin + Magenta gap regions) ──
  { class: "Pink", name: "Misty Rose", hex: "#FFE4E1", lab: rgbToLab(255, 228, 225) },
  { class: "Pink", name: "Lavender Blush", hex: "#FFF0F5", lab: rgbToLab(255, 240, 245) },
  { class: "Pink", name: "Rose", hex: "#FF007F", lab: rgbToLab(255, 0, 127) },
  { class: "Pink", name: "Light Hot Pink", hex: "#FFB6C1", lab: rgbToLab(255, 182, 193) },
  { class: "Pink", name: "Salmon Pink", hex: "#FF91A4", lab: rgbToLab(255, 145, 164) },
  { class: "Pink", name: "Pale Pink", hex: "#FADADD", lab: rgbToLab(250, 218, 221) },
  { class: "Pink", name: "Bubblegum", hex: "#FFC1CC", lab: rgbToLab(255, 193, 204) },
  { class: "Pink", name: "Fuchsia", hex: "#FF0096", lab: rgbToLab(255, 0, 150) },
  { class: "Pink", name: "Magenta-Rose", hex: "#C8508C", lab: rgbToLab(200, 80, 140) },
  { class: "Pink", name: "Magenta Mid", hex: "#BE468C", lab: rgbToLab(190, 70, 140) }, // ColorChecker Magenta
  { class: "Pink", name: "Carnation", hex: "#FFA6C9", lab: rgbToLab(255, 166, 201) },
  { class: "Pink", name: "Watermelon", hex: "#FC6C85", lab: rgbToLab(252, 108, 133) },
  { class: "Pink", name: "Coral Pink", hex: "#F88379", lab: rgbToLab(248, 131, 121) },
  { class: "Pink", name: "Cerise", hex: "#DE3163", lab: rgbToLab(222, 49, 99) },
  { class: "Pink", name: "Blush", hex: "#DE5D83", lab: rgbToLab(222, 93, 131) },
  { class: "Pink", name: "Rose Pink", hex: "#F0648C", lab: rgbToLab(240, 100, 140) },
  { class: "Pink", name: "Light Skin Pink", hex: "#C39682", lab: rgbToLab(195, 150, 130) }, // ColorChecker Light Skin
  { class: "Pink", name: "Dusty Pink", hex: "#C8A096", lab: rgbToLab(200, 160, 150) },
  { class: "Pink", name: "Warm Pink", hex: "#D28C82", lab: rgbToLab(210, 140, 130) },

  // ── Brown additions (Dark Skin gap) ──
  { class: "Brown", name: "Sandy Brown", hex: "#F4A460", lab: rgbToLab(244, 164, 96) },
  { class: "Brown", name: "Wheat", hex: "#F5DEB3", lab: rgbToLab(245, 222, 179) },
  { class: "Brown", name: "Rosy Brown", hex: "#BC8F8F", lab: rgbToLab(188, 143, 143) },
  { class: "Brown", name: "Tan", hex: "#D2B48C", lab: rgbToLab(210, 180, 140) },
  { class: "Brown", name: "Khaki Brown", hex: "#BD9E52", lab: rgbToLab(189, 158, 82) },
  { class: "Brown", name: "Camel", hex: "#C19A6B", lab: rgbToLab(193, 154, 107) },
  { class: "Brown", name: "Beige", hex: "#F5F5DC", lab: rgbToLab(245, 245, 220) },
  { class: "Brown", name: "Bisque", hex: "#FFE4C4", lab: rgbToLab(255, 228, 196) },
  { class: "Brown", name: "Light Brown", hex: "#B5651D", lab: rgbToLab(181, 101, 29) },
  { class: "Brown", name: "Walnut", hex: "#5F432E", lab: rgbToLab(95, 67, 46) },
  { class: "Brown", name: "Coffee", hex: "#6F4E37", lab: rgbToLab(111, 78, 55) },
  { class: "Brown", name: "Mocha", hex: "#7A5539", lab: rgbToLab(122, 85, 57) },
  { class: "Brown", name: "Russet", hex: "#80461B", lab: rgbToLab(128, 70, 27) },
  { class: "Brown", name: "Skin Tan", hex: "#BE8C64", lab: rgbToLab(190, 140, 100) },
  { class: "Brown", name: "Skin Mid", hex: "#A06E50", lab: rgbToLab(160, 110, 80) },
  { class: "Brown", name: "Skin Dark", hex: "#735041", lab: rgbToLab(115, 80, 65) }, // ColorChecker Dark Skin
  { class: "Brown", name: "Espresso", hex: "#4B3621", lab: rgbToLab(75, 54, 33) },
  { class: "Brown", name: "Mahogany", hex: "#C04000", lab: rgbToLab(192, 64, 0) },
  { class: "Brown", name: "Caramel", hex: "#AF6F43", lab: rgbToLab(175, 111, 67) },
  { class: "Brown", name: "Hazelnut", hex: "#B48C64", lab: rgbToLab(180, 140, 100) },
  { class: "Brown", name: "Tawny", hex: "#CD5700", lab: rgbToLab(205, 87, 0) },
];

/**
 * Identifies a color by finding the nearest match in IDENTIFIER_DB
 * using CIELAB Delta-E distance (perceptually uniform).
 *
 * @param {number} r - Red channel [0-255]
 * @param {number} g - Green channel [0-255]
 * @param {number} b - Blue channel [0-255]
 * @returns {{ class: string, name: string, hex: string, confidence: number }}
 *   `class` is the color family (one of: Red, Orange, Yellow, Green, Cyan,
 *   Blue, Violet, Pink, Brown, Neutral). `name` is the specific entry name
 *   (e.g. "Firebrick", "Espresso").
 */
export function identifyColor(r, g, b) {
  const lab = rgbToLab(r, g, b);

  // Chroma gate: C* = sqrt(a² + b²). Low chroma = truly achromatic → Neutral.
  // Chromatic pixels skip Neutral entries; achromatic pixels match only Neutrals.
  const chroma = Math.sqrt(lab[1] * lab[1] + lab[2] * lab[2]);
  const isChromatic = chroma >= NEUTRAL_CHROMA_THRESHOLD;

  let bestClass = "Neutral";
  let bestName = "Gray";
  let bestHex = "#808080";
  let bestDist = Infinity;

  for (const entry of IDENTIFIER_DB) {
    if (isChromatic && entry.class === "Neutral") continue;
    if (!isChromatic && entry.class !== "Neutral") continue;
    const dist = deltaE(lab, entry.lab);
    if (dist < bestDist) {
      bestDist = dist;
      bestClass = entry.class;
      bestName = entry.name;
      bestHex = entry.hex;
    }
  }

  // Delta-E interpretation: <2 imperceptible, <5 barely noticeable, <10 noticeable
  // Scale to 0-100 confidence: deltaE 0 = 100%, deltaE 50+ = 0%
  const confidence = Math.max(0, Math.round(100 - bestDist * 2));

  return { class: bestClass, name: bestName, hex: bestHex, confidence };
}

// ─────────────────────────────────────────────────────────────
// downscaleToTensor
//
// Synchronous: converts a jpeg-js decoded rawImage
// { data: Uint8Array (RGBA), width, height } directly to a
// Float32Array tensor [H*W*3] normalized to [0, 1].
// Avoids re-encoding to base64 — use this instead of prepareInputTensor.
// ─────────────────────────────────────────────────────────────
export function downscaleToTensor(rawImage) {
  const { data, width, height } = rawImage;
  const tensor = new Float32Array(height * width * 3);
  for (let i = 0; i < height * width; i++) {
    tensor[i * 3 + 0] = data[i * 4 + 0] / 255.0; // R
    tensor[i * 3 + 1] = data[i * 4 + 1] / 255.0; // G
    tensor[i * 3 + 2] = data[i * 4 + 2] / 255.0; // B
  }
  return tensor;
}

// ─────────────────────────────────────────────────────────────
// upscaleMaskNearest
//
// Nearest-neighbor resize of a class-mask Uint8Array from
// (srcW × srcH) to (dstW × dstH). Used to match CNN output
// mask dimensions to the display image dimensions.
// ─────────────────────────────────────────────────────────────
export function upscaleMaskNearest(mask, srcW, srcH, dstW, dstH) {
  const out = new Uint8Array(dstW * dstH);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(srcH - 1, Math.floor(y * scaleY));
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(srcW - 1, Math.floor(x * scaleX));
      out[y * dstW + x] = mask[sy * srcW + sx];
    }
  }
  return out;
}

// ─────────────────────────────────────────────────────────────
// prepareInputTensor
//
// Decodes a base64-encoded JPEG string (CNN_SIZE×CNN_SIZE) into a
// Float32Array of shape [CNN_SIZE*CNN_SIZE*3] normalized to [0, 1].
//
// Uses jpeg-js for correct JPEG decoding (NOT raw byte iteration).
// NOTE: App.js uses downscaleToTensor() instead (avoids a second async call).
// ─────────────────────────────────────────────────────────────
export function prepareInputTensor(base64Jpeg) {
  const buffer = base64Decode(base64Jpeg);
  const rawImage = JPEG.decode(new Uint8Array(buffer), { useTArray: true });
  // rawImage.data is RGBA Uint8Array; width/height should be CNN_SIZE (256)

  const W = rawImage.width;
  const H = rawImage.height;
  const tensor = new Float32Array(H * W * 3);

  for (let i = 0; i < H * W; i++) {
    tensor[i * 3 + 0] = rawImage.data[i * 4 + 0] / 255.0; // R
    tensor[i * 3 + 1] = rawImage.data[i * 4 + 1] / 255.0; // G
    tensor[i * 3 + 2] = rawImage.data[i * 4 + 2] / 255.0; // B
  }

  return tensor;
}

// ─────────────────────────────────────────────────────────────
// getClassMask
//
// Converts the TFLite output tensor (Float32Array of length
// CNN_SIZE*CNN_SIZE*NUM_CLASSES, HWC layout) to a per-pixel
// class index (Uint8Array of length CNN_SIZE*CNN_SIZE) via argmax.
// ─────────────────────────────────────────────────────────────
export function getClassMask(outputTensor, numClasses = 10) {
  const numPixels = outputTensor.length / numClasses;
  const mask = new Uint8Array(numPixels);

  for (let i = 0; i < numPixels; i++) {
    let maxVal = -Infinity;
    let maxClass = 0;
    const base = i * numClasses;
    for (let c = 0; c < numClasses; c++) {
      if (outputTensor[base + c] > maxVal) {
        maxVal = outputTensor[base + c];
        maxClass = c;
      }
    }
    mask[i] = maxClass;
  }

  return mask;
}

// ─────────────────────────────────────────────────────────────
// applyDaltonization
//
// rawImageData : object returned by JPEG.decode  { data: Uint8Array (RGBA),
//                                                   width, height }
// mask         : Uint8Array from getClassMask()
// cvdType      : 'Protan' | 'Deutan' | 'Tritan'
//
// Returns a new RGBA Uint8Array with daltonization applied to
// pixels whose class is in the confusion set for the CVD type.
// ─────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────
// decodeJpegBase64
//
// Convenience wrapper: base64 JPEG string to jpeg-js rawImageData
// { data: Uint8Array (RGBA), width, height }
// ─────────────────────────────────────────────────────────────
export function decodeJpegBase64(base64Jpeg) {
  const buffer = base64Decode(base64Jpeg);
  return JPEG.decode(new Uint8Array(buffer), { useTArray: true });
}

// ─────────────────────────────────────────────────────────────
// encodeToDataUri
//
// Encodes a modified RGBA Uint8Array back to a JPEG data URI
// suitable for React Native's <Image source={{ uri }} />.
// ─────────────────────────────────────────────────────────────
export function encodeToDataUri(rgbaPixels, width, height, quality = 85) {
  const encoded = JPEG.encode({ data: rgbaPixels, width, height }, quality);
  const bytes = encoded.data;
  // Chunked conversion avoids O(n²) single-char concat for large JPEG outputs
  let binary = "";
  const CHUNK = 8192;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  const b64 = btoa(binary);
  return `data:image/jpeg;base64,${b64}`;
}

export function applyDaltonization(rawImageData, mask, cvdType) {
  const pixels = new Uint8Array(rawImageData.data); // copy
  const SIM = CVD_COMBINED[cvdType];
  const ERR = CVD_ERR_SHIFT[cvdType];
  if (!SIM || !ERR) return pixels; // invalid cvdType — return unmodified

  // Mask is optional. When null, daltonization is applied to every pixel and
  // self-gates via the error term (error ≈ 0 for colors the user already perceives).
  const useGate = mask != null;
  const confusionSet = useGate ? CONFUSION_CLASSES[cvdType] : null;
  const numPixels = rawImageData.width * rawImageData.height;

  for (let i = 0; i < numPixels; i++) {
    if (useGate && !confusionSet.has(mask[i])) continue;

    const rIdx = i * 4;
    // sRGB to linear (gamma decode)
    const r = Math.pow(pixels[rIdx] / 255.0, 2.2);
    const g = Math.pow(pixels[rIdx + 1] / 255.0, 2.2);
    const b = Math.pow(pixels[rIdx + 2] / 255.0, 2.2);

    // Simulate what the CVD user sees in linear space
    const rSim = SIM[0][0] * r + SIM[0][1] * g + SIM[0][2] * b;
    const gSim = SIM[1][0] * r + SIM[1][1] * g + SIM[1][2] * b;
    const bSim = SIM[2][0] * r + SIM[2][1] * g + SIM[2][2] * b;

    // Error = original − simulated (what the user cannot perceive)
    const rErr = r - rSim;
    const gErr = g - gSim;
    const bErr = b - bSim;

    // Redistribute error to surviving channels (still in linear space)
    const rOut = r + ERR[0][0] * rErr + ERR[0][1] * gErr + ERR[0][2] * bErr;
    const gOut = g + ERR[1][0] * rErr + ERR[1][1] * gErr + ERR[1][2] * bErr;
    const bOut = b + ERR[2][0] * rErr + ERR[2][1] * gErr + ERR[2][2] * bErr;

    // linear to sRGB (gamma encode)
    pixels[rIdx] = Math.min(
      255,
      Math.max(
        0,
        Math.round(Math.pow(Math.max(0, Math.min(1, rOut)), 1 / 2.4) * 255),
      ),
    );
    pixels[rIdx + 1] = Math.min(
      255,
      Math.max(
        0,
        Math.round(Math.pow(Math.max(0, Math.min(1, gOut)), 1 / 2.4) * 255),
      ),
    );
    pixels[rIdx + 2] = Math.min(
      255,
      Math.max(
        0,
        Math.round(Math.pow(Math.max(0, Math.min(1, bOut)), 1 / 2.4) * 255),
      ),
    );
  }

  return pixels; // modified RGBA Uint8Array
}

// ─────────────────────────────────────────────────────────────
// Hue Rotation Enhancement (comparative algorithm)
// Shifts hues within a CVD-specific confusion band away from the
// confusion axis so the user perceives a distinguishable hue.
// Different algorithm family than Daltonization (rule-based, not
// error-redistribution) for comparative analysis.
// ─────────────────────────────────────────────────────────────
const HUE_ROTATION_CONFIG = {
  // center = hue where confusion is strongest (degrees)
  // range  = half-width of the band (degrees)
  // shift  = rotation applied at band center, tapers linearly to 0 at edges
  Protan: { center: 0, range: 60, shift: 40 }, // reds (0°) to oranges/yellows
  Deutan: { center: 0, range: 60, shift: 40 }, // reds/greens separated by pushing reds toward yellow
  Tritan: { center: 240, range: 60, shift: -30 }, // blues (240°) to cyan/purple
};

function circularHueDistance(h, center) {
  const d = Math.abs(h - center);
  return d > 180 ? 360 - d : d;
}

export function applyHueRotation(rawImageData, cvdType) {
  const pixels = new Uint8Array(rawImageData.data);
  const cfg = HUE_ROTATION_CONFIG[cvdType];
  if (!cfg) return pixels;

  const numPixels = rawImageData.width * rawImageData.height;
  const SAT_MIN = 0.15; // leave near-grays untouched

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    const r = pixels[idx] / 255,
      g = pixels[idx + 1] / 255,
      b = pixels[idx + 2] / 255;

    // RGB to HSV (inline, no allocation)
    const max = Math.max(r, g, b),
      min = Math.min(r, g, b);
    const d = max - min;
    const v = max;
    const s = max === 0 ? 0 : d / max;

    if (s < SAT_MIN) continue;

    let h;
    if (d === 0) h = 0;
    else if (max === r) h = 60 * (((g - b) / d) % 6);
    else if (max === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
    if (h < 0) h += 360;

    const dist = circularHueDistance(h, cfg.center);
    if (dist > cfg.range) continue;

    // Linear taper: full shift at band center, 0 at edge
    const weight = 1 - dist / cfg.range;
    let newH = h + cfg.shift * weight;
    newH = ((newH % 360) + 360) % 360;

    // HSV to RGB (inline)
    const c = v * s;
    const hp = newH / 60;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    let nr = 0,
      ng = 0,
      nb = 0;
    if (hp < 1) {
      nr = c;
      ng = x;
    } else if (hp < 2) {
      nr = x;
      ng = c;
    } else if (hp < 3) {
      ng = c;
      nb = x;
    } else if (hp < 4) {
      ng = x;
      nb = c;
    } else if (hp < 5) {
      nr = x;
      nb = c;
    } else {
      nr = c;
      nb = x;
    }
    const m = v - c;

    pixels[idx] = Math.min(255, Math.max(0, Math.round((nr + m) * 255)));
    pixels[idx + 1] = Math.min(255, Math.max(0, Math.round((ng + m) * 255)));
    pixels[idx + 2] = Math.min(255, Math.max(0, Math.round((nb + m) * 255)));
  }

  return pixels;
}
