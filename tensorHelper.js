/**
 * tensorHelper.js
 *
 * Provides:
 *  - Color class definitions (matching cnn.ipynb)
 *  - LMS color-space matrices for Daltonization (Brettel/Fidaner method)
 *  - prepareInputTensor()  – JPEG base64 → Float32Array [256*256*3]
 *  - getClassMask()        – TFLite output → Uint8Array class mask [256*256]
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

// ─────────────────────────────────────────────────────────────
// Error-shift matrices
// Redistribute the missing-cone error into the surviving channels
// so the user can perceive the difference.
// ─────────────────────────────────────────────────────────────
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

/**
 * Converts sRGB [0-255] to CIELAB using D65 illuminant.
 */
function rgbToLab(r, g, b) {
  // sRGB → linear RGB → XYZ (D65)
  let rl = srgbToLinear(r / 255);
  let gl = srgbToLinear(g / 255);
  let bl = srgbToLinear(b / 255);

  // Linear RGB → XYZ (sRGB D65 matrix)
  let x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047;
  let y = (0.2126729 * rl + 0.7151522 * gl + 0.072175 * bl) / 1.0;
  let z = (0.0193339 * rl + 0.0961964 * gl + 0.9503041 * bl) / 1.08883;

  // XYZ → Lab
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

/**
 * Weighted Delta-E distance between two CIELAB colors.
 * L* (lightness) is down-weighted so dark red and bright red both match "Red".
 * a* and b* (chroma/hue) are full-weight since they determine color name.
 */
const L_WEIGHT = 0.5; // lightness matters less for color naming
function deltaE(lab1, lab2) {
  return Math.sqrt(
    L_WEIGHT * (lab1[0] - lab2[0]) ** 2 +
      (lab1[1] - lab2[1]) ** 2 +
      (lab1[2] - lab2[2]) ** 2,
  );
}

/**
 * Representative colors for each of the 10 classes.
 * Multiple entries per class cover common shades for robust matching.
 * LAB values are pre-computed from the RGB values.
 */
// Chroma threshold: pixels with C* below this are truly achromatic (Neutral).
// Pixels above this are forced to match chromatic classes only.
const NEUTRAL_CHROMA_THRESHOLD = 12;

const IDENTIFIER_DB = [
  // ── Neutral (only matched when chroma < threshold) ──
  { name: "black", hex: "#000000", lab: rgbToLab(0, 0, 0) },
  { name: "dark gray", hex: "#404040", lab: rgbToLab(64, 64, 64) },
  { name: "gray", hex: "#808080", lab: rgbToLab(128, 128, 128) },
  { name: "light gray", hex: "#C0C0C0", lab: rgbToLab(192, 192, 192) },
  { name: "white", hex: "#FFFFFF", lab: rgbToLab(255, 255, 255) },

  // ── Red (saturated + muted) ──
  { name: "Red", hex: "#FF0000", lab: rgbToLab(255, 0, 0) },
  { name: "Red", hex: "#CC0000", lab: rgbToLab(204, 0, 0) },
  { name: "Red", hex: "#8B0000", lab: rgbToLab(139, 0, 0) },
  { name: "Crimson", hex: "#DC143C", lab: rgbToLab(220, 20, 60) }, // crimson
  { name: "Firebrick", hex: "#B22222", lab: rgbToLab(178, 34, 34) }, // firebrick
  { name: "Red", hex: "#FF3333", lab: rgbToLab(255, 51, 51) },
  { name: "Indian Red", hex: "#CD5C5C", lab: rgbToLab(205, 92, 92) }, // indian red (muted)
  { name: "Dark Muted Red", hex: "#8B3A3A", lab: rgbToLab(139, 58, 58) }, // dark muted red
  { name: "Soft Red", hex: "#E06060", lab: rgbToLab(224, 96, 96) }, // soft red

  // ── Orange (saturated + muted) ──
  { name: "Dark Orange", hex: "#FF8C00", lab: rgbToLab(255, 140, 0) }, // dark orange
  { name: "Orange", hex: "#FFA500", lab: rgbToLab(255, 165, 0) }, // orange
  { name: "Coral", hex: "#FF7F50", lab: rgbToLab(255, 127, 80) }, // coral
  { name: "Orange", hex: "#E8751A", lab: rgbToLab(232, 117, 26) },
  { name: "Orange", hex: "#CC7000", lab: rgbToLab(204, 112, 0) },
  { name: "Muted Orange", hex: "#C48040", lab: rgbToLab(196, 128, 64) }, // muted orange
  { name: "Peach", hex: "#E0976E", lab: rgbToLab(224, 151, 110) }, // peach/salmon
  { name: "Dusty Orange", hex: "#B8743A", lab: rgbToLab(184, 116, 58) }, // dusty orange

  // ── Yellow (saturated + muted) ──
  { name: "Yellow", hex: "#FFFF00", lab: rgbToLab(255, 255, 0) },
  { name: "Gold", hex: "#FFD700", lab: rgbToLab(255, 215, 0) }, // gold
  { name: "Light Goldenrod", hex: "#FFEC8B", lab: rgbToLab(255, 236, 139) }, // light goldenrod
  { name: "Goldenrod", hex: "#DAA520", lab: rgbToLab(218, 165, 32) }, // goldenrod
  { name: "Khaki", hex: "#F0E68C", lab: rgbToLab(240, 230, 140) }, // khaki
  { name: "Dark Khaki", hex: "#BDB76B", lab: rgbToLab(189, 183, 107) }, // dark khaki (muted)
  { name: "Muted Yellow", hex: "#D4CC6A", lab: rgbToLab(212, 204, 106) }, // muted yellow

  // ── Green (saturated + muted) ──
  { name: "Green", hex: "#008000", lab: rgbToLab(0, 128, 0) },
  { name: "Lime", hex: "#00FF00", lab: rgbToLab(0, 255, 0) }, // lime
  { name: "Forest Green", hex: "#228B22", lab: rgbToLab(34, 139, 34) }, // forest green
  { name: "Dark Green", hex: "#006400", lab: rgbToLab(0, 100, 0) }, // dark green
  { name: "Lime Green", hex: "#32CD32", lab: rgbToLab(50, 205, 50) }, // lime green
  { name: "Light Green", hex: "#90EE90", lab: rgbToLab(144, 238, 144) }, // light green
  { name: "Olive Drab", hex: "#6B8E23", lab: rgbToLab(107, 142, 35) }, // olive drab (muted)
  { name: "Dark Olive Green", hex: "#556B2F", lab: rgbToLab(85, 107, 47) }, // dark olive green
  { name: "Dark Sea Green", hex: "#8FBC8F", lab: rgbToLab(143, 188, 143) }, // dark sea green (muted)
  { name: "Muted Green", hex: "#4A7A4A", lab: rgbToLab(74, 122, 74) }, // muted green

  // ── Cyan (saturated + muted) ──
  { name: "Cyan", hex: "#00FFFF", lab: rgbToLab(0, 255, 255) },
  { name: "Dark Cyan", hex: "#008B8B", lab: rgbToLab(0, 139, 139) }, // dark cyan
  { name: "Light Sea Green", hex: "#20B2AA", lab: rgbToLab(32, 178, 170) }, // light sea green
  { name: "Dark Turquoise", hex: "#00CED1", lab: rgbToLab(0, 206, 209) }, // dark turquoise
  { name: "Turquoise", hex: "#40E0D0", lab: rgbToLab(64, 224, 208) }, // turquoise
  { name: "Cadet Blue", hex: "#5F9EA0", lab: rgbToLab(95, 158, 160) }, // cadet blue (muted)
  { name: "Muted Teal", hex: "#6B9B9B", lab: rgbToLab(107, 155, 155) }, // muted teal

  // ── Blue (saturated + muted) ──
  { name: "Blue", hex: "#0000FF", lab: rgbToLab(0, 0, 255) },
  { name: "Navy", hex: "#000080", lab: rgbToLab(0, 0, 128) }, // navy
  { name: "Dodger Blue", hex: "#1E90FF", lab: rgbToLab(30, 144, 255) }, // dodger blue
  { name: "Royal Blue", hex: "#4169E1", lab: rgbToLab(65, 105, 225) }, // royal blue
  { name: "Sky Blue", hex: "#87CEEB", lab: rgbToLab(135, 206, 235) }, // sky blue
  { name: "Steel Blue", hex: "#4682B4", lab: rgbToLab(70, 130, 180) }, // steel blue
  { name: "Slate Blue", hex: "#6A7B8D", lab: rgbToLab(106, 123, 141) }, // slate (muted blue)
  { name: "Denim", hex: "#4A6A8A", lab: rgbToLab(74, 106, 138) }, // denim (muted)
  { name: "Light Steel Blue", hex: "#B0C4DE", lab: rgbToLab(176, 196, 222) }, // light steel blue

  // ── Violet (saturated + muted) ──
  { name: "Violet", hex: "#8B00FF", lab: rgbToLab(139, 0, 255) },
  { name: "Purple", hex: "#800080", lab: rgbToLab(128, 0, 128) }, // purple
  { name: "Dark Violet", hex: "#9400D3", lab: rgbToLab(148, 0, 211) }, // dark violet
  { name: "Medium Orchid", hex: "#BA55D3", lab: rgbToLab(186, 85, 211) }, // medium orchid
  { name: "Indigo", hex: "#4B0082", lab: rgbToLab(75, 0, 130) }, // indigo
  { name: "Rebecca Purple", hex: "#663399", lab: rgbToLab(102, 51, 153) }, // rebecca purple
  { name: "Medium Purple", hex: "#9370DB", lab: rgbToLab(147, 112, 219) }, // medium purple (muted)
  { name: "Muted Lavender", hex: "#7B68A5", lab: rgbToLab(123, 104, 165) }, // muted lavender
  { name: "Dusty Purple", hex: "#5D4E7A", lab: rgbToLab(93, 78, 122) }, // dusty purple

  // ── Pink (saturated + muted) ──
  { name: "Pink", hex: "#FFC0CB", lab: rgbToLab(255, 192, 203) },
  { name: "Hot Pink", hex: "#FF69B4", lab: rgbToLab(255, 105, 180) }, // hot pink
  { name: "Deep Pink", hex: "#FF1493", lab: rgbToLab(255, 20, 147) }, // deep pink
  { name: "Pale Violet Red", hex: "#DB7093", lab: rgbToLab(219, 112, 147) }, // pale violet red
  { name: "Light Pink", hex: "#FFB6C1", lab: rgbToLab(255, 182, 193) }, // light pink
  { name: "Magenta", hex: "#FF00FF", lab: rgbToLab(255, 0, 255) }, // magenta
  { name: "Dusty Rose", hex: "#C48A9A", lab: rgbToLab(196, 138, 154) }, // dusty rose (muted)
  { name: "Muted Pink", hex: "#D4A0A0", lab: rgbToLab(212, 160, 160) }, // muted pink
  { name: "Mauve", hex: "#B07080", lab: rgbToLab(176, 112, 128) }, // mauve

  // ── Brown (saturated + muted) ──
  { name: "Saddle Brown", hex: "#8B4513", lab: rgbToLab(139, 69, 19) }, // saddle brown
  { name: "Sienna", hex: "#A0522D", lab: rgbToLab(160, 82, 45) }, // sienna
  { name: "Chocolate", hex: "#D2691E", lab: rgbToLab(210, 105, 30) }, // chocolate
  { name: "Dark Brown", hex: "#654321", lab: rgbToLab(101, 67, 33) }, // dark brown
  { name: "Brown", hex: "#A52A2A", lab: rgbToLab(165, 42, 42) }, // brown
  { name: "Burlywood", hex: "#DEB887", lab: rgbToLab(222, 184, 135) }, // burlywood
  { name: "Muted Tan", hex: "#8B7355", lab: rgbToLab(139, 115, 85) }, // muted tan
  { name: "Muted Brown", hex: "#6B4F3A", lab: rgbToLab(107, 79, 58) }, // muted brown
  { name: "Sand/Beige-Brown", hex: "#C4A882", lab: rgbToLab(196, 168, 130) }, // sand/beige-brown
  { name: "Medium Brown", hex: "#806040", lab: rgbToLab(128, 96, 64) }, // medium brown
];

/**
 * Identifies a color by finding the nearest match in IDENTIFIER_DB
 * using CIELAB Delta-E distance (perceptually uniform).
 *
 * @param {number} r - Red channel [0-255]
 * @param {number} g - Green channel [0-255]
 * @param {number} b - Blue channel [0-255]
 * @returns {{ className: string, hex: string, confidence: number }}
 */
export function identifyColor(r, g, b) {
  const lab = rgbToLab(r, g, b);

  // Chroma gate: C* = sqrt(a² + b²). Low chroma = truly achromatic → Neutral
  const chroma = Math.sqrt(lab[1] * lab[1] + lab[2] * lab[2]);
  const isChromatic = chroma >= NEUTRAL_CHROMA_THRESHOLD;

  let bestName = "Neutral";
  let bestHex = "#808080";
  let bestDist = Infinity;

  for (const entry of IDENTIFIER_DB) {
    // If pixel has color, skip Neutral entries; if achromatic, allow all
    if (isChromatic && entry.name === "Neutral") continue;
    const dist = deltaE(lab, entry.lab);
    if (dist < bestDist) {
      bestDist = dist;
      bestName = entry.name;
      bestHex = entry.hex;
    }
  }

  // Delta-E interpretation: <2 imperceptible, <5 barely noticeable, <10 noticeable
  // Scale to 0-100 confidence: deltaE 0 = 100%, deltaE 50+ = 0%
  const confidence = Math.max(0, Math.round(100 - bestDist * 2));

  return { className: bestName, hex: bestHex, confidence };
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
// Convenience wrapper: base64 JPEG string → jpeg-js rawImageData
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
    // sRGB → linear (gamma decode)
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

    // linear → sRGB (gamma encode)
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
  Protan: { center: 0, range: 60, shift: 40 }, // reds (0°) → oranges/yellows
  Deutan: { center: 0, range: 60, shift: 40 }, // reds/greens separated by pushing reds toward yellow
  Tritan: { center: 240, range: 60, shift: -30 }, // blues (240°) → cyan/purple
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

    // RGB → HSV (inline, no allocation)
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

    // HSV → RGB (inline)
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
