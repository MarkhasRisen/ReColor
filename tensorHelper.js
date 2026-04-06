/**
 * tensorHelper.js
 *
 * Provides:
 *  - Color class definitions (matching cnn.ipynb)
 *  - LMS color-space matrices for Daltonization (Brettel/Fidaner method)
 *  - prepareInputTensor()  – JPEG base64 → Float32Array [128*128*3]
 *  - getClassMask()        – TFLite output → Uint8Array class mask [128*128]
 *  - applyDaltonization()  – Pixel-level CVD compensation
 */

import JPEG from 'jpeg-js';
import { decode as base64Decode } from 'base64-arraybuffer';
import { Buffer } from 'buffer';
if (typeof global.Buffer === 'undefined') global.Buffer = Buffer;

// ─────────────────────────────────────────────────────────────
// Color class definitions (must match algo/cnn.ipynb)
// ─────────────────────────────────────────────────────────────
export const COLOR_CLASSES = [
  'Neutral', // 0
  'Red',     // 1
  'Orange',  // 2
  'Yellow',  // 3
  'Green',   // 4
  'Cyan',    // 5
  'Blue',    // 6
  'Violet',  // 7
  'Pink',    // 8
  'Brown',   // 9
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
// LMS matrices – Hunt-Pointer-Estevez adapted to D65
// Operate on linear (gamma-decoded) RGB in [0, 1]
// ─────────────────────────────────────────────────────────────
const RGB_TO_LMS = [
  [0.31399022, 0.63951294, 0.04649755],
  [0.15537241, 0.75789446, 0.08670142],
  [0.01775239, 0.10944209, 0.87256922],
];

const LMS_TO_RGB = [
  [ 5.47221206, -4.64196010,  0.16963708],
  [-1.12524190,  2.29317094, -0.16789520],
  [ 0.02980165, -0.19318073,  1.16364789],
];

// ─────────────────────────────────────────────────────────────
// CVD simulation matrices (in LMS space)
// Reconstruct the missing cone channel from the surviving two.
// ─────────────────────────────────────────────────────────────
const CVD_SIM = {
  // Protanopia – L cone absent, L reconstructed from M and S
  Protan: [
    [0.00000,  2.02344, -2.52581],
    [0.00000,  1.00000,  0.00000],
    [0.00000,  0.00000,  1.00000],
  ],
  // Deuteranopia – M cone absent, M reconstructed from L and S
  Deutan: [
    [1.00000,  0.00000,  0.00000],
    [0.49421,  0.00000,  1.24827],
    [0.00000,  0.00000,  1.00000],
  ],
  // Tritanopia – S cone absent, S reconstructed from L and M
  Tritan: [
    [ 1.00000,  0.00000,  0.00000],
    [ 0.00000,  1.00000,  0.00000],
    [-0.86744,  1.86744,  0.00000],
  ],
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
    [0.114503, 0.786281,  0.099216],
    [-0.003882, -0.048116, 1.051998],
  ],
  Deutan: [
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501,  0.047413],
    [-0.011820, 0.042940, 0.968881],
  ],
  Tritan: [
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809,  0.147602],
    [0.004733, 0.691367,  0.303900],
  ],
};

/**
 * Returns a 20-element array for Skia.ColorFilter.MakeMatrix()
 * Format: 4x5 row-major [R_row, G_row, B_row, A_row] with offsets
 */
export function getCVDColorMatrix(cvdType) {
  const m = CVD_COMBINED[cvdType];
  if (!m) return [1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0]; // identity
  return [
    m[0][0], m[0][1], m[0][2], 0, 0,
    m[1][0], m[1][1], m[1][2], 0, 0,
    m[2][0], m[2][1], m[2][2], 0, 0,
    0,       0,       0,       1, 0,
  ];
}

// ─────────────────────────────────────────────────────────────
// Error-shift matrices
// Redistribute the missing-cone error into the surviving channels
// so the user can perceive the difference.
// ─────────────────────────────────────────────────────────────
const CVD_ERR_SHIFT = {
  Protan: [
    [0.00000, 0.00000, 0.00000],
    [0.70000, 1.00000, 0.00000],
    [0.70000, 0.00000, 1.00000],
  ],
  Deutan: [
    [1.00000, 0.60000, 0.00000],
    [0.00000, 0.00000, 0.00000],
    [0.00000, 0.60000, 1.00000],
  ],
  Tritan: [
    [1.00000, 0.00000, 0.70000],
    [0.00000, 1.00000, 0.70000],
    [0.00000, 0.00000, 0.00000],
  ],
};

// ─────────────────────────────────────────────────────────────
// Gamma helpers – IEC 61966-2-1 sRGB standard
// ─────────────────────────────────────────────────────────────
function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function linearToSrgb(c) {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
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
  let y = (0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl) / 1.00000;
  let z = (0.0193339 * rl + 0.0961964 * gl + 0.9503041 * bl) / 1.08883;

  // XYZ → Lab
  const f = t => t > 0.008856 ? Math.cbrt(t) : (7.787 * t + 16 / 116);
  const fx = f(x), fy = f(y), fz = f(z);

  return [
    116 * fy - 16,       // L
    500 * (fx - fy),     // a
    200 * (fy - fz),     // b
  ];
}

/**
 * CIE76 Delta-E distance between two CIELAB colors.
 */
function deltaE(lab1, lab2) {
  return Math.sqrt(
    (lab1[0] - lab2[0]) ** 2 +
    (lab1[1] - lab2[1]) ** 2 +
    (lab1[2] - lab2[2]) ** 2
  );
}

/**
 * Representative colors for each of the 10 classes.
 * Multiple entries per class cover common shades for robust matching.
 * LAB values are pre-computed from the RGB values.
 */
const IDENTIFIER_DB = [
  // Neutral (black, dark gray, gray, light gray, white)
  { name: 'Neutral', hex: '#000000', lab: rgbToLab(0, 0, 0) },
  { name: 'Neutral', hex: '#404040', lab: rgbToLab(64, 64, 64) },
  { name: 'Neutral', hex: '#808080', lab: rgbToLab(128, 128, 128) },
  { name: 'Neutral', hex: '#C0C0C0', lab: rgbToLab(192, 192, 192) },
  { name: 'Neutral', hex: '#FFFFFF', lab: rgbToLab(255, 255, 255) },
  { name: 'Neutral', hex: '#F5F5DC', lab: rgbToLab(245, 245, 220) }, // beige

  // Red
  { name: 'Red', hex: '#FF0000', lab: rgbToLab(255, 0, 0) },
  { name: 'Red', hex: '#CC0000', lab: rgbToLab(204, 0, 0) },
  { name: 'Red', hex: '#8B0000', lab: rgbToLab(139, 0, 0) },
  { name: 'Red', hex: '#DC143C', lab: rgbToLab(220, 20, 60) },      // crimson
  { name: 'Red', hex: '#B22222', lab: rgbToLab(178, 34, 34) },      // firebrick
  { name: 'Red', hex: '#FF3333', lab: rgbToLab(255, 51, 51) },

  // Orange
  { name: 'Orange', hex: '#FF8C00', lab: rgbToLab(255, 140, 0) },   // dark orange
  { name: 'Orange', hex: '#FFA500', lab: rgbToLab(255, 165, 0) },   // orange
  { name: 'Orange', hex: '#FF7F50', lab: rgbToLab(255, 127, 80) },  // coral
  { name: 'Orange', hex: '#E8751A', lab: rgbToLab(232, 117, 26) },
  { name: 'Orange', hex: '#CC7000', lab: rgbToLab(204, 112, 0) },

  // Yellow
  { name: 'Yellow', hex: '#FFFF00', lab: rgbToLab(255, 255, 0) },
  { name: 'Yellow', hex: '#FFD700', lab: rgbToLab(255, 215, 0) },   // gold
  { name: 'Yellow', hex: '#FFEC8B', lab: rgbToLab(255, 236, 139) }, // light goldenrod
  { name: 'Yellow', hex: '#DAA520', lab: rgbToLab(218, 165, 32) },  // goldenrod
  { name: 'Yellow', hex: '#F0E68C', lab: rgbToLab(240, 230, 140) }, // khaki

  // Green
  { name: 'Green', hex: '#008000', lab: rgbToLab(0, 128, 0) },
  { name: 'Green', hex: '#00FF00', lab: rgbToLab(0, 255, 0) },      // lime
  { name: 'Green', hex: '#228B22', lab: rgbToLab(34, 139, 34) },    // forest green
  { name: 'Green', hex: '#006400', lab: rgbToLab(0, 100, 0) },      // dark green
  { name: 'Green', hex: '#32CD32', lab: rgbToLab(50, 205, 50) },    // lime green
  { name: 'Green', hex: '#90EE90', lab: rgbToLab(144, 238, 144) },  // light green

  // Cyan
  { name: 'Cyan', hex: '#00FFFF', lab: rgbToLab(0, 255, 255) },
  { name: 'Cyan', hex: '#008B8B', lab: rgbToLab(0, 139, 139) },    // dark cyan
  { name: 'Cyan', hex: '#20B2AA', lab: rgbToLab(32, 178, 170) },   // light sea green
  { name: 'Cyan', hex: '#00CED1', lab: rgbToLab(0, 206, 209) },    // dark turquoise
  { name: 'Cyan', hex: '#40E0D0', lab: rgbToLab(64, 224, 208) },   // turquoise

  // Blue
  { name: 'Blue', hex: '#0000FF', lab: rgbToLab(0, 0, 255) },
  { name: 'Blue', hex: '#000080', lab: rgbToLab(0, 0, 128) },      // navy
  { name: 'Blue', hex: '#1E90FF', lab: rgbToLab(30, 144, 255) },   // dodger blue
  { name: 'Blue', hex: '#4169E1', lab: rgbToLab(65, 105, 225) },   // royal blue
  { name: 'Blue', hex: '#87CEEB', lab: rgbToLab(135, 206, 235) },  // sky blue
  { name: 'Blue', hex: '#4682B4', lab: rgbToLab(70, 130, 180) },   // steel blue

  // Violet
  { name: 'Violet', hex: '#8B00FF', lab: rgbToLab(139, 0, 255) },
  { name: 'Violet', hex: '#800080', lab: rgbToLab(128, 0, 128) },   // purple
  { name: 'Violet', hex: '#9400D3', lab: rgbToLab(148, 0, 211) },   // dark violet
  { name: 'Violet', hex: '#BA55D3', lab: rgbToLab(186, 85, 211) },  // medium orchid
  { name: 'Violet', hex: '#4B0082', lab: rgbToLab(75, 0, 130) },    // indigo
  { name: 'Violet', hex: '#663399', lab: rgbToLab(102, 51, 153) },  // rebecca purple

  // Pink
  { name: 'Pink', hex: '#FFC0CB', lab: rgbToLab(255, 192, 203) },
  { name: 'Pink', hex: '#FF69B4', lab: rgbToLab(255, 105, 180) },   // hot pink
  { name: 'Pink', hex: '#FF1493', lab: rgbToLab(255, 20, 147) },    // deep pink
  { name: 'Pink', hex: '#DB7093', lab: rgbToLab(219, 112, 147) },   // pale violet red
  { name: 'Pink', hex: '#FFB6C1', lab: rgbToLab(255, 182, 193) },   // light pink
  { name: 'Pink', hex: '#FF00FF', lab: rgbToLab(255, 0, 255) },     // magenta

  // Brown
  { name: 'Brown', hex: '#8B4513', lab: rgbToLab(139, 69, 19) },    // saddle brown
  { name: 'Brown', hex: '#A0522D', lab: rgbToLab(160, 82, 45) },    // sienna
  { name: 'Brown', hex: '#D2691E', lab: rgbToLab(210, 105, 30) },   // chocolate
  { name: 'Brown', hex: '#654321', lab: rgbToLab(101, 67, 33) },    // dark brown
  { name: 'Brown', hex: '#A52A2A', lab: rgbToLab(165, 42, 42) },    // brown
  { name: 'Brown', hex: '#DEB887', lab: rgbToLab(222, 184, 135) },  // burlywood
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
  let bestName = 'Neutral';
  let bestHex = '#808080';
  let bestDist = Infinity;

  for (const entry of IDENTIFIER_DB) {
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
// prepareInputTensor
//
// Decodes a base64-encoded JPEG string (CNN_SIZE×CNN_SIZE) into a
// Float32Array of shape [CNN_SIZE*CNN_SIZE*3] normalized to [0, 1].
//
// Uses jpeg-js for correct JPEG decoding (NOT raw byte iteration).
// NOTE: App.js uses downscaleToTensor() instead (avoids a second async call).
// ─────────────────────────────────────────────────────────────
export function prepareInputTensor(base64Jpeg) {
  const buffer    = base64Decode(base64Jpeg);
  const rawImage  = JPEG.decode(new Uint8Array(buffer), { useTArray: true });
  // rawImage.data is RGBA Uint8Array; width/height should be 128

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
    let maxVal   = -Infinity;
    let maxClass = 0;
    const base = i * numClasses;
    for (let c = 0; c < numClasses; c++) {
      if (outputTensor[base + c] > maxVal) {
        maxVal   = outputTensor[base + c];
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
// downscaleToTensor
//
// Downscales RGBA pixel data directly to a Float32Array tensor
// suitable for TFLite input, using nearest-neighbor sampling.
// Avoids a second ImageManipulator async call.
// ─────────────────────────────────────────────────────────────
export function downscaleToTensor(rgbaData, srcW, srcH, dstW, dstH) {
  const tensor = new Float32Array(dstW * dstH * 3);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    const srcY = Math.floor(y * yRatio);
    for (let x = 0; x < dstW; x++) {
      const srcX = Math.floor(x * xRatio);
      const srcIdx = (srcY * srcW + srcX) * 4;
      const dstIdx = (y * dstW + x) * 3;
      tensor[dstIdx]     = rgbaData[srcIdx]     / 255.0;
      tensor[dstIdx + 1] = rgbaData[srcIdx + 1] / 255.0;
      tensor[dstIdx + 2] = rgbaData[srcIdx + 2] / 255.0;
    }
  }
  return tensor;
}

// ─────────────────────────────────────────────────────────────
// upscaleMaskNearest
//
// Upscales a Uint8Array class mask using nearest-neighbor.
// Used to match the CNN's CNN_SIZE×CNN_SIZE mask to the actual
// display/capture resolution for the daltonization overlay.
// ─────────────────────────────────────────────────────────────
export function upscaleMaskNearest(mask, srcW, srcH, dstW, dstH) {
  const out = new Uint8Array(dstW * dstH);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    const srcY = Math.floor(y * yRatio);
    for (let x = 0; x < dstW; x++) {
      out[y * dstW + x] = mask[srcY * srcW + Math.floor(x * xRatio)];
    }
  }
  return out;
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
  let binary = '';
  const CHUNK = 8192;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  const b64 = btoa(binary);
  return `data:image/jpeg;base64,${b64}`;
}

// ─────────────────────────────────────────────────────────────
// applyCVDSimulation
//
// Like applyDaltonization but stops at simulation — shows what
// a colorblind person actually sees (no error redistribution).
// Used by CVDSimulationScreen.
// ─────────────────────────────────────────────────────────────
export function applyCVDSimulation(rawImageData, mask, cvdType) {
  const pixels       = new Uint8Array(rawImageData.data); // copy
  const confusionSet = CONFUSION_CLASSES[cvdType];
  const SIM          = CVD_SIM[cvdType];
  const numPixels    = rawImageData.width * rawImageData.height;

  for (let i = 0; i < numPixels; i++) {
    if (!confusionSet.has(mask[i])) continue;

    const rIdx = i * 4;
    const rLin = srgbToLinear(pixels[rIdx]     / 255.0);
    const gLin = srgbToLinear(pixels[rIdx + 1] / 255.0);
    const bLin = srgbToLinear(pixels[rIdx + 2] / 255.0);

    // RGB → LMS
    const L = RGB_TO_LMS[0][0] * rLin + RGB_TO_LMS[0][1] * gLin + RGB_TO_LMS[0][2] * bLin;
    const M = RGB_TO_LMS[1][0] * rLin + RGB_TO_LMS[1][1] * gLin + RGB_TO_LMS[1][2] * bLin;
    const S = RGB_TO_LMS[2][0] * rLin + RGB_TO_LMS[2][1] * gLin + RGB_TO_LMS[2][2] * bLin;

    // Simulate CVD (just show what they see — NO error redistribution)
    const Ls = SIM[0][0] * L + SIM[0][1] * M + SIM[0][2] * S;
    const Ms = SIM[1][0] * L + SIM[1][1] * M + SIM[1][2] * S;
    const Ss = SIM[2][0] * L + SIM[2][1] * M + SIM[2][2] * S;

    // LMS → RGB (simulated values directly)
    const rOut = LMS_TO_RGB[0][0] * Ls + LMS_TO_RGB[0][1] * Ms + LMS_TO_RGB[0][2] * Ss;
    const gOut = LMS_TO_RGB[1][0] * Ls + LMS_TO_RGB[1][1] * Ms + LMS_TO_RGB[1][2] * Ss;
    const bOut = LMS_TO_RGB[2][0] * Ls + LMS_TO_RGB[2][1] * Ms + LMS_TO_RGB[2][2] * Ss;

    pixels[rIdx]     = Math.round(linearToSrgb(Math.max(0, Math.min(1, rOut))) * 255);
    pixels[rIdx + 1] = Math.round(linearToSrgb(Math.max(0, Math.min(1, gOut))) * 255);
    pixels[rIdx + 2] = Math.round(linearToSrgb(Math.max(0, Math.min(1, bOut))) * 255);
  }

  return pixels;
}

export function applyDaltonization(rawImageData, mask, cvdType) {
  const pixels       = new Uint8Array(rawImageData.data); // copy
  const confusionSet = CONFUSION_CLASSES[cvdType];
  const SIM          = CVD_SIM[cvdType];
  const ERR          = CVD_ERR_SHIFT[cvdType];

  const numPixels = rawImageData.width * rawImageData.height;

  for (let i = 0; i < numPixels; i++) {
    if (!confusionSet.has(mask[i])) continue; // Leave non-confused pixels untouched

    const rIdx = i * 4;

    // Normalize to [0,1] and gamma-decode (sRGB → linear)
    const rLin = srgbToLinear(pixels[rIdx]     / 255.0);
    const gLin = srgbToLinear(pixels[rIdx + 1] / 255.0);
    const bLin = srgbToLinear(pixels[rIdx + 2] / 255.0);

    // RGB → LMS
    const L = RGB_TO_LMS[0][0] * rLin + RGB_TO_LMS[0][1] * gLin + RGB_TO_LMS[0][2] * bLin;
    const M = RGB_TO_LMS[1][0] * rLin + RGB_TO_LMS[1][1] * gLin + RGB_TO_LMS[1][2] * bLin;
    const S = RGB_TO_LMS[2][0] * rLin + RGB_TO_LMS[2][1] * gLin + RGB_TO_LMS[2][2] * bLin;

    // Simulate what the CVD user sees (missing cone reconstructed)
    const Lsim = SIM[0][0] * L + SIM[0][1] * M + SIM[0][2] * S;
    const Msim = SIM[1][0] * L + SIM[1][1] * M + SIM[1][2] * S;
    const Ssim = SIM[2][0] * L + SIM[2][1] * M + SIM[2][2] * S;

    // Error = original − simulated (what the user cannot perceive)
    const Lerr = L - Lsim;
    const Merr = M - Msim;
    const Serr = S - Ssim;

    // Redistribute error to surviving channels
    const Ldalt = L + ERR[0][0] * Lerr + ERR[0][1] * Merr + ERR[0][2] * Serr;
    const Mdalt = M + ERR[1][0] * Lerr + ERR[1][1] * Merr + ERR[1][2] * Serr;
    const Sdalt = S + ERR[2][0] * Lerr + ERR[2][1] * Merr + ERR[2][2] * Serr;

    // LMS → RGB (linear)
    const rOut = LMS_TO_RGB[0][0] * Ldalt + LMS_TO_RGB[0][1] * Mdalt + LMS_TO_RGB[0][2] * Sdalt;
    const gOut = LMS_TO_RGB[1][0] * Ldalt + LMS_TO_RGB[1][1] * Mdalt + LMS_TO_RGB[1][2] * Sdalt;
    const bOut = LMS_TO_RGB[2][0] * Ldalt + LMS_TO_RGB[2][1] * Mdalt + LMS_TO_RGB[2][2] * Sdalt;

    // Gamma re-encode (linear → sRGB), clamp, write back
    pixels[rIdx]     = Math.round(linearToSrgb(Math.max(0, Math.min(1, rOut))) * 255);
    pixels[rIdx + 1] = Math.round(linearToSrgb(Math.max(0, Math.min(1, gOut))) * 255);
    pixels[rIdx + 2] = Math.round(linearToSrgb(Math.max(0, Math.min(1, bOut))) * 255);
    // Alpha (pixels[rIdx + 3]) stays unchanged
  }

  return pixels; // modified RGBA Uint8Array
}
