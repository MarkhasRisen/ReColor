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
// prepareInputTensor
//
// Decodes a base64-encoded JPEG string (128×128) into a
// Float32Array of shape [128*128*3] normalized to [0, 1].
//
// Uses jpeg-js for correct JPEG decoding (NOT raw byte iteration).
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
// 128*128*NUM_CLASSES) to a per-pixel class index (Uint8Array
// of length 128*128) via argmax.
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
// encodeToDataUri
//
// Encodes a modified RGBA Uint8Array back to a JPEG data URI
// suitable for React Native's <Image source={{ uri }} />.
// ─────────────────────────────────────────────────────────────
export function encodeToDataUri(rgbaPixels, width, height, quality = 85) {
  const encoded = JPEG.encode({ data: rgbaPixels, width, height }, quality);
  // Convert raw byte buffer to base64 manually (no native Buffer available in RN)
  const bytes = encoded.data;
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  const b64 = btoa(binary);
  return `data:image/jpeg;base64,${b64}`;
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
