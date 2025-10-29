/**
 * On-device daltonization processing for color blindness correction.
 * Implements the same algorithms as the backend but runs locally on the device.
 */

export type CVDType = 'protan' | 'deutan' | 'tritan' | 'normal';

/**
 * Simulation matrices for different types of color blindness
 * Based on Brettel daltonization algorithm
 */
const SIMULATION_MATRICES: Record<CVDType, number[][]> = {
  protan: [
    [0.0, 2.02344, -2.52581],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ],
  deutan: [
    [1.0, 0.0, 0.0],
    [0.494207, 0.0, 1.24827],
    [0.0, 0.0, 1.0],
  ],
  tritan: [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-0.395913, 0.801109, 0.0],
  ],
  normal: [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ],
};

/**
 * Correction matrices for daltonization
 */
const CORRECTION_MATRICES: Record<CVDType, number[][]> = {
  protan: [
    [0.0, 0.7, 0.7],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ],
  deutan: [
    [1.0, 0.0, 0.0],
    [0.7, 0.0, 0.7],
    [0.0, 0.0, 1.0],
  ],
  tritan: [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.7, 0.7, 0.0],
  ],
  normal: [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ],
};

/**
 * Matrix multiplication for a single RGB pixel
 */
function matrixMultiply(rgb: number[], matrix: number[][]): number[] {
  return [
    rgb[0] * matrix[0][0] + rgb[1] * matrix[0][1] + rgb[2] * matrix[0][2],
    rgb[0] * matrix[1][0] + rgb[1] * matrix[1][1] + rgb[2] * matrix[1][2],
    rgb[0] * matrix[2][0] + rgb[1] * matrix[2][1] + rgb[2] * matrix[2][2],
  ];
}

/**
 * Clamp value between 0 and 1
 */
function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/**
 * Apply daltonization correction to a single RGB pixel
 * @param rgb - RGB values normalized to [0, 1]
 * @param cvdType - Type of color vision deficiency
 * @param severity - Correction strength (0-1)
 * @returns Corrected RGB values
 */
export function daltonizePixel(
  rgb: number[],
  cvdType: CVDType,
  severity: number
): number[] {
  if (cvdType === 'normal' || severity === 0) {
    return rgb;
  }

  // Simulate how the color appears to someone with CVD
  const simulated = matrixMultiply(rgb, SIMULATION_MATRICES[cvdType]);

  // Calculate the error (difference between original and simulated)
  const error = [
    rgb[0] - simulated[0],
    rgb[1] - simulated[1],
    rgb[2] - simulated[2],
  ];

  // Apply correction matrix to the error
  const correction = matrixMultiply(error, CORRECTION_MATRICES[cvdType]);

  // Add weighted correction to original
  const corrected = [
    rgb[0] + severity * correction[0],
    rgb[1] + severity * correction[1],
    rgb[2] + severity * correction[2],
  ];

  // Clamp values to valid range
  return corrected.map(clamp);
}

/**
 * Process an entire image buffer
 * @param imageData - RGBA image data (0-255)
 * @param cvdType - Type of color vision deficiency
 * @param severity - Correction strength (0-1)
 * @returns Processed RGBA image data
 */
export function daltonizeImage(
  imageData: Uint8ClampedArray,
  cvdType: CVDType,
  severity: number
): Uint8ClampedArray {
  const output = new Uint8ClampedArray(imageData.length);

  for (let i = 0; i < imageData.length; i += 4) {
    // Normalize RGB to [0, 1]
    const rgb = [
      imageData[i] / 255,
      imageData[i + 1] / 255,
      imageData[i + 2] / 255,
    ];

    // Apply daltonization
    const corrected = daltonizePixel(rgb, cvdType, severity);

    // Convert back to [0, 255]
    output[i] = Math.round(corrected[0] * 255);
    output[i + 1] = Math.round(corrected[1] * 255);
    output[i + 2] = Math.round(corrected[2] * 255);
    output[i + 3] = imageData[i + 3]; // Copy alpha channel
  }

  return output;
}

/**
 * Simulate color blindness (how colors appear to someone with CVD)
 * Useful for the CVD simulation feature
 */
export function simulateCVD(
  imageData: Uint8ClampedArray,
  cvdType: CVDType
): Uint8ClampedArray {
  const output = new Uint8ClampedArray(imageData.length);

  for (let i = 0; i < imageData.length; i += 4) {
    // Normalize RGB to [0, 1]
    const rgb = [
      imageData[i] / 255,
      imageData[i + 1] / 255,
      imageData[i + 2] / 255,
    ];

    // Simulate CVD
    const simulated = matrixMultiply(rgb, SIMULATION_MATRICES[cvdType]);

    // Convert back to [0, 255]
    output[i] = Math.round(clamp(simulated[0]) * 255);
    output[i + 1] = Math.round(clamp(simulated[1]) * 255);
    output[i + 2] = Math.round(clamp(simulated[2]) * 255);
    output[i + 3] = imageData[i + 3]; // Copy alpha channel
  }

  return output;
}

/**
 * Identify the dominant color in a region
 * Used for the color identifier feature
 */
export function identifyColor(rgb: number[]): {
  name: string;
  hex: string;
  rgb: number[];
} {
  const [r, g, b] = rgb;

  // Convert to hex
  const hex = `#${Math.round(r).toString(16).padStart(2, '0')}${Math.round(g)
    .toString(16)
    .padStart(2, '0')}${Math.round(b).toString(16).padStart(2, '0')}`;

  // Simple color name detection
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;

  let name = 'Unknown';

  if (delta < 20) {
    // Grayscale
    if (max < 50) name = 'Black';
    else if (max > 200) name = 'White';
    else name = 'Gray';
  } else {
    // Chromatic colors
    if (r === max) {
      if (g > b) name = g > r * 0.7 ? 'Yellow' : 'Orange';
      else name = 'Red';
    } else if (g === max) {
      if (r > b) name = 'Yellow-Green';
      else name = b > g * 0.5 ? 'Cyan' : 'Green';
    } else {
      if (r > g) name = 'Magenta';
      else name = g > b * 0.7 ? 'Cyan' : 'Blue';
    }
  }

  return { name, hex, rgb: [r, g, b] };
}

/**
 * Calculate color difference for better correction
 */
export function colorDistance(rgb1: number[], rgb2: number[]): number {
  const dr = rgb1[0] - rgb2[0];
  const dg = rgb1[1] - rgb2[1];
  const db = rgb1[2] - rgb2[2];
  return Math.sqrt(dr * dr + dg * dg + db * db);
}
