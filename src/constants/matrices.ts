// =============================================================================
// RECOLOR MATRIX LIBRARY
// Based on Viénot, Brettel & Mollon (1999) and Hunt-Pointer-Estevez (HPE)
// =============================================================================
//
// CRITICAL NOTE FOR REACT NATIVE SKIA / GLSL:
// GLSL matrices are COLUMN-MAJOR. 
// The arrays below are defined such that the first 3 numbers represent COLUMN 1,
// the next 3 represent COLUMN 2, etc.
//
// Effectively, these are the TRANSPOSES of the standard math matrices found in papers.
// This ensures they work instantly in the shader without runtime processing.

// 1. RGB to LMS (Hunt-Pointer-Estevez)
// Standard Row-Major:
// [17.8824, 43.5161, 4.11935]
// [ 3.45565, 27.1554, 3.86714]
// [ 0.02996,  0.18431, 1.46709]
export const RGB_TO_LMS: number[] = [
  17.8824,  3.45565,  0.02996, // Column 1
  43.5161, 27.1554,   0.18431, // Column 2
   4.11935, 3.86714,  1.46709  // Column 3
];

// 2. LMS to RGB (Inverse of above)
export const LMS_TO_RGB: number[] = [
   0.0809, -0.1305,  0.1167, // Column 1
  -0.0102,  0.0540, -0.1136, // Column 2
  -0.0004, -0.0041,  0.6935  // Column 3
];

// =============================================================================
// PROTANOPIA (Red-Blind)
// =============================================================================

// Simulation: Reduces L-cone information to a mix of M and S.
// Standard Row-Major (Viénot 1999):
// [0.0, 2.02344, -2.52581]
// [0.0, 1.0,      0.0]
// [0.0, 0.0,      1.0]
export const SIM_PROTAN: number[] = [
  0.0,     0.0, 0.0,  // Column 1 (Red input is zeroed out/replaced)
  2.02344, 1.0, 0.0,  // Column 2
 -2.52581, 0.0, 1.0   // Column 3
];

// Correction: Shifts error found in L/M into S (Blue)
// Standard Row-Major:
// [0.0, 0.0, 0.0]
// [0.7, 1.0, 0.0]
// [0.7, 0.0, 1.0]
export const CORR_PROTAN: number[] = [
  0.0, 0.7, 0.7, // Column 1
  0.0, 1.0, 0.0, // Column 2
  0.0, 0.0, 1.0  // Column 3
];

// =============================================================================
// DEUTERANOPIA (Green-Blind)
// =============================================================================

// Simulation: Reduces M-cone information to a mix of L and S.
// Standard Row-Major (Viénot 1999):
// [1.0,      0.0, 0.0]
// [0.494207, 0.0, 1.24827]
// [0.0,      0.0, 1.0]
export const SIM_DEUTAN: number[] = [
  1.0, 0.494207, 0.0, // Column 1
  0.0, 0.0,      0.0, // Column 2 (Green input is replaced)
  0.0, 1.24827,  1.0  // Column 3
];

// Correction: Shifts error found in L/M into S (Blue)
export const CORR_DEUTAN: number[] = [
  1.0, 0.0, 0.0, // Column 1
  0.7, 0.0, 0.7, // Column 2 (Shift Green Error to R/B)
  0.0, 0.0, 1.0  // Column 3
];

// =============================================================================
// TRITANOPIA (Blue-Blind)
// =============================================================================

// Simulation: Reduces S-cone information to a mix of L and M.
// Standard Row-Major (Brettel 1997):
// [1.0,       0.0,      0.0]
// [0.0,       1.0,      0.0]
// [-0.395913, 0.801109, 0.0]
export const SIM_TRITAN: number[] = [
   1.0,       0.0,      -0.395913, // Column 1
   0.0,       1.0,       0.801109, // Column 2
   0.0,       0.0,       0.0       // Column 3 (Blue input replaced)
];

// Correction: Shifts error found in S into L/M (Red/Green)
export const CORR_TRITAN: number[] = [
  1.0, 0.0, 0.0, // Column 1
  0.0, 1.0, 0.0, // Column 2
  0.7, 0.7, 0.0  // Column 3 (Shift Blue Error to R/G)
];

// =============================================================================
// EXPORTS
// =============================================================================

export type CVDType = 'protan' | 'deutan' | 'tritan';

export const MATRICES_BY_TYPE: Record<CVDType, {
  RGB_TO_LMS: number[];
  LMS_TO_RGB: number[];
  SIMULATION: number[];
  CORRECTION: number[];
}> = {
  protan: {
    RGB_TO_LMS, LMS_TO_RGB, SIMULATION: SIM_PROTAN, CORRECTION: CORR_PROTAN
  },
  deutan: {
    RGB_TO_LMS, LMS_TO_RGB, SIMULATION: SIM_DEUTAN, CORRECTION: CORR_DEUTAN
  },
  tritan: {
    RGB_TO_LMS, LMS_TO_RGB, SIMULATION: SIM_TRITAN, CORRECTION: CORR_TRITAN
  },
};
