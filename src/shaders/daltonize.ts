export const DALTONIZE_SKSL = `
// Daltonization Runtime Shader (SKSL)
// Inputs:
// - u_Image: the input camera frame
// - u_Strength: correction strength [0..1]
// - u_RGB_TO_LMS, u_LMS_TO_RGB: color space transforms
// - u_SIMULATION: CVD simulation matrix in LMS space
// - u_CORRECTION: compensation matrix in LMS space

uniform shader u_Image;
uniform float u_Strength; // Correction strength/gain [0..1]
uniform float3x3 u_RGB_TO_LMS;
uniform float3x3 u_LMS_TO_RGB;
uniform float3x3 u_SIMULATION;
uniform float3x3 u_CORRECTION;
uniform int u_OutputSimulated; // 1 = output simulated, 0 = output corrected

// sRGB <-> Linear helpers
float3 srgbToLinear(float3 c) {
  // IEC 61966-2-1 sRGB EOTF
  float3 cutoff = step(float3(0.04045), c);
  float3 low = c / 12.92;
  float3 high = pow((c + 0.055) / 1.055, 2.4);
  return mix(low, high, cutoff);
}

float3 linearToSrgb(float3 c) {
  float3 cutoff = step(float3(0.0031308), c);
  float3 low = 12.92 * c;
  float3 high = 1.055 * pow(c, 1.0/2.4) - 0.055;
  return mix(low, high, cutoff);
}

half4 main(float2 xy) {
  half4 inC = u_Image.eval(xy);

  // Work in float
  float3 srgb = float3(inC.rgb);
  // Convert to linear
  float3 rgbLin = srgbToLinear(srgb);

  // RGB -> LMS
  float3 lms = u_RGB_TO_LMS * rgbLin;

  // Simulate full (100%) CVD in LMS
  float3 lmsSim = u_SIMULATION * lms;

  // Error in LMS
  float3 err = lms - lmsSim;

  // Compensation (in LMS)
  float3 comp = u_CORRECTION * err;

  // Recombine with adjustable strength
  float3 lmsDaltonized = lms + u_Strength * comp;

  // Back to RGB (linear)
  float3 rgbCorrLin = u_LMS_TO_RGB * lmsDaltonized;
  float3 rgbSimLin = u_LMS_TO_RGB * lmsSim;

  // Choose output: simulated vs corrected
  float3 outLin = (u_OutputSimulated == 1) ? rgbSimLin : rgbCorrLin;
  outLin = clamp(outLin, 0.0, 1.0);

  // Convert to sRGB for display
  float3 srgbOut = linearToSrgb(outLin);

  return half4(half3(srgbOut), inC.a);
}
`;
