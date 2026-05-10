import { Dimensions } from 'react-native';
import { Skia } from '@shopify/react-native-skia';

const { width } = Dimensions.get('window');

export const DISPLAY_SIZE = Math.min(720, width);
export const CAPTURE_SIZE = Math.min(1024, width);

// Gamma-aware CVD simulation shader (SkSL).
// Pipeline: per-channel calibration → sRGB → linear → CVD matrix → linear → sRGB.
// `calib` is the per-channel scalar from Settings → Camera Calibration.
// Pass (1,1,1) when no manual calibration is set.
export const CVD_SHADER_SOURCE = `
uniform shader contents;
uniform half3 row0;
uniform half3 row1;
uniform half3 row2;
uniform half3 calib;

half4 main(float2 coord) {
  half4 c = contents.eval(coord);
  half3 calibrated = clamp(c.rgb * calib, half3(0.0), half3(1.0));
  half3 lin = pow(calibrated, half3(2.2));
  half3 sim = half3(dot(row0, lin), dot(row1, lin), dot(row2, lin));
  sim = clamp(sim, half3(0.0), half3(1.0));
  return half4(pow(sim, half3(0.4545)), c.a);
}
`;
export const CVD_EFFECT = Skia.RuntimeEffect.Make(CVD_SHADER_SOURCE);

// ─────────────────────────────────────────────────────────────────────────
// Daltonization shader (Brettel/Fidaner) — GPU port of applyDaltonization.
// Pipeline (per pixel):
//   1. calibrate (per-channel multiply from Settings)
//   2. sRGB → linear (γ 2.2)
//   3. simulate CVD perception with sim matrix (sim0, sim1, sim2)
//   4. error = original_linear − simulated
//   5. redistribute error into surviving channels via err matrix
//   6. linear → sRGB
//   7. blend with calibrated original by `intensity` (0..1)
//
// Uniforms must be supplied by `getDaltonizationUniforms(cvdType, calib, intensity)`.
// ─────────────────────────────────────────────────────────────────────────
export const DALTONIZATION_SHADER_SOURCE = `
uniform shader contents;
uniform half3 sim0;
uniform half3 sim1;
uniform half3 sim2;
uniform half3 err0;
uniform half3 err1;
uniform half3 err2;
uniform half3 calib;
uniform half intensity;

half4 main(float2 coord) {
  half4 c = contents.eval(coord);
  half3 calibrated = clamp(c.rgb * calib, half3(0.0), half3(1.0));
  half3 lin = pow(calibrated, half3(2.2));

  half3 sim = half3(dot(sim0, lin), dot(sim1, lin), dot(sim2, lin));
  half3 err = lin - sim;
  half3 outLin = lin + half3(dot(err0, err), dot(err1, err), dot(err2, err));
  outLin = clamp(outLin, half3(0.0), half3(1.0));

  half3 enhanced = pow(outLin, half3(0.4545));
  half3 blended = mix(calibrated, enhanced, intensity);
  return half4(blended, c.a);
}
`;
export const DALTONIZATION_EFFECT = Skia.RuntimeEffect.Make(
  DALTONIZATION_SHADER_SOURCE,
);

// ─────────────────────────────────────────────────────────────────────────
// Hue Rotation shader — GPU port of applyHueRotation.
// HSV-band rotation with saturation gate and linear taper.
//
// Uniforms must be supplied by `getHueRotationUniforms(cvdType, calib, intensity)`.
// ─────────────────────────────────────────────────────────────────────────
export const HUE_ROTATION_SHADER_SOURCE = `
uniform shader contents;
uniform half3 calib;
uniform half center;
uniform half range;
uniform half shift;
uniform half satMin;
uniform half intensity;

half3 rgb2hsv(half3 c) {
  half cmax = max(c.r, max(c.g, c.b));
  half cmin = min(c.r, min(c.g, c.b));
  half d = cmax - cmin;
  half h = 0.0;
  if (d > 0.0) {
    if (cmax == c.r) {
      h = mod((c.g - c.b) / d, 6.0);
    } else if (cmax == c.g) {
      h = (c.b - c.r) / d + 2.0;
    } else {
      h = (c.r - c.g) / d + 4.0;
    }
    h = h * 60.0;
    if (h < 0.0) h += 360.0;
  }
  half s = cmax == 0.0 ? 0.0 : d / cmax;
  return half3(h, s, cmax);
}

half3 hsv2rgb(half3 hsv) {
  half h = hsv.x;
  half s = hsv.y;
  half v = hsv.z;
  half c = v * s;
  half hp = h / 60.0;
  half x = c * (1.0 - abs(mod(hp, 2.0) - 1.0));
  half3 rgb;
  if (hp < 1.0) rgb = half3(c, x, 0.0);
  else if (hp < 2.0) rgb = half3(x, c, 0.0);
  else if (hp < 3.0) rgb = half3(0.0, c, x);
  else if (hp < 4.0) rgb = half3(0.0, x, c);
  else if (hp < 5.0) rgb = half3(x, 0.0, c);
  else rgb = half3(c, 0.0, x);
  return rgb + half3(v - c);
}

half4 main(float2 coord) {
  half4 c = contents.eval(coord);
  half3 calibrated = clamp(c.rgb * calib, half3(0.0), half3(1.0));
  half3 hsv = rgb2hsv(calibrated);
  half h = hsv.x;
  half s = hsv.y;
  half v = hsv.z;

  half3 result = calibrated;
  if (s >= satMin) {
    half diff = abs(h - center);
    half dist = diff > 180.0 ? 360.0 - diff : diff;
    if (dist <= range) {
      half weight = 1.0 - dist / range;
      half newH = mod(h + shift * weight + 360.0, 360.0);
      result = hsv2rgb(half3(newH, s, v));
    }
  }

  half3 blended = mix(calibrated, result, intensity);
  return half4(blended, c.a);
}
`;
export const HUE_ROTATION_EFFECT = Skia.RuntimeEffect.Make(
  HUE_ROTATION_SHADER_SOURCE,
);
