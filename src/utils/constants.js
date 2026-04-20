import { Dimensions } from 'react-native';
import { Skia } from '@shopify/react-native-skia';

const { width } = Dimensions.get('window');

export const DISPLAY_SIZE = Math.min(720, width);
export const CAPTURE_SIZE = Math.min(1024, width);

// Gamma-aware CVD simulation shader (SkSL).
// Applies sRGB → linear → CVD matrix → linear → sRGB on the GPU.
export const CVD_SHADER_SOURCE = `
uniform shader contents;
uniform half3 row0;
uniform half3 row1;
uniform half3 row2;

half4 main(float2 coord) {
  half4 c = contents.eval(coord);
  half3 lin = pow(c.rgb, half3(2.2));
  half3 sim = half3(dot(row0, lin), dot(row1, lin), dot(row2, lin));
  sim = clamp(sim, half3(0.0), half3(1.0));
  return half4(pow(sim, half3(0.4545)), c.a);
}
`;
export const CVD_EFFECT = Skia.RuntimeEffect.Make(CVD_SHADER_SOURCE);
