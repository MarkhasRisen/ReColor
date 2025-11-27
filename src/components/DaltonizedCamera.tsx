import React, { useCallback, useMemo } from 'react';
import { Platform, StyleSheet, View } from 'react-native';
import { useWindowDimensions } from 'react-native';
import { Camera, useCameraDevice, useFrameProcessor } from 'react-native-vision-camera';
import { runOnJS } from 'react-native-reanimated';
import type { SharedValue } from 'react-native-reanimated';
import { Canvas, Fill, ImageShader, RuntimeShader, Skia, Image, Rect, Mask } from '@shopify/react-native-skia';
import PerfHUD from './PerfHUD';
import { DALTONIZE_SKSL } from '../shaders/daltonize';
import { MATRICES_BY_TYPE, type CVDType } from '../constants/matrices';

// Optional TS augmentation for a Skia integration on Frame Processors
// This assumes a VisionCamera plugin that exposes frame.toSkImage()
// eslint-disable-next-line @typescript-eslint/no-namespace
declare module 'react-native-vision-camera' {
  interface Frame {
    // Returns a JSI-backed SkImage (platform plugin required)
    toSkImage?: () => any | null;
  }
}

export type CvdMatrices = {
  RGB_TO_LMS: number[]; // 9 elements
  LMS_TO_RGB: number[]; // 9 elements
  SIMULATION: number[]; // 9 elements
  CORRECTION: number[]; // 9 elements
};

export type DaltonizedCameraProps = {
  strength: SharedValue<number>; // Reanimated shared value in [0..1]
  matrices?: CvdMatrices; // If provided, overrides cvdType
  cvdType?: CVDType; // Defaults to 'protan'
  fps?: number; // Desired frame processor fps (default 30)
  style?: any;
  showPerfHUD?: boolean;
  compare?: boolean; // show split comparison overlay
  compareSplit?: number; // 0..1 position of split (left portion shows original)
};

export const DaltonizedCamera = ({
  strength,
  matrices,
  cvdType = 'protan',
  fps: _fps = 30,
  style,
  showPerfHUD = __DEV__,
  compare = false,
  compareSplit = 0.5,
}: DaltonizedCameraProps) => {
  const device = useCameraDevice('back');
  const { width, height } = useWindowDimensions();

  const effect = useMemo(() => Skia.RuntimeEffect.Make(DALTONIZE_SKSL)!, []);

  // Skia uniforms as SkiaValues (live on UI thread and avoid React re-renders)
  const { useValue } = require('@shopify/react-native-skia');
  const uStrength = useValue(0.5);
  const initial = useMemo(() => (matrices ?? MATRICES_BY_TYPE[cvdType as CVDType]) || MATRICES_BY_TYPE['protan'], [matrices, cvdType]);
  const uRGB2LMS = useValue(new Float32Array(initial.RGB_TO_LMS));
  const uLMS2RGB = useValue(new Float32Array(initial.LMS_TO_RGB));
  const uSim = useValue(new Float32Array(initial.SIMULATION));
  const uCorr = useValue(new Float32Array(initial.CORRECTION));
  const uOutputSim = useValue(0); // always corrected in camera view

  // Image coming from VisionCamera frame processor (SkImage JSI object)
  const skImage: any = useValue(null);

  // Bridge Reanimated SharedValue -> SkiaValue without React setState
  // We intentionally use a small range clamp on UI thread
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const { useSharedValueEffect } = require('@shopify/react-native-skia');
  // Keep it dynamic to avoid type issues if versions differ.
  if (useSharedValueEffect) {
    useSharedValueEffect(() => {
      const v = Math.min(1, Math.max(0, strength.value));
      uStrength.current = v;
    }, strength);
  } else {
    // Fallback to React effect (less ideal, but functional)
    React.useEffect(() => {
      const id = setInterval(() => {
        // Pull occasionally; better replaced by useSharedValueEffect
        uStrength.current = Math.min(1, Math.max(0, strength.value));
      }, 16);
      return () => clearInterval(id);
    }, [strength, uStrength]);
  }

  // Allow updating matrices at runtime (e.g., switching CVD type)
  React.useEffect(() => {
    const m = matrices ?? MATRICES_BY_TYPE[cvdType as CVDType] ?? MATRICES_BY_TYPE['protan'];
    uRGB2LMS.current = new Float32Array(m.RGB_TO_LMS);
    uLMS2RGB.current = new Float32Array(m.LMS_TO_RGB);
    uSim.current = new Float32Array(m.SIMULATION);
    uCorr.current = new Float32Array(m.CORRECTION);
  }, [cvdType, matrices, uRGB2LMS, uLMS2RGB, uSim, uCorr]);

  // Receive frames from camera and turn them into a SkImage on the JS side
  const setImage = useCallback((img: any | null) => {
    skImage.current = img;
  }, [skImage]);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    // Requires a VisionCamera plugin that exposes toSkImage()
    try {
      const img = frame.toSkImage ? frame.toSkImage() : null;
      if (img) {
        runOnJS(setImage)(img);
      }
    } catch (_) {
      // Swallow errors if plugin not present
    }
  }, [setImage]);

  // Compose uniforms for the runtime shader
  const uniforms = useMemo(() => ({
    u_Strength: uStrength,
    u_RGB_TO_LMS: uRGB2LMS,
    u_LMS_TO_RGB: uLMS2RGB,
    u_SIMULATION: uSim,
    u_CORRECTION: uCorr,
    u_OutputSimulated: uOutputSim,
  }), [uCorr, uLMS2RGB, uRGB2LMS, uSim, uStrength, uOutputSim]);

  if (!device) {
    return <View style={[styles.container, style]} />;
  }

  return (
    <View style={[styles.container, style]}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        pixelFormat={Platform.OS === 'ios' ? 'rgb' : 'yuv'}
        photo={false}
        video={false}
        frameProcessor={frameProcessor}
      />

      {/* Skia overlay with Runtime Shader sampling the camera SkImage */}
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill>
          <RuntimeShader source={effect} uniforms={uniforms}>
            {skImage.current && (
              <ImageShader
                image={skImage.current}
                fit="cover"
                x={0}
                y={0}
                width={width}
                height={height}
              />
            )}
          </RuntimeShader>
        </Fill>

        {/* Comparison overlay: left side original (clipped) over processed */}
        {compare && skImage.current ? (
          <>
            <Mask
              mask={<Rect x={0} y={0} width={width * Math.max(0, Math.min(1, compareSplit))} height={height} color="white" />}
            >
              <Image
                image={skImage.current}
                fit="cover"
                x={0}
                y={0}
                width={width}
                height={height}
              />
            </Mask>
            {/* Divider line */}
            <Rect x={width * Math.max(0, Math.min(1, compareSplit)) - 0.5} y={0} width={1} height={height} color="white" />
          </>
        ) : null}
      </Canvas>
      {showPerfHUD ? <PerfHUD /> : null}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
});

export default DaltonizedCamera;
