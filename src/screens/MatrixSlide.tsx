import React, { useMemo } from 'react';
import { View, Text, StyleSheet, useWindowDimensions, ScrollView, Pressable, Alert } from 'react-native';
import { Canvas, RuntimeShader, ImageShader, Skia } from '@shopify/react-native-skia';
import { DALTONIZE_SKSL } from '../shaders/daltonize';
import { MATRICES_BY_TYPE, type CVDType } from '../constants/matrices';

// Two sources: provided default and optional custom
const PROVIDED = require('../assets/colorful-flower-1.jpg');
const CUSTOM = require('../assets/source.jpg');

export default function MatrixSlide() {
  const effect = useMemo(() => Skia.RuntimeEffect.Make(DALTONIZE_SKSL)!, []);
  const { useImage, useValue } = require('@shopify/react-native-skia');
  const providedImg = useImage(PROVIDED);
  const customImg = useImage(CUSTOM);
  const [useProvided, setUseProvided] = React.useState(true);
  const img = useProvided ? providedImg : customImg;

  const { width: sw } = useWindowDimensions();

  // Desired output size in pixels (scaled by device width for on-screen preview)
  const OUT_W = Math.min(sw, 1200);
  const COLS = 3;
  const ROWS = 2;
  const GAP = 12;
  const PANEL_W = Math.floor((OUT_W - GAP * (COLS - 1)) / COLS);
  const PANEL_H = Math.floor(PANEL_W * 0.75);
  const OUT_H = ROWS * PANEL_H + (ROWS - 1) * GAP;

  const makeUniforms = (cvd: CVDType, mode: 'sim' | 'corr') => {
    const m = MATRICES_BY_TYPE[cvd];
    const u_Strength = useValue(mode === 'corr' ? 1.0 : 0.0);
    const u_OutputSimulated = useValue(mode === 'sim' ? 1 : 0);
    const u_RGB_TO_LMS = useValue(new Float32Array(m.RGB_TO_LMS));
    const u_LMS_TO_RGB = useValue(new Float32Array(m.LMS_TO_RGB));
    const u_SIMULATION = useValue(new Float32Array(m.SIMULATION));
    const u_CORRECTION = useValue(new Float32Array(m.CORRECTION));
    return { u_Strength, u_OutputSimulated, u_RGB_TO_LMS, u_LMS_TO_RGB, u_SIMULATION, u_CORRECTION };
  };

  const U = {
    protanSim: makeUniforms('protan', 'sim'),
    deutanSim: makeUniforms('deutan', 'sim'),
    tritanSim: makeUniforms('tritan', 'sim'),
    protanCorr: makeUniforms('protan', 'corr'),
    deutanCorr: makeUniforms('deutan', 'corr'),
    tritanCorr: makeUniforms('tritan', 'corr'),
  } as const;

  const labels = [
    'Protanopia (Simulated)',
    'Deuteranopia (Simulated)',
    'Tritanopia (Simulated)',
    'Protanopia (Corrected)',
    'Deuteranopia (Corrected)',
    'Tritanopia (Corrected)'
  ];

  const uniformGrid = [
    U.protanSim,
    U.deutanSim,
    U.tritanSim,
    U.protanCorr,
    U.deutanCorr,
    U.tritanCorr,
  ];

  const onSave = async () => {
    Alert.alert('Export', 'For a high-res export, capture a screenshot or integrate react-native-fs + Skia makeImageSnapshot to save PNG from the Canvas. I can wire that up if you want.');
  };

  if (!img) {
    return (
      <ScrollView contentContainerStyle={styles.root} style={{ flex: 1 }}>
        <View style={{ width: OUT_W, height: OUT_H, alignItems: 'center', justifyContent: 'center', backgroundColor: 'white' }}>
          <Text style={{ color: '#222', fontWeight: '600', textAlign: 'center', paddingHorizontal: 24 }}>
            {useProvided ? 'Loading provided image…' : 'Add your custom image at\nsrc/assets/source.jpg'}
          </Text>
        </View>
      </ScrollView>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.root} style={{ flex: 1 }}>
      <View style={{ width: OUT_W, height: OUT_H }}>
        <Canvas style={{ width: OUT_W, height: OUT_H, backgroundColor: 'white' }}>
          {uniformGrid.map((uni, idx) => {
            const row = Math.floor(idx / COLS);
            const col = idx % COLS;
            const x = col * (PANEL_W + GAP);
            const y = row * (PANEL_H + GAP);
            return (
              <RuntimeShader key={idx} source={effect} uniforms={uni} x={x} y={y} width={PANEL_W} height={PANEL_H}>
                <ImageShader image={img} fit="cover" x={0} y={0} width={PANEL_W} height={PANEL_H} />
              </RuntimeShader>
            );
          })}
        </Canvas>

        {/* RN overlay labels for clarity; for export, I can move to Skia text with a font asset */}
        {labels.map((t, idx) => {
          const row = Math.floor(idx / COLS);
          const col = idx % COLS;
          const x = col * (PANEL_W + GAP) + 8;
          const y = row * (PANEL_H + GAP) + 8;
          return (
            <View key={t} style={[styles.labelWrap, { left: x, top: y }]}> 
              <Text style={styles.labelText}>{t}</Text>
            </View>
          );
        })}
      </View>

      <Pressable onPress={onSave} style={styles.button}>
        <Text style={styles.buttonText}>Save PNG</Text>
      </Pressable>

      <View style={{ height: 12 }} />
      <Pressable onPress={() => setUseProvided((v: boolean) => !v)} style={[styles.button, { backgroundColor: '#2c2c2c' }]}>
        <Text style={styles.buttonText}>{useProvided ? 'Using: colorful-flower-1.jpg (tap to switch to source.jpg)' : 'Using: source.jpg (tap to switch to colorful-flower-1.jpg)'}</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 24,
    backgroundColor: '#f0f0f0'
  },
  label: {
    position: 'absolute'
  },
  labelWrap: {
    position: 'absolute',
    backgroundColor: 'rgba(0,0,0,0.45)',
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  labelText: {
    color: 'white',
    fontWeight: '700',
  },
  button: {
    marginTop: 16,
    backgroundColor: '#111',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
  },
  buttonText: {
    color: 'white',
    fontWeight: '600'
  }
});
