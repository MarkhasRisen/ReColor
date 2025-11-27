# Daltonization – Thesis Artifacts & Figure Generator

High-performance, GPU-accelerated daltonization for live camera **plus** a set of
CPU-based export scripts for generating all thesis figures.

This repository includes:
- A Skia Runtime Shader in SKSL performing LMS-based daltonization (with sRGB linearization).
- A `DaltonizedCamera` component that renders the device camera and overlays a Skia canvas using the shader.
- CVD matrices (Protan/Deutan/Tritan) for simulation and correction.
- Node/Sharp scripts under `scripts/` to regenerate every figure used in the thesis
  (2×3 matrix, 1×3 strips, research grid, algorithm flow diagram).

## 1. Requirements (Live App)

- React Native 0.72+ (or Expo with custom dev client)
- react-native-vision-camera (latest)
- @shopify/react-native-skia (latest)
- react-native-reanimated (latest)
- A Vision Camera plugin that exposes `frame.toSkImage()` from the frame processor. This is required to feed the live camera frame to Skia at 30+ FPS on the UI/JSI thread without React re-renders.

Note: The code compiles even without this plugin, but live GPU filtering requires it. Without the plugin, the overlay won’t receive the camera frame as a SkImage.

For the **figure-generation scripts** only, you just need:

```sh
npm install
```

The scripts run in Node and do not require a React Native runtime.

## 2. Shader Overview (Algorithm)

The shader operates per pixel:

1. Convert sRGB -> linear RGB.
2. RGB -> LMS via uniform `u_RGB_TO_LMS`.
3. Simulate full (100%) CVD in LMS via uniform `u_SIMULATION`.
4. Error = Original LMS − Simulated LMS.
5. Compensation = `u_CORRECTION * Error`.
6. Recombine = LMS + `u_Strength * Compensation`.
7. LMS -> RGB via `u_LMS_TO_RGB`.
8. Convert linear RGB -> sRGB and clamp.

Source: `src/shaders/daltonize.ts` (`DALTONIZE_SKSL`). See also `src/shaders/daltonize.sksl` for a standalone SKSL version.

### 📝 Methodology & Technical Rationale

The ReColor algorithm uses a custom fragment shader approach to ensure high performance and color accuracy. Our methodology adheres to the following principles:

| Principle | Implementation | Rationale for Thesis |
| :--- | :--- | :--- |
| Color Space Accuracy | sRGB Linearization: The shader includes explicit sRGB EOTF/OETF functions to convert the camera's gamma-corrected input to linear light before all matrix operations. | Ensures the Daltonization math (which is linear) is physically correct, avoiding washed-out colors and mathematical error. |
| Scientific Standard | HPE LMS Matrices: We use the Hunt-Pointer-Estevez (HPE) based LMS transformation matrices (as published by Viénot et al., 1999) for simulation. | Provides a robust, academically verifiable foundation for the color vision model. |
| GPU Optimization | Column-Major Matrices: All `float3x3` matrices are transposed in the TypeScript layer to match the Column-Major convention of SKSL/GLSL. | Maximizes GPU efficiency by eliminating runtime CPU matrix transposition, a critical performance choice for mobile real-time video. |
| Correction Model | Full Simulation + Gain Control: The algorithm simulates full Dichromacy (100% severity) to derive the maximum possible error vector. The single `u_Strength` slider then applies a gain factor to this correction vector. | Provides the user with a single, clear control that scales assistance strength without introducing confusing, mathematically redundant compounding effects. |
| System Constraint Aware | Calibration Guard: The UI enforces maximum screen brightness and educates the user to disable protected OS color layers (like Night Shift/True Tone) that cannot be controlled programmatically. | Proves the software is aware of platform security boundaries and manages the physical display environment for reliable color output. |

## 3. Live Camera Component API

`src/components/DaltonizedCamera.tsx`

Props:
- `strength: SharedValue<number>` – Reanimated shared value in [0..1]
- `cvdType?: 'protan' | 'deutan' | 'tritan'` – selects CVD matrices (default 'protan')
- `matrices?: { RGB_TO_LMS, LMS_TO_RGB, SIMULATION, CORRECTION }` – overrides `cvdType` when provided
- `style?: ViewStyle`
- `showPerfHUD?: boolean` – show a tiny FPS overlay (defaults to `true` in dev)
- `compare?: boolean` – overlay a split-view comparison (original vs daltonized)
- `compareSplit?: number` – position of split (0..1, left portion shows original)

Internals:
- Renders `<Camera />` for the preview.
- Uses a frame processor to convert frames to a `SkImage` (via `frame.toSkImage()`), then renders a full-screen `<Canvas>` with a `RuntimeShader` sampling from an `<ImageShader>`.
- Uses `useSharedValueEffect` (from Skia) to map the Reanimated `strength` SharedValue directly to the `u_Strength` uniform on the UI thread, avoiding React re-renders.

### Compare Mode

Enable a live split-view to show the unprocessed camera on the left and the daltonized output on the right. Set `compare` to true and control the split with `compareSplit`:

```tsx
<DaltonizedCamera
  strength={strength}
  cvdType={cvdType}
  compare
  compareSplit={0.4}
/>
```

Implementation details:
- The shader-rendered (daltonized) image fills the screen.
- The original image is drawn on top and clipped to `[0, split * width]` using a Skia `Mask` with a `Rect`.
- A thin white divider marks the boundary for clear visual comparison.

## 4. Matrices

`src/constants/matrices.ts` provides:
- HPE RGB↔LMS transforms for linearized sRGB (column-major uniforms for SKSL)
- Protanopia, Deutanopia, Tritanopia simulation matrices
- Matching correction matrices to redistribute lost cone signals
- `MATRICES_BY_TYPE` and `CVDType` helpers to switch quickly

## 5. Usage Example (Live App)

```tsx
import React from 'react';
import { SafeAreaView, View, Text } from 'react-native';
import Slider from '@react-native-community/slider';
import { useSharedValue } from 'react-native-reanimated';
import { DaltonizedCamera } from './src/components/DaltonizedCamera';
import MatrixSlide from './src/screens/MatrixSlide';

export default function App() {
  const strength = useSharedValue(0.6);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: 'black' }}>
      <DaltonizedCamera strength={strength} cvdType={'protan'} showPerfHUD />

      {/* UI overlay for strength control */}
      <View style={{ position: 'absolute', left: 12, right: 12, bottom: 24 }}>
        <Text style={{ color: 'white', marginBottom: 8 }}>Strength</Text>
        <Slider
          minimumValue={0}
          maximumValue={1}
          value={strength.value}
          step={0.01}
          onValueChange={(v) => (strength.value = v)}
          minimumTrackTintColor="#6cf"
          maximumTrackTintColor="#888"
          thumbTintColor="#6cf"
        />
      </View>
    </SafeAreaView>
  );
}
```

## 6. Thesis Figure Generation (scripts/export_* )

All heavy PNG outputs used in the thesis are intentionally **not** committed to git.
Instead, regenerate them on demand using the scripts below. By default, outputs are
written to the `out/` directory.

### 6.1 2×3 Matrix: Simulation (Row 1) + Correction (Row 2)

Six panels: Protan/Deutan/Tritan simulated on top row, corrected on bottom row.

```sh
npm run export:matrix
# => out/matrix.png
```

Optional sizing:

```sh
node scripts/export_matrix.js --size 3300x2550
node scripts/export_matrix.js --letter --orientation portrait --dpi 300
```

### 6.2 1×3 Simulation Strip

```sh
npm run export:strip
# => out/strip.png
```

### 6.3 1×3 Corrected Strip

```sh
npm run export:corrected:strip
# => out/corrected_strip.png
```

Custom strength / sizing:

```sh
node scripts/export_corrected_strip.js --strength 0.8 --output out/corrected_strip_s0_8.png
node scripts/export_corrected_strip.js --letter --orientation landscape --dpi 300
```

### 6.4 Individual Corrected Images (Per CVD Type)

```sh
npm run export:corrected
# => out/protan_corrected.png
# => out/deutan_corrected.png
# => out/tritan_corrected.png
```

### 6.5 Research-Style Grid (Severity vs. Strength)

Multi-row grid showing simulation severities (`s`) and correction strengths (`k`)
for Protan/Deutan/Tritan:

```sh
npm run export:grid
# => out/research_grid.png
```

Custom levels and sizing:

```sh
node scripts/export_research_grid.js --sim 0.25,0.5,0.75,1.0 --corr 0.25,0.5,0.75,1.0
node scripts/export_research_grid.js --letter --orientation portrait --dpi 300
```

### 6.6 Algorithm Flow Diagram

Thesis-grade flow diagram for the full pipeline
(Linearization → LMS Transform → Simulation → Error → Correction):

```sh
npm run export:flow
# => out/flow_diagram.svg
# => out/flow_diagram.png
```

### 6.7 Combined 2×3 Matrix from Separate Rows (Optional)

When generating rows separately:

```sh
node scripts/export_strip.js --output out/row_simulated.png
node scripts/export_corrected_strip.js --output out/row_corrected.png --strength 1.0
npm run export:combine
# => out/matrix_combined.png
```

## 7. Installation Notes (React Native App)

1. Install packages:

```sh
# Vision Camera + Reanimated + Skia
npm install react-native-vision-camera react-native-reanimated @shopify/react-native-skia @react-native-community/slider
# Calibration helpers: brightness + keep-awake
npm install react-native-screen-brightness react-native-keep-awake
# Persist calibration preference
npm install @react-native-async-storage/async-storage
```

2. Reanimated config: add to `babel.config.js`:

```js
plugins: ['react-native-reanimated/plugin']
```

3. iOS/Android Native setup:
- Follow Vision Camera installation (permissions, ProGuard, etc.)
- Follow Skia installation (Hermes recommended)
- Enable Reanimated (JSI) in `MainApplication`
- For brightness/keep-awake, autolinking should suffice; no extra permissions needed for app-level brightness.

4. Frame → Skia bridge:
- Integrate or implement a Vision Camera frame processor plugin that exposes `frame.toSkImage()` returning a JSI-backed `SkImage` consumable by Skia on the UI thread. This avoids React re-renders and supports 30+ FPS.
- If your Vision Camera version supports setting frame processor FPS, configure it as needed.

## Performance Tips

- Keep `u_Strength` driven by a Reanimated SharedValue with Skia’s `useSharedValueEffect` to avoid React re-renders.
- Avoid `setState` or prop changes on every frame.
- Prefer device-native camera preview (no snapshots), always pass GPU texture or SkImage handles across JSI.
- Use linear color conversions inside the shader (already implemented) for better color fidelity.
- In dev, use the React Native Perf Monitor, and optionally enable `showPerfHUD` to see a lightweight FPS counter.

## Calibration Mode

- The `CalibrationGuard` component always sets app brightness to maximum and keeps the screen awake in the background.
- It shows a one-time modal educating the user to disable: Night Shift / Eye Comfort (blue light), True Tone (iOS), and Accessibility color inversion/correction.
- Acknowledgment is persisted via AsyncStorage; you can reset it from a Settings screen.

Limitations:
- Apps cannot programmatically disable Night Shift / blue light filters or accessibility color transforms due to OS restrictions. We provide clear guidance and shortcuts to the relevant Settings pages.

## Switching CVD Types

- Replace `SIMULATION` and `CORRECTION` with your desired CVD set (Deutan/Tritan). The shader does not need to change—only uniforms.

## Known Limitations

- Without a `frame.toSkImage()` bridge, live GPU filtering won’t be applied; add the plugin.
- Provided CVD simulation matrices are standard single-matrix approximations; for research-grade results, consider piecewise Brettel/Machado models.
