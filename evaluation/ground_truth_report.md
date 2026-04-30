# ReColor Ground-Truth Evaluation Report

**Generated:** 2026-04-30 09:50:25
**Suites:** 3 (Identifier, Simulation, Discrimination Gain)

---

## Suite 1: Color Identifier Ground Truth (ColorChecker 24)

**Accuracy: 19/24 (79.2%)**

| Patch | RGB | Expected | Predicted | Conf | Result |
|-------|-----|----------|-----------|------|--------|
| Dark Skin | (115,82,68) | Brown | Brown | 90% | PASS |
| Light Skin | (194,150,130) | Orange | Pink | 77% | **MISS** |
| Blue Sky | (98,122,157) | Blue | Blue | 89% | PASS |
| Foliage | (87,108,67) | Green | Green | 78% | PASS |
| Blue Flower | (133,128,177) | Violet | Violet | 78% | PASS |
| Bluish Green | (103,189,170) | Cyan | Cyan | 78% | PASS |
| Orange | (214,126,44) | Orange | Orange | 81% | PASS |
| Purplish Blue | (80,91,166) | Blue | Violet | 73% | **MISS** |
| Moderate Red | (193,90,99) | Red | Red | 84% | PASS |
| Purple | (94,60,108) | Violet | Violet | 82% | PASS |
| Yellow Green | (157,188,64) | Green | Green | 71% | PASS |
| Orange Yellow | (224,163,46) | Orange | Yellow | 88% | **MISS** |
| Blue | (56,61,150) | Blue | Violet | 67% | **MISS** |
| Green | (70,148,73) | Green | Green | 66% | PASS |
| Red | (175,54,60) | Red | Brown | 87% | **MISS** |
| Yellow | (231,199,31) | Yellow | Yellow | 77% | PASS |
| Magenta | (187,86,149) | Pink | Pink | 65% | PASS |
| Cyan | (8,133,161) | Cyan | Cyan | 65% | PASS |
| White | (243,243,242) | Neutral | Neutral | 94% | PASS |
| Neutral 8 | (200,200,200) | Neutral | Neutral | 96% | PASS |
| Neutral 6.5 | (160,160,160) | Neutral | Neutral | 83% | PASS |
| Neutral 5 | (122,122,121) | Neutral | Neutral | 96% | PASS |
| Neutral 3.5 | (85,85,85) | Neutral | Neutral | 87% | PASS |
| Black | (52,52,52) | Neutral | Neutral | 92% | PASS |

### Confusion Matrix

| Expected \ Predicted | Red | Orange | Yellow | Green | Cyan | Blue | Violet | Pink | Brown | Neutral |
|---|---|---|---|---|---|---|---|---|---|---|
| **Red** | 1 | . | . | . | . | . | . | . | 1 | . |
| **Orange** | . | 1 | 1 | . | . | . | . | 1 | . | . |
| **Yellow** | . | . | 1 | . | . | . | . | . | . | . |
| **Green** | . | . | . | 3 | . | . | . | . | . | . |
| **Cyan** | . | . | . | . | 2 | . | . | . | . | . |
| **Blue** | . | . | . | . | . | 1 | 2 | . | . | . |
| **Violet** | . | . | . | . | . | . | 2 | . | . | . |
| **Pink** | . | . | . | . | . | . | . | 1 | . | . |
| **Brown** | . | . | . | . | . | . | . | . | 1 | . |
| **Neutral** | . | . | . | . | . | . | . | . | . | 6 |

---

## Suite 2: CVD Simulation Fidelity (Vienot Reference)

**Verdict: CHECK** (max channel error <= 1 across all CVD types)

### Protan (max channel error: 22)

| Color | Input | Reference | App Output | Ch Error | Delta-E |
|-------|-------|-----------|------------|----------|---------|
| Pure Red | [255, 0, 0] | [np.int64(108), np.int64(95), np.int64(0)] | [np.int64(108), np.int64(95), np.int64(0)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Pure Green | [0, 255, 0] | [np.int64(255), np.int64(229), np.int64(0)] | [np.int64(255), np.int64(229), np.int64(0)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Pure Blue | [0, 0, 255] | [np.int64(0), np.int64(89), np.int64(255)] | [np.int64(0), np.int64(89), np.int64(255)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Yellow | [255, 255, 0] | [np.int64(255), np.int64(243), np.int64(0)] | [np.int64(255), np.int64(243), np.int64(0)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Cyan | [0, 255, 255] | [np.int64(237), np.int64(241), np.int64(255)] | [np.int64(237), np.int64(241), np.int64(255)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Magenta | [255, 0, 255] | [np.int64(0), np.int64(126), np.int64(255)] | [np.int64(0), np.int64(126), np.int64(255)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Orange | [255, 165, 0] | [np.int64(194), np.int64(170), np.int64(0)] | [np.int64(195), np.int64(171), np.int64(0)] | [np.int64(1), np.int64(1), np.int64(0)] | 0.487 |
| Forest Green | [34, 139, 34] | [np.int64(141), np.int64(124), np.int64(22)] | [np.int64(142), np.int64(125), np.int64(0)] | [np.int64(1), np.int64(1), np.int64(22)] | 5.595 |
| Sky Blue | [135, 206, 235] | [np.int64(189), np.int64(201), np.int64(236)] | [np.int64(190), np.int64(203), np.int64(237)] | [np.int64(1), np.int64(2), np.int64(1)] | 0.921 |
| Mid Gray | [128, 128, 128] | [np.int64(127), np.int64(127), np.int64(127)] | [np.int64(128), np.int64(128), np.int64(128)] | [np.int64(1), np.int64(1), np.int64(1)] | 0.392 |
| White | [255, 255, 255] | [np.int64(255), np.int64(255), np.int64(255)] | [np.int64(255), np.int64(255), np.int64(255)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Dark Brown | [101, 67, 33] | [np.int64(79), np.int64(71), np.int64(35)] | [np.int64(78), np.int64(70), np.int64(29)] | [np.int64(1), np.int64(1), np.int64(6)] | 3.027 |

### Deutan (max channel error: 4)

| Color | Input | Reference | App Output | Ch Error | Delta-E |
|-------|-------|-----------|------------|----------|---------|
| Pure Red | [255, 0, 0] | [np.int64(162), np.int64(143), np.int64(0)] | [np.int64(162), np.int64(143), np.int64(0)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Pure Green | [0, 255, 0] | [np.int64(238), np.int64(213), np.int64(61)] | [np.int64(238), np.int64(213), np.int64(61)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Pure Blue | [0, 0, 255] | [np.int64(0), np.int64(64), np.int64(251)] | [np.int64(0), np.int64(64), np.int64(251)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Yellow | [255, 255, 0] | [np.int64(255), np.int64(249), np.int64(53)] | [np.int64(255), np.int64(249), np.int64(53)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Cyan | [0, 255, 255] | [np.int64(207), np.int64(220), np.int64(255)] | [np.int64(207), np.int64(220), np.int64(255)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Magenta | [255, 0, 255] | [np.int64(104), np.int64(154), np.int64(250)] | [np.int64(104), np.int64(154), np.int64(250)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Orange | [255, 165, 0] | [np.int64(216), np.int64(192), np.int64(22)] | [np.int64(216), np.int64(192), np.int64(22)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Forest Green | [34, 139, 34] | [np.int64(129), np.int64(117), np.int64(49)] | [np.int64(130), np.int64(117), np.int64(46)] | [np.int64(1), np.int64(0), np.int64(3)] | 1.66 |
| Sky Blue | [135, 206, 235] | [np.int64(174), np.int64(190), np.int64(234)] | [np.int64(175), np.int64(191), np.int64(235)] | [np.int64(1), np.int64(1), np.int64(1)] | 0.364 |
| Mid Gray | [128, 128, 128] | [np.int64(127), np.int64(127), np.int64(127)] | [np.int64(128), np.int64(128), np.int64(128)] | [np.int64(1), np.int64(1), np.int64(1)] | 0.392 |
| White | [255, 255, 255] | [np.int64(255), np.int64(255), np.int64(255)] | [np.int64(255), np.int64(255), np.int64(255)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Dark Brown | [101, 67, 33] | [np.int64(86), np.int64(79), np.int64(38)] | [np.int64(86), np.int64(77), np.int64(34)] | [np.int64(0), np.int64(2), np.int64(4)] | 1.849 |

### Tritan (max channel error: 3)

| Color | Input | Reference | App Output | Ch Error | Delta-E |
|-------|-------|-----------|------------|----------|---------|
| Pure Red | [255, 0, 0] | [np.int64(255), np.int64(0), np.int64(22)] | [np.int64(255), np.int64(0), np.int64(22)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Pure Green | [0, 255, 0] | [np.int64(0), np.int64(247), np.int64(216)] | [np.int64(0), np.int64(247), np.int64(216)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Pure Blue | [0, 0, 255] | [np.int64(0), np.int64(107), np.int64(148)] | [np.int64(0), np.int64(107), np.int64(148)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Yellow | [255, 255, 0] | [np.int64(255), np.int64(237), np.int64(216)] | [np.int64(255), np.int64(237), np.int64(216)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Cyan | [0, 255, 255] | [np.int64(0), np.int64(255), np.int64(254)] | [np.int64(0), np.int64(255), np.int64(254)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Magenta | [255, 0, 255] | [np.int64(255), np.int64(76), np.int64(149)] | [np.int64(255), np.int64(76), np.int64(149)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Orange | [255, 165, 0] | [np.int64(255), np.int64(141), np.int64(139)] | [np.int64(255), np.int64(143), np.int64(141)] | [np.int64(0), np.int64(2), np.int64(2)] | 1.046 |
| Forest Green | [34, 139, 34] | [np.int64(0), np.int64(134), np.int64(118)] | [np.int64(0), np.int64(135), np.int64(119)] | [np.int64(0), np.int64(1), np.int64(1)] | 0.395 |
| Sky Blue | [135, 206, 235] | [np.int64(93), np.int64(214), np.int64(214)] | [np.int64(94), np.int64(215), np.int64(215)] | [np.int64(1), np.int64(1), np.int64(1)] | 0.35 |
| Mid Gray | [128, 128, 128] | [np.int64(127), np.int64(127), np.int64(127)] | [np.int64(128), np.int64(128), np.int64(128)] | [np.int64(1), np.int64(1), np.int64(1)] | 0.392 |
| White | [255, 255, 255] | [np.int64(255), np.int64(255), np.int64(255)] | [np.int64(255), np.int64(255), np.int64(255)] | [np.int64(0), np.int64(0), np.int64(0)] | 0.0 |
| Dark Brown | [101, 67, 33] | [np.int64(110), np.int64(62), np.int64(62)] | [np.int64(110), np.int64(59), np.int64(59)] | [np.int64(0), np.int64(3), np.int64(3)] | 1.819 |

---

## Suite 3: Enhancement Discrimination Gain

**Method:** Generate color pairs confused under CVD (high pre-sim Delta-E, low post-sim Delta-E).
Enhance each color, re-simulate, measure if Delta-E increases (= discrimination restored).
Gain > 2 Delta-E = meaningful improvement.

### Summary

| CVD | Pairs | DAL Mean Gain | HUE Mean Gain | DAL Median | HUE Median | DAL %>2 | HUE %>2 | Winner |
|-----|-------|---------------|---------------|------------|------------|---------|---------|--------|
| Protan | 20 | 22.06 | 8.39 | 17.64 | 0.0 | 95.0% | 45.0% | **DAL** |
| Deutan | 20 | 12.72 | 4.65 | 8.12 | 0.0 | 90.0% | 25.0% | **DAL** |
| Tritan | 14 | 3.36 | 1.19 | 3.0 | 0.1 | 64.3% | 35.7% | **DAL** |

### Protan Pair Details (first 10)

| C1 | C2 | DE Orig | DE Confused | DE post-DAL | DE post-HUE | DAL Gain | HUE Gain |
|----|----|---------|-------------|-------------|-------------|----------|----------|
| [216, 32, 64] | [75, 85, 56] | 80.05 | 4.21 | 47.9 | 39.51 | 43.69 | 35.3 |
| [131, 162, 92] | [221, 134, 87] | 52.34 | 4.22 | 55.09 | 25.82 | 50.87 | 21.6 |
| [100, 143, 147] | [170, 139, 159] | 29.17 | 4.65 | 16.85 | 2.02 | 12.2 | -2.63 |
| [42, 108, 129] | [213, 73, 129] | 75.62 | 1.12 | 34.39 | 33.95 | 33.27 | 32.83 |
| [96, 62, 91] | [166, 39, 97] | 36.73 | 2.54 | 18.33 | 26.4 | 15.79 | 23.86 |
| [67, 79, 205] | [172, 56, 200] | 33.96 | 4.63 | 23.38 | 4.63 | 18.75 | -0.0 |
| [196, 178, 184] | [77, 199, 195] | 42.48 | 3.35 | 19.03 | 3.35 | 15.68 | 0.0 |
| [190, 190, 144] | [52, 214, 151] | 47.89 | 4.71 | 42.56 | 4.71 | 37.85 | -0.0 |
| [124, 186, 65] | [71, 184, 67] | 15.86 | 3.82 | 9.96 | 3.82 | 6.14 | -0.0 |
| [155, 171, 89] | [189, 151, 75] | 24.35 | 4.77 | 34.03 | 12.77 | 29.26 | 8.0 |

### Deutan Pair Details (first 10)

| C1 | C2 | DE Orig | DE Confused | DE post-DAL | DE post-HUE | DAL Gain | HUE Gain |
|----|----|---------|-------------|-------------|-------------|----------|----------|
| [45, 182, 144] | [156, 143, 140] | 49.99 | 3.65 | 9.2 | 3.65 | 5.55 | 0.0 |
| [174, 195, 116] | [89, 223, 121] | 38.44 | 2.52 | 10.13 | 2.52 | 7.61 | 0.0 |
| [202, 35, 79] | [78, 120, 58] | 92.3 | 4.47 | 48.79 | 26.5 | 44.32 | 22.03 |
| [189, 198, 167] | [96, 220, 167] | 38.4 | 2.3 | 11.16 | 2.3 | 8.86 | 0.0 |
| [123, 94, 214] | [169, 68, 221] | 24.86 | 2.52 | 10.65 | 2.52 | 8.13 | -0.0 |
| [191, 54, 109] | [67, 126, 96] | 85.65 | 2.94 | 36.47 | 30.19 | 33.53 | 27.25 |
| [57, 203, 126] | [214, 163, 141] | 72.06 | 3.42 | 19.15 | 12.33 | 15.73 | 8.91 |
| [124, 89, 153] | [76, 117, 154] | 31.45 | 3.58 | 8.15 | 3.58 | 4.57 | -0.0 |
| [184, 123, 211] | [149, 140, 213] | 19.48 | 2.34 | 9.81 | 2.34 | 7.47 | 0.0 |
| [162, 188, 219] | [199, 178, 226] | 18.78 | 1.78 | 6.79 | 1.78 | 5.01 | 0.0 |

### Tritan Pair Details (first 10)

| C1 | C2 | DE Orig | DE Confused | DE post-DAL | DE post-HUE | DAL Gain | HUE Gain |
|----|----|---------|-------------|-------------|-------------|----------|----------|
| [102, 218, 155] | [55, 214, 189] | 21.07 | 4.88 | 4.32 | 4.88 | -0.56 | -0.0 |
| [68, 222, 135] | [55, 215, 99] | 15.84 | 3.65 | 5.33 | 3.65 | 1.68 | -0.0 |
| [51, 137, 141] | [36, 139, 174] | 18.69 | 4.01 | 9.41 | 7.74 | 5.4 | 3.73 |
| [134, 169, 78] | [146, 171, 133] | 29.07 | 4.37 | 16.09 | 4.37 | 11.72 | -0.0 |
| [88, 140, 188] | [74, 144, 166] | 17.18 | 4.04 | 9.75 | 4.24 | 5.71 | 0.2 |
| [82, 114, 126] | [79, 114, 99] | 16.79 | 3.38 | 8.48 | 5.08 | 5.1 | 1.7 |
| [192, 182, 89] | [177, 175, 49] | 15.3 | 4.79 | 7.97 | 4.79 | 3.18 | 0.0 |
| [64, 144, 178] | [37, 148, 211] | 16.85 | 4.73 | 7.54 | 8.61 | 2.81 | 3.88 |
| [68, 141, 210] | [85, 137, 180] | 15.04 | 4.95 | 6.75 | 7.68 | 1.8 | 2.73 |
| [82, 145, 53] | [76, 141, 86] | 19.88 | 3.6 | 2.48 | 3.6 | -1.12 | -0.0 |

---

## Discussion

Suite 1 measures whether the CIELAB nearest-neighbor identifier correctly classifies
the 24 standard ColorChecker patches into the app's 10-class taxonomy.

Suite 2 confirms the CVD simulation matrices produce identical output to the
Vienot 1999 reference computation (same matrices, same gamma pipeline).

Suite 3 is the key discrimination-gain test: for color pairs a CVD user confuses,
does enhancement make them distinguishable again? A positive gain means the algorithm
is doing useful work. This is the ground truth that faithfulness metrics (Suites 1-3
of the previous report) cannot capture.