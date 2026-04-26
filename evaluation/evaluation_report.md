# ReColor — Daltonization vs Hue Rotation Evaluation

**Generated:** 2026-04-25 16:45:21
**Test images:** 6 synthetic images (720×480)
**CVD types:** Protan, Deutan, Tritan
**Algorithms:** Daltonization (Brettel/Fidaner), Hue Rotation (HSV band)

## Tools

| Purpose | Tool | Notes |
|---|---|---|
| Perceptual accuracy (ΔE) | `colour-science` 0.4+ | sRGB → XYZ (D65) → CIE Lab → CIEDE2000 |
| Structural preservation (SSIM) | `scikit-image` | Per-channel mean SSIM, `data_range=255` |
| Pixel maths | `numpy` | Vectorised ports of tensorHelper.js |
| Image I/O | `Pillow` | PNG, lossless |

## Pipeline

For each (image, CVD type, algorithm):

```
original (sRGB)
  → enhanced = algorithm(original, cvd_type)
  → cvd_view = simulate_cvd(enhanced, cvd_type)
  → ΔE(cvd_view, original)        ← perceptual accuracy
  → SSIM(cvd_view, original)       ← structural preservation
```

`simulate_cvd` uses the same Viénot 1999 matrices and γ=2.2 used inside the
shipped Daltonization function — so the CVD perception model is consistent
across the eval.

## Headline

| Metric | Daltonization | Hue Rotation | Winner |
|---|---|---|---|
| Mean ΔE (lower = better) | 15.721 | 15.312 | **HUE** |
| Mean SSIM (higher = better) | 0.7638 | 0.7878 | **HUE** |
| Per-CVD wins (out of 6) | 2 | 4 | **HUE** |

> **Important caveat (read before citing):** these metrics measure how close
> the CVD-perceived enhanced image is to the *original normal-vision view*.
> That is a faithfulness target, not a discrimination-gain target. Both
> algorithms can do well on faithfulness by changing pixels minimally. They
> can do well on discrimination gain by changing pixels intelligently. The
> two goals can disagree — see the comments in the report's discussion.

---

## Suite 1 — Perceptual Accuracy (CIEDE2000 ΔE)

**Method:** Enhance the image, then simulate CVD perception of the enhanced
image. Compare per-pixel against the original via CIEDE2000. Report mean,
median, 95th percentile and max ΔE per (image, CVD, algorithm).

**Interpretation:** Lower ΔE means the CVD-perceived enhanced image is closer
to the original normal-vision view. ΔE < 1 is imperceptible; ΔE 1-2 is barely
noticeable; ΔE > 5 is clearly different. Note: physiological loss makes
ΔE = 0 unreachable for any algorithm.

| Image | CVD | Algorithm | Mean ΔE | Median ΔE | P95 ΔE | Max ΔE |
|---|---|---|---|---|---|---|
| gradient_red_green.png | Protan | Daltonization | 26.21 | 28.76 | 39.01 | 39.80 |
| gradient_red_green.png | Protan | Hue Rotation | 27.07 | 27.05 | 46.91 | 47.64 |
| gradient_red_green.png | Deutan | Daltonization | 25.38 | 25.11 | 43.70 | 44.15 |
| gradient_red_green.png | Deutan | Hue Rotation | 27.44 | 26.00 | 49.62 | 50.58 |
| gradient_red_green.png | Tritan | Daltonization | 22.85 | 25.49 | 28.59 | 29.32 |
| gradient_red_green.png | Tritan | Hue Rotation | 19.86 | 23.49 | 27.30 | 28.07 |
| gradient_blue_yellow.png | Protan | Daltonization | 9.05 | 7.41 | 18.64 | 19.28 |
| gradient_blue_yellow.png | Protan | Hue Rotation | 7.55 | 6.95 | 13.82 | 14.97 |
| gradient_blue_yellow.png | Deutan | Daltonization | 7.40 | 6.11 | 14.48 | 15.52 |
| gradient_blue_yellow.png | Deutan | Hue Rotation | 5.57 | 5.89 | 8.84 | 10.30 |
| gradient_blue_yellow.png | Tritan | Daltonization | 27.55 | 28.37 | 48.45 | 49.71 |
| gradient_blue_yellow.png | Tritan | Hue Rotation | 27.45 | 27.89 | 40.36 | 41.06 |
| color_checker_24.png | Protan | Daltonization | 12.67 | 14.29 | 29.67 | 31.26 |
| color_checker_24.png | Protan | Hue Rotation | 12.80 | 12.23 | 36.84 | 40.13 |
| color_checker_24.png | Deutan | Daltonization | 13.18 | 12.48 | 39.36 | 42.63 |
| color_checker_24.png | Deutan | Hue Rotation | 12.98 | 11.62 | 38.98 | 42.25 |
| color_checker_24.png | Tritan | Daltonization | 12.09 | 12.69 | 27.47 | 27.98 |
| color_checker_24.png | Tritan | Hue Rotation | 13.14 | 12.12 | 31.36 | 34.23 |
| natural_scene.png | Protan | Daltonization | 16.98 | 17.02 | 26.17 | 29.06 |
| natural_scene.png | Protan | Hue Rotation | 16.78 | 17.04 | 24.30 | 36.23 |
| natural_scene.png | Deutan | Daltonization | 15.83 | 16.44 | 22.64 | 36.83 |
| natural_scene.png | Deutan | Hue Rotation | 16.62 | 17.23 | 23.18 | 37.65 |
| natural_scene.png | Tritan | Daltonization | 16.57 | 17.08 | 23.58 | 26.16 |
| natural_scene.png | Tritan | Hue Rotation | 18.17 | 19.15 | 22.55 | 27.68 |
| saturated_primaries.png | Protan | Daltonization | 26.72 | 30.70 | 39.80 | 39.80 |
| saturated_primaries.png | Protan | Hue Rotation | 26.02 | 27.33 | 47.55 | 47.55 |
| saturated_primaries.png | Deutan | Daltonization | 25.25 | 27.15 | 44.15 | 44.15 |
| saturated_primaries.png | Deutan | Hue Rotation | 25.18 | 27.48 | 50.49 | 50.49 |
| saturated_primaries.png | Tritan | Daltonization | 25.25 | 28.20 | 49.71 | 49.71 |
| saturated_primaries.png | Tritan | Hue Rotation | 19.00 | 21.94 | 41.06 | 41.06 |
| neutral_grays.png | Protan | Daltonization | 0.00 | 0.00 | 0.00 | 0.00 |
| neutral_grays.png | Protan | Hue Rotation | 0.00 | 0.00 | 0.00 | 0.00 |
| neutral_grays.png | Deutan | Daltonization | 0.00 | 0.00 | 0.00 | 0.00 |
| neutral_grays.png | Deutan | Hue Rotation | 0.00 | 0.00 | 0.00 | 0.00 |
| neutral_grays.png | Tritan | Daltonization | 0.00 | 0.00 | 0.00 | 0.00 |
| neutral_grays.png | Tritan | Hue Rotation | 0.00 | 0.00 | 0.00 | 0.00 |

---

## Suite 2 — Structural Preservation (SSIM)

**Method:** Per-channel SSIM on the (cvd_view, original) pair, averaged across
R, G, B. SSIM ranges [0,1]; higher = more structurally similar. Penalises
luminance shifts, contrast loss, and pattern distortion.

**Interpretation:** SSIM > 0.95 is structurally faithful; SSIM > 0.85 is
acceptable; SSIM < 0.7 indicates noticeable degradation.

| Image | CVD | Algorithm | Mean SSIM | R | G | B |
|---|---|---|---|---|---|---|
| gradient_red_green.png | Protan | Daltonization | 0.7011 | 0.802 | 0.779 | 0.522 |
| gradient_red_green.png | Protan | Hue Rotation | 0.8590 | 0.763 | 0.814 | 1.000 |
| gradient_red_green.png | Deutan | Daltonization | 0.6598 | 0.770 | 0.825 | 0.384 |
| gradient_red_green.png | Deutan | Hue Rotation | 0.5282 | 0.780 | 0.798 | 0.007 |
| gradient_red_green.png | Tritan | Daltonization | 0.4933 | 0.545 | 0.934 | 0.001 |
| gradient_red_green.png | Tritan | Hue Rotation | 0.5107 | 0.780 | 0.750 | 0.002 |
| gradient_blue_yellow.png | Protan | Daltonization | 0.7645 | 0.698 | 0.881 | 0.714 |
| gradient_blue_yellow.png | Protan | Hue Rotation | 0.7933 | 0.686 | 0.896 | 0.797 |
| gradient_blue_yellow.png | Deutan | Daltonization | 0.8463 | 0.710 | 0.894 | 0.934 |
| gradient_blue_yellow.png | Deutan | Hue Rotation | 0.8447 | 0.677 | 0.922 | 0.935 |
| gradient_blue_yellow.png | Tritan | Daltonization | 0.7990 | 0.770 | 0.818 | 0.809 |
| gradient_blue_yellow.png | Tritan | Hue Rotation | 0.7622 | 0.685 | 0.821 | 0.780 |
| color_checker_24.png | Protan | Daltonization | 0.8715 | 0.885 | 0.925 | 0.805 |
| color_checker_24.png | Protan | Hue Rotation | 0.8978 | 0.862 | 0.957 | 0.874 |
| color_checker_24.png | Deutan | Daltonization | 0.9094 | 0.884 | 0.948 | 0.896 |
| color_checker_24.png | Deutan | Hue Rotation | 0.9360 | 0.876 | 0.944 | 0.988 |
| color_checker_24.png | Tritan | Daltonization | 0.9055 | 0.865 | 0.956 | 0.895 |
| color_checker_24.png | Tritan | Hue Rotation | 0.9068 | 0.900 | 0.950 | 0.870 |
| natural_scene.png | Protan | Daltonization | 0.6008 | 0.645 | 0.689 | 0.468 |
| natural_scene.png | Protan | Hue Rotation | 0.7347 | 0.592 | 0.788 | 0.824 |
| natural_scene.png | Deutan | Daltonization | 0.7241 | 0.630 | 0.836 | 0.706 |
| natural_scene.png | Deutan | Hue Rotation | 0.7870 | 0.644 | 0.747 | 0.969 |
| natural_scene.png | Tritan | Daltonization | 0.7159 | 0.611 | 0.900 | 0.636 |
| natural_scene.png | Tritan | Hue Rotation | 0.7192 | 0.715 | 0.926 | 0.517 |
| saturated_primaries.png | Protan | Daltonization | 0.6383 | 0.639 | 0.464 | 0.811 |
| saturated_primaries.png | Protan | Hue Rotation | 0.6592 | 0.478 | 0.500 | 1.000 |
| saturated_primaries.png | Deutan | Daltonization | 0.5821 | 0.591 | 0.495 | 0.660 |
| saturated_primaries.png | Deutan | Hue Rotation | 0.5327 | 0.598 | 0.493 | 0.507 |
| saturated_primaries.png | Tritan | Daltonization | 0.5376 | 0.643 | 0.486 | 0.483 |
| saturated_primaries.png | Tritan | Hue Rotation | 0.7084 | 1.000 | 0.666 | 0.459 |
| neutral_grays.png | Protan | Daltonization | 1.0000 | 1.000 | 1.000 | 1.000 |
| neutral_grays.png | Protan | Hue Rotation | 1.0000 | 1.000 | 1.000 | 1.000 |
| neutral_grays.png | Deutan | Daltonization | 1.0000 | 1.000 | 1.000 | 1.000 |
| neutral_grays.png | Deutan | Hue Rotation | 1.0000 | 1.000 | 1.000 | 1.000 |
| neutral_grays.png | Tritan | Daltonization | 1.0000 | 1.000 | 1.000 | 1.000 |
| neutral_grays.png | Tritan | Hue Rotation | 1.0000 | 1.000 | 1.000 | 1.000 |

---

## Suite 3 — Comparative Analysis (per-CVD aggregate)

**Method:** For each CVD type, mean ΔE and mean SSIM across all 6 test images.
Winner is the algorithm with the lower aggregate ΔE / higher aggregate SSIM.

| CVD | DAL Mean ΔE | HUE Mean ΔE | ΔE Winner | DAL Mean SSIM | HUE Mean SSIM | SSIM Winner |
|---|---|---|---|---|---|---|
| Protan | 15.271 | 15.036 | HUE | 0.7627 | 0.8240 | HUE |
| Deutan | 14.507 | 14.631 | DAL | 0.7869 | 0.7714 | DAL |
| Tritan | 17.384 | 16.269 | HUE | 0.7419 | 0.7679 | HUE |

---

## Discussion

### What "winning" means here

Hue Rotation tends to win on both metrics because it operates on a narrow hue
band (60° around the CVD's confusion centre) with a saturation gate that
excludes near-greys. Pixels outside the band and below the saturation
threshold are bit-for-bit unchanged. SSIM rewards this — most of the image is
preserved exactly. ΔE-vs-original rewards this for the same reason: untouched
pixels contribute zero ΔE.

Daltonization touches every pixel that has a non-zero error term in linear
space, which is most of them. The redistribution shifts the chromaticity of
many pixels by a few JNDs each. This raises the aggregate ΔE-vs-original even
though the redistribution is precisely what gives a CVD user back the
discrimination they lost.

### Why these metrics are not the full story

Faithfulness to the *normal-vision* original is unattainable for an
enhancement algorithm — the cones that distinguish the confused colours are
physiologically missing. A perfect score on these two metrics would mean
"the algorithm did nothing", which would also mean "the user got no help".
The right metric for enhancement is **discrimination gain**: for color pairs
the CVD user previously confused (low ΔE under CVD simulation of the
unenhanced image), does the enhanced version increase the ΔE between those
same pairs after CVD simulation? Suites 1-3 above do not measure this; a
follow-up suite using LMS-space confusion-line endpoints would.

### What to take from this evaluation

* Both algorithms are well-behaved on neutral greys (ΔE = 0, SSIM = 1) — the
  saturation gate / self-gating both trigger correctly.
* On gradient_red_green and saturated_primaries, ΔE is high for both because
  these images are dominated by colours in the confusion set and any
  non-trivial correction necessarily changes them.
* Hue Rotation's higher SSIM is a structural artefact of its narrow band, not
  evidence that the corrected image is more useful.
* Daltonization's lower SSIM is the cost of its broader correction, not
  evidence that it is incorrect.
* For a clean comparative verdict, run a discrimination-gain benchmark next.

---

## Reproducibility

* All test images are deterministically synthesised in
  [scripts/eval_enhancement.py](../scripts/eval_enhancement.py) (`make_test_images`).
* Algorithm constants (matrices, gamma, hue config) are pinned in the script
  with the same numerical values as `tensorHelper.js`.
* CIEDE2000 implementation is the reference one in `colour-science` —
  no homegrown ΔE.
* SSIM uses `skimage.metrics.structural_similarity` with explicit
  `data_range=255` to avoid the silent default-range bug.
* Random seed: 42 (used only for the synthesised noise in
  `natural_scene.png`).

To regenerate this report:

```bash
python scripts/eval_enhancement.py
```
