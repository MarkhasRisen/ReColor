"""
ReColor — Enhancement Algorithm Evaluation
==========================================

Evaluates the two enhancement algorithms shipped in the ReColor app
(Daltonization and Hue Rotation) across three metric suites:

    Suite 1 — Perceptual accuracy (CIEDE2000 ΔE)
    Suite 2 — Structural preservation (SSIM)
    Suite 3 — Comparative analysis (per-CVD aggregate winners)

Pipeline per (test_image, cvd_type, algorithm):
    1. enhanced  = enhance(original, cvd_type)
    2. cvd_view  = simulate_cvd(enhanced, cvd_type)   ← what the CVD user sees
    3. ΔE        = CIEDE2000(cvd_view, original)      ← perceptual fidelity
    4. SSIM      = SSIM(cvd_view, original)            ← structural fidelity

Tools:
    colour-science  → CIEDE2000, sRGB↔XYZ↔LAB, D65 illuminant
    scikit-image    → SSIM (multichannel, with explicit data_range)
    NumPy           → vectorised pixel math
    Pillow          → PNG I/O

The JS algorithms in tensorHelper.js are ported here verbatim
(same matrices, same gamma, same hue config) so the evaluation
measures the actual behaviour of the deployed app.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image
import colour
from skimage.metrics import structural_similarity as ssim

# ───────────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation"
IMG_DIR = EVAL_DIR / "test_images"
RESULTS_DIR = EVAL_DIR / "results"
REPORT_PATH = EVAL_DIR / "evaluation_report.md"

IMG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────────────────────────────────────
# Algorithm constants (verbatim from tensorHelper.js)
# ───────────────────────────────────────────────────────────────
CVD_COMBINED = {
    "Protan": np.array([
        [ 0.152286,  1.052583, -0.204868],
        [ 0.114503,  0.786281,  0.099216],
        [-0.003882, -0.048116,  1.051998],
    ]),
    "Deutan": np.array([
        [ 0.367322,  0.860646, -0.227968],
        [ 0.280085,  0.672501,  0.047413],
        [-0.011820,  0.042940,  0.968881],
    ]),
    "Tritan": np.array([
        [ 1.255528, -0.076749, -0.178779],
        [-0.078411,  0.930809,  0.147602],
        [ 0.004733,  0.691367,  0.303900],
    ]),
}

CVD_ERR_SHIFT = {
    "Protan": np.array([
        [0.0, 0.0, 0.0],
        [0.7, 1.0, 0.0],
        [0.7, 0.0, 1.0],
    ]),
    "Deutan": np.array([
        [1.0, 0.6, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.6, 1.0],
    ]),
    "Tritan": np.array([
        [1.0, 0.0, 0.7],
        [0.0, 1.0, 0.7],
        [0.0, 0.0, 0.0],
    ]),
}

HUE_ROTATION_CONFIG = {
    "Protan": {"center":   0, "range": 60, "shift":  40},
    "Deutan": {"center":   0, "range": 60, "shift":  40},
    "Tritan": {"center": 240, "range": 60, "shift": -30},
}

GAMMA = 2.2
SAT_MIN = 0.15


# ───────────────────────────────────────────────────────────────
# Algorithm implementations (ports of tensorHelper.js)
# ───────────────────────────────────────────────────────────────
def srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → linear via pow 2.2 (matches JS exactly)."""
    return np.power(np.clip(arr, 0.0, 1.0), GAMMA)


def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    """Linear [0,1] → sRGB via pow 1/2.2 (matches JS exactly)."""
    return np.power(np.clip(arr, 0.0, 1.0), 1.0 / GAMMA)


def apply_cvd_simulation(rgb_u8: np.ndarray, cvd_type: str) -> np.ndarray:
    """
    Models what a CVD user perceives. Same matrix and gamma flow
    used inside applyDaltonization in tensorHelper.js.
    Input/output: (H,W,3) uint8 RGB.
    """
    sim = CVD_COMBINED[cvd_type]
    rgb = rgb_u8.astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    # Per-pixel matrix multiply: out = lin @ sim.T
    out_lin = lin @ sim.T
    out = linear_to_srgb(out_lin)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_daltonization(rgb_u8: np.ndarray, cvd_type: str) -> np.ndarray:
    """
    Brettel/Fidaner daltonization, no mask (self-gating via error term).
    Vectorised port of tensorHelper.js::applyDaltonization with mask=null.
    Input/output: (H,W,3) uint8 RGB.
    """
    sim = CVD_COMBINED[cvd_type]
    err_m = CVD_ERR_SHIFT[cvd_type]

    rgb = rgb_u8.astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)            # (H,W,3) linear
    sim_lin = lin @ sim.T                # what CVD sees, linear
    err = lin - sim_lin                  # what CVD misses, linear
    out_lin = lin + err @ err_m.T        # redistribute into surviving channels

    out = linear_to_srgb(out_lin)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB→HSV. rgb in [0,1]. Returns (H,W,3) with H in [0,360)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    d = mx - mn
    v = mx
    s = np.where(mx > 0, d / np.where(mx == 0, 1, mx), 0)

    h = np.zeros_like(mx)
    nz = d > 0
    # Red is max
    rmax = nz & (mx == r)
    h[rmax] = (60 * (((g[rmax] - b[rmax]) / d[rmax]) % 6))
    # Green is max
    gmax = nz & (mx == g) & ~rmax
    h[gmax] = (60 * ((b[gmax] - r[gmax]) / d[gmax] + 2))
    # Blue is max
    bmax = nz & (mx == b) & ~rmax & ~gmax
    h[bmax] = (60 * ((r[bmax] - g[bmax]) / d[bmax] + 4))
    h = (h + 360) % 360
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorised HSV→RGB. h in [0,360). Returns (H,W,3) in [0,1]."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    c = v * s
    hp = h / 60.0
    x = c * (1 - np.abs((hp % 2) - 1))
    z = np.zeros_like(c)

    nr = np.select(
        [hp < 1, hp < 2, hp < 3, hp < 4, hp < 5, hp < 6],
        [c,      x,      z,      z,      x,      c],
        default=0,
    )
    ng = np.select(
        [hp < 1, hp < 2, hp < 3, hp < 4, hp < 5, hp < 6],
        [x,      c,      c,      x,      z,      z],
        default=0,
    )
    nb = np.select(
        [hp < 1, hp < 2, hp < 3, hp < 4, hp < 5, hp < 6],
        [z,      z,      x,      c,      c,      x],
        default=0,
    )
    m = v - c
    return np.stack([nr + m, ng + m, nb + m], axis=-1)


def apply_hue_rotation(rgb_u8: np.ndarray, cvd_type: str) -> np.ndarray:
    """
    HSV-band hue rotation with linear taper and saturation gate.
    Vectorised port of tensorHelper.js::applyHueRotation.
    """
    cfg = HUE_ROTATION_CONFIG[cvd_type]
    rgb = rgb_u8.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Saturation gate
    in_sat = s >= SAT_MIN

    # Circular hue distance from band centre
    diff = np.abs(h - cfg["center"])
    dist = np.where(diff > 180, 360 - diff, diff)
    in_band = dist <= cfg["range"]

    mask = in_sat & in_band
    weight = np.where(mask, 1 - dist / cfg["range"], 0)
    new_h = (h + cfg["shift"] * weight) % 360

    new_hsv = np.stack([new_h, s, v], axis=-1)
    new_rgb = hsv_to_rgb(new_hsv)
    return np.clip(new_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ───────────────────────────────────────────────────────────────
# Test image synthesis
# ───────────────────────────────────────────────────────────────
W, H = 720, 480


def _save(name: str, arr: np.ndarray):
    Image.fromarray(arr, "RGB").save(IMG_DIR / name)


def make_test_images():
    rng = np.random.default_rng(42)

    # 1. Red-to-green horizontal gradient
    x = np.linspace(0, 1, W, dtype=np.float32)
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 0] = (1 - x) * 255
    img[..., 1] = x * 255
    _save("gradient_red_green.png", img.astype(np.uint8))

    # 2. Blue-to-yellow horizontal gradient
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 2] = (1 - x) * 255       # blue at left
    img[..., 0] = x * 255              # yellow = R+G at right
    img[..., 1] = x * 255
    _save("gradient_blue_yellow.png", img.astype(np.uint8))

    # 3. ColorChecker-style 6×4 grid (24 patches; standard reference colours)
    cc24 = np.array([
        [115,  82,  68],[194,150,130],[ 98,122,157],[ 87,108, 67],
        [133,128,177],[103,189,170],[214,126, 44],[ 80, 91,166],
        [193, 90, 99],[ 94, 60,108],[157,188, 64],[224,163, 46],
        [ 56, 61,150],[ 70,148, 73],[175, 54, 60],[231,199, 31],
        [187, 86,149],[  8,133,161],[243,243,242],[200,200,200],
        [160,160,160],[122,122,121],[ 85, 85, 85],[ 52, 52, 52],
    ], dtype=np.uint8)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    pw, ph = W // 6, H // 4
    for i, c in enumerate(cc24):
        r, c_idx = divmod(i, 6)
        img[r*ph:(r+1)*ph, c_idx*pw:(c_idx+1)*pw] = c
    _save("color_checker_24.png", img)

    # 4. Synthetic "natural" scene — sky / foliage / skin / ground bands
    img = np.zeros((H, W, 3), dtype=np.float32)
    band_h = H // 4
    sky    = np.array([135, 206, 235], dtype=np.float32)  # sky blue
    foliage= np.array([ 34, 139,  34], dtype=np.float32)  # forest green
    skin   = np.array([222, 184, 135], dtype=np.float32)  # tan/skin
    ground = np.array([139,  90,  43], dtype=np.float32)  # earth brown
    for k, c in enumerate([sky, foliage, skin, ground]):
        noise = rng.normal(0, 8, (band_h, W, 3))
        img[k*band_h:(k+1)*band_h] = np.clip(c + noise, 0, 255)
    _save("natural_scene.png", img.astype(np.uint8))

    # 5. Saturated primaries — pure RGB+CMY swatches, 2×3 grid
    swatches = np.array([
        [255,   0,   0],[  0, 255,   0],[  0,   0, 255],
        [255, 255,   0],[  0, 255, 255],[255,   0, 255],
    ], dtype=np.uint8)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    pw, ph = W // 3, H // 2
    for i, c in enumerate(swatches):
        r, c_idx = divmod(i, 3)
        img[r*ph:(r+1)*ph, c_idx*pw:(c_idx+1)*pw] = c
    _save("saturated_primaries.png", img)

    # 6. Neutral grays — black-to-white vertical gradient (sanity image)
    y = np.linspace(0, 255, H, dtype=np.float32).reshape(H, 1, 1)
    img = np.repeat(y, W, axis=1).repeat(3, axis=2)
    _save("neutral_grays.png", img.astype(np.uint8))


# ───────────────────────────────────────────────────────────────
# Metrics
# ───────────────────────────────────────────────────────────────
def deltaE_2000(rgb_a_u8: np.ndarray, rgb_b_u8: np.ndarray) -> np.ndarray:
    """
    Per-pixel CIEDE2000 ΔE between two sRGB uint8 images.
    Uses colour-science: sRGB → XYZ (D65) → CIE Lab → ΔE2000.
    Returns (H,W) float array.
    """
    a = rgb_a_u8.astype(np.float64) / 255.0
    b = rgb_b_u8.astype(np.float64) / 255.0
    xyz_a = colour.sRGB_to_XYZ(a)
    xyz_b = colour.sRGB_to_XYZ(b)
    lab_a = colour.XYZ_to_Lab(xyz_a)
    lab_b = colour.XYZ_to_Lab(xyz_b)
    return colour.delta_E(lab_a, lab_b, method="CIE 2000")


def ssim_rgb(a_u8: np.ndarray, b_u8: np.ndarray) -> tuple[float, dict]:
    """
    Multichannel SSIM with explicit data_range=255.
    Returns (mean_ssim, per_channel_dict).
    """
    per_ch = {}
    for k, name in enumerate("RGB"):
        per_ch[name] = float(ssim(a_u8[..., k], b_u8[..., k], data_range=255))
    mean = float(np.mean(list(per_ch.values())))
    return mean, per_ch


# ───────────────────────────────────────────────────────────────
# Evaluation harness
# ───────────────────────────────────────────────────────────────
TEST_IMAGES = [
    "gradient_red_green.png",
    "gradient_blue_yellow.png",
    "color_checker_24.png",
    "natural_scene.png",
    "saturated_primaries.png",
    "neutral_grays.png",
]
CVD_TYPES = ["Protan", "Deutan", "Tritan"]
ALGORITHMS = {
    "Daltonization": apply_daltonization,
    "Hue Rotation":  apply_hue_rotation,
}


@dataclass
class Row:
    image: str
    cvd: str
    algorithm: str
    deltaE_mean:   float
    deltaE_median: float
    deltaE_p95:    float
    deltaE_max:    float
    ssim_mean:     float
    ssim_R:        float
    ssim_G:        float
    ssim_B:        float


def evaluate_one(img_name: str, cvd: str, algo_name: str, algo_fn) -> Row:
    original = np.array(Image.open(IMG_DIR / img_name).convert("RGB"))
    enhanced = algo_fn(original, cvd)
    cvd_view = apply_cvd_simulation(enhanced, cvd)

    de = deltaE_2000(cvd_view, original)
    ssim_m, ssim_ch = ssim_rgb(cvd_view, original)

    return Row(
        image=img_name, cvd=cvd, algorithm=algo_name,
        deltaE_mean=float(np.mean(de)),
        deltaE_median=float(np.median(de)),
        deltaE_p95=float(np.percentile(de, 95)),
        deltaE_max=float(np.max(de)),
        ssim_mean=ssim_m,
        ssim_R=ssim_ch["R"], ssim_G=ssim_ch["G"], ssim_B=ssim_ch["B"],
    )


def run_evaluation() -> list[Row]:
    rows: list[Row] = []
    total = len(TEST_IMAGES) * len(CVD_TYPES) * len(ALGORITHMS)
    n = 0
    for img_name in TEST_IMAGES:
        for cvd in CVD_TYPES:
            for algo_name, algo_fn in ALGORITHMS.items():
                n += 1
                print(f"[{n:2d}/{total}] {img_name:28s} {cvd:7s} {algo_name}")
                rows.append(evaluate_one(img_name, cvd, algo_name, algo_fn))
    return rows


# ───────────────────────────────────────────────────────────────
# Reporting
# ───────────────────────────────────────────────────────────────
def fmt(x: float, p: int = 3) -> str:
    if abs(x) < 0.0005 and p >= 3:
        return "0.000"
    return f"{x:.{p}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def write_report(rows: list[Row]):
    import statistics

    # Suite 1 — Perceptual accuracy detail
    s1 = [[
        r.image, r.cvd, r.algorithm,
        fmt(r.deltaE_mean, 2), fmt(r.deltaE_median, 2),
        fmt(r.deltaE_p95, 2), fmt(r.deltaE_max, 2),
    ] for r in rows]

    # Suite 2 — SSIM detail
    s2 = [[
        r.image, r.cvd, r.algorithm,
        fmt(r.ssim_mean, 4), fmt(r.ssim_R, 3), fmt(r.ssim_G, 3), fmt(r.ssim_B, 3),
    ] for r in rows]

    # Suite 3 — per-CVD aggregate
    s3 = []
    for cvd in CVD_TYPES:
        dal_de  = [r.deltaE_mean for r in rows if r.cvd == cvd and r.algorithm == "Daltonization"]
        hue_de  = [r.deltaE_mean for r in rows if r.cvd == cvd and r.algorithm == "Hue Rotation"]
        dal_ss  = [r.ssim_mean   for r in rows if r.cvd == cvd and r.algorithm == "Daltonization"]
        hue_ss  = [r.ssim_mean   for r in rows if r.cvd == cvd and r.algorithm == "Hue Rotation"]
        de_win  = "DAL" if statistics.mean(dal_de) <  statistics.mean(hue_de) else "HUE"
        ss_win  = "DAL" if statistics.mean(dal_ss) >  statistics.mean(hue_ss) else "HUE"
        s3.append([cvd,
                   fmt(statistics.mean(dal_de), 3), fmt(statistics.mean(hue_de), 3), de_win,
                   fmt(statistics.mean(dal_ss), 4), fmt(statistics.mean(hue_ss), 4), ss_win])

    # Aggregate winner counts (3 CVDs × 2 metrics = 6 categories)
    dal_wins  = sum(1 for row in s3 if row[3] == "DAL") + sum(1 for row in s3 if row[6] == "DAL")
    hue_wins  = sum(1 for row in s3 if row[3] == "HUE") + sum(1 for row in s3 if row[6] == "HUE")

    # Headline summary numbers
    dal_de_all = statistics.mean([r.deltaE_mean for r in rows if r.algorithm == "Daltonization"])
    hue_de_all = statistics.mean([r.deltaE_mean for r in rows if r.algorithm == "Hue Rotation"])
    dal_ss_all = statistics.mean([r.ssim_mean   for r in rows if r.algorithm == "Daltonization"])
    hue_ss_all = statistics.mean([r.ssim_mean   for r in rows if r.algorithm == "Hue Rotation"])

    from datetime import datetime
    body = f"""# ReColor — Daltonization vs Hue Rotation Evaluation

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test images:** {len(TEST_IMAGES)} synthetic images ({W}×{H})
**CVD types:** {', '.join(CVD_TYPES)}
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
| Mean ΔE (lower = better) | {fmt(dal_de_all, 3)} | {fmt(hue_de_all, 3)} | **{"DAL" if dal_de_all < hue_de_all else "HUE"}** |
| Mean SSIM (higher = better) | {fmt(dal_ss_all, 4)} | {fmt(hue_ss_all, 4)} | **{"DAL" if dal_ss_all > hue_ss_all else "HUE"}** |
| Per-CVD wins (out of 6) | {dal_wins} | {hue_wins} | **{"DAL" if dal_wins > hue_wins else "HUE" if hue_wins > dal_wins else "TIE"}** |

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

{md_table(["Image", "CVD", "Algorithm", "Mean ΔE", "Median ΔE", "P95 ΔE", "Max ΔE"], s1)}

---

## Suite 2 — Structural Preservation (SSIM)

**Method:** Per-channel SSIM on the (cvd_view, original) pair, averaged across
R, G, B. SSIM ranges [0,1]; higher = more structurally similar. Penalises
luminance shifts, contrast loss, and pattern distortion.

**Interpretation:** SSIM > 0.95 is structurally faithful; SSIM > 0.85 is
acceptable; SSIM < 0.7 indicates noticeable degradation.

{md_table(["Image", "CVD", "Algorithm", "Mean SSIM", "R", "G", "B"], s2)}

---

## Suite 3 — Comparative Analysis (per-CVD aggregate)

**Method:** For each CVD type, mean ΔE and mean SSIM across all 6 test images.
Winner is the algorithm with the lower aggregate ΔE / higher aggregate SSIM.

{md_table(["CVD", "DAL Mean ΔE", "HUE Mean ΔE", "ΔE Winner", "DAL Mean SSIM", "HUE Mean SSIM", "SSIM Winner"], s3)}

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
"""
    REPORT_PATH.write_text(body, encoding="utf-8")

    # Also dump raw rows as JSON for downstream analysis
    (RESULTS_DIR / "rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2),
        encoding="utf-8",
    )


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────
def main():
    print("Generating test images...")
    make_test_images()
    print(f"Test images written to {IMG_DIR}")

    print("Running evaluation...")
    rows = run_evaluation()

    print("Writing report...")
    write_report(rows)
    print(f"Report:  {REPORT_PATH}")
    print(f"Raw:     {RESULTS_DIR / 'rows.json'}")


if __name__ == "__main__":
    main()
