# ReColor — Ground-Truth-Anchored Evaluation

**Generated:** 2026-04-30 23:37:52

This report answers the adviser's question: **"Where is the ground truth, and
what is the actual truth produced by the algorithms?"** Every score below is
computed against an EXTERNAL reference that does not depend on the algorithm
under test.

| Camera | Ground truth | Result |
|---|---|---|
| Color Identifier | X-Rite ColorChecker 24 + 30 CSS named colors with literature-published color classes | **88.9% accuracy** |
| CVD Simulation | Viénot/Brettel 1999 confusion-line invariants | **90.9% invariants satisfied** |
| Camera Enhancement | Synthetic confused color pairs (post-CVD ΔE < 5) — discrimination gain | DAL **+6.91** / HUE **+1.88** ΔE gain |

---

## Suite GT-1 — Color Identifier vs Known-Lab Ground Truth

**Ground truth source:** X-Rite ColorChecker 24 (industry standard, used in
print/photography for 40+ years) + 30 CSS-named colors (W3C CSS Color Module
Level 3, the canonical sRGB named-color list).

**Method:** for each labeled sample, run `identifyColor(r,g,b)` and check
whether the predicted class matches the ground-truth class. Failures are split:

* **low-ΔE failures** (ΔE < 8 to nearest DB match) → DB has a confidently-wrong
  nearest neighbour at the class boundary. Symptom: DB labels disagree with
  ground-truth labels (re-labeling fix).
* **high-ΔE failures** (ΔE ≥ 8) → genuinely ambiguous color, far from any DB
  entry. Symptom: DB sparsity (add-entries fix).

### Headline
* **Total samples:** 54 (24 ColorChecker + 30 CSS)
* **Correct:** 48
* **Accuracy:** **88.9%**
* **Failure split:** 2 low-ΔE (boundary), 4 high-ΔE (sparsity)

### Per-class accuracy
| Class | Correct | Total | Accuracy |
|---|---|---|---|
| Blue | 5 | 6 | 83.3% |
| Brown | 4 | 4 | 100.0% |
| Cyan | 4 | 5 | 80.0% |
| Green | 5 | 6 | 83.3% |
| Neutral | 12 | 12 | 100.0% |
| Orange | 3 | 3 | 100.0% |
| Pink | 4 | 5 | 80.0% |
| Red | 2 | 4 | 50.0% |
| Violet | 5 | 5 | 100.0% |
| Yellow | 4 | 4 | 100.0% |

### Confusion matrix (rows = expected, columns = predicted)

| expected\predicted | Blue | Brown | Cyan | Green | Neutral | Orange | Pink | Red | Violet | Yellow |
|---|---|---|---|---|---|---|---|---|---|---|
| Blue | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| Brown | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cyan | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Green | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 1 |
| Neutral | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 |
| Orange | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| Pink | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 0 | 0 | 0 |
| Red | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 |
| Violet | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| Yellow | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |


### Hypothesis test: "Is the 88.9% accuracy due to DB sparsity?"
Failures break down as:
* **4 high-ΔE failures (DB sparsity):** the sample is far from any DB entry.
  Adding new DB entries near these regions would fix them.
* **2 low-ΔE failures (boundary mislabeling):** the algorithm found a confident
  match in the DB, but the DB entry has the wrong class label. Adding entries
  alone will NOT fix these — they need re-labeling.

**Verdict for your hypothesis:**

**You are mostly right.** 4 of 6 failures are sparsity-driven.
Adding DB entries in the under-represented color regions (see high-ΔE table
below) would lift accuracy meaningfully. The remaining 2 are boundary
issues that need re-labeling, not new entries.


#### High-ΔE failures (DB sparsity):

| sample | expected | predicted | ΔE |
|---|---|---|---|
| Light Skin | Pink | Orange | 8.2 |
| Yellow Green | Green | Yellow | 10.05 |
| Blue | Blue | Violet | 9.79 |
| Cyan | Cyan | Blue | 10.86 |


#### Low-ΔE failures (DB boundary mislabeling):

| sample | expected | predicted | ΔE |
|---|---|---|---|
| Red | Red | Brown | 4.52 |
| tomato | Red | Orange | 7.1 |


---

## Suite GT-2 — CVD Simulation vs Viénot 1999 Invariants

**Ground truth source:** the Viénot/Brettel/Mollon 1999 papers established that
under correct CVD simulation, color pairs ON a CVD's confusion line should
COLLAPSE to nearly identical perception, while pairs OFF the confusion line
should be preserved. We test both directions.

**Method:** for each CVD type, run clinically-known confusion pairs through the
simulation matrix and verify the post-simulation ΔE drops to <50% of the
original (collapse). Run a control pair off the confusion line and verify the
post-simulation ΔE stays above 50% (preservation).

### Headline
* **Total invariants tested:** 11
* **Passed:** 10
* **Pass rate:** **90.9%**

| type | cvd | pair | ΔE before | ΔE after | ratio | expected | pass |
|---|---|---|---|---|---|---|---|
| confusion | Protan | Red 255,0,0 vs Green 0,128,0 | 72.18 | 7.77 | 0.108 | collapse | ✓ |
| confusion | Protan | Red 200,40,40 vs Green 50,150,50 | 68.44 | 22.78 | 0.333 | collapse | ✓ |
| confusion | Protan | Pure Red vs Pure Green | 86.61 | 42.29 | 0.488 | collapse | ✓ |
| control | Protan | Red vs Blue | 52.88 | 64.16 | 1.213 | preserved | ✓ |
| confusion | Deutan | Red 255,0,0 vs Green 0,128,0 | 72.18 | 15.38 | 0.213 | collapse | ✓ |
| confusion | Deutan | Red 200,40,40 vs Green 50,150,50 | 68.44 | 5.3 | 0.077 | collapse | ✓ |
| confusion | Deutan | Pure Red vs Pure Green | 86.61 | 19.61 | 0.226 | collapse | ✓ |
| control | Deutan | Red vs Blue | 52.88 | 74.64 | 1.412 | preserved | ✓ |
| confusion | Tritan | Pure Blue vs Pure Yellow | 103.43 | 51.14 | 0.494 | collapse | ✓ |
| confusion | Tritan | Sky Blue vs Khaki | 40.7 | 36.49 | 0.897 | collapse | ✗ |
| control | Tritan | Red vs Green | 86.61 | 76.2 | 0.88 | preserved | ✓ |

---

## Suite GT-3 — Enhancement Discrimination Gain (the missing metric)

**Ground truth source:** synthetic color pairs that are genuinely
distinguishable to normal vision (ΔE > 8) but **become confused under CVD
simulation** (post-simulation ΔE < 5; the Sharma & Bala 2002 confusion
threshold). Each such pair is a verified case of "CVD will confuse these".

**Why this is real ground truth:** the input pairs are objectively confused —
verified BEFORE enhancement is applied, by simulating CVD on the unenhanced
versions. The output is a measurable change in ΔE between the same pair after
enhancement+simulation. Either the gap closed (positive gain = algorithm
helped) or it didn't.

**Why this is the right metric for enhancement:** the previous reports
measured ΔE-vs-original-normal-view (faithfulness). That target is unattainable
because the missing cone information is physiologically lost. Discrimination
gain measures the *actual goal* — did previously-confused pairs become
distinguishable.

### Per-CVD discrimination gain (mean ΔE increase, higher = better)

| CVD | n pairs | DAL mean gain | DAL % positive | HUE mean gain | HUE % positive | Winner |
|---|---|---|---|---|---|---|
| Protan | 60 | 13.62 | 98.3% | 3.29 | 61.7% | DAL |
| Deutan | 60 | 4.03 | 86.7% | 1.88 | 55.0% | DAL |
| Tritan | 36 | 3.08 | 83.3% | 0.46 | 50.0% | DAL |

### Aggregate
* **Daltonization mean gain:** **+6.91** ΔE across all CVD types
* **Hue Rotation mean gain:** **+1.88** ΔE across all CVD types
* **Winner on discrimination gain:** **Daltonization**

### Reading this
A positive mean gain means previously-confused color pairs become more
distinguishable to a CVD viewer after enhancement. **A higher gain means the
algorithm restored more discrimination.**

The percentage-positive column tells you how often each algorithm helped at
all (vs hurt or did nothing). An algorithm with high mean gain and high %
positive is consistently helpful. High mean / low % means it helps a few
extreme cases dramatically but often does nothing.

---

## What this report tells your adviser

1. **Color Identifier ground truth: external, named, peer-reviewed.**
   The X-Rite ColorChecker 24 is the industry-standard color reference (used
   for camera calibration in cinema and print since 1976). The CSS Color
   Module Level 3 is the W3C-published canonical named-color list. We feed
   their published Lab values to the algorithm and check class agreement.
   Result: **88.9% accuracy**.

2. **CVD Simulation ground truth: literature invariants, not algorithm self-checks.**
   The Viénot 1999 paper *defines* what correct CVD simulation must do: pairs
   on the confusion line must collapse, pairs off it must be preserved. We
   verify both directions. Result: **90.9% of invariants satisfied**.

3. **Camera Enhancement ground truth: synthetic confused pairs + discrimination gain.**
   We don't measure faithfulness anymore — that target is impossible. We
   measure whether previously-confused pairs become distinguishable. The input
   confused-ness is verified objectively (post-simulation ΔE < 5), the gain is
   a direct measurement. Result: **Daltonization** wins on
   discrimination gain (6.91 vs 1.88 ΔE).

This is the answer to *"where is the ground truth?"* — three external
references (ColorChecker, Viénot invariants, confusion-pair construction),
three measurements against them, three numbers.
