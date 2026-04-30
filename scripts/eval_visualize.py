"""
ReColor Ground-Truth Evaluation — Visualizations
Generates publication-quality charts for all 3 suites.
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
VIS_DIR = ROOT / "evaluation" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Load data (run eval_ground_truth.py first if missing)
gt_path = RESULTS / "ground_truth.json"
if not gt_path.exists():
    print("Running eval_ground_truth.py first...")
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_ground_truth as egt
    egt.main()

with open(gt_path, encoding="utf-8") as f:
    data = json.load(f)

s1, s2, s3 = data["suite1"], data["suite2"], data["suite3"]

# ── Theme ──
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "text.color": "#c9d1d9", "grid.color": "#21262d",
    "font.family": "sans-serif", "font.size": 11,
})
ACCENT = ["#58a6ff","#3fb950","#f0883e","#f85149","#bc8cff","#79c0ff","#d2a8ff","#ff7b72","#ffa657","#7ee787"]
CVD_COLORS = {"Protan":"#f85149","Deutan":"#3fb950","Tritan":"#58a6ff"}

def save(fig, name):
    fig.savefig(VIS_DIR / name, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  OK: {name}")

# ═══════════════════════════════════════════════
# SUITE 1: Identifier Ground Truth
# ═══════════════════════════════════════════════
def vis_suite1():
    print("Suite 1 visualizations...")
    details = s1["details"]

    # 1a — ColorChecker grid with pass/fail overlay
    fig, ax = plt.subplots(figsize=(14, 7))
    cols, rows_g = 6, 4
    for i, d in enumerate(details):
        r, c = divmod(i, cols)
        rgb = [v/255 for v in d["rgb"]]
        rect = mpatches.FancyBboxPatch((c, rows_g-1-r), 0.92, 0.88, boxstyle="round,pad=0.04",
                                        facecolor=rgb, edgecolor="#3fb950" if d["correct"] else "#f85149", linewidth=3)
        ax.add_patch(rect)
        lum = 0.299*d["rgb"][0] + 0.587*d["rgb"][1] + 0.114*d["rgb"][2]
        tc = "white" if lum < 128 else "black"
        ax.text(c+0.46, rows_g-1-r+0.58, d["patch"], ha="center", va="center", fontsize=7, fontweight="bold", color=tc)
        mark = "✓" if d["correct"] else "✗"
        ax.text(c+0.46, rows_g-1-r+0.3, f"{d['predicted']} {mark}", ha="center", va="center", fontsize=6.5, color=tc, style="italic")
        ax.text(c+0.46, rows_g-1-r+0.12, f"{d['confidence']}%", ha="center", va="center", fontsize=6, color=tc, alpha=0.7)
    ax.set_xlim(-0.1, cols+0.1); ax.set_ylim(-0.2, rows_g+0.1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Suite 1 — ColorChecker 24 Identification: {s1['correct']}/{s1['total']} ({s1['accuracy_pct']}%)",
                 fontsize=15, fontweight="bold", pad=15)
    legend_els = [mpatches.Patch(edgecolor="#3fb950", facecolor="none", linewidth=2, label="Correct"),
                  mpatches.Patch(edgecolor="#f85149", facecolor="none", linewidth=2, label="Misclassified")]
    ax.legend(handles=legend_els, loc="lower center", ncol=2, frameon=False, fontsize=10)
    save(fig, "suite1_colorchecker_grid.png")

    # 1b — Confusion matrix heatmap
    classes = ["Red","Orange","Yellow","Green","Cyan","Blue","Violet","Pink","Brown","Neutral"]
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for d in details:
        if d["expected"] in classes and d["predicted"] in classes:
            cm[classes.index(d["expected"])][classes.index(d["predicted"])] += 1
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Expected", fontsize=12)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i][j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                        fontweight="bold", color="white" if v >= 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Count")
    ax.set_title("Suite 1 — Confusion Matrix (ColorChecker 24)", fontsize=14, fontweight="bold", pad=12)
    save(fig, "suite1_confusion_matrix.png")

    # 1c — Confidence bar chart
    fig, ax = plt.subplots(figsize=(14, 5))
    names = [d["patch"] for d in details]
    confs = [d["confidence"] for d in details]
    colors = ["#3fb950" if d["correct"] else "#f85149" for d in details]
    bars = ax.bar(range(len(names)), confs, color=colors, edgecolor="#30363d", linewidth=0.5)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("Confidence (%)", fontsize=11); ax.set_ylim(0, 110)
    ax.axhline(y=np.mean(confs), color="#58a6ff", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(confs):.0f}%")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("Suite 1 — Per-Patch Confidence", fontsize=14, fontweight="bold", pad=12)
    save(fig, "suite1_confidence_bars.png")

# ═══════════════════════════════════════════════
# SUITE 2: Simulation Fidelity
# ═══════════════════════════════════════════════
def vis_suite2():
    print("Suite 2 visualizations...")
    suites = s2["suites"]

    # 2a — Delta-E comparison across CVD types
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for idx, suite in enumerate(suites):
        ax = axes[idx]
        cvd = suite["cvd_type"]
        colors_data = suite["colors"]
        names = [c["color"] for c in colors_data]
        des = [c["deltaE"] for c in colors_data]
        clr = CVD_COLORS[cvd]
        bars = ax.barh(range(len(names)), des, color=clr, alpha=0.8, edgecolor="#30363d")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("ΔE₇₆ (Reference vs App)", fontsize=10)
        ax.set_title(f"{cvd}", fontsize=13, fontweight="bold", color=clr)
        ax.axvline(x=1.0, color="#ffa657", linestyle="--", linewidth=1, alpha=0.7, label="JND (ΔE=1)")
        for i, v in enumerate(des):
            if v > 0:
                ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8, color="#8b949e")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Suite 2 — Simulation Fidelity: Viénot Reference vs App Output (Verdict: {s2['verdict']})",
                 fontsize=14, fontweight="bold", y=1.02)
    save(fig, "suite2_simulation_fidelity.png")

    # 2b — Input vs Reference vs App color swatches (visual comparison)
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    for idx, suite in enumerate(suites):
        ax = axes[idx]
        cvd = suite["cvd_type"]
        colors_data = suite["colors"]
        n = len(colors_data)
        for i, c in enumerate(colors_data):
            inp = [int(x) if isinstance(x, (int, float)) else int(x) for x in c["input_rgb"]]
            ref = [int(x) if isinstance(x, (int, float)) else int(str(x).replace("np.int64(","").replace(")","")) for x in c["reference"]]
            app = [int(x) if isinstance(x, (int, float)) else int(str(x).replace("np.int64(","").replace(")","")) for x in c["app_output"]]
            w = 0.9/n
            for row_idx, (rgb_vals, label) in enumerate([(inp,"Input"),(ref,"Reference"),(app,"App")]):
                rgb_n = [v/255 for v in rgb_vals]
                rect = mpatches.Rectangle((i*w, 2-row_idx), w*0.95, 0.9, facecolor=rgb_n, edgecolor="#30363d", linewidth=0.5)
                ax.add_patch(rect)
            if idx == 0:
                ax.text(i*w + w*0.47, 3.1, c["color"], ha="center", va="bottom", fontsize=6.5, rotation=45)
        ax.set_xlim(0, 1); ax.set_ylim(-0.1, 3.5 if idx==0 else 3.1)
        ax.set_yticks([0.45, 1.45, 2.45]); ax.set_yticklabels(["App","Reference","Input"], fontsize=9)
        ax.set_xticks([]); ax.set_title(f"{cvd}", fontsize=12, fontweight="bold", color=CVD_COLORS[cvd])
    fig.suptitle("Suite 2 — Visual Swatch Comparison (Input → Viénot Reference → App)", fontsize=14, fontweight="bold", y=1.01)
    save(fig, "suite2_swatch_comparison.png")

# ═══════════════════════════════════════════════
# SUITE 3: Enhancement Discrimination Gain
# ═══════════════════════════════════════════════
def vis_suite3():
    print("Suite 3 visualizations...")

    # 3a — Summary grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    cvd_names = [r["cvd_type"] for r in s3]
    dal_means = [r["dal_mean_gain"] for r in s3]
    hue_means = [r["hue_mean_gain"] for r in s3]
    x = np.arange(len(cvd_names))
    w = 0.35
    b1 = ax.bar(x - w/2, dal_means, w, label="Daltonization", color="#f0883e", edgecolor="#30363d")
    b2 = ax.bar(x + w/2, hue_means, w, label="Hue Rotation", color="#58a6ff", edgecolor="#30363d")
    ax.set_xticks(x); ax.set_xticklabels(cvd_names, fontsize=12)
    ax.set_ylabel("Mean ΔE Gain", fontsize=12)
    ax.axhline(y=2, color="#ffa657", linestyle="--", linewidth=1, alpha=0.7, label="Threshold (ΔE gain = 2)")
    ax.legend(frameon=False, fontsize=10)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Suite 3 — Mean Discrimination Gain by Algorithm & CVD Type", fontsize=14, fontweight="bold", pad=12)
    save(fig, "suite3_mean_gain_comparison.png")

    # 3b — Positive improvement % stacked bars
    fig, ax = plt.subplots(figsize=(10, 5))
    dal_pos = [r["dal_positive_pct"] for r in s3]
    hue_pos = [r["hue_positive_pct"] for r in s3]
    x = np.arange(len(cvd_names))
    ax.bar(x - w/2, dal_pos, w, label="Daltonization", color="#f0883e", edgecolor="#30363d")
    ax.bar(x + w/2, hue_pos, w, label="Hue Rotation", color="#58a6ff", edgecolor="#30363d")
    ax.set_xticks(x); ax.set_xticklabels(cvd_names, fontsize=12)
    ax.set_ylabel("% Pairs with Gain > 2 ΔE", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, fontsize=10)
    for i in range(len(cvd_names)):
        ax.text(i - w/2, dal_pos[i]+2, f"{dal_pos[i]}%", ha="center", fontsize=9, fontweight="bold", color="#f0883e")
        ax.text(i + w/2, hue_pos[i]+2, f"{hue_pos[i]}%", ha="center", fontsize=9, fontweight="bold", color="#58a6ff")
    ax.set_title("Suite 3 — Positive Improvement Rate (Gain > 2 ΔE)", fontsize=14, fontweight="bold", pad=12)
    save(fig, "suite3_positive_rate.png")

    # 3c — Per-pair scatter: confused ΔE vs post-enhancement ΔE (per CVD)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    for idx, r in enumerate(s3):
        ax = axes[idx]
        cvd = r["cvd_type"]
        pairs = r["pairs"]
        de_confused = [p["de_confused"] for p in pairs]
        de_dal = [p["de_after_dal"] for p in pairs]
        de_hue = [p["de_after_hue"] for p in pairs]
        ax.scatter(de_confused, de_dal, c="#f0883e", s=60, alpha=0.8, edgecolors="#30363d", linewidths=0.5, label="DAL", zorder=3)
        ax.scatter(de_confused, de_hue, c="#58a6ff", s=60, alpha=0.8, edgecolors="#30363d", linewidths=0.5, label="HUE", marker="^", zorder=3)
        lim = max(max(de_dal+de_hue+de_confused)+5, 10)
        ax.plot([0, lim], [0, lim], color="#484f58", linestyle="--", linewidth=1, alpha=0.5, label="No gain line")
        ax.set_xlabel("ΔE Post-Sim (Confused)", fontsize=10)
        if idx == 0: ax.set_ylabel("ΔE Post-Enhancement+Sim", fontsize=10)
        ax.set_title(f"{cvd} ({len(pairs)} pairs)", fontsize=12, fontweight="bold", color=CVD_COLORS[cvd])
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(0, max(de_confused)+1)
    fig.suptitle("Suite 3 — Discrimination Gain: Confused ΔE → Post-Enhancement ΔE", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "suite3_scatter_gain.png")

    # 3d — Color pair swatches for Protan (top 8 pairs)
    protan = s3[0]
    pairs = protan["pairs"][:8]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, len(pairs)*1.2))
    if len(pairs) == 1: axes = [axes]
    for i, p in enumerate(pairs):
        ax = axes[i]
        ax.set_xlim(0, 10); ax.set_ylim(0, 1)
        ax.axis("off")
        c1 = [v/255 for v in p["c1"]]
        c2 = [v/255 for v in p["c2"]]
        # C1 swatch
        ax.add_patch(mpatches.Rectangle((0, 0.1), 0.8, 0.8, facecolor=c1, edgecolor="#30363d"))
        # C2 swatch
        ax.add_patch(mpatches.Rectangle((1, 0.1), 0.8, 0.8, facecolor=c2, edgecolor="#30363d"))
        # Labels
        ax.text(2.1, 0.5, f"ΔE orig: {p['de_original']}", va="center", fontsize=8, color="#c9d1d9")
        ax.text(3.5, 0.5, f"ΔE confused: {p['de_confused']}", va="center", fontsize=8, color="#f85149")
        ax.text(5.3, 0.5, f"DAL gain: +{p['dal_gain']}", va="center", fontsize=8, color="#f0883e", fontweight="bold")
        ax.text(7.0, 0.5, f"HUE gain: +{p['hue_gain']}", va="center", fontsize=8, color="#58a6ff", fontweight="bold")
        # Gain bar
        max_g = max(abs(p["dal_gain"]), abs(p["hue_gain"]), 1)
        dal_w = max(0, p["dal_gain"]/max_g) * 1.2
        hue_w = max(0, p["hue_gain"]/max_g) * 1.2
        ax.add_patch(mpatches.Rectangle((8.5, 0.55), dal_w, 0.35, facecolor="#f0883e", alpha=0.8))
        ax.add_patch(mpatches.Rectangle((8.5, 0.1), hue_w, 0.35, facecolor="#58a6ff", alpha=0.8))
    fig.suptitle("Suite 3 — Protan: Top Confused Pairs & Discrimination Gain", fontsize=13, fontweight="bold", y=1.01)
    save(fig, "suite3_protan_pair_swatches.png")

    # 3e — Median gain comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    dal_med = [r["dal_median_gain"] for r in s3]
    hue_med = [r["hue_median_gain"] for r in s3]
    x = np.arange(len(cvd_names))
    ax.bar(x - w/2, dal_med, w, label="Daltonization (Median)", color="#f0883e", edgecolor="#30363d")
    ax.bar(x + w/2, hue_med, w, label="Hue Rotation (Median)", color="#58a6ff", edgecolor="#30363d")
    ax.set_xticks(x); ax.set_xticklabels(cvd_names, fontsize=12)
    ax.set_ylabel("Median ΔE Gain", fontsize=11)
    ax.axhline(y=2, color="#ffa657", linestyle="--", linewidth=1, alpha=0.7)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("Suite 3 — Median Discrimination Gain", fontsize=14, fontweight="bold", pad=12)
    save(fig, "suite3_median_gain.png")

# ═══════════════════════════════════════════════
# DASHBOARD — Combined overview
# ═══════════════════════════════════════════════
def vis_dashboard():
    print("Dashboard...")
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # Panel 1: Suite 1 accuracy donut
    ax1 = fig.add_subplot(gs[0, 0])
    correct = s1["correct"]; total = s1["total"]; miss = total - correct
    wedges, _ = ax1.pie([correct, miss], colors=["#3fb950","#f85149"], startangle=90,
                         wedgeprops=dict(width=0.35, edgecolor="#161b22", linewidth=2))
    ax1.text(0, 0, f"{s1['accuracy_pct']}%", ha="center", va="center", fontsize=22, fontweight="bold", color="#c9d1d9")
    ax1.set_title("Suite 1: Identifier\nAccuracy", fontsize=11, fontweight="bold", pad=10)

    # Panel 2: Suite 2 max errors
    ax2 = fig.add_subplot(gs[0, 1])
    cvd_names = [s["cvd_type"] for s in s2["suites"]]
    max_errs = [float(str(s["max_channel_error"])) for s in s2["suites"]]
    colors = [CVD_COLORS[c] for c in cvd_names]
    ax2.bar(cvd_names, max_errs, color=colors, edgecolor="#30363d")
    ax2.axhline(y=1, color="#ffa657", linestyle="--", linewidth=1, label="Pass threshold")
    ax2.set_ylabel("Max Channel Error")
    ax2.set_title(f"Suite 2: Sim Fidelity\n({s2['verdict']})", fontsize=11, fontweight="bold", pad=10)
    ax2.legend(frameon=False, fontsize=8)

    # Panel 3: Suite 3 winners
    ax3 = fig.add_subplot(gs[0, 2])
    dal_wins = sum(1 for r in s3 if r["dal_mean_gain"] > r["hue_mean_gain"])
    hue_wins = len(s3) - dal_wins
    ax3.pie([dal_wins, hue_wins], labels=["DAL","HUE"], colors=["#f0883e","#58a6ff"],
            autopct="%1.0f%%", startangle=90, wedgeprops=dict(edgecolor="#161b22", linewidth=2),
            textprops=dict(fontsize=10, fontweight="bold"))
    ax3.set_title("Suite 3: Enhancement\nWinner by CVD", fontsize=11, fontweight="bold", pad=10)

    # Panel 4-6: Suite 3 gain bars across bottom
    for idx, r in enumerate(s3):
        ax = fig.add_subplot(gs[1, idx])
        cvd = r["cvd_type"]
        pairs = r["pairs"]
        dal_gains = [p["dal_gain"] for p in pairs]
        hue_gains = [p["hue_gain"] for p in pairs]
        x = np.arange(len(pairs))
        ax.bar(x - 0.2, dal_gains, 0.4, color="#f0883e", alpha=0.85, label="DAL")
        ax.bar(x + 0.2, hue_gains, 0.4, color="#58a6ff", alpha=0.85, label="HUE")
        ax.axhline(y=2, color="#ffa657", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(y=0, color="#484f58", linewidth=0.5)
        ax.set_xlabel("Pair #", fontsize=9)
        ax.set_ylabel("ΔE Gain", fontsize=9)
        ax.set_title(f"{cvd} Per-Pair Gain", fontsize=10, fontweight="bold", color=CVD_COLORS[cvd])
        if idx == 0: ax.legend(frameon=False, fontsize=8)

    # Panel 7: Confusion matrix mini
    ax7 = fig.add_subplot(gs[2, 0:2])
    classes = ["Red","Org","Yel","Grn","Cyn","Blu","Vio","Pnk","Brn","Neu"]
    full_classes = ["Red","Orange","Yellow","Green","Cyan","Blue","Violet","Pink","Brown","Neutral"]
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for d in s1["details"]:
        if d["expected"] in full_classes and d["predicted"] in full_classes:
            cm[full_classes.index(d["expected"])][full_classes.index(d["predicted"])] += 1
    im = ax7.imshow(cm, cmap="YlOrRd", aspect="auto", vmin=0)
    ax7.set_xticks(range(len(classes))); ax7.set_yticks(range(len(classes)))
    ax7.set_xticklabels(classes, fontsize=8, rotation=45, ha="right")
    ax7.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i][j] > 0:
                ax7.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=10, fontweight="bold")
    ax7.set_title("Confusion Matrix", fontsize=10, fontweight="bold")

    # Panel 8: Summary table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    summary_text = (
        f"EVALUATION SUMMARY\n"
        f"{'─'*30}\n"
        f"Suite 1: {s1['correct']}/{s1['total']} ({s1['accuracy_pct']}%)\n"
        f"Suite 2: {s2['verdict']}\n\n"
    )
    for r in s3:
        summary_text += f"{r['cvd_type']}: DAL={r['dal_mean_gain']:+.1f}  HUE={r['hue_mean_gain']:+.1f}\n"
    summary_text += f"\nDAL wins {dal_wins}/{len(s3)} CVD types"
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=11, va="top",
             fontfamily="monospace", color="#c9d1d9",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1117", edgecolor="#30363d"))

    fig.suptitle("ReColor Ground-Truth Evaluation Dashboard", fontsize=18, fontweight="bold", y=0.98, color="#58a6ff")
    save(fig, "dashboard_overview.png")

# ═══════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output: {VIS_DIR}\n")
    vis_suite1()
    vis_suite2()
    vis_suite3()
    vis_dashboard()
    print(f"\nDone! {len(list(VIS_DIR.glob('*.png')))} images generated in {VIS_DIR}")
