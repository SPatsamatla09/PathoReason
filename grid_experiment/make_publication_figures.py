"""Create publication-quality figures from completed Track-A analyses.

The script operates on image-level or image-clustered summaries to avoid
treating correlated perturbations as independent observations. It creates:

1. Track-A MHIST ROC curve from each image's original PLIP probability.
2. Cited-removal versus tissue-matched-control probability-drop distributions.
3. Per-feature control-adjusted effects with clustered bootstrap 95% CIs.
4. Cross-dataset control-adjusted effects with bootstrap 95% CIs.
5. A combined four-panel figure containing the above.

Example
-------
python3 grid_experiment/make_publication_figures.py \
  --mhist-dir grid_experiment/runs/track_a_faithfulness_gpt4o_full_corrected \
  --cross-dir grid_experiment/runs/cross_dataset_comparison \
  --out-dir grid_experiment/figures/publication
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OCCLUSION_ORDER = ["black", "blur", "mean"]
OCCLUSION_LABELS = {"black": "Black", "blur": "Blur", "mean": "Mean color"}
COLORS = {"black": "#3D405B", "blur": "#2A9D8F", "mean": "#E07A5F"}


def roc_curve_manual(y_true: np.ndarray, scores: np.ndarray):
    """Return FPR, TPR, and trapezoidal AUC for binary labels."""
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order].astype(int)
    s = scores[order]
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires both HP and SSA ground-truth examples")

    distinct = np.r_[True, s[1:] != s[:-1]]
    tp = np.cumsum(y)[distinct]
    fp = np.cumsum(1 - y)[distinct]
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def load_original_predictions(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    needed = {"image", "true_label", "original_label", "original_probability"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"all_predictions.csv missing columns: {sorted(missing)}")
    original = raw[list(needed)].drop_duplicates("image").copy()
    if original.image.duplicated().any():
        raise ValueError("original prediction fields are not unique within image")
    original["ssa_probability"] = np.where(
        original.original_label.eq("SSA"),
        original.original_probability,
        1.0 - original.original_probability,
    )
    original["is_ssa"] = original.true_label.eq("SSA").astype(int)
    return original


def panel_roc(ax, originals: pd.DataFrame):
    fpr, tpr, auc = roc_curve_manual(
        originals.is_ssa.to_numpy(), originals.ssa_probability.to_numpy()
    )
    ax.plot(fpr, tpr, color="#264653", lw=2.2, label=f"PLIP (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#888888", label="Chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Track-A classification ROC")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.02, 0.02, f"n = {len(originals):,} images", transform=ax.transAxes, fontsize=9)
    return auc


def panel_drop_distributions(ax, image_effects: pd.DataFrame):
    mhist = image_effects[image_effects.dataset.eq("MHIST")].copy()
    positions, values, colors, labels = [], [], [], []
    pos = 1
    for occ in OCCLUSION_ORDER:
        group = mhist[mhist.occlusion.eq(occ)]
        if group.empty:
            continue
        positions.extend([pos, pos + 0.75])
        values.extend([group.cited_drop.to_numpy(), group.control_drop.to_numpy()])
        colors.extend([COLORS[occ], "#B8B8B8"])
        labels.append((pos + 0.375, OCCLUSION_LABELS[occ]))
        pos += 2.25
    bp = ax.boxplot(
        values, positions=positions, widths=0.55, patch_artist=True,
        showfliers=False, medianprops={"color": "white", "linewidth": 1.4},
        whiskerprops={"linewidth": 1}, capprops={"linewidth": 1},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    ax.set_xticks([x for x, _ in labels], [lab for _, lab in labels])
    ax.set_ylabel("Original-class probability drop")
    ax.set_title("Cited removal versus matched control")
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor="#3D405B", label="Cited cells"),
                 Patch(facecolor="#B8B8B8", label="Tissue-matched control")],
        frameon=False, fontsize=8, loc="upper right",
    )


def panel_features(ax, feature_table: pd.DataFrame):
    table = feature_table[feature_table.inferentially_eligible.fillna(False)].copy()
    feature_order = (
        table.groupby("feature").n_images.max().sort_values(ascending=True).index.tolist()
    )
    y = np.arange(len(feature_order), dtype=float)
    offsets = {"black": -0.22, "blur": 0.0, "mean": 0.22}
    for occ in OCCLUSION_ORDER:
        part = table[table.occlusion.eq(occ)].set_index("feature").reindex(feature_order)
        valid = part.mean_effect_over_control.notna().to_numpy()
        means = part.mean_effect_over_control.to_numpy(float)
        lo = part.effect_ci_low.to_numpy(float)
        hi = part.effect_ci_high.to_numpy(float)
        ax.errorbar(
            means[valid], y[valid] + offsets[occ],
            xerr=np.vstack([means[valid] - lo[valid], hi[valid] - means[valid]]),
            fmt="o", ms=4.8, capsize=2, color=COLORS[occ],
            label=OCCLUSION_LABELS[occ], linewidth=1.2,
        )
    ax.axvline(0, color="#777777", ls="--", lw=1)
    ax.set_yticks(y, feature_order)
    ax.set_xlabel("Cited drop − control drop")
    ax.set_title("Per-feature control-adjusted effects")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.text(
        0.99, 0.01, "Positive favors cited evidence",
        ha="right", va="bottom", transform=ax.transAxes, fontsize=8, color="#555555"
    )


def panel_cross_dataset(ax, summary: pd.DataFrame):
    datasets = [d for d in ["MHIST", "PatchCamelyon"] if d in set(summary.dataset)]
    x = np.arange(len(OCCLUSION_ORDER), dtype=float)
    width = 0.34
    dataset_colors = {"MHIST": "#457B9D", "PatchCamelyon": "#F4A261"}
    for i, dataset in enumerate(datasets):
        part = summary[summary.dataset.eq(dataset)].set_index("occlusion").reindex(OCCLUSION_ORDER)
        means = part.mean_effect_over_control.to_numpy(float)
        lo = part.effect_ci_low.to_numpy(float)
        hi = part.effect_ci_high.to_numpy(float)
        offset = (i - (len(datasets) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, color=dataset_colors[dataset], label=dataset)
        ax.errorbar(
            x + offset, means, yerr=np.vstack([means - lo, hi - means]),
            fmt="none", ecolor="#222222", elinewidth=1, capsize=2,
        )
    ax.axhline(0, color="#777777", ls="--", lw=1)
    ax.set_xticks(x, [OCCLUSION_LABELS[o] for o in OCCLUSION_ORDER])
    ax.set_ylabel("Cited drop − control drop")
    ax.set_title("Cross-dataset comparison")
    ax.legend(frameon=False, fontsize=8)


def save_single(draw, data, path: Path, size=(6.5, 4.6)):
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    draw(ax, data)
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhist-dir", type=Path, required=True)
    ap.add_argument("--cross-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    originals = load_original_predictions(args.mhist_dir / "all_predictions.csv")
    feature_table = pd.read_csv(args.mhist_dir / "robust_per_feature.csv")
    image_effects = pd.read_csv(args.cross_dir / "image_level_effects.csv")
    dataset_summary = pd.read_csv(args.cross_dir / "dataset_summary.csv")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    save_single(panel_roc, originals, args.out_dir / "figure_roc")
    save_single(panel_drop_distributions, image_effects, args.out_dir / "figure_drop_distributions")
    save_single(panel_features, feature_table, args.out_dir / "figure_per_feature", size=(7.2, 5.2))
    save_single(panel_cross_dataset, dataset_summary, args.out_dir / "figure_cross_dataset")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    auc = panel_roc(axes[0, 0], originals)
    panel_drop_distributions(axes[0, 1], image_effects)
    panel_features(axes[1, 0], feature_table)
    panel_cross_dataset(axes[1, 1], dataset_summary)
    for label, ax in zip("ABCD", axes.flat):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=14, fontweight="bold")
    fig.savefig(args.out_dir / "figure_combined.png", dpi=400, bbox_inches="tight")
    fig.savefig(args.out_dir / "figure_combined.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"MHIST ROC AUC: {auc:.6f} (n={len(originals)})")
    print(f"outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
