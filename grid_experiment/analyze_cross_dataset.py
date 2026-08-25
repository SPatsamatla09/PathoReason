"""Compare Track-A citation effects between MHIST and PatchCamelyon.

The comparison unit is the image. MHIST feature rows are averaged within each
image before inference; PCam already has one cited/control pair per image.
Positive effects mean cited removal lowered the original-class probability
more than tissue-matched control removal. Negative effects mean the matched
control caused the larger drop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

SEED = 20260824


def bootstrap_ci(values, n_bootstrap=50_000, seed=SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [.025, .975])


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_mhist(path):
    table = pd.read_csv(path)
    required = {"image", "occlusion", "comprehensiveness", "control_drop_mean"}
    missing = required - set(table)
    if missing:
        raise ValueError(f"MHIST table missing columns: {sorted(missing)}")
    return table.groupby(["image", "occlusion"], as_index=False).agg(
        cited_drop=("comprehensiveness", "mean"),
        control_drop=("control_drop_mean", "mean"),
        n_features=("feature", "nunique"),
    ).assign(dataset="MHIST")


def load_pcam(masking_path, baseline_path):
    baselines = {
        row["image"]: row for row in load_jsonl(baseline_path)
        if not row.get("error")
    }
    rows = []
    for row in load_jsonl(masking_path):
        if row.get("error") or row["image"] not in baselines:
            continue
        baseline = baselines[row["image"]]
        original_label = baseline["label"]
        original_probability = float(baseline["class_probabilities"][original_label])
        masked_probability = float(row["class_probabilities"][original_label])
        rows.append({
            "image": row["image"], "occlusion": row["occlusion"],
            "arm": row["arm"], "drop": original_probability - masked_probability,
        })
    table = pd.DataFrame(rows)
    wide = table.pivot(index=["image", "occlusion"], columns="arm", values="drop").reset_index()
    return wide.rename(columns={"cited": "cited_drop", "tissue_matched": "control_drop"}).assign(
        dataset="PatchCamelyon", n_features=1
    )


def summarize(table):
    rows = []
    for (dataset, occlusion), group in table.groupby(["dataset", "occlusion"], sort=True):
        effects = group.effect_over_control.to_numpy(float)
        lo, hi = bootstrap_ci(effects)
        p = 1.0 if np.allclose(effects, 0) else float(wilcoxon(effects, zero_method="pratt").pvalue)
        rows.append({
            "dataset": dataset, "occlusion": occlusion, "n_images": group.image.nunique(),
            "mean_cited_drop": group.cited_drop.mean(),
            "mean_control_drop": group.control_drop.mean(),
            "mean_effect_over_control": effects.mean(),
            "effect_ci_low": lo, "effect_ci_high": hi, "wilcoxon_p": p,
        })
    return pd.DataFrame(rows)


def between_dataset(table):
    rows = []
    rng = np.random.default_rng(SEED)
    for occlusion, group in table.groupby("occlusion", sort=True):
        mhist = group[group.dataset == "MHIST"].effect_over_control.to_numpy(float)
        pcam = group[group.dataset == "PatchCamelyon"].effect_over_control.to_numpy(float)
        draws = (
            rng.choice(mhist, size=(50_000, len(mhist)), replace=True).mean(axis=1)
            - rng.choice(pcam, size=(50_000, len(pcam)), replace=True).mean(axis=1)
        )
        test = mannwhitneyu(mhist, pcam, alternative="two-sided")
        rows.append({
            "occlusion": occlusion, "mhist_n": len(mhist), "pcam_n": len(pcam),
            "mhist_mean_effect": mhist.mean(), "pcam_mean_effect": pcam.mean(),
            "mean_difference_mhist_minus_pcam": mhist.mean() - pcam.mean(),
            "difference_ci_low": np.quantile(draws, .025),
            "difference_ci_high": np.quantile(draws, .975),
            "mannwhitney_u": float(test.statistic), "mannwhitney_p": float(test.pvalue),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhist", type=Path, required=True)
    ap.add_argument("--pcam-masking", type=Path, required=True)
    ap.add_argument("--pcam-baseline", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mhist = load_mhist(args.mhist)
    pcam = load_pcam(args.pcam_masking, args.pcam_baseline)
    combined = pd.concat([mhist, pcam], ignore_index=True)
    combined["effect_over_control"] = combined.cited_drop - combined.control_drop
    summary = summarize(combined)
    comparison = between_dataset(combined)

    args.out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out / "image_level_effects.csv", index=False)
    summary.to_csv(args.out / "dataset_summary.csv", index=False)
    comparison.to_csv(args.out / "between_dataset_tests.csv", index=False)
    print(summary.to_string(index=False))
    print("\nBetween-dataset comparison")
    print(comparison.to_string(index=False))
    print(f"\noutputs -> {args.out}")


if __name__ == "__main__":
    main()
