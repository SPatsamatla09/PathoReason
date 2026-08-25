"""Complete the proposal's statistical analysis for the causal masking sweep.

Outputs paired cited-vs-control effects, bootstrap confidence intervals,
Wilcoxon tests, BH corrections, agreement strata, exploratory feature strata,
and a tidy image x feature x occlusion table.

The current sweep masks one combined citation set per image. Consequently,
feature rows are explicitly exploratory, not feature-specific causal ablations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE = ROOT / "runs" / "cte_p1.jsonl"
DEFAULT_MASKED = ROOT / "runs" / "masking_cte_p1_k3.jsonl"
DEFAULT_MANIFEST = ROOT / "selection_manifest.json"
DEFAULT_OUT = ROOT / "runs" / "faithfulness_stats"
SEED = 20260806


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def probability_of(label: str, predicted_label: str, confidence: float) -> float:
    """Binary-class probability assigned to ``label`` by a prediction."""
    confidence = float(confidence)
    return confidence if predicted_label == label else 1.0 - confidence


def bootstrap_mean_ci(values, *, n_bootstrap=10_000, seed=SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def wilcoxon_p(values) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, zero_method="pratt", alternative="two-sided").pvalue)


def bh_adjust(pvalues) -> list[float]:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return []
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0, 1)
    return out.tolist()


def load_tables(baseline_path: Path, masked_path: Path, manifest_path: Path):
    baseline_records = [
        r for r in read_jsonl(baseline_path)
        if r.get("replicate") == 1 and r.get("parsed") and not r.get("error")
    ]
    baselines = {r["image"]: r for r in baseline_records}
    metadata = {r["image"]: r for r in json.loads(manifest_path.read_text())["tiles"]}

    rows = []
    for r in read_jsonl(masked_path):
        if r.get("error") or not r.get("parsed"):
            continue
        base = baselines[r["image"]]
        meta = metadata[r["image"]]
        base_label = base["parsed"]["label"]
        base_conf = float(base["parsed"]["confidence"])
        new_label = r["parsed"]["label"]
        new_conf = float(r["parsed"]["confidence"])
        base_score = probability_of(base_label, base_label, base_conf)
        new_score = probability_of(base_label, new_label, new_conf)
        rows.append({
            "image": r["image"], "arm": r["arm"], "occlusion": r["occlusion"],
            "baseline_label": base_label, "baseline_confidence": base_conf,
            "new_label": new_label, "new_confidence": new_conf,
            "baseline_class_score": base_score, "new_baseline_class_score": new_score,
            "confidence_drop": base_score - new_score,
            "label_flip": int(new_label != base_label), "label_true": base["label_true"],
            "ssa_votes_out_of_7": meta["ssa_votes_out_of_7"],
            "majority_agreement": max(meta["ssa_votes_out_of_7"], 7 - meta["ssa_votes_out_of_7"]),
            "agreement_band": meta["agreement_band"],
            "masked_cells": "|".join(r["masked_cells"]),
            "tissue_match_gap_pp": r["match_quality"]["tissue_gap_pp"],
            "control_usable": bool(r["match_quality"].get("usable", False)),
            "model": r["model"], "prompt_id": r["prompt_id"],
        })
    outcomes = pd.DataFrame(rows)

    feature_rows = []
    for image, base in baselines.items():
        for evidence in base["parsed"].get("evidence", []):
            feature_rows.append({
                "image": image,
                "feature": evidence["feature"].strip().lower(),
                "feature_cells": "|".join(evidence.get("grid_cells_valid", [])),
            })
    features = pd.DataFrame(feature_rows).drop_duplicates(["image", "feature"])
    return outcomes, features


def pair_arms(outcomes: pd.DataFrame) -> pd.DataFrame:
    key = ["image", "occlusion"]
    values = ["confidence_drop", "label_flip", "new_label", "new_confidence"]
    wide = outcomes.pivot(index=key, columns="arm", values=values).reset_index()
    wide.columns = [
        "_".join(str(x) for x in col if x).rstrip("_") if isinstance(col, tuple) else col
        for col in wide.columns
    ]
    meta_cols = [
        "image", "agreement_band", "majority_agreement", "ssa_votes_out_of_7",
        "label_true", "baseline_label", "baseline_confidence", "masked_cells",
        "tissue_match_gap_pp", "control_usable", "model", "prompt_id",
    ]
    cited_meta = outcomes[outcomes.arm == "cited"][meta_cols].drop_duplicates("image")
    paired = wide.merge(cited_meta, on="image", validate="many_to_one")
    paired["effect_over_control"] = paired["confidence_drop_cited"] - paired["confidence_drop_tissue_matched"]
    paired["flip_effect_over_control"] = paired["label_flip_cited"].astype(int) - paired["label_flip_tissue_matched"].astype(int)
    # mask.py marks controls with unacceptable tissue mismatch as unusable.
    # Excluding them was pre-specified in the experiment README.
    return paired[paired["control_usable"]].reset_index(drop=True)


def summarize_groups(paired: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, group in paired.groupby(grouper, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        effects = group["effect_over_control"].to_numpy(float)
        flips = group["flip_effect_over_control"].to_numpy(float)
        lo, hi = bootstrap_mean_ci(effects)
        row = dict(zip(group_cols, keys))
        row.update({
            "n": len(group), "mean_effect_over_control": float(np.mean(effects)),
            "effect_ci_low": lo, "effect_ci_high": hi,
            "median_effect_over_control": float(np.median(effects)),
            "wilcoxon_p": wilcoxon_p(effects),
            "cited_flip_rate": float(group["label_flip_cited"].mean()),
            "control_flip_rate": float(group["label_flip_tissue_matched"].mean()),
            "mean_flip_effect_over_control": float(np.mean(flips)),
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    result["wilcoxon_p_bh"] = bh_adjust(result["wilcoxon_p"])
    return result


def make_feature_table(paired: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    table = paired.merge(features, on="image", how="inner", validate="many_to_many")
    table["feature_specific_ablation"] = False
    table["analysis_note"] = "Exploratory: image-level combined citation mask, not a feature-unique mask."
    return table


def markdown_report(control, by_feature, by_agreement, n_records) -> str:
    def md(df, columns):
        shown = df[columns].copy()
        for col in shown.select_dtypes(include=["float"]):
            shown[col] = shown[col].map(lambda x: f"{x:.4f}")
        header = "| " + " | ".join(columns) + " |"
        rule = "| " + " | ".join("---" for _ in columns) + " |"
        rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in shown.itertuples(index=False, name=None)]
        return "\n".join([header, rule, *rows])

    return f"""# Faithfulness analysis: cte_p1 combined-citation mask

## Scope

- {n_records} usable paired image/occlusion comparisons (18 images x 3 methods) from 120 successful Gemma calls.
- Control: equal-size, disjoint, tissue-matched cells (stronger than an unstratified random mask).
- Positive effect means cited-cell masking reduced the baseline-class score more than control masking.
- These are Track-B self-reported confidences, not Track-A softmax probabilities.
- Per-feature results are exploratory because the committed sweep masked one combined citation set per image.

## Control-adjusted effects

{md(control, ['occlusion', 'n', 'mean_effect_over_control', 'effect_ci_low', 'effect_ci_high', 'wilcoxon_p', 'wilcoxon_p_bh', 'cited_flip_rate', 'control_flip_rate'])}

## By annotator agreement

{md(by_agreement, ['agreement_band', 'occlusion', 'n', 'mean_effect_over_control', 'effect_ci_low', 'effect_ci_high', 'wilcoxon_p_bh'])}

## Exploratory per-feature breakdown

{md(by_feature, ['feature', 'occlusion', 'n', 'mean_effect_over_control', 'effect_ci_low', 'effect_ci_high', 'wilcoxon_p_bh'])}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--masked", type=Path, default=DEFAULT_MASKED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    outcomes, features = load_tables(args.baseline, args.masked, args.manifest)
    paired = pair_arms(outcomes)
    feature_table = make_feature_table(paired, features)
    control = summarize_groups(paired, ["occlusion"])
    agreement = summarize_groups(paired, ["agreement_band", "occlusion"])
    per_feature = summarize_groups(feature_table, ["feature", "occlusion"])
    args.out.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(args.out / "masking_outcomes.csv", index=False)
    paired.to_csv(args.out / "paired_results.csv", index=False)
    feature_table.to_csv(args.out / "image_feature_results.csv", index=False)
    control.to_csv(args.out / "control_effects.csv", index=False)
    agreement.to_csv(args.out / "agreement_breakdown.csv", index=False)
    per_feature.to_csv(args.out / "feature_breakdown.csv", index=False)
    report = markdown_report(control, per_feature, agreement, len(paired))
    (args.out / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
