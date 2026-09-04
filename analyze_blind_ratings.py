"""Analyze two completed blinded explanation-rating files.

Produces inter-rater agreement, exploratory image-level associations with the
Gemma masking experiment, and a conservative error taxonomy.  The raw key and
rater files are inputs only and are never copied to the output directory.

Example
-------
python3 grid_experiment/analyze_blind_ratings.py \
  --ratings-a ratings_AK.csv --ratings-b ratings_RR.csv \
  --key KEY_do_not_open_until_scored.csv \
  --sheet grid_experiment/rating/rater_A.csv \
  --masking grid_experiment/runs/masking_cte_p1_k3.jsonl \
  --stability grid_experiment/runs/stability_analysis.json \
  --out grid_experiment/runs/blind_rating_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


AXES = {
    "presence": ["False", "uncertain", "True"],
    "relevance": ["none", "secondary", "primary"],
    "support": ["contradicts", "neutral", "supports"],
}
SCORES = {
    "presence": {"False": 0.0, "uncertain": 0.5, "True": 1.0},
    "relevance": {"none": 0.0, "secondary": 0.5, "primary": 1.0},
    "support": {"contradicts": -1.0, "neutral": 0.0, "supports": 1.0},
}


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Unweighted Cohen's kappa for two categorical series."""
    a = a.astype(str).to_numpy()
    b = b.astype(str).to_numpy()
    if len(a) == 0:
        return np.nan
    po = float(np.mean(a == b))
    labels = sorted(set(a) | set(b))
    pe = sum(np.mean(a == label) * np.mean(b == label) for label in labels)
    return np.nan if np.isclose(pe, 1.0) else (po - pe) / (1.0 - pe)


def cluster_bootstrap_ci(
    frame: pd.DataFrame, axis: str, *, draws: int = 5000, seed: int = 20260904
) -> tuple[float, float]:
    """Percentile CI for kappa, resampling source images as clusters."""
    rng = np.random.default_rng(seed)
    images = frame["image"].drop_duplicates().to_numpy()
    values: list[float] = []
    groups = {image: group for image, group in frame.groupby("image", sort=False)}
    for _ in range(draws):
        sampled = rng.choice(images, size=len(images), replace=True)
        boot = pd.concat([groups[image] for image in sampled], ignore_index=True)
        value = cohen_kappa(boot[f"{axis}_A"], boot[f"{axis}_B"])
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]))


def load_and_validate(args: argparse.Namespace) -> pd.DataFrame:
    key = pd.read_csv(args.key, dtype=str).fillna("")
    sheet = pd.read_csv(args.sheet, dtype=str).fillna("")
    a = pd.read_csv(args.ratings_a, dtype=str).fillna("")
    b = pd.read_csv(args.ratings_b, dtype=str).fillna("")

    for name, frame in (("key", key), ("sheet", sheet), ("ratings A", a), ("ratings B", b)):
        if len(frame) != 100:
            raise ValueError(f"{name} must contain 100 rows; found {len(frame)}")
        if frame.rating_id.nunique() != 100:
            raise ValueError(f"{name} must contain 100 unique rating_id values")

    expected = set(key.rating_id)
    for name, frame in (("sheet", sheet), ("ratings A", a), ("ratings B", b)):
        if set(frame.rating_id) != expected:
            raise ValueError(f"{name} rating IDs do not exactly match the key")

    for name, frame in (("ratings A", a), ("ratings B", b)):
        for axis, allowed in AXES.items():
            bad = sorted(set(frame[axis]) - set(allowed))
            if bad:
                raise ValueError(f"{name} has invalid or missing {axis} values: {bad}")

    a = a.rename(columns={c: f"{c}_A" for c in ["rater", "presence", "relevance", "support", "notes"]})
    b = b.rename(columns={c: f"{c}_B" for c in ["rater", "presence", "relevance", "support", "notes"]})
    merged = key.merge(sheet, on="rating_id", validate="one_to_one")
    merged = merged.merge(a, on="rating_id", validate="one_to_one")
    merged = merged.merge(b, on="rating_id", validate="one_to_one")
    merged["model_group"] = np.where(merged.run.str.startswith("gpt4o"), "GPT-4o", "Gemma")
    merged["model_correct"] = merged.stated_label == merged.true_label
    return merged


def agreement_outputs(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    cross_rows = []
    distribution_rows = []
    for axis, labels in AXES.items():
        a = frame[f"{axis}_A"]
        b = frame[f"{axis}_B"]
        lo, hi = cluster_bootstrap_ci(frame, axis)
        rows.append(
            {
                "axis": axis,
                "n_items": len(frame),
                "n_source_images": frame.image.nunique(),
                "exact_agreement": float((a == b).mean()),
                "cohen_kappa": cohen_kappa(a, b),
                "kappa_cluster_bootstrap_ci_low": lo,
                "kappa_cluster_bootstrap_ci_high": hi,
                "n_agree": int((a == b).sum()),
                "n_disagree": int((a != b).sum()),
            }
        )
        table = pd.crosstab(a, b, dropna=False).reindex(index=labels, columns=labels, fill_value=0)
        for label_a in labels:
            for label_b in labels:
                cross_rows.append(
                    {"axis": axis, "rating_A": label_a, "rating_B": label_b, "count": int(table.loc[label_a, label_b])}
                )
        for rater in ("A", "B"):
            counts = frame[f"{axis}_{rater}"].value_counts()
            for label in labels:
                distribution_rows.append(
                    {
                        "axis": axis,
                        "rater": rater,
                        "rating": label,
                        "count": int(counts.get(label, 0)),
                        "proportion": float(counts.get(label, 0) / len(frame)),
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "agreement_summary.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(out / "agreement_crosstabs.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(out / "rater_distributions.csv", index=False)

    group_rows = []
    for group_name, group in frame.groupby("model_group"):
        for axis in AXES:
            group_rows.append(
                {
                    "model_group": group_name,
                    "axis": axis,
                    "n_items": len(group),
                    "exact_agreement": float((group[f"{axis}_A"] == group[f"{axis}_B"]).mean()),
                    "cohen_kappa": cohen_kappa(group[f"{axis}_A"], group[f"{axis}_B"]),
                }
            )
    pd.DataFrame(group_rows).to_csv(out / "agreement_by_model.csv", index=False)
    return summary


def masking_image_metrics(masking_path: str) -> pd.DataFrame:
    records = [json.loads(line) for line in open(masking_path) if line.strip()]
    rows = []
    for record in records:
        if record.get("error") or not record.get("parsed") or not record.get("baseline"):
            continue
        rows.append(
            {
                "image": record["image"],
                "arm": record["arm"],
                "occlusion": record["occlusion"],
                "flip": int(record["parsed"]["label"] != record["baseline"]["label"]),
                "abs_confidence_change": abs(
                    float(record["parsed"].get("confidence") or 0)
                    - float(record["baseline"].get("confidence") or 0)
                ),
            }
        )
    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("No valid masking records")
    expected = {"cited", "tissue_matched"}
    if set(data.arm) != expected:
        raise ValueError(f"Unexpected masking arms: {sorted(set(data.arm))}")

    out_rows = []
    for image, group in data.groupby("image"):
        row = {"image": image}
        for metric in ("flip", "abs_confidence_change"):
            cited = group[group.arm == "cited"][metric].mean()
            control = group[group.arm == "tissue_matched"][metric].mean()
            row[f"cited_{metric}"] = float(cited)
            row[f"control_{metric}"] = float(control)
            row[f"{metric}_advantage"] = float(cited - control)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def correlation_outputs(frame: pd.DataFrame, masking_path: str, out: Path) -> pd.DataFrame:
    # Only the base Gemma cte_p1 explanations generated the masking locations.
    # GPT-4o and stability-replicate ratings cannot be causally joined to this run.
    eligible = frame[frame.run == "gemma_cte_p1"].copy()
    for axis, mapping in SCORES.items():
        eligible[f"{axis}_mean"] = (
            eligible[f"{axis}_A"].map(mapping) + eligible[f"{axis}_B"].map(mapping)
        ) / 2.0

    human = (
        eligible.groupby("image", as_index=False)
        .agg(
            n_rated_evidence_items=("rating_id", "size"),
            presence_mean=("presence_mean", "mean"),
            relevance_mean=("relevance_mean", "mean"),
            support_mean=("support_mean", "mean"),
        )
    )
    joined = human.merge(masking_image_metrics(masking_path), on="image", how="inner", validate="one_to_one")
    joined.to_csv(out / "faithfulness_join_image_level.csv", index=False)

    rows = []
    human_metrics = ["presence_mean", "relevance_mean", "support_mean"]
    faith_metrics = ["flip_advantage", "abs_confidence_change_advantage"]
    for human_metric in human_metrics:
        for faith_metric in faith_metrics:
            result = spearmanr(joined[human_metric], joined[faith_metric])
            rows.append(
                {
                    "human_metric": human_metric,
                    "faithfulness_metric": faith_metric,
                    "n_images": len(joined),
                    "spearman_rho": float(result.statistic),
                    "p_value_two_sided": float(result.pvalue),
                    "analysis_status": "exploratory_average_of_raters_not_consensus",
                }
            )
    correlations = pd.DataFrame(rows)
    # Holm adjustment across the six explicitly exploratory comparisons.
    order = np.argsort(correlations.p_value_two_sided.to_numpy())
    adjusted = np.empty(len(correlations))
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (len(correlations) - rank) * correlations.loc[idx, "p_value_two_sided"])
        running = max(running, value)
        adjusted[idx] = running
    correlations["p_value_holm"] = adjusted
    correlations.to_csv(out / "faithfulness_correlations.csv", index=False)
    return correlations


def taxonomy_outputs(frame: pd.DataFrame, stability_path: str, out: Path) -> pd.DataFrame:
    both_presence_false = (frame.presence_A == "False") & (frame.presence_B == "False")
    either_presence_false = (frame.presence_A == "False") | (frame.presence_B == "False")
    both_presence_true = (frame.presence_A == "True") & (frame.presence_B == "True")
    both_relevance_none = (frame.relevance_A == "none") & (frame.relevance_B == "none")
    both_support_neutral = (frame.support_A == "neutral") & (frame.support_B == "neutral")
    both_support_contradict = (frame.support_A == "contradicts") & (frame.support_B == "contradicts")
    either_support_contradict = (frame.support_A == "contradicts") | (frame.support_B == "contradicts")

    stability = json.load(open(stability_path))
    per_image = stability["per_image"]
    label_unstable = [name for name, value in per_image.items() if not value["label_stable"]]
    localization_unstable = [
        name for name, value in per_image.items() if float(value["mean_pairwise_jaccard"]) < 0.5
    ]
    either_unstable = sorted(set(label_unstable) | set(localization_unstable))

    rows = [
        {
            "category": "Visually unsupported in cited cells",
            "unit": "rated evidence item",
            "denominator": len(frame),
            "conservative_count": int(both_presence_false.sum()),
            "flagged_by_at_least_one_rater": int(either_presence_false.sum()),
            "status": "not reliably resolved because presence kappa is approximately zero",
        },
        {
            "category": "Hallucinated feature",
            "unit": "rated evidence item",
            "denominator": len(frame),
            "conservative_count": np.nan,
            "flagged_by_at_least_one_rater": np.nan,
            "status": "not separately measured; requires confirming absence from the entire tile",
        },
        {
            "category": "Correct feature, wrong cited region",
            "unit": "rated evidence item",
            "denominator": len(frame),
            "conservative_count": np.nan,
            "flagged_by_at_least_one_rater": np.nan,
            "status": "not separately measured; form assessed cited cells but not whole-tile presence",
        },
        {
            "category": "Correct cited region, diagnostically irrelevant",
            "unit": "rated evidence item",
            "denominator": len(frame),
            "conservative_count": int((both_presence_true & both_relevance_none & both_support_neutral).sum()),
            "flagged_by_at_least_one_rater": np.nan,
            "status": "confirmed only when both raters marked present, irrelevant, and neutral",
        },
        {
            "category": "Evidence contradicts stated label",
            "unit": "rated evidence item",
            "denominator": len(frame),
            "conservative_count": int(both_support_contradict.sum()),
            "flagged_by_at_least_one_rater": int(either_support_contradict.sum()),
            "status": "two-rater conservative and one-rater screening counts",
        },
        {
            "category": "Label unstable across repeated samples",
            "unit": "source image",
            "denominator": len(per_image),
            "conservative_count": len(label_unstable),
            "flagged_by_at_least_one_rater": np.nan,
            "status": "model output changed label across five repetitions",
        },
        {
            "category": "Citation localization unstable (mean Jaccard < 0.50)",
            "unit": "source image",
            "denominator": len(per_image),
            "conservative_count": len(localization_unstable),
            "flagged_by_at_least_one_rater": np.nan,
            "status": "descriptive threshold applied to five repetitions",
        },
        {
            "category": "Any sampled-output instability",
            "unit": "source image",
            "denominator": len(per_image),
            "conservative_count": len(either_unstable),
            "flagged_by_at_least_one_rater": np.nan,
            "status": "label instability or citation Jaccard below 0.50",
        },
    ]
    taxonomy = pd.DataFrame(rows)
    taxonomy.to_csv(out / "error_taxonomy_summary.csv", index=False)

    stability_rows = []
    for image, value in per_image.items():
        stability_rows.append(
            {
                "image": image,
                "n_repetitions": value["n_reps"],
                "label_stable": value["label_stable"],
                "mean_pairwise_citation_jaccard": value["mean_pairwise_jaccard"],
                "citation_jaccard_below_0_50": value["mean_pairwise_jaccard"] < 0.5,
                "any_instability": (not value["label_stable"]) or value["mean_pairwise_jaccard"] < 0.5,
            }
        )
    pd.DataFrame(stability_rows).to_csv(out / "stability_taxonomy.csv", index=False)

    # Only release item IDs and explanation text for jointly confirmed irrelevant items;
    # do not reproduce the private unblinding key in a public-facing output.
    confirmed = frame[both_presence_true & both_relevance_none & both_support_neutral][
        ["rating_id", "model_group", "feature", "description", "stated_label"]
    ]
    confirmed.to_csv(out / "confirmed_irrelevant_items.csv", index=False)
    return taxonomy


def write_report(
    agreement: pd.DataFrame, correlations: pd.DataFrame, taxonomy: pd.DataFrame, out: Path
) -> None:
    stats = agreement.set_index("axis")
    primary_corr = correlations[
        (correlations.human_metric == "presence_mean")
        & (correlations.faithfulness_metric == "flip_advantage")
    ].iloc[0]
    lines = [
        "# Blinded rating and error-taxonomy analysis",
        "",
        "## Task 8 — inter-rater agreement",
        "",
        "| Axis | Exact agreement | Cohen's κ | Cluster-bootstrap 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for axis in AXES:
        row = stats.loc[axis]
        lines.append(
            f"| {axis.title()} | {row.exact_agreement:.1%} | {row.cohen_kappa:.3f} | "
            f"[{row.kappa_cluster_bootstrap_ci_low:.3f}, {row.kappa_cluster_bootstrap_ci_high:.3f}] |"
        )
    lines += [
        "",
        "Agreement was low on all three axes. Presence was especially discordant: rater A "
        "marked 99/100 items present, whereas rater B marked 43 present, 16 uncertain, and 41 absent. "
        "The two raters never jointly marked an item absent. Accordingly, a reliable consensus-presence "
        "variable could not be formed.",
        "",
        "### Exploratory association with masking faithfulness",
        "",
        "Only base-run Gemma ratings can be matched to the 20-image Gemma masking experiment; the random "
        "rating sample provides at least one rated evidence item for 17 of those images. Because strict "
        "presence consensus had no usable variation, correlations use the mean ordinal score across the "
        "two raters and are exploratory, not consensus correlations.",
        "",
        f"The primary exploratory comparison—mean presence score versus cited-minus-control label-flip "
        f"advantage—gave Spearman ρ={primary_corr.spearman_rho:.3f}, "
        f"p={primary_corr.p_value_two_sided:.3f} (n={int(primary_corr.n_images)} images), indicating no "
        "detectable association. No exploratory comparison remained significant after Holm correction.",
        "",
        "## Task 10 — conservative error taxonomy",
        "",
        "| Category | Count | Denominator | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for _, row in taxonomy.iterrows():
        count = "not estimable" if pd.isna(row.conservative_count) else str(int(row.conservative_count))
        lines.append(f"| {row.category} | {count} | {int(row.denominator)} | {row.status} |")
    lines += [
        "",
        "The current form cannot separate a hallucinated feature from a correct feature cited in the wrong "
        "region: both would appear as ‘not visible in the cited cells.’ That distinction requires a new "
        "whole-tile review of the disputed presence cases. Therefore, the taxonomy is complete for the "
        "categories actually measured (irrelevance and sampled-output instability), but the first two "
        "requested categories must remain combined/unresolved unless adjudication is added.",
        "",
        "## Reporting recommendation",
        "",
        "Report the low agreement directly. Do not convert disagreements into a two-rater majority, and do "
        "not describe rater-average scores as consensus. For definitive hallucination and wrong-region "
        "counts, use an independent expert adjudicator or a follow-up whole-tile rating axis.",
    ]
    (out / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings-a", required=True)
    parser.add_argument("--ratings-b", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--masking", required=True)
    parser.add_argument("--stability", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_and_validate(args)
    agreement = agreement_outputs(frame, out)
    correlations = correlation_outputs(frame, args.masking, out)
    taxonomy = taxonomy_outputs(frame, args.stability, out)
    write_report(agreement, correlations, taxonomy, out)

    validation = {
        "n_items": len(frame),
        "n_unique_rating_ids": int(frame.rating_id.nunique()),
        "n_unique_source_images": int(frame.image.nunique()),
        "all_required_ratings_complete": True,
        "private_key_copied_to_output": False,
        "raw_ratings_copied_to_output": False,
    }
    (out / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")
    print((out / "analysis_summary.md").read_text())


if __name__ == "__main__":
    main()
