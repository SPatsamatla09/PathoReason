"""Analyze PLIP causal masking on PatchCamelyon.

Primary question:
Does masking model-cited evidence perturb PLIP more than masking
tissue-matched uncited evidence?

Confidence perturbation is measured on the ORIGINAL BASELINE CLASS:
    masked P(baseline class) - baseline P(baseline class)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


ROOT = Path(__file__).resolve().parent

RUN = ROOT / "pcam" / "runs" / "masking_cte_p1_k3_plip.jsonl"
BASELINES = ROOT / "pcam" / "runs" / "plip_baseline.jsonl"
OUT = ROOT / "pcam" / "runs" / "masking_analysis_plip.json"

ARMS = ("cited", "tissue_matched")
OCCS = ("mean", "blur", "black")


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sign_test_p(pos: int, neg: int) -> float:
    """Exact two-sided sign test over discordant pairs."""
    n = pos + neg
    if n == 0:
        return 1.0

    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)

    return min(1.0, 2 * tail)


def main():
    recs = read_jsonl(RUN)
    baseline_rows = read_jsonl(BASELINES)

    baselines = {
        r["image"]: r
        for r in baseline_rows
        if not r.get("error")
        and r.get("label")
        and r.get("class_probabilities")
    }

    valid = [
        r for r in recs
        if not r.get("error")
        and r.get("label")
        and r.get("class_probabilities")
        and r.get("image") in baselines
    ]

    if recs and not valid:
        raise RuntimeError(
            "Loaded masking records but scored none. "
            f"Example masking keys: {sorted(recs[0].keys())}"
        )

    if len(baselines) != 20:
        raise RuntimeError(
            f"Expected 20 PLIP baselines, found {len(baselines)}"
        )

    if len(recs) != 120:
        raise RuntimeError(
            f"Expected 120 masking records, found {len(recs)}"
        )

    if len(valid) != 120:
        raise RuntimeError(
            f"Expected 120 scoreable masking records, found {len(valid)}"
        )

    # ------------------------------------------------------------
    # Canonicalize records
    # ------------------------------------------------------------

    rows = []

    for r in valid:
        b = baselines[r["image"]]

        baseline_label = b["label"]

        baseline_probs = b["class_probabilities"]
        masked_probs = r["class_probabilities"]

        baseline_class_prob = float(baseline_probs[baseline_label])
        masked_baseline_class_prob = float(masked_probs[baseline_label])

        signed_dconf = (
            masked_baseline_class_prob
            - baseline_class_prob
        )

        rows.append({
            "image": r["image"],
            "tile_dir": r["tile_dir"],
            "arm": r["arm"],
            "occlusion": r["occlusion"],

            "baseline_label": baseline_label,
            "masked_label": r["label"],

            "baseline_class_prob": baseline_class_prob,
            "masked_baseline_class_prob": masked_baseline_class_prob,

            "signed_dconf": signed_dconf,
            "abs_dconf": abs(signed_dconf),

            "flip": r["label"] != baseline_label,

            "masked_cells": r.get("masked_cells", []),
            "match_quality": r.get("match_quality", {}),
        })

    # ------------------------------------------------------------
    # Validate paired design
    # ------------------------------------------------------------

    by_image = defaultdict(list)
    for r in rows:
        by_image[r["image"]].append(r)

    design_errors = {}

    for image, rs in sorted(by_image.items()):
        conditions = {
            (r["arm"], r["occlusion"])
            for r in rs
        }

        expected = {
            (arm, occ)
            for arm in ARMS
            for occ in OCCS
        }

        if len(rs) != 6 or conditions != expected:
            design_errors[image] = {
                "n": len(rs),
                "conditions": sorted(
                    [list(x) for x in conditions]
                ),
            }

    if design_errors:
        raise RuntimeError(
            "Paired design validation failed:\n"
            + json.dumps(design_errors, indent=2)
        )

    # ------------------------------------------------------------
    # Mask-quality diagnostics
    # ------------------------------------------------------------

    unique_match = {}

    for r in rows:
        q = r["match_quality"]
        unique_match[r["image"]] = q

    gaps = [
        float(q.get("tissue_gap_pp", 0))
        for q in unique_match.values()
    ]

    overlaps = [
        int(q.get("control_overlap_cells", 0))
        for q in unique_match.values()
    ]

    unusable = [
        image
        for image, q in unique_match.items()
        if q.get("usable") is False
    ]

    # ------------------------------------------------------------
    # Per arm × occlusion descriptive results
    # ------------------------------------------------------------

    per_arm_occ = {}

    for arm in ARMS:
        for occ in OCCS:
            rs = [
                r for r in rows
                if r["arm"] == arm
                and r["occlusion"] == occ
            ]

            flips = [r for r in rs if r["flip"]]
            abs_d = [r["abs_dconf"] for r in rs]
            signed_d = [r["signed_dconf"] for r in rs]

            per_arm_occ[f"{arm}/{occ}"] = {
                "n": len(rs),
                "flips": len(flips),
                "flip_rate": round(
                    len(flips) / len(rs), 4
                ),
                "flipped_images": sorted(
                    r["image"] for r in flips
                ),
                "mean_abs_dconf": round(
                    sum(abs_d) / len(abs_d), 6
                ),
                "median_abs_dconf": round(
                    median(abs_d), 6
                ),
                "mean_signed_dconf": round(
                    sum(signed_d) / len(signed_d), 6
                ),
            }

    # ------------------------------------------------------------
    # Paired cited vs control results
    # ------------------------------------------------------------

    cell = {
        (r["image"], r["arm"], r["occlusion"]): r
        for r in rows
    }

    contrasts = {}

    for occ in OCCS:
        cited_only = 0
        control_only = 0
        both = 0
        neither = 0

        deltas = []

        for image in sorted(by_image):
            cited = cell[(image, "cited", occ)]
            control = cell[(image, "tissue_matched", occ)]

            fc = cited["flip"]
            ft = control["flip"]

            if fc and not ft:
                cited_only += 1
            elif ft and not fc:
                control_only += 1
            elif fc and ft:
                both += 1
            else:
                neither += 1

            deltas.append(
                cited["abs_dconf"]
                - control["abs_dconf"]
            )

        positive = sum(x > 0 for x in deltas)
        negative = sum(x < 0 for x in deltas)
        zero = sum(x == 0 for x in deltas)

        if wilcoxon is not None and any(x != 0 for x in deltas):
            try:
                w = wilcoxon(
                    deltas,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                wilcoxon_stat = float(w.statistic)
                wilcoxon_p = float(w.pvalue)
            except ValueError:
                wilcoxon_stat = None
                wilcoxon_p = None
        else:
            wilcoxon_stat = None
            wilcoxon_p = None

        contrasts[occ] = {
            "n_pairs": len(deltas),

            "cited_flip_only": cited_only,
            "control_flip_only": control_only,
            "both_flip": both,
            "neither": neither,

            "flip_sign_test_p": round(
                sign_test_p(
                    cited_only,
                    control_only,
                ),
                6,
            ),

            "mean_abs_dconf_paired_delta": round(
                sum(deltas) / len(deltas),
                6,
            ),
            "median_abs_dconf_paired_delta": round(
                median(deltas),
                6,
            ),

            "delta_gt_0": positive,
            "delta_lt_0": negative,
            "delta_eq_0": zero,

            "wilcoxon_statistic": (
                round(wilcoxon_stat, 6)
                if wilcoxon_stat is not None
                else None
            ),
            "wilcoxon_p": (
                round(wilcoxon_p, 6)
                if wilcoxon_p is not None
                else None
            ),
        }

    # ------------------------------------------------------------
    # Image-level aggregate across the 3 occlusions
    # Avoid pretending repeated occlusions are independent.
    # ------------------------------------------------------------

    image_effects = []

    for image in sorted(by_image):
        cited_mean = sum(
            cell[(image, "cited", occ)]["abs_dconf"]
            for occ in OCCS
        ) / len(OCCS)

        control_mean = sum(
            cell[(image, "tissue_matched", occ)]["abs_dconf"]
            for occ in OCCS
        ) / len(OCCS)

        image_effects.append({
            "image": image,
            "cited_mean_abs_dconf": cited_mean,
            "control_mean_abs_dconf": control_mean,
            "delta": cited_mean - control_mean,
        })

    aggregate_deltas = [
        r["delta"]
        for r in image_effects
    ]

    if wilcoxon is not None and any(
        x != 0 for x in aggregate_deltas
    ):
        try:
            agg_w = wilcoxon(
                aggregate_deltas,
                alternative="two-sided",
                zero_method="wilcox",
            )
            agg_stat = float(agg_w.statistic)
            agg_p = float(agg_w.pvalue)
        except ValueError:
            agg_stat = None
            agg_p = None
    else:
        agg_stat = None
        agg_p = None

    aggregate = {
        "unit": "image",
        "n_images": len(image_effects),

        "definition": (
            "mean |delta confidence| across mean/blur/black "
            "for cited minus tissue-matched"
        ),

        "mean_delta": round(
            sum(aggregate_deltas)
            / len(aggregate_deltas),
            6,
        ),
        "median_delta": round(
            median(aggregate_deltas),
            6,
        ),

        "delta_gt_0": sum(
            x > 0 for x in aggregate_deltas
        ),
        "delta_lt_0": sum(
            x < 0 for x in aggregate_deltas
        ),
        "delta_eq_0": sum(
            x == 0 for x in aggregate_deltas
        ),

        "wilcoxon_statistic": (
            round(agg_stat, 6)
            if agg_stat is not None
            else None
        ),
        "wilcoxon_p": (
            round(agg_p, 6)
            if agg_p is not None
            else None
        ),
    }

    report = {
        "dataset": "pcam",
        "model": "plip",

        "masking_run": RUN.name,
        "baseline_run": BASELINES.name,

        "masking_records": len(recs),
        "scored": len(rows),

        "baseline_images": len(baselines),
        "unique_masked_images": len(by_image),

        "errors": sum(
            1 for r in recs
            if r.get("error")
        ),

        "confidence_definition": (
            "Probability assigned to the original unmasked "
            "baseline-predicted class."
        ),

        "design_validation": {
            "expected_conditions_per_image": 6,
            "all_images_have_6_conditions": True,
        },

        "mask_matching": {
            "mean_abs_tissue_gap_pp": round(
                sum(abs(x) for x in gaps) / len(gaps),
                4,
            ),
            "max_abs_tissue_gap_pp": round(
                max(abs(x) for x in gaps),
                4,
            ),
            "images_with_control_overlap": sum(
                x > 0 for x in overlaps
            ),
            "unusable_images": sorted(unusable),
        },

        "per_arm_occlusion": per_arm_occ,

        "cited_vs_control": contrasts,

        "image_level_aggregate": aggregate,

        "image_level_effects": [
            {
                "image": r["image"],
                "cited_mean_abs_dconf": round(
                    r["cited_mean_abs_dconf"], 6
                ),
                "control_mean_abs_dconf": round(
                    r["control_mean_abs_dconf"], 6
                ),
                "delta": round(r["delta"], 6),
            }
            for r in image_effects
        ],
    }

    OUT.write_text(
        json.dumps(report, indent=2)
    )

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
