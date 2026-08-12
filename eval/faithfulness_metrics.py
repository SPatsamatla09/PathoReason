from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class FaithfulnessResult:
    image_id: str
    comprehensiveness: float
    flipped: bool
    control_drop: float
    faithful: bool


def comprehensiveness(orig_conf: float, new_conf: float) -> float:
    return orig_conf - new_conf


def flipped(orig_label: str, new_label: str) -> bool:
    return orig_label != new_label


def sufficiency(*args, **kwargs):
    raise NotImplementedError(
        "sufficiency requires a keep-only-the-evidence masked image and a matching "
        "confidence value, neither of which exist yet in mask.py or the runs schema. "
        "Needs a new masking function (inverse of the current removal masks) and a "
        "new schema field before this can be computed."
    )


def faithful_threshold(orig_confs: Sequence[float], control_confs: Sequence[float], percentile: float = 95.0) -> float:
    orig_confs = np.asarray(orig_confs, dtype=float)
    control_confs = np.asarray(control_confs, dtype=float)
    control_drops = orig_confs - control_confs
    return float(np.percentile(control_drops, percentile))


def evaluate_faithfulness(
    image_id: str,
    orig_conf: float,
    new_conf: float,
    orig_label: str,
    new_label: str,
    control_conf: float,
    control_threshold: float,
) -> FaithfulnessResult:
    comp = comprehensiveness(orig_conf, new_conf)
    flip = flipped(orig_label, new_label)
    ctrl_drop = orig_conf - control_conf
    is_faithful = comp > control_threshold
    return FaithfulnessResult(
        image_id=image_id,
        comprehensiveness=comp,
        flipped=flip,
        control_drop=ctrl_drop,
        faithful=is_faithful,
    )


def evaluate_faithfulness_batch(runs_df) -> list:
    threshold = faithful_threshold(runs_df["orig_conf"].values, runs_df["control_conf"].values)
    results = []
    for _, row in runs_df.iterrows():
        results.append(
            evaluate_faithfulness(
                image_id=row["image_id"],
                orig_conf=row["orig_conf"],
                new_conf=row["new_conf"],
                orig_label=row["orig_label"],
                new_label=row["new_label"],
                control_conf=row["control_conf"],
                control_threshold=threshold,
            )
        )
    return results


def faithfulness_summary(results: list) -> str:
    n = len(results)
    n_faithful = sum(r.faithful for r in results)
    n_flipped = sum(r.flipped for r in results)
    avg_comp = np.mean([r.comprehensiveness for r in results])
    lines = []
    lines.append(f"n_images: {n}")
    lines.append(f"faithful: {n_faithful}/{n} ({100 * n_faithful / n:.1f}%)")
    lines.append(f"flipped label: {n_flipped}/{n} ({100 * n_flipped / n:.1f}%)")
    lines.append(f"avg comprehensiveness (confidence drop): {avg_comp:.4f}")
    lines.append("")
    lines.append(f"{'image_id':<10}{'comp':>10}{'flipped':>10}{'faithful':>10}")
    for r in results:
        lines.append(f"{r.image_id:<10}{r.comprehensiveness:>10.4f}{str(r.flipped):>10}{str(r.faithful):>10}")
    return "\n".join(lines)


if __name__ == "__main__":
    import pandas as pd

    sample = pd.DataFrame([
        {"image_id": "001", "orig_label": "Adenoma", "orig_conf": 0.987, "new_label": "Adenoma", "new_conf": 0.612, "control_conf": 0.95},
        {"image_id": "002", "orig_label": "Hyperplastic", "orig_conf": 0.964, "new_label": "Hyperplastic", "new_conf": 0.938, "control_conf": 0.955},
        {"image_id": "003", "orig_label": "Adenoma", "orig_conf": 0.991, "new_label": "Hyperplastic", "new_conf": 0.524, "control_conf": 0.97},
        {"image_id": "004", "orig_label": "Hyperplastic", "orig_conf": 0.949, "new_label": "Hyperplastic", "new_conf": 0.941, "control_conf": 0.94},
        {"image_id": "005", "orig_label": "Adenoma", "orig_conf": 0.976, "new_label": "Adenoma", "new_conf": 0.705, "control_conf": 0.96},
    ])

    results = evaluate_faithfulness_batch(sample)
    print(faithfulness_summary(results))
