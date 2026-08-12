from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class ReliabilityBin:
    bin_low: float
    bin_high: float
    count: int
    avg_confidence: Optional[float]
    accuracy: Optional[float]


@dataclass
class CalibrationResult:
    ece: float
    bins: list
    n_samples: int
    calibrated: bool
    track_b_usable: bool


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not np.all((confidences >= 0.0) & (confidences <= 1.0)):
        raise ValueError("confidences must all be between 0 and 1")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        if count == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10):
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(ReliabilityBin(lo, hi, 0, None, None))
        else:
            bins.append(ReliabilityBin(lo, hi, count, float(confidences[mask].mean()), float(correct[mask].mean())))
    return bins


def run_confidence_audit(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10, ece_threshold: float = 0.05) -> CalibrationResult:
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(confidences)
    if n < 30:
        raise ValueError(f"audit expects a reasonably sized sample (got {n}); the plan calls for 100 images")

    ece = expected_calibration_error(confidences, correct, n_bins)
    bins = reliability_curve(confidences, correct, n_bins)
    calibrated = ece <= ece_threshold

    return CalibrationResult(
        ece=ece,
        bins=bins,
        n_samples=n,
        calibrated=calibrated,
        track_b_usable=calibrated,
    )


def audit_report(result: CalibrationResult) -> str:
    lines = []
    lines.append(f"n_samples: {result.n_samples}")
    lines.append(f"ECE: {result.ece:.4f}")
    lines.append(f"calibrated (ECE <= threshold): {result.calibrated}")
    if result.track_b_usable:
        lines.append("Decision: Track B confidences are usable for the faithfulness experiment.")
    else:
        lines.append("Decision: Track B is uncalibrated. Faithfulness experiment should run on Track A probabilities; Track B is used for explanation quality only.")
    lines.append("")
    lines.append(f"{'bin range':<16}{'count':>8}{'avg conf':>12}{'accuracy':>12}")
    for b in result.bins:
        conf_str = f"{b.avg_confidence:.3f}" if b.avg_confidence is not None else "-"
        acc_str = f"{b.accuracy:.3f}" if b.accuracy is not None else "-"
        lines.append(f"{b.bin_low:.2f}-{b.bin_high:.2f}{'':<4}{b.count:>8}{conf_str:>12}{acc_str:>12}")
    return "\n".join(lines)


def audit_from_runs(runs_df, ground_truth: dict, track_value: str = "B", label_col: str = "orig_label", conf_col: str = "orig_conf", n_bins: int = 10, ece_threshold: float = 0.05) -> CalibrationResult:
    df = runs_df[runs_df["track"] == track_value].drop_duplicates(subset="image_id").copy()
    df["y_true"] = df["image_id"].map(ground_truth)
    missing = df["y_true"].isna().sum()
    if missing:
        raise ValueError(f"{missing} image_id(s) have no entry in ground_truth")

    correct = (df[label_col] == df["y_true"]).values
    confidences = df[conf_col].values
    return run_confidence_audit(confidences, correct, n_bins=n_bins, ece_threshold=ece_threshold)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    confidences = rng.uniform(0.95, 0.99, n)
    correct = rng.random(n) < 0.80

    result = run_confidence_audit(confidences, correct)
    print(audit_report(result))
