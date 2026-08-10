from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


@dataclass
class MetricResult:
    value: float
    ci_low: float
    ci_high: float

    def __repr__(self):
        return f"{self.value:.4f} (95% CI: {self.ci_low:.4f}-{self.ci_high:.4f})"


def _binarize(labels, positive_label):
    return np.array([1 if l == positive_label else 0 for l in labels])


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence,
    y_pred: Sequence,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = metric_fn(y_true, y_pred)

    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
        except ValueError:
            continue
        boot_scores.append(score)

    alpha = (1 - ci) / 2
    lo = np.percentile(boot_scores, 100 * alpha)
    hi = np.percentile(boot_scores, 100 * (1 - alpha))
    return point, lo, hi


def compute_classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    y_score: Optional[Sequence[float]] = None,
    positive_label: Optional[Union[str, int]] = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = sorted(set(y_true) | set(y_pred))
    if positive_label is None:
        if len(labels) != 2:
            raise ValueError(f"positive_label must be given for non-binary label sets, got {labels}")
        positive_label = labels[1]

    y_true_bin = _binarize(y_true, positive_label)
    y_pred_bin = _binarize(y_pred, positive_label)

    results = {}

    def _acc(yt, yp): return accuracy_score(yt, yp)
    def _prec(yt, yp): return precision_score(yt, yp, zero_division=0)
    def _rec(yt, yp): return recall_score(yt, yp, zero_division=0)
    def _f1(yt, yp): return f1_score(yt, yp, zero_division=0)

    for name, fn in [("accuracy", _acc), ("precision", _prec), ("recall", _rec), ("f1", _f1)]:
        point, lo, hi = bootstrap_ci(fn, y_true_bin, y_pred_bin, n_bootstrap, ci, seed)
        results[name] = MetricResult(point, lo, hi)

    if y_score is not None:
        y_score = np.asarray(y_score)

        def _auc(yt, ys): return roc_auc_score(yt, ys)

        point, lo, hi = bootstrap_ci(_auc, y_true_bin, y_score, n_bootstrap, ci, seed)
        results["roc_auc"] = MetricResult(point, lo, hi)

    return results


def metrics_report(results: dict) -> str:
    lines = [f"{'metric':<10}{'value':>10}   95% CI"]
    for name, r in results.items():
        lines.append(f"{name:<10}{r.value:>10.4f}   ({r.ci_low:.4f}, {r.ci_high:.4f})")
    return "\n".join(lines)


def metrics_from_runs(runs_df, ground_truth: dict, label_col: str = "orig_label",
                       score_col: Optional[str] = None, positive_label: Optional[str] = None, **kwargs):
    df = runs_df.drop_duplicates(subset="image_id").copy()
    df["y_true"] = df["image_id"].map(ground_truth)
    missing = df["y_true"].isna().sum()
    if missing:
        raise ValueError(f"{missing} image_id(s) in runs_df have no entry in ground_truth")

    y_score = df[score_col].values if score_col else None
    return compute_classification_metrics(
        y_true=df["y_true"].values,
        y_pred=df[label_col].values,
        y_score=y_score,
        positive_label=positive_label,
        **kwargs,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.choice(["HP", "SSA"], size=200)
    y_pred = np.where(rng.random(200) < 0.85, y_true, rng.choice(["HP", "SSA"], size=200))
    y_score = np.where(y_pred == "SSA", rng.uniform(0.5, 1.0, 200), rng.uniform(0.0, 0.5, 200))

    results = compute_classification_metrics(y_true, y_pred, y_score, positive_label="SSA", seed=0)
    print(metrics_report(results))
