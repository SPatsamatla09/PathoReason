"""Full-test-set report for cte_p1: accuracy with exact CI, cell citations.

    python3 analyze_full.py            ->  runs/cte_p1_full_analysis.json
"""

import json
import math
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, "runs", "cte_p1_full.jsonl")
OUT = os.path.join(ROOT, "runs", "cte_p1_full_analysis.json")
ALL_CELLS = [f"{r}{c}" for r in "ABCD" for c in range(1, 5)]


def bincdf(k, n, p):
    return sum(math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def clopper(k, n, alpha=0.05):
    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if 1 - bincdf(k - 1, n, m) > alpha / 2:
                b = m
            else:
                a = m
        lo = a
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if bincdf(k, n, m) < alpha / 2:
                b = m
            else:
                a = m
        hi = a
    return lo, hi


def main():
    recs = [json.loads(l) for l in open(RUN) if l.strip()]
    ok = [r for r in recs if r.get("replicate", 1) == 1 and (r.get("parsed") or {}).get("label") in ("HP", "SSA")]
    n = len(ok)
    k = sum(1 for r in ok if r["parsed"]["label"] == r["label_true"])
    lo, hi = clopper(k, n)
    p2 = 2 * min(bincdf(k, n, 0.5), 1 - bincdf(k - 1, n, 0.5)) if n else 1.0

    conf_mat = Counter((r["label_true"], r["parsed"]["label"]) for r in ok)
    by_band = defaultdict(lambda: [0, 0])
    for r in ok:
        b = by_band[r["agreement_band"]]
        b[1] += 1
        if r["parsed"]["label"] == r["label_true"]:
            b[0] += 1
    by_votes = defaultdict(lambda: [0, 0])
    for r in ok:
        b = by_votes[r["ssa_votes_out_of_7"]]
        b[1] += 1
        if r["parsed"]["label"] == r["label_true"]:
            b[0] += 1

    tp = conf_mat[("SSA", "SSA")]; fn = conf_mat[("SSA", "HP")]
    tn = conf_mat[("HP", "HP")]; fp = conf_mat[("HP", "SSA")]
    bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))

    cited = Counter(); empty_hits = 0; total_cites = 0
    per_resp_cells = []
    for r in ok:
        cells = sorted({c for e in r["parsed"]["evidence"] for c in e.get("grid_cells_valid", [])})
        per_resp_cells.append(len(cells))
        cited.update(cells)
        total_cites += len(cells)
        empty_hits += sum(1 for c in cells if c in set(r["empty_cells"]))
    invalid = sum(len(e.get("grid_cells_invalid") or []) for r in ok for e in r["parsed"]["evidence"])

    report = {
        "n_records": len(recs),
        "n_scored": n,
        "errors": sum(1 for r in recs if r.get("error")),
        "parse_failures": sum(1 for r in recs if not r.get("error") and not (r.get("parsed") or {}).get("label")),
        "accuracy": {
            "correct": k, "n": n, "point": round(k / max(n, 1), 4),
            "exact_95ci": [round(lo, 4), round(hi, 4)],
            "two_sided_p_vs_chance": round(min(p2, 1.0), 6),
            "balanced_accuracy": round(bal, 4),
            "confusion_true_pred": {f"{a}->{b}": v for (a, b), v in sorted(conf_mat.items())},
            "predicted_ssa_rate": round((tp + fp) / max(n, 1), 4),
            "mean_confidence": round(sum(float(r["parsed"]["confidence"] or 0) for r in ok) / max(n, 1), 3),
            "by_band": {kk: f"{v[0]}/{v[1]} ({v[0]/max(v[1],1)*100:.1f}%)" for kk, v in sorted(by_band.items())},
            "by_ssa_votes": {kk: f"{v[0]}/{v[1]}" for kk, v in sorted(by_votes.items())},
        },
        "citations": {
            "total_unique_cell_citations": total_cites,
            "invalid_cells": invalid,
            "mean_cells_per_response": round(sum(per_resp_cells) / max(n, 1), 2),
            "cell_frequency": {c: cited[c] for c in ALL_CELLS},
            "cell_frequency_pct_of_responses": {c: round(cited[c] / max(n, 1) * 100, 1) for c in ALL_CELLS},
            "empty_cell_citations": empty_hits,
            "empty_citation_rate": round(empty_hits / max(total_cites, 1), 4),
        },
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
