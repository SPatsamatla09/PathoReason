"""Stability probe: 5 tiles x 5 samples, same prompt, same temperature.

The user's criterion: an explanation that changes run to run cannot be
faithful. Metric is within-image Jaccard over cited cell sets across the five
replicates, read against two references from the main runs:

  between-image floor  ~0.39 (cte_p1) -- what unrelated citations look like
  1.0                  perfectly stable citations

Also reports label stability and per-cell citation persistence (cells cited in
5/5 reps vs cells that come and go).

    python3 analyze_stability.py
"""

import itertools
import json
import os
from collections import Counter

from score import cited_cells, jaccard

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, "runs", "cte_p1_stability.jsonl")
BASELINE = os.path.join(ROOT, "runs", "cte_p1_score.json")
OUT = os.path.join(ROOT, "runs", "stability_analysis.json")


def main():
    recs = [json.loads(l) for l in open(RUN) if l.strip()]
    ok = [r for r in recs if r.get("parsed")]
    by = {}
    for r in ok:
        by.setdefault(r["image"], []).append(r)

    floor = None
    if os.path.exists(BASELINE):
        floor = json.load(open(BASELINE))["q2_cross_image_variation"][
            "mean_pairwise_jaccard_between_images"
        ]

    per = {}
    all_j = []
    for img, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["replicate"])
        sets = [set(cited_cells(r)) for r in rs]
        js = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
        all_j += js
        labels = [r["parsed"]["label"] for r in rs]
        cellcount = Counter(c for s in sets for c in s)
        per[img] = {
            "n_reps": len(rs),
            "temperature": rs[0].get("temperature"),
            "labels": labels,
            "label_stable": len(set(labels)) == 1,
            "confidences": [r["parsed"]["confidence"] for r in rs],
            "mean_pairwise_jaccard": round(sum(js) / max(len(js), 1), 4),
            "cited_per_rep": [sorted(s) for s in sets],
            "cells_in_all_reps": sorted(c for c, k in cellcount.items() if k == len(rs)),
            "cells_in_one_rep_only": sorted(c for c, k in cellcount.items() if k == 1),
            "mean_cells_per_rep": round(sum(len(s) for s in sets) / len(sets), 2),
        }

    core = sum(len(v["cells_in_all_reps"]) for v in per.values())
    total_distinct = sum(
        len(set(c for s in v["cited_per_rep"] for c in s)) for v in per.values()
    )
    report = {
        "run": os.path.basename(RUN),
        "records": len(recs),
        "parse_failures": len(recs) - len(ok),
        "mean_within_image_jaccard": round(sum(all_j) / max(len(all_j), 1), 4),
        "between_image_floor_cte_p1": floor,
        "core_cells_cited_in_every_rep": core,
        "distinct_cells_cited_anywhere": total_distinct,
        "core_fraction": round(core / max(total_distinct, 1), 4),
        "per_image": per,
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2))
    for img, v in per.items():
        print(f"{img[6:9]}: J={v['mean_pairwise_jaccard']:.2f} labels={v['labels']} "
              f"core={v['cells_in_all_reps']}")


if __name__ == "__main__":
    main()
