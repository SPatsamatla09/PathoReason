"""Score the biological masking sweep against the grid-square sweep.

Contrasts, all vs the unmasked cte_p1 baseline answer:

  1. struct vs matched control, per mask source -- does occluding the actual
     structure beat occluding the same area placed randomly? (exact sign test
     on per-tile discordant pairs)
  2. named vs unnamed structures -- gemma named epithelium-mapped features on
     every tile and nuclei on none, so the nuclei arm doubles as the
     "structure it never cited" probe.
  3. biological vs grid-square masking -- flip rates side by side with the
     runs/masking_cte_p1_k3 results. The area caveat is printed with the
     numbers: epithelium/stroma masks cover ~2x the tissue the 3-cell grid
     subsets did, so a HIGHER flip rate alone proves nothing; the informative
     comparison is struct-minus-control within each sweep.

    python3 analyze_biological.py
"""

import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, "runs", "biological_masking.jsonl")
GRID = os.path.join(ROOT, "runs", "masking_analysis.json")
OUT = os.path.join(ROOT, "runs", "biological_analysis.json")

SOURCES = ["epithelium", "stroma", "nuclei"]


def sign_test_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main():
    recs = [json.loads(l) for l in open(RUN) if l.strip()]
    ok = [r for r in recs if r.get("parsed") and r.get("baseline") and not r.get("error")]
    cell = {(r["image"], r["mask_source"], r["arm"]): r for r in ok}
    images = sorted({r["image"] for r in ok})

    def flipped(r):
        return r["parsed"]["label"] != r["baseline"]["label"]

    def dconf(r):
        return abs(float(r["parsed"]["confidence"] or 0) - float(r["baseline"]["confidence"] or 0))

    per = {}
    contrasts = {}
    for src in SOURCES:
        for arm in ("struct", "control"):
            rows = [r for r in ok if r["mask_source"] == src and r["arm"] == arm]
            fl = [r for r in rows if flipped(r)]
            per[f"{src}/{arm}"] = {
                "n": len(rows),
                "flips": len(fl),
                "flip_rate": round(len(fl) / max(len(rows), 1), 4),
                "flipped_images": sorted(r["image"][6:9] for r in fl),
                "mean_abs_dconf": round(sum(dconf(r) for r in rows) / max(len(rows), 1), 4),
                "mean_pct_of_tissue_masked": round(
                    sum(
                        (r["struct_tissue_pct"] if arm == "struct" else r["control_tissue_pct"])
                        for r in rows
                    )
                    / max(len(rows), 1),
                    1,
                ),
            }
        pos = neg = both = 0
        for img in images:
            a, b = cell.get((img, src, "struct")), cell.get((img, src, "control"))
            if not a or not b:
                continue
            fa, fb = flipped(a), flipped(b)
            if fa and not fb:
                pos += 1
            elif fb and not fa:
                neg += 1
            elif fa:
                both += 1
        contrasts[src] = {
            "struct_flip_only": pos,
            "control_flip_only": neg,
            "both_flip": both,
            "sign_test_p": round(sign_test_p(pos, neg), 4),
        }

    pooled_pos = sum(c["struct_flip_only"] for c in contrasts.values())
    pooled_neg = sum(c["control_flip_only"] for c in contrasts.values())

    named_split = {}
    for src in SOURCES:
        for named in (True, False):
            rows = [
                r for r in ok
                if r["mask_source"] == src and r["arm"] == "struct" and r["named"] == named
            ]
            if rows:
                named_split[f"{src}/named={named}"] = {
                    "n": len(rows),
                    "flip_rate": round(sum(flipped(r) for r in rows) / len(rows), 4),
                }

    grid = None
    if os.path.exists(GRID):
        g = json.load(open(GRID))
        grid = {
            "cited/mean_flip_rate": g["per_arm_occlusion"]["cited/mean"]["flip_rate"],
            "control/mean_flip_rate": g["per_arm_occlusion"]["tissue_matched/mean"]["flip_rate"],
            "pooled_contrast": g["pooled_contrast"],
            "note": "grid subsets masked ~19% of tile / ~35% of cited-cell tissue; "
            "epithelium and stroma masks here cover ~40-55% of ALL tissue, so raw "
            "flip-rate differences across sweeps partly reflect area, not targeting. "
            "Compare struct-minus-control within each sweep instead.",
        }

    report = {
        "run": os.path.basename(RUN),
        "records": len(recs),
        "scored": len(ok),
        "errors": sum(1 for r in recs if r.get("error")),
        "parse_failures": sum(1 for r in recs if not r.get("error") and not r.get("parsed")),
        "per_source_arm": per,
        "struct_vs_control": contrasts,
        "pooled_struct_vs_control": {
            "struct_flip_only": pooled_pos,
            "control_flip_only": pooled_neg,
            "sign_test_p": round(sign_test_p(pooled_pos, pooled_neg), 4),
        },
        "named_vs_unnamed_struct": named_split,
        "grid_sweep_reference": grid,
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
