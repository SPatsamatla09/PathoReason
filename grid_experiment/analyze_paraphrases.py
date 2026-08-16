"""Cross the ordering effect against prompt wording.

The p1 finding to test: 6/20 tiles flipped cte=HP -> etc=SSA and none flipped
the other way (McNemar p=0.031). If that survives p2 and p3, it is an ordering
effect; if it vanishes or reverses under rewording, it was a wording artifact.

Reads runs/{co,cte,etc}_{p1,p2,p3}.jsonl, reports per-run accuracy / SSA rate,
per-paraphrase directional flip counts, and the pooled McNemar across all
three paraphrases.

    python3 analyze_paraphrases.py
"""

import json
import math
import os
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(RUNS, "paraphrase_analysis.json")

CONDS = ["co", "cte", "etc"]
PARAS = ["p1", "p2", "p3"]


def mcnemar_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_firsts(run_id):
    path = os.path.join(RUNS, f"{run_id}.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["replicate"] == 1 and r.get("parsed") and r["parsed"].get("label") in ("HP", "SSA"):
            out[r["image"]] = r
    return out


def main():
    per_run = {}
    firsts = {}
    for c in CONDS:
        for p in PARAS:
            rid = f"{c}_{p}"
            d = load_firsts(rid)
            if d is None:
                per_run[rid] = "MISSING"
                continue
            firsts[rid] = d
            n = len(d)
            acc = sum(1 for r in d.values() if r["parsed"]["label"] == r["label_true"])
            ssa = sum(1 for r in d.values() if r["parsed"]["label"] == "SSA")
            conf = [float(r["parsed"]["confidence"] or 0) for r in d.values()]
            per_run[rid] = {
                "n": n,
                "accuracy": f"{acc}/{n}",
                "ssa_predictions": ssa,
                "mean_confidence": round(sum(conf) / max(n, 1), 3),
            }

    flips = {}
    pooled_pos = pooled_neg = 0
    for p in PARAS:
        a, b = firsts.get(f"cte_{p}"), firsts.get(f"etc_{p}")
        if not a or not b:
            flips[p] = "MISSING"
            continue
        shared = sorted(set(a) & set(b))
        hp2ssa = [i for i in shared
                  if a[i]["parsed"]["label"] == "HP" and b[i]["parsed"]["label"] == "SSA"]
        ssa2hp = [i for i in shared
                  if a[i]["parsed"]["label"] == "SSA" and b[i]["parsed"]["label"] == "HP"]
        pooled_pos += len(hp2ssa)
        pooled_neg += len(ssa2hp)
        flips[p] = {
            "n_shared": len(shared),
            "cte_HP_to_etc_SSA": len(hp2ssa),
            "cte_SSA_to_etc_HP": len(ssa2hp),
            "flipped_tiles": sorted(t[6:9] for t in hp2ssa + ssa2hp),
            "mcnemar_p": round(mcnemar_p(len(hp2ssa), len(ssa2hp)), 4),
        }

    # wording sensitivity within a condition: same condition, different words
    wording = {}
    for c in CONDS:
        runs = [firsts.get(f"{c}_{p}") for p in PARAS]
        runs = [r for r in runs if r]
        if len(runs) < 2:
            continue
        shared = sorted(set.intersection(*[set(r) for r in runs]))
        unanimous = sum(
            1 for i in shared if len({r[i]["parsed"]["label"] for r in runs}) == 1
        )
        wording[c] = {
            "n_shared": len(shared),
            "label_unanimous_across_paraphrases": unanimous,
        }

    report = {
        "per_run": per_run,
        "cte_vs_etc_flips_by_paraphrase": flips,
        "pooled_mcnemar": {
            "cte_HP_to_etc_SSA": pooled_pos,
            "cte_SSA_to_etc_HP": pooled_neg,
            "p": round(mcnemar_p(pooled_pos, pooled_neg), 4),
        },
        "label_agreement_across_wordings": wording,
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
