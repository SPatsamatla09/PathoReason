"""Competence pilot: can any candidate VLM clear the majority-class baseline on MHIST?

Sample: 100 test tiles stratified in proportion to label x agreement band (seed
20260930; runs/.pilot100_tiles.json). It holds 63 HP and 37 SSA, so always answering
HP scores 63%, the same as the full test set's 63.2%.

Rule, written before the results: a model is competent if its Wilson 95% lower bound
exceeds 0.632, i.e. at least 73/100 correct. Also reported: exact one-sided binomial p
against 0.632, balanced accuracy, sensitivity and specificity, SSA-call rate,
unusable responses, reasoning tokens (must be 0) and billed cost. gemma-4-31b is run
on the same tiles as the reference.

    python3 analyze_pilot.py  ->  runs/PILOT_RESULTS.json (+ printed table)
"""

import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = 0.632


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(c - h, 4), round(c + h, 4)


def p_one_sided(k, n, p0=BASE):
    return sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1))


def main():
    tiles = json.load(open(os.path.join(ROOT, "runs", ".pilot100_tiles.json")))["tiles"]
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "cte_p1_pilot__*.jsonl"))):
        name = os.path.basename(f)[len("cte_p1_pilot__"):-len(".jsonl")]
        recs = {}
        bad = 0
        cost = 0.0
        rtok = 0
        for l in open(f):
            r = json.loads(l)
            u = (r.get("meta") or {}).get("usage") or {}
            cost += u.get("cost") or 0
            rtok += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            if r.get("error") or (r.get("parsed") or {}).get("label") not in ("HP", "SSA"):
                bad += 1
                continue
            recs[r["image"]] = r
        v = [recs[t] for t in tiles if t in recs]
        n = len(v)
        if not n:
            continue
        k = sum(r["parsed"]["label"] == r["label_true"] for r in v)
        hp = [r for r in v if r["label_true"] == "HP"]
        ssa = [r for r in v if r["label_true"] == "SSA"]
        spec = sum(r["parsed"]["label"] == "HP" for r in hp) / max(len(hp), 1)
        sens = sum(r["parsed"]["label"] == "SSA" for r in ssa) / max(len(ssa), 1)
        lo, hi = wilson(k, n)
        out[name] = {
            "n_valid": n, "missing": len(tiles) - n, "unusable_responses": bad,
            "accuracy": round(k / n, 4), "correct": k, "wilson95": [lo, hi],
            "p_one_sided_vs_0.632": float(f"{p_one_sided(k, n):.3g}"),
            "balanced_accuracy": round((sens + spec) / 2, 4),
            "sensitivity_SSA": round(sens, 4), "specificity_HP": round(spec, 4),
            "ssa_call_rate": round(sum(r["parsed"]["label"] == "SSA" for r in v) / n, 4),
            "always_HP_on_these_tiles": round(len(hp) / n, 4),
            "reasoning_tokens_total": rtok, "billed_cost_usd": round(cost, 4),
            "verdict": "COMPETENT (lower 95% bound > 0.632)" if n == len(tiles) and lo > BASE else
                       ("incomplete" if n < len(tiles) else "not shown competent"),
        }
    json.dump(out, open(os.path.join(ROOT, "runs", "PILOT_RESULTS.json"), "w"), indent=2)
    print(f"{'model':70s} {'n':>3} {'acc':>6} {'95% CI':>15} {'p>0.632':>8} {'bal':>6} {'sens':>5} {'spec':>5} {'SSA%':>5} {'cost':>7}  verdict")
    for m, d in out.items():
        print(f"{m:70s} {d['n_valid']:>3} {d['accuracy']:>6.3f} {str(d['wilson95']):>15} {d['p_one_sided_vs_0.632']:>8.3g} "
              f"{d['balanced_accuracy']:>6.3f} {d['sensitivity_SSA']:>5.2f} {d['specificity_HP']:>5.2f} "
              f"{d['ssa_call_rate']:>5.2f} ${d['billed_cost_usd']:>6.4f}  {d['verdict']}")


if __name__ == "__main__":
    main()
