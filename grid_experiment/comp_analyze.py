"""Score every competence configuration against runs/competence/PLAN.md.

    python3 comp_analyze.py  ->  runs/competence/RESULTS.json + runs/competence/CONFIG_TABLE.md

Unusable answers count as wrong (accuracy) and as a miss for the tile's class (balanced accuracy).
"""

import glob
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "runs", "competence")


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def bal(recs):
    hp = [r for r in recs if r["label_true"] == "HP"]
    ssa = [r for r in recs if r["label_true"] == "SSA"]
    rh = sum(r["label"] == "HP" for r in hp) / max(len(hp), 1)
    rs = sum(r["label"] == "SSA" for r in ssa) / max(len(ssa), 1)
    return (rh + rs) / 2, rh, rs


def boot_bal(recs, seed=20261003, B=5000):
    rng = random.Random(seed)
    vals = sorted(bal([recs[rng.randrange(len(recs))] for _ in recs])[0] for _ in range(B))
    return [round(vals[int(0.025 * B)], 4), round(vals[int(0.975 * B) - 1], 4)]


def summarize(recs):
    n = len(recs)
    k = sum(r["label"] == r["label_true"] for r in recs)
    b, rh, rs = bal(recs)
    usable = [r for r in recs if r["label"] in ("HP", "SSA")]
    out = {"n": n, "correct": k, "accuracy": round(k / n, 4) if n else None, "accuracy_wilson95": wilson(k, n),
           "balanced_accuracy": round(b, 4), "balanced_boot95": boot_bal(recs) if n >= 20 else None,
           "recall_HP": round(rh, 4), "recall_SSA": round(rs, 4),
           "parse_rate": round(len(usable) / n, 4) if n else None,
           "ssa_call_rate": round(sum(r["label"] == "SSA" for r in usable) / max(len(usable), 1), 4),
           "cost_usd": round(sum(r.get("cost_usd") or 0 for r in recs), 4),
           "thinking_trace_rate": (round(sum(bool(r.get("thinking_trace")) for r in recs) / n, 4)
                                   if any("thinking_trace" in r for r in recs) else None),
           "finish_length": sum(r.get("finish_reason") == "length" for r in recs),
           "cited_ge1": (round(sum((r.get("n_valid_cited_cells") or 0) >= 1 for r in recs) / n, 4)
                         if any("n_valid_cited_cells" in r for r in recs) else None),
           "cited_ge3": (round(sum((r.get("n_valid_cited_cells") or 0) >= 3 for r in recs) / n, 4)
                         if any("n_valid_cited_cells" in r for r in recs) else None),
           "providers": sorted({str(r.get("provider_served")) for r in recs})}
    halves = {}
    for a in ("HP", "SSA"):
        sub = [r for r in recs if r.get("class_a_is") == a]
        if sub:
            halves[f"class_A_is_{a}"] = {"n": len(sub), "accuracy": round(sum(r['label'] == r['label_true'] for r in sub) / len(sub), 4),
                                         "balanced_accuracy": round(bal(sub)[0], 4)}
    if halves:
        out["counterbalance_halves_descriptive"] = halves
    return out


def main():
    groups = {}
    for f in sorted(glob.glob(os.path.join(D, "*.jsonl"))):
        recs = [json.loads(l) for l in open(f)]
        if not recs:
            continue
        r0 = recs[0]
        key = (r0["config"], r0["model"], r0["provider_pinned"] + (f" [{r0['sampling_tag']}]" if r0.get("sampling_tag") else ""),
               r0["control"])
        groups.setdefault(key, {})[r0["tiles"]] = recs
    res = []
    for (config, model, prov, control), parts in sorted(groups.items()):
        row = {"config": config, "model": model, "provider": prov, "control": control}
        if "screen" in parts:
            row["screen"] = summarize(parts["screen"])
            row["screen_advances"] = row["screen"]["n"] == 100 and row["screen"]["accuracy"] >= 0.65
        dev = parts.get("screen", []) + parts.get("dev_rest", []) + parts.get("dev", [])
        if len(dev) == 300:
            row["dev"] = summarize(dev)
            if control == "none":
                row["dev_pass"] = row["dev"]["accuracy"] >= 0.72 and row["dev"]["balanced_accuracy"] >= 0.65
            else:
                # one-sided (clarified in the PLAN addendum before any control call): not above chance
                lo, hi = row["dev"]["balanced_boot95"]
                row["control_ok_falls_to_chance"] = lo <= 0.5 and row["dev"]["balanced_accuracy"] <= 0.58
        if "test" in parts:
            t = summarize(parts["test"])
            row["test"] = t
            row["test_competent"] = (t["n"] == 977 and t["accuracy_wilson95"][0] > 0.632 and t["balanced_boot95"][0] > 0.5)
        res.append(row)
    json.dump(res, open(os.path.join(D, "RESULTS.json"), "w"), indent=2)
    lines = ["| config | model | provider | control | set | n | acc | acc 95% CI | bal acc | recall HP | recall SSA | parse | SSA calls | cost | decision |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        for s in ("screen", "dev", "test"):
            if s not in r:
                continue
            v = r[s]
            dec = ("advances" if r.get("screen_advances") else "stops") if s == "screen" else \
                  ("PASS" if r.get("dev_pass") else ("control OK" if r.get("control_ok_falls_to_chance") else
                   ("control FAILS (holds up without the right image)" if "control_ok_falls_to_chance" in r else "fail"))) if s == "dev" else \
                  ("COMPETENT" if r.get("test_competent") else "not competent")
            lines.append(f"| {r['config']} | {r['model']} | {r['provider']} | {r['control']} | {s} | {v['n']} | {v['accuracy']:.3f} | "
                         f"{v['accuracy_wilson95'][0]:.3f}–{v['accuracy_wilson95'][1]:.3f} | {v['balanced_accuracy']:.3f} | "
                         f"{v['recall_HP']:.2f} | {v['recall_SSA']:.2f} | {v['parse_rate']:.2f} | {v['ssa_call_rate']:.2f} | ${v['cost_usd']:.3f} | {dec} |")
    open(os.path.join(D, "CONFIG_TABLE.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
