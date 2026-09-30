"""Analyze the prompt diagnostic against the rules fixed in runs/prompt_diagnostic/PLAN.md.

    python3 analyze_diag.py  ->  runs/prompt_diagnostic/RESULTS.json (+ printed table)
"""

import glob
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "runs", "prompt_diagnostic")
PILOT = {"openai/gpt-4.1": "openai-gpt-4.1__openai",
         "qwen/qwen3-vl-235b-a22b-instruct": "qwen-qwen3-vl-235b-a22b-instruct__alibaba",
         "google/gemini-2.5-flash": "google-gemini-2.5-flash__google-ai-studio__x4955da",
         "google/gemma-4-31b-it": "google-gemma-4-31b-it__friendli"}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def fisher(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    pr = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    obs = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= obs * (1 + 1e-9)))


def auroc(scores, truth):
    pos = [s for s, t in zip(scores, truth) if t]
    neg = [s for s, t in zip(scores, truth) if not t]
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def summarize(recs):
    usable = [r for r in recs if r["label"] in ("HP", "SSA")]
    n = len(usable)
    if not n:
        return {"n_records": len(recs), "n_usable": 0}
    k = sum(r["label"] == r["label_true"] for r in usable)
    hp = [r for r in usable if r["label_true"] == "HP"]
    ssa = [r for r in usable if r["label_true"] == "SSA"]
    sens = sum(r["label"] == "SSA" for r in ssa) / max(len(ssa), 1)
    spec = sum(r["label"] == "HP" for r in hp) / max(len(hp), 1)
    a = sum(r["label"] == "SSA" for r in ssa)
    b = sum(r["label"] == "SSA" for r in hp)
    out = {"n_records": len(recs), "n_usable": n, "correct": k, "accuracy": round(k / n, 4),
           "wilson95": wilson(k, n), "balanced_accuracy": round((sens + spec) / 2, 4),
           "sensitivity_SSA": round(sens, 4), "specificity_HP": round(spec, 4),
           "ssa_call_rate": round(sum(r["label"] == "SSA" for r in usable) / n, 4),
           "label_vs_truth_fisher_p": float(f"{fisher(a, len(ssa) - a, b, len(hp) - b):.3g}"),
           "competent_by_pilot_bar": n == 100 and wilson(k, n)[0] > 0.632,
           "cost_usd": round(sum(((r.get("usage") or {}).get("cost") or 0) for r in recs), 4)}
    lp = [(r["p_ssa_logprob"], r["label_true"] == "SSA") for r in recs if r.get("p_ssa_logprob") is not None]
    if lp:
        sc, tr = [x for x, _ in lp], [y for _, y in lp]
        auc = auroc(sc, tr)
        rng = random.Random(20260930)
        boots = []
        for _ in range(5000):
            idx = [rng.randrange(len(sc)) for _ in sc]
            v = auroc([sc[i] for i in idx], [tr[i] for i in idx])
            if v is not None:
                boots.append(v)
        boots.sort()
        out["logprob_auroc"] = round(auc, 4)
        out["logprob_auroc_ci95"] = [round(boots[int(0.025 * len(boots))], 4), round(boots[int(0.975 * len(boots)) - 1], 4)]
        out["n_logprob"] = len(lp)
        med = sorted(sc)[len(sc) // 2]
        out["logprob_p_ssa_median"] = med
        out["logprob_median_log10_p_hp"] = round(math.log10(max(1 - med, 1e-300)), 2)
    return out


def main():
    res = {}
    for model, ptag in PILOT.items():
        pf = os.path.join(ROOT, "runs", f"cte_p1_pilot__openrouter__{ptag}.jsonl")
        if os.path.exists(pf):
            recs = []
            for l in open(pf):
                r = json.loads(l)
                lab = (r.get("parsed") or {}).get("label") if not r.get("error") else None
                recs.append({"label": lab if lab in ("HP", "SSA") else None, "label_true": r["label_true"],
                             "usage": (r.get("meta") or {}).get("usage")})
            res.setdefault(model, {})["grid_cte (pilot)"] = summarize(recs)
    import diag_run
    for f in sorted(glob.glob(os.path.join(D, "*.jsonl"))):
        recs = [json.loads(l) for l in open(f)]
        for r in recs:  # re-parse every record with the strict bare-answer parser
            r["label"] = diag_run.parse(r["prompt_kind"], r.get("raw_response"))
            if r.get("p_ssa_logprob") is not None and r.get("p_ssa_rule") != "exact-token v2":
                r["p_ssa_rule"] = "prefix v1 (first diagnostic run)"
        if recs:
            res.setdefault(recs[0]["model"], {})[recs[0]["arm"]] = summarize(recs)
    decisions = {}
    for model, arms in res.items():
        cm = arms.get("clean_min")
        if not cm or not cm.get("n_usable"):
            continue
        rule1 = cm["ssa_call_rate"] <= 0.75 and cm["balanced_accuracy"] >= 0.60
        if "logprob_auroc" in cm:
            rule1 = rule1 and cm["logprob_auroc"] >= 0.65 and cm["logprob_auroc_ci95"][0] > 0.5
        al = arms.get("clean_alias")
        decisions[model] = {
            "rule1_prompt_grid_induced_default": bool(rule1),
            "rule2_competent_arms": [a for a, v in arms.items() if v.get("competent_by_pilot_bar")],
            "rule3_label_name_shift_pts": round(100 * (al["ssa_call_rate"] - cm["ssa_call_rate"]), 1) if al and al.get("n_usable") else None,
            "rule3_label_names_shift_answer": (bool(abs(al["ssa_call_rate"] - cm["ssa_call_rate"]) >= 0.20)
                                               if al and al.get("n_usable") else "not applicable (no alias arm)"),
        }
    # ---- addendum rules 4 and 5 (PLAN.md addendum)
    for model, arms in res.items():
        al, ar, cm, ni = arms.get("clean_alias"), arms.get("clean_alias_rev"), arms.get("clean_min"), arms.get("noimage_min")
        dec = decisions.setdefault(model, {})
        if al and ar and al.get("n_usable") and ar.get("n_usable"):
            drop = al["ssa_call_rate"] - ar["ssa_call_rate"]
            dec["rule4_ssa_rate_alias_vs_alias_rev"] = [al["ssa_call_rate"], ar["ssa_call_rate"]]
            dec["rule4_reading"] = ("letter/position drives the answer" if drop >= 0.20 else
                                    "preference follows SSA content regardless of letter/position"
                                    if al["ssa_call_rate"] >= 0.90 and ar["ssa_call_rate"] >= 0.90 else "neither criterion met")
        if ni and ni.get("n_records"):
            dec["rule5_noimage_usable_answers"] = f"{ni.get('n_usable', 0)}/{ni['n_records']}"
            dec["rule5_noimage_ssa_rate"] = ni.get("ssa_call_rate")
            dec["rule5_as_written"] = ("SSA default exists without any image" if ni.get("n_usable") and ni["ssa_call_rate"] >= 0.90
                                       else "no image-independent SSA default")
            dec["rule5_reading"] = ("SSA default exists without any image" if ni.get("n_usable") and ni["ssa_call_rate"] >= 0.90
                                    and ni["n_usable"] >= 50 else
                                    "not assessable from labels (mostly refusals)" if (ni.get("n_usable") or 0) < 50 else
                                    "no image-independent SSA default")
            if "logprob_p_ssa_median" in ni and cm and "logprob_p_ssa_median" in cm:
                dec["rule5_logprob_median_p_ssa_noimage_vs_clean_min"] = [ni["logprob_p_ssa_median"], cm["logprob_p_ssa_median"]]
                dec["rule5_logprob_reading"] = (
                    "image adds no SSA evidence beyond the prior" if ni["logprob_p_ssa_median"] >= cm["logprob_p_ssa_median"]
                    else "not triggered: with the image the answer-token preference is MORE SSA than without")
                dec["rule5_logprob_caveat"] = ("clean_min scored with the prefix rule v1 (as registered), noimage with the "
                                               "exact-token rule v2; the two rules agreed to 1e-16 on alias_rev")
            if "rule5_reading" in dec and dec["rule5_reading"] != dec.get("rule5_as_written"):
                dec["rule5_deviation"] = ("the >= 50 usable-answer threshold is NOT in PLAN.md; added after seeing 96/100 refusals. "
                                          "As written the rule reads '" + dec["rule5_as_written"] + "' on "
                                          + dec["rule5_noimage_usable_answers"] + " answers.")
    json.dump({"arms": res, "decisions": decisions}, open(os.path.join(D, "RESULTS.json"), "w"), indent=2)
    print(f"{'model':34s} {'arm':18s} {'n':>3} {'acc':>6} {'wilson95':>16} {'bal':>6} {'SSA%':>5} {'fisherp':>8} {'AUROC [CI]':>22} {'$':>6}")
    for model, arms in res.items():
        for arm, v in arms.items():
            if not v.get("n_usable"):
                print(f"{model:34s} {arm:18s} no usable answers ({v['n_records']} records)")
                continue
            au = f"{v['logprob_auroc']} {v['logprob_auroc_ci95']}" if "logprob_auroc" in v else ""
            print(f"{model[:34]:34s} {arm:18s} {v['n_usable']:>3} {v['accuracy']:>6.3f} {str(v['wilson95']):>16} "
                  f"{v['balanced_accuracy']:>6.3f} {v['ssa_call_rate']:>5.2f} {v['label_vs_truth_fisher_p']:>8.3g} {au:>22} {v['cost_usd']:>6.3f}")
    print(json.dumps(decisions, indent=1))


if __name__ == "__main__":
    main()
