"""Analyze the ordering-control conditions (pre-registered in runs/ordering_controls/PLAN.md).

Within ONE host, on the abl100 tiles:

Gates (exit non-zero unless --allow-incomplete / --allow-mixed-provider):
  completeness  every condition has a valid HP/SSA label for all 100 tiles x K reps
  provider      one upstream provider across every record (router hosts)

Requested format (replicate 1): paired flips vs classify-only (HP->SSA, SSA->HP),
exact McNemar.

Pre-registered primary estimand, per control condition c:
  Delta_c = mean over tiles of [p_c(t) - p_co(t)], where p(t) = fraction of the K
  replicates labelled SSA. 95% tile-bootstrap CI; two-sided sign-flip
  permutation p, Holm-corrected over the fixed family of four primary controls
  (ins_filler, ins_checklist, ins_copied, describe_first). Classification:
    shift        95% CI excludes 0 AND Holm p < .05
    equivalent   90% CI inside (-0.10, +0.10)   (TOST, alpha .05, margin 10 pts)
    inconclusive otherwise -- never reported as "no shift"

Reproduction gate (pre-registered): same-host co vs etc, replicate 1, McNemar
p < .05 with HP->SSA > SSA->HP. If it fails, the controls cannot be read as
controls for that effect and the report says so first.

    python3 analyze_ordering_controls.py --host openrouter__google-gemma-4-31b-it__<provider>

Writes runs/ordering_controls/analysis__<host>.json and .md
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, "runs", "ordering_controls")
PRIMARY = ["ins_filler", "ins_checklist", "ins_copied", "describe_first"]
ALL = ["co", "cte", "etc"] + PRIMARY + ["ins_self"]
PREAMBLE_CONDS = {"ins_filler", "ins_checklist", "ins_copied", "ins_self", "describe_first"}
COPY_CONDS = {"ins_filler", "ins_checklist", "ins_copied", "ins_self"}
MARGIN = 0.10
MIN_COMPLETE_TILES = 95  # pre-registered: analyse tiles complete in every condition if >= 95 remain
EMPTY = {"recs": {}, "counts": {}, "providers": Counter()}
COPY_FIDELITY_MIN = 0.90
N_BOOT, N_PERM = 10000, 20000


# ---------------------------------------------------------------- loading
def load_condition(path):
    recs, counts, providers = {}, Counter(), Counter()
    if not os.path.exists(path):
        return None
    for line in open(path):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            counts["unreadable_line"] += 1
            continue
        counts["records"] += 1
        key = (r["image"], r.get("replicate", 1))
        if r.get("error"):
            kind = str(r["error"]).split(" ")[0].split(":")[0]
            counts[f"error:{kind}"] += 1
            continue
        if (r.get("parsed") or {}).get("label") not in ("HP", "SSA"):
            counts["invalid_label"] += 1
            continue
        if key in recs:
            counts["duplicate_valid"] += 1
            continue
        recs[key] = r
        counts["valid"] += 1
        if r.get("parse_path") == "lenient":
            counts["lenient_parse"] += 1
        if (r.get("meta") or {}).get("finish_reason") == "length":
            counts["truncated"] += 1
        providers[(r.get("base_url"), r.get("model"), (r.get("meta") or {}).get("provider"))] += 1
    return {"recs": recs, "counts": dict(counts), "providers": providers}


def legacy(files):
    """Cerebras gemma-4-31b files (replicate 1 only) for the legacy reference."""
    out = {"recs": {}, "counts": Counter(), "providers": Counter()}
    for f in files:
        d = load_condition(os.path.join(ROOT, f))
        if d:
            for k, r in d["recs"].items():
                if k[1] == 1:
                    out["recs"][k] = r
    return out


# ---------------------------------------------------------------- statistics
def mcnemar(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def holm_fixed(pvals, m):
    """Holm step-down with a FIXED family size m (missing members count against it)."""
    present = [(p, k) for k, p in pvals.items() if p is not None]
    present.sort()
    adj, running = {}, 0.0
    for rank, (p, k) in enumerate(present):
        running = max(running, min(1.0, (m - rank) * p))
        adj[k] = round(running, 6)
    return adj


def p_ssa(d, tile, reps):
    vals = [d["recs"][(tile, r)]["parsed"]["label"] == "SSA" for r in reps if (tile, r) in d["recs"]]
    return sum(vals) / len(vals) if vals else None


def boot_ci(xs, seed, level=0.95):
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    lo = means[int((1 - level) / 2 * N_BOOT)]
    hi = means[int((1 + level) / 2 * N_BOOT) - 1]
    return round(lo, 4), round(hi, 4)


def signflip_p(xs, seed):
    rng = random.Random(seed)
    obs = abs(sum(xs))
    if obs == 0:
        return 1.0
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in xs)) >= obs - 1e-12 for _ in range(N_PERM))
    return (hits + 1) / (N_PERM + 1)


def paired_rep1(a, b, tiles):
    shared = [t for t in tiles if (t, 1) in a["recs"] and (t, 1) in b["recs"]]
    la = {t: a["recs"][(t, 1)]["parsed"]["label"] for t in shared}
    lb = {t: b["recs"][(t, 1)]["parsed"]["label"] for t in shared}
    h2s = sorted(t[6:9] for t in shared if la[t] == "HP" and lb[t] == "SSA")
    s2h = sorted(t[6:9] for t in shared if la[t] == "SSA" and lb[t] == "HP")
    if not shared:
        return "MISSING"
    return {"n_shared": len(shared), "HP_to_SSA": len(h2s), "SSA_to_HP": len(s2h),
            "flip_rate": round((len(h2s) + len(s2h)) / len(shared), 3),
            "mcnemar_p": mcnemar(len(h2s), len(s2h)),
            "flipped_HP_to_SSA": h2s, "flipped_SSA_to_HP": s2h}


def delta(a, b, tiles, reps_a, reps_b, seed, keep=None):
    """Mean over tiles of p_b(t) - p_a(t), with CIs and a sign-flip p."""
    ds = []
    for t in tiles:
        if keep is not None and t not in keep:
            continue
        pa, pb = p_ssa(a, t, reps_a), p_ssa(b, t, reps_b)
        if pa is not None and pb is not None:
            ds.append(pb - pa)
    if not ds:
        return "MISSING"
    return {"n_tiles": len(ds), "delta_ssa_rate": round(sum(ds) / len(ds), 4),
            "ci95": boot_ci(ds, seed, 0.95), "ci90": boot_ci(ds, seed + 1, 0.90),
            "signflip_p": signflip_p(ds, seed + 2)}


def classify(dl, holm_p):
    if not isinstance(dl, dict):
        return "missing"
    lo95, hi95 = dl["ci95"]
    lo90, hi90 = dl["ci90"]
    if (lo95 > 0 or hi95 < 0) and holm_p is not None and holm_p < 0.05:
        return "shift toward SSA" if dl["delta_ssa_rate"] > 0 else "shift toward HP"
    if -MARGIN < lo90 and hi90 < MARGIN:
        return f"equivalent to classify-only (within +/-{int(MARGIN * 100)} pts)"
    return "inconclusive"


def cluster_perm(y, group, cluster, strata, seed):
    """Stratified mean difference y[group==1] - y[group==0], p by permuting group
    labels across CLUSTERS (all members of a cluster move together)."""
    def stat(g):
        diffs, weights = [], []
        for s in set(strata.values()):
            a = [y[t] for t in y if strata[t] == s and g[cluster[t]] == 1]
            b = [y[t] for t in y if strata[t] == s and g[cluster[t]] == 0]
            if a and b:
                diffs.append(sum(a) / len(a) - sum(b) / len(b))
                weights.append(len(a) + len(b))
        return sum(d * w for d, w in zip(diffs, weights)) / sum(weights) if weights else None
    obs_g = {}
    for t in y:
        obs_g[cluster[t]] = group[t]
    obs = stat(obs_g)
    if obs is None:
        return None, None
    rng = random.Random(seed)
    clusters = sorted(obs_g)
    n1 = sum(obs_g.values())
    hits = 0
    for _ in range(N_PERM // 4):
        pick = set(rng.sample(clusters, n1))
        s = stat({c: int(c in pick) for c in clusters})
        if s is not None and abs(s) >= abs(obs) - 1e-12:
            hits += 1
    return round(obs, 4), (hits + 1) / (N_PERM // 4 + 1)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--allow-mixed-provider", action="store_true")
    args = ap.parse_args()
    host = args.host
    tiles = json.load(open(os.path.join(ROOT, "runs", ".abl100_tiles.json")))
    index = json.load(open(os.path.join(ROOT, "prompts", "rendered_ordering_controls", "index.json")))

    data = {c: load_condition(os.path.join(OUTDIR, f"{c}__{host}.jsonl")) for c in ALL}
    K = max((k[1] for d in data.values() if d for k in d["recs"]), default=1)
    reps = list(range(1, K + 1))

    # ---- gates
    problems = []
    for c in ALL:
        if not data[c]:
            problems.append(f"{c}: no output file")
            data[c] = dict(EMPTY)
    # completeness is judged over the conditions that have output; a wholly missing
    # condition is its own gate failure, and under --allow-incomplete it is reported
    # as MISSING rather than emptying every other analysis
    present = [c for c in ALL if data[c]["recs"]]
    complete = [t for t in tiles if all((t, r) in data[c]["recs"] for c in present for r in reps)]
    dropped = [t for t in tiles if t not in complete]
    if dropped:
        per_cond = {c: sum(1 for t in tiles for r in reps if (t, r) not in data[c]["recs"]) for c in present}
        msg = (f"{len(dropped)} tiles lack a valid label in some condition/replicate "
               f"(missing tile-replicates by condition {per_cond}); {len(complete)} tiles complete")
        if len(complete) < MIN_COMPLETE_TILES:
            problems.append(f"only {len(complete)} tiles complete in every condition (< {MIN_COMPLETE_TILES}): {msg}")
    provs = Counter()
    for d in data.values():
        if d:
            provs.update(d["providers"])
    distinct = {p[2] for p in provs}
    base_urls = {p[0] for p in provs}
    router = any("openrouter" in (u or "") for u in base_urls)
    if len({(p[0], p[1]) for p in provs}) > 1:
        problems.append(f"records span several endpoints/models: {sorted({(p[0], p[1]) for p in provs})}")
    if router and (len(distinct) != 1 or None in distinct):
        problems.append(f"router host with upstream providers {sorted(map(str, distinct))}: not one pinned provider")
    gate_fail = [p for p in problems if not (("complete in every condition" in p or "no output file" in p) and args.allow_incomplete)
                 and not ("provider" in p and args.allow_mixed_provider)]
    if gate_fail:
        print("GATE FAILED -- not analysing:\n  " + "\n  ".join(gate_fail))
        sys.exit(1)

    rep = {"host": host, "K_replicates": K, "n_tiles_planned": len(tiles),
           "n_tiles_analysed": len(complete), "dropped_incomplete_tiles": dropped,
           "gate_warnings": problems + ([msg] if dropped else []),
           "providers": {str(k): v for k, v in provs.items()},
           "counts": {c: data[c].get("counts") for c in ALL}}
    planned = tiles
    tiles = complete if complete else tiles

    co, etc_, cte = data["co"], data["etc"], data["cte"]

    # ---- summaries
    def summ(d, reps_use):
        labs = [(t, d["recs"][(t, r)]["parsed"]["label"]) for t in tiles for r in reps_use if (t, r) in d["recs"]]
        if not labs:
            return "MISSING"
        truth = {t: next(iter(d["recs"][k]["label_true"] for k in d["recs"] if k[0] == t)) for t in {x[0] for x in labs}}
        k = sum(l == truth[t] for t, l in labs)
        return {"n_calls": len(labs), "accuracy": round(k / len(labs), 4),
                "ssa_rate": round(sum(l == "SSA" for _, l in labs) / len(labs), 4)}
    rep["conditions"] = {c: summ(data[c], reps) for c in ALL}

    # ---- legacy reference (Cerebras, rep 1) and same-host reproduction gate
    L = {"co": legacy(["runs/co_p1.jsonl", "runs/co_p1_abl100.jsonl"]),
         "etc": legacy(["runs/etc_p1.jsonl", "runs/etc_p1_abl100.jsonl"]),
         "cte": legacy(["runs/cte_p1_full.jsonl"])}
    rep["legacy_cerebras_reference"] = {"co_vs_etc": paired_rep1(L["co"], L["etc"], planned),
                                        "cte_vs_etc": paired_rep1(L["cte"], L["etc"], planned),
                                        "note": "Cerebras gemma-4-31b (now archived), all planned tiles; reference only, never compared statistically"}
    co_etc = paired_rep1(co, etc_, tiles)
    rep["reproduction"] = {
        "co_vs_etc_rep1": co_etc,
        "cte_vs_etc_rep1": paired_rep1(cte, etc_, tiles),
        "co_vs_cte_rep1": paired_rep1(co, cte, tiles),
        "co_vs_etc_delta": delta(co, etc_, tiles, reps, reps, 11),
        "gate": "PASS" if isinstance(co_etc, dict) and co_etc["mcnemar_p"] < 0.05
        and co_etc["HP_to_SSA"] > co_etc["SSA_to_HP"] else "FAIL",
    }

    # ---- noise floor: co replicate vs co replicate
    if K >= 2:
        nf = {}
        for r2 in reps[1:]:
            a = {"recs": {(t, 1): co["recs"][(t, r2)] for t in tiles if (t, r2) in co["recs"]}}
            nf[f"co_rep1_vs_rep{r2}"] = paired_rep1(co, a, tiles)
        unstable = sum(len({co["recs"][(t, r)]["parsed"]["label"] for r in reps if (t, r) in co["recs"]}) > 1 for t in tiles)
        nf["co_tiles_label_unstable_across_reps"] = unstable
        rep["noise_floor"] = nf
    else:
        rep["noise_floor"] = "needs --reps >= 2"

    # ---- primary: requested format + pre-registered estimand
    prim, pvals = {}, {}
    for i, c in enumerate(PRIMARY):
        dl = delta(co, data[c], tiles, reps, reps, 100 + i)
        prim[c] = {"rep1_vs_classify_only": paired_rep1(co, data[c], tiles), "estimand": dl}
        pvals[c] = dl["signflip_p"] if isinstance(dl, dict) else None
    adj = holm_fixed(pvals, m=len(PRIMARY))
    for c in PRIMARY:
        prim[c]["holm_p"] = adj.get(c)
        prim[c]["classification"] = classify(prim[c]["estimand"], adj.get(c))
    rep["primary_vs_classify_only"] = prim

    # ---- secondary contrasts
    sec = {"ins_self_vs_classify_only": delta(co, data["ins_self"], tiles, reps, reps, 201),
           "ins_self_minus_ins_copied": delta(data["ins_copied"], data["ins_self"], tiles, reps, reps, 202),
           "etc_minus_ins_self": delta(data["ins_self"], etc_, tiles, reps, reps, 203)}
    for i, c in enumerate(PRIMARY):
        sec[f"etc_minus_{c}"] = delta(data[c], etc_, tiles, reps, reps, 210 + i)
    # ins_self vs ins_copied is NOT a clean tile-specificity contrast: ins_self texts
    # are 80% SSA-direction, donors are 50/50, 15 self texts name SSA lexically, and
    # the texts were generated on Cerebras. Direction-matched, unpaired versions:
    own = {t: index["ins_self"][t]["own_etc_label"] for t in tiles}
    don = {t: index["donors"][t]["donor_etc_label"] for t in tiles}
    for lab in ("HP", "SSA"):
        s_set = {t for t in tiles if own[t] == lab}
        c_set = {t for t in tiles if don[t] == lab}
        ds = delta(co, data["ins_self"], tiles, reps, reps, 220, keep=s_set)
        dc = delta(co, data["ins_copied"], tiles, reps, reps, 221, keep=c_set)
        sec[f"direction_matched_{lab}_text"] = {
            "ins_self_shift": ds, "ins_copied_shift": dc,
            "difference_self_minus_copied": round(ds["delta_ssa_rate"] - dc["delta_ssa_rate"], 4)
            if isinstance(ds, dict) and isinstance(dc, dict) else "MISSING",
            "note": "unpaired (different tiles); descriptive"}
    lexical = {t for t in tiles if index["ins_self"][t]["names_ssa_lexically"]}
    sec["ins_self_without_lexical_SSA_texts"] = delta(co, data["ins_self"], tiles, reps, reps, 222,
                                                      keep=set(tiles) - lexical)
    sec["ins_self_n_lexical_SSA_texts"] = len(lexical)
    # copied explanations cite grid cells of ANOTHER tile; some land on this tile's blank glass
    empty = {t: set(json.load(open(os.path.join(ROOT, "coverage", t.replace(".png", ".json"))))["empty_cells"]) for t in tiles}
    donor_cells = {}
    for t in tiles:
        txt = index["donors"][t]["inserted_text"]
        donor_cells[t] = set(__import__("re").findall(r"\b([A-D][1-4])\b", txt))
    hits = {t for t in tiles if donor_cells[t] & empty[t]}
    sec["ins_copied_by_donor_cells_on_recipient_background"] = {
        "cells_hit_background": delta(co, data["ins_copied"], tiles, reps, reps, 223, keep=hits),
        "cells_on_tissue": delta(co, data["ins_copied"], tiles, reps, reps, 224, keep=set(tiles) - hits),
        "note": "descriptive"}
    rep["secondary"] = sec
    rep["secondary_note"] = (
        "All secondary contrasts are descriptive. etc minus each control differs in instruction wording and output "
        "format as well as content. ins_self_minus_ins_copied is NOT a tile-specificity test: ins_self texts are 80% "
        "SSA-direction vs 50/50 for donors, 15 name SSA lexically, and both were generated on Cerebras; read the "
        "direction-matched and lexical-split rows instead.")

    # ---- exploratory: does the direction of inserted content matter?
    def shift(d):
        return {t: p_ssa(d, t, reps) - p_ssa(co, t, reps) for t in tiles
                if p_ssa(d, t, reps) is not None and p_ssa(co, t, reps) is not None}
    co_major = {t: ("SSA" if (p_ssa(co, t, reps) or 0) >= 0.5 else "HP") for t in tiles}
    donors = index["donors"]
    y = shift(data["ins_copied"])
    # unit = donor TEXT: each text's mean recipient shift, adjusted within
    # classify-only-majority strata; Welch t between HP-text and SSA-text means,
    # p by permuting the HP/SSA label over texts
    strat_mean = {s: sum(y[t] for t in y if co_major[t] == s) / max(1, sum(1 for t in y if co_major[t] == s))
                  for s in ("HP", "SSA")}
    by_text = defaultdict(list)
    for t in y:
        by_text[donors[t]["donor_image"]].append(y[t] - strat_mean[co_major[t]])
    text_mean = {d: sum(v) / len(v) for d, v in by_text.items()}
    text_lab = {d: index["donors"][next(t for t in y if donors[t]["donor_image"] == d)]["donor_etc_label"] for d in text_mean}

    def welch(lab):
        a = [text_mean[d] for d in text_mean if lab[d] == "HP"]
        b = [text_mean[d] for d in text_mean if lab[d] == "SSA"]
        if len(a) < 2 or len(b) < 2:
            return None
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
        vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
        se = math.sqrt(va / len(a) + vb / len(b)) or 1e-9
        return (ma - mb) / se, ma - mb
    obs = welch(text_lab)
    pval = None
    if obs:
        rng = random.Random(301)
        texts = sorted(text_mean)
        n_hp = sum(1 for d in texts if text_lab[d] == "HP")
        hits = 0
        for _ in range(N_PERM // 4):
            pick = set(rng.sample(texts, n_hp))
            w = welch({d: ("HP" if d in pick else "SSA") for d in texts})
            if w and abs(w[0]) >= abs(obs[0]) - 1e-12:
                hits += 1
        pval = (hits + 1) / (N_PERM // 4 + 1)
    rep["exploratory_donor_direction"] = {
        "statistic": "HP-text minus SSA-text mean recipient SSA-shift (stratum-adjusted); Welch t over donor texts",
        "difference": round(obs[1], 4) if obs else None, "welch_t": round(obs[0], 3) if obs else None,
        "text_level_permutation_p": pval,
        "texts": {"HP": sum(1 for d in text_lab if text_lab[d] == "HP"), "SSA": sum(1 for d in text_lab if text_lab[d] == "SSA")},
        "note": "exploratory and underpowered: 9 HP texts"}
    y = shift(data["ins_checklist"])
    gv = {t: int(index["checklist_variants"][t] == "hp_first") for t in y}
    obs, p = cluster_perm(y, gv, {t: t for t in y}, {t: co_major[t] for t in y}, 302)
    rep["exploratory_checklist_order"] = {"statistic": "SSA-shift hp_first minus ssa_first", "value": obs, "permutation_p": p}

    # ---- manipulation checks + sensitivity (drop failed manipulations; post-treatment caveat)
    def order_ok(r, first):
        ko = r.get("raw_key_order") or []
        return first in ko and "label" in ko and ko.index(first) < ko.index("label")
    checks, keeps = {}, {}
    for c in PREAMBLE_CONDS | {"etc"}:
        d = data[c]
        if not d["recs"]:
            checks[c] = "MISSING"
            continue
        first = "evidence" if c == "etc" else "preamble"
        bad = {k for k, r in d["recs"].items() if not order_ok(r, first)}
        if c in COPY_CONDS:
            bad |= {k for k, r in d["recs"].items() if (r.get("copy_similarity") or 0) < COPY_FIDELITY_MIN}
        if c == "describe_first":
            bad |= {k for k, r in d["recs"].items()
                    if r.get("describe_diagnostic_hits") or r.get("describe_paraphrase_hits")}
        checks[c] = {"order_fail": sum(1 for r in d["recs"].values() if not order_ok(r, first)),
                     "failed_manipulation_total": len(bad), "of": len(d["recs"])}
        if c in COPY_CONDS:
            sims = sorted(r.get("copy_similarity") or 0 for r in d["recs"].values())
            checks[c]["median_copy_similarity"] = sims[len(sims) // 2]
        if c == "describe_first":
            checks[c]["with_diagnostic_terms"] = sum(1 for r in d["recs"].values() if r.get("describe_diagnostic_hits"))
            checks[c]["with_criterion_paraphrases"] = sum(1 for r in d["recs"].values() if r.get("describe_paraphrase_hits"))
            term_counts = Counter(t for r in d["recs"].values()
                                  for t in (r.get("describe_diagnostic_hits") or []) + (r.get("describe_paraphrase_hits") or []))
            checks[c]["most_common_terms"] = term_counts.most_common(10)
        if c in PREAMBLE_CONDS:
            w = sorted(r.get("preamble_words") or len(str(r.get("preamble") or "").split()) for r in d["recs"].values())
            checks[c]["preamble_words_median"] = w[len(w) // 2] if w else None
        keeps[c] = {"recs": {k: r for k, r in d["recs"].items() if k not in bad}}
    rep["manipulation_checks"] = checks
    rep["sensitivity_failed_manipulations_dropped"] = {
        c: delta(co, keeps[c], tiles, reps, reps, 400 + i) for i, c in enumerate(PRIMARY + ["ins_self"]) if c in keeps}
    rep["sensitivity_etc_order_ok_only"] = delta(co, keeps.get("etc", etc_), tiles, reps, reps, 450)

    os.makedirs(OUTDIR, exist_ok=True)
    jpath = os.path.join(OUTDIR, f"analysis__{host}.json")
    json.dump(rep, open(jpath, "w"), indent=2, default=str)

    # ---- markdown
    md = [f"# Ordering controls, host `{host}` (K = {K} replicates per tile)", ""]
    g1 = rep["reproduction"]
    md += [f"**Reproduction gate: {g1['gate']}.** Same-host classify-only vs explain-then-classify, replicate 1: "
           f"{g1['co_vs_etc_rep1']['HP_to_SSA'] if isinstance(g1['co_vs_etc_rep1'], dict) else '?'} HP→SSA / "
           f"{g1['co_vs_etc_rep1']['SSA_to_HP'] if isinstance(g1['co_vs_etc_rep1'], dict) else '?'} SSA→HP "
           f"(Cerebras reference 28 / 3; classify-then-explain vs explain-then-classify reference 41 / 0).", ""]
    if g1["gate"] != "PASS":
        md += ["> The explain-first effect did not reproduce on this host, so the controls below are NOT "
               "interpretable as controls for it.", ""]
    md += ["| condition | rep-1 HP→SSA / SSA→HP vs classify-only | McNemar p | Δ SSA rate (K reps) [95% CI] | Holm p | reading |",
           "|---|---|---|---|---|---|"]
    for c in PRIMARY:
        v = prim[c]
        r1, dl = v["rep1_vs_classify_only"], v["estimand"]
        if not isinstance(dl, dict) or not isinstance(r1, dict):
            md.append(f"| {c} | missing | | | | |")
            continue
        md.append(f"| {c} | {r1['HP_to_SSA']} / {r1['SSA_to_HP']} | {r1['mcnemar_p']:.3g} | "
                  f"{dl['delta_ssa_rate']:+.3f} [{dl['ci95'][0]:+.3f}, {dl['ci95'][1]:+.3f}] | "
                  f"{v['holm_p']:.3g} | {v['classification']} |")
    md += ["", f"Tiles analysed: {rep['n_tiles_analysed']} of {rep['n_tiles_planned']}"
           + (f" (dropped as incomplete: {', '.join(t[6:9] for t in dropped)})" if dropped else "") + ".", ""]
    nf = rep.get("noise_floor")
    if isinstance(nf, dict) and isinstance(nf.get("co_rep1_vs_rep2"), dict):
        md += [f"Noise floor, classify-only replicate 1 vs 2: {nf['co_rep1_vs_rep2']['HP_to_SSA']} / "
               f"{nf['co_rep1_vs_rep2']['SSA_to_HP']} flips (rate {nf['co_rep1_vs_rep2']['flip_rate']}). "
               "Compare each condition's total rep-1 flip rate against this, descriptively:", ""]
        for c in PRIMARY:
            r1 = prim[c]["rep1_vs_classify_only"]
            if isinstance(r1, dict):
                md.append(f"- {c}: flip rate {r1['flip_rate']}")
        md.append("")
    md += ["Secondary (outside Holm):", ""]
    for k, v in sec.items():
        if isinstance(v, dict) and "delta_ssa_rate" in v:
            md.append(f"- {k}: {v['delta_ssa_rate']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] (n={v['n_tiles']})")
    def fmt(v):
        return (f"{v['delta_ssa_rate']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] (n={v['n_tiles']})"
                if isinstance(v, dict) and "delta_ssa_rate" in v else str(v))
    for lab in ("HP", "SSA"):
        v = sec[f"direction_matched_{lab}_text"]
        md.append(f"- {lab}-direction inserted text, unpaired: ins_self {fmt(v['ins_self_shift'])}; "
                  f"ins_copied {fmt(v['ins_copied_shift'])}")
    v = sec["ins_copied_by_donor_cells_on_recipient_background"]
    md.append(f"- ins_copied, donor cells land on recipient background: {fmt(v['cells_hit_background'])}; "
              f"all on tissue: {fmt(v['cells_on_tissue'])}")
    md += ["", "> " + rep["secondary_note"], "", "Exploratory:", ""]
    ed, ec = rep["exploratory_donor_direction"], rep["exploratory_checklist_order"]
    md.append(f"- donor-text direction (HP-text minus SSA-text recipient shift): {ed['difference']}, "
              f"Welch t {ed['welch_t']}, text-level permutation p {ed['text_level_permutation_p']} "
              f"({ed['texts']['HP']} HP / {ed['texts']['SSA']} SSA texts)")
    md.append(f"- checklist order (hp_first minus ssa_first): {ec['value']}, permutation p {ec['permutation_p']}")
    md += ["", "Counts per condition (records / valid / errors by kind):", ""]
    for c in ALL:
        md.append(f"- {c}: {rep['counts'].get(c)}")
    md += ["", "Manipulation checks: " + json.dumps(rep.get("manipulation_checks", {}))]
    open(os.path.join(OUTDIR, f"analysis__{host}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("->", jpath)


if __name__ == "__main__":
    main()
