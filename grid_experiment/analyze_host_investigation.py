"""Why does the masking effect differ between Cerebras and Friendli (same weights)?

Part A: what differs in the request and response path. Same 340 tiles, same cte_p1 request:
    input tokens, output length, finish reason, parse path, key order, code-fence rate.
Part B: sampling probe. The 100 abl100 tiles on Friendli, cte_p1, with Gemma's
    recommended top_p 0.95 / top_k 64 (K = 2), and at temperature 0.5 (K = 2) and
    0 (greedy, K = 1), compared with Friendli at host defaults
    (ordering-run cte, K = 3) and with Cerebras (cte_p1_full, replicate 1). Metrics:
    code-fence rate, replicate disagreement, SSA rate, agreement with Cerebras labels.
Part C: crossover. The 185 legacy tiles with their ORIGINAL Cerebras-built masks,
    re-run on Friendli (fresh Friendli baselines), with the same tile-level sign test
    as the pooled analysis. It is compared with Cerebras on the same tiles (54 vs 47)
    and with the Friendli extension (102 vs 49).

Part D: the same 185 legacy tiles on Friendli, with masks rebuilt from FRIENDLI's own
    citations (its crossover baselines). This separates "which tiles" from "whose
    citations are masked".

    python3 analyze_host_investigation.py  ->  runs/HOST_INVESTIGATION.json
"""

import json
import math
import os
import statistics
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
H = "openrouter__google-gemma-4-31b-it__friendli"
R = lambda *p: os.path.join(ROOT, "runs", *p)
OCCS = ("mean", "blur", "black")


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def valid(r):
    return not r.get("error") and (r.get("parsed") or {}).get("label") in ("HP", "SSA")


def fence(r):
    return (r.get("raw_response") or "").lstrip().startswith("```")


def binom_p(k, n):
    """Exact two-sided sign test."""
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n)


def fisher(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    pr = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    obs = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= obs * (1 + 1e-9)))


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def profile(recs, name):
    v = [r for r in recs if valid(r)]
    f = sum(fence(r) for r in v)
    by = {}
    for r in v:
        by.setdefault(r["image"], {})[r.get("replicate", 1)] = r["parsed"]["label"]
    pairs = [(d[1], d[2]) for d in by.values() if 1 in d and 2 in d]
    dis = sum(a != b for a, b in pairs)
    ct = sorted(((r.get("meta") or {}).get("usage") or {}).get("completion_tokens") or 0 for r in v)
    pt = Counter(((r.get("meta") or {}).get("usage") or {}).get("prompt_tokens") for r in v)
    return {"name": name, "n_valid": len(v), "code_fence": f, "code_fence_rate": round(f / max(len(v), 1), 4),
            "code_fence_rate_ci95": wilson(f, len(v)),
            "rep1_vs_rep2_disagreement": f"{dis}/{len(pairs)}" if pairs else None,
            "rep1_vs_rep2_rate": round(dis / len(pairs), 4) if pairs else None,
            "ssa_rate": round(sum(r["parsed"]["label"] == "SSA" for r in v) / max(len(v), 1), 4),
            "completion_tokens_median": statistics.median(ct) if ct else None,
            "prompt_tokens": dict(pt), "labels_by_tile": by}


def tile_test(recs):
    cell = {(r["image"], r["arm"], r["occlusion"]): r for r in recs
            if valid(r) and (r.get("baseline") or {}).get("label") in ("HP", "SSA")}
    arms = sorted({k[1] for k in cell} - {"cited"})
    ctrl = arms[0] if arms else None
    pos = neg = tie = 0
    cflip = kflip = ncalls = 0
    for img in sorted({k[0] for k in cell}):
        pairs = []
        for o in OCCS:
            a, b = cell.get((img, "cited", o)), cell.get((img, ctrl, o))
            if not a or not b:
                break
            pairs.append((a["parsed"]["label"] != a["baseline"]["label"], b["parsed"]["label"] != b["baseline"]["label"]))
        if len(pairs) != 3:
            continue
        net = sum(int(x) - int(y) for x, y in pairs)
        pos += net > 0
        neg += net < 0
        tie += net == 0
        cflip += sum(x for x, _ in pairs)
        kflip += sum(y for _, y in pairs)
        ncalls += 3
    return {"control_arm": ctrl, "n_tiles": pos + neg + tie, "cited_more": pos, "control_more": neg, "tied": tie,
            "sign_test_p": float(f"{binom_p(pos, pos + neg):.3g}"),
            "cited_flip_rate": round(cflip / max(ncalls, 1), 4), "control_flip_rate": round(kflip / max(ncalls, 1), 4)}


def main():
    out = {}
    # ---------------- Part A: request/response path, same 340 tiles
    cer = {r["image"]: r for r in rows(R("cte_p1_full.jsonl")) if r.get("replicate", 1) == 1 and valid(r)}
    fr = {r["image"]: r for r in rows(R(f"cte_p1_ext__{H}.jsonl")) if valid(r)}
    shared = sorted(set(cer) & set(fr))
    A = {}
    for name, d in (("cerebras", cer), ("friendli_default", fr)):
        v = [d[t] for t in shared]
        A[name] = {
            "prompt_tokens": dict(Counter(((r.get("meta") or {}).get("usage") or {}).get("prompt_tokens") for r in v)),
            "completion_tokens_median": statistics.median(((r.get("meta") or {}).get("usage") or {}).get("completion_tokens") or 0 for r in v),
            "finish_reason": dict(Counter((r.get("meta") or {}).get("finish_reason") for r in v)),
            "key_order": dict(Counter(",".join(r.get("raw_key_order") or []) for r in v)),
            "code_fence": sum(fence(r) for r in v), "n": len(v),
            "confidence": dict(Counter((r.get("parsed") or {}).get("confidence") for r in v).most_common(4)),
        }
    both_f = sum(fence(cer[t]) and fence(fr[t]) for t in shared)
    only_c = sum(fence(cer[t]) and not fence(fr[t]) for t in shared)
    only_f = sum(fence(fr[t]) and not fence(cer[t]) for t in shared)
    A["code_fence_mcnemar"] = {"cerebras_only": only_c, "friendli_only": only_f, "both": both_f,
                               "p": float(f"{binom_p(only_c, only_c + only_f):.3g}")}
    A["label_agreement"] = f"{sum(cer[t]['parsed']['label'] == fr[t]['parsed']['label'] for t in shared)}/{len(shared)}"
    A["request_body_fields"] = ("same runner code on both hosts: one user message (image_url data-URL PNG + the same "
                                "prompt text), max_completion_tokens, temperature 1.0; no system role, response_format, "
                                "penalties or logit bias; top_p/top_k/min_p/seed never sent, so each host applied its own "
                                "defaults. Differences: model id (Cerebras 'gemma-4-31b' vs OpenRouter "
                                "'google/gemma-4-31b-it') and OpenRouter's provider-routing object. Request bodies are "
                                "not stored in the records; prompt tokens (825) match on every call.")
    out["A_request_response_path"] = A

    # ---------------- Part B: sampling probe (abl100 tiles)
    abl = set(json.load(open(R(".abl100_tiles.json"))))
    probe = [r for r in rows(R("cte_p1_decprobe.jsonl")) if r["image"] in abl]
    fdef = [r for r in rows(R("ordering_controls", f"cte__{H}.jsonl")) if r["image"] in abl]
    cer100 = [dict(cer[t], replicate=1) for t in abl if t in cer]
    B = {"friendli_top_p0.95_top_k64": profile(probe, "Friendli, top_p 0.95 / top_k 64"),
         "friendli_default": profile(fdef, "Friendli, host defaults (ordering cte run)"),
         "cerebras_rep1": profile(cer100, "Cerebras cte_p1_full replicate 1"),
         "friendli_temperature0.5": profile([r for r in rows(R("cte_p1_tempprobe_t05.jsonl")) if r["image"] in abl],
                                            "Friendli, temperature 0.5, host-default top_p/top_k"),
         "friendli_temperature0": profile([r for r in rows(R("cte_p1_tempprobe_t0.jsonl")) if r["image"] in abl],
                                          "Friendli, temperature 0 (greedy)")}
    for k in ("friendli_top_p0.95_top_k64", "friendli_default", "friendli_temperature0.5", "friendli_temperature0"):
        by = B[k]["labels_by_tile"]
        ag = [by[t][1] == cer[t]["parsed"]["label"] for t in by if 1 in by[t] and t in cer]
        B[k]["agreement_with_cerebras_rep1"] = f"{sum(ag)}/{len(ag)}"
    p1, p0 = B["friendli_top_p0.95_top_k64"], B["friendli_default"]
    B["fence_rate_probe_vs_default_fisher_p"] = float(f"{fisher(p1['code_fence'], p1['n_valid'] - p1['code_fence'], p0['code_fence'], p0['n_valid'] - p0['code_fence']):.3g}")
    for k in list(B):
        if isinstance(B[k], dict):
            B[k].pop("labels_by_tile", None)
    out["B_sampling_probe"] = B

    # ---------------- Part C: crossover
    xo = rows(R(f"masking_xover__{H}_k3.jsonl"))
    C = {"friendli_on_legacy185_with_cerebras_masks": tile_test(xo)}
    pooled = json.load(open(R("masking_pooled_analysis.json")))
    C["cerebras_on_same_185_tiles"] = pooled["per_host_unique_tiles"].get("legacy-cerebras__gemma-4-31b")
    C["friendli_extension_319"] = {k: v for k, v in pooled["per_sweep_as_run"][f"ext:{H}"].items()
                                   if k in ("n_tiles", "tiles_cited_more", "tiles_control_more", "tiles_tied", "sign_test_p")}
    xb = {r["image"]: r for r in rows(R(f"cte_p1_xover__{H}.jsonl")) if valid(r)}
    ag = [xb[t]["parsed"]["label"] == cer[t]["parsed"]["label"] for t in xb if t in cer]
    C["friendli_baseline_agreement_with_cerebras_on_185"] = f"{sum(ag)}/{len(ag)}"
    x = C["friendli_on_legacy185_with_cerebras_masks"]
    if x["n_tiles"]:
        C["vs_cerebras_same_tiles_fisher_p"] = float(f"{fisher(x['cited_more'], x['control_more'], 54, 47):.3g}")
        C["vs_friendli_extension_fisher_p"] = float(f"{fisher(x['cited_more'], x['control_more'], 102, 49):.3g}")
    out["C_crossover"] = C

    # ---------------- Part D: same 185 legacy tiles on Friendli, masks rebuilt from
    # FRIENDLI's own citations (its crossover baseline answers), not Cerebras's
    own = rows(R(f"masking_xover_own__{H}_k3.jsonl"))
    D = {"friendli_own_citations_on_legacy_tiles": tile_test(own)}
    d = D["friendli_own_citations_on_legacy_tiles"]
    if d["n_tiles"]:
        D["vs_crossover_with_cerebras_masks_fisher_p"] = float(f"{fisher(d['cited_more'], d['control_more'], x['cited_more'], x['control_more']):.3g}")
        D["vs_friendli_extension_fisher_p"] = float(f"{fisher(d['cited_more'], d['control_more'], 102, 49):.3g}")
        B_tiles = {json.loads(l)["tile_dir"] for l in open(R("masking_cte_p1_full_k3.jsonl"))}
        D["sweep_B_tiles"] = tile_test([r for r in own if r["tile_dir"] in B_tiles])
        D["sweep_C_unique_tiles"] = tile_test([r for r in own if r["tile_dir"] not in B_tiles])
    out["D_own_citation_crossover"] = D
    Cx = {json.loads(l)["tile_dir"] for l in open(R("masking_cte_p1_full_k3.jsonl"))}
    out["C_crossover"]["sweep_B_tiles"] = tile_test([r for r in xo if r["tile_dir"] in Cx])
    out["C_crossover"]["sweep_C_unique_tiles"] = tile_test([r for r in xo if r["tile_dir"] not in Cx])
    json.dump(out, open(R("HOST_INVESTIGATION.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
