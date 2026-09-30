"""Masking on the remaining 450 test tiles: pre-specified analysis (runs/masking_ext2/PLAN.md).

    python3 analyze_ext2.py  ->  runs/masking_ext2/RESULTS.json (+ printed summary)
"""

import json
import math
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
H = "openrouter__google-gemma-4-31b-it__friendli"
OCCS = ("mean", "blur", "black")
R = lambda *p: os.path.join(ROOT, "runs", *p)


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def ok(r):
    return (not r.get("error") and (r.get("parsed") or {}).get("label") in ("HP", "SSA")
            and (r.get("baseline") or {}).get("label") in ("HP", "SSA"))


def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def fisher(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    pr = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    obs = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= obs * (1 + 1e-9)))


def power(n, share, alpha=0.05):
    """Power of the two-sided exact sign test on n non-tied tiles at a true cited share."""
    crit = next((k for k in range(n // 2, n + 1) if sign_p(k, n - k) <= alpha), None)
    if crit is None:
        return None
    return sum(math.comb(n, i) * share ** i * (1 - share) ** (n - i) for i in range(crit, n + 1))


def tiles_from(cells, cited_key, ctrl_arm):
    out = {}
    for img in sorted({k[0] for k in cells}):
        pr = []
        for o in OCCS:
            a, b = cells.get((img, cited_key, o)), cells.get((img, ctrl_arm, o))
            if not a or not b:
                break
            pr.append((int(a["parsed"]["label"] != a["baseline"]["label"]), int(b["parsed"]["label"] != b["baseline"]["label"])))
        if len(pr) == 3:
            out[img] = pr
    return out


def summary(t):
    nets = [sum(c - k for c, k in v) for v in t.values()]
    pos, neg = sum(n > 0 for n in nets), sum(n < 0 for n in nets)
    calls = 3 * len(t)
    return {"n_tiles": len(t), "cited_more": pos, "control_more": neg, "tied": len(t) - pos - neg,
            "sign_test_p": float(f"{sign_p(pos, neg):.3g}"),
            "cited_flip_rate": round(sum(c for v in t.values() for c, _ in v) / max(calls, 1), 4),
            "control_flip_rate": round(sum(k for v in t.values() for _, k in v) / max(calls, 1), 4)}


def load(path):
    return {(r["image"], r["arm"], r["occlusion"]): r for r in rows(path) if ok(r)}


def main():
    rep = {}
    area = load(R(f"masking_ext2__{H}_k3.jsonl"))
    typ = load(R(f"masking_typematched2__{H}_k3.jsonl"))
    t_area = tiles_from(area, "cited", "tissue_matched")
    s = summary(t_area)
    nt = s["cited_more"] + s["control_more"]
    s["reading"] = ("REPLICATES (cited > control, p < .05)" if s["sign_test_p"] < 0.05 and s["cited_more"] > s["control_more"]
                    else "OPPOSITE direction (p < .05)" if s["sign_test_p"] < 0.05 else "NOT replicated")
    s["power_at_ext_share_0.675"] = round(power(nt, 0.675), 3) if nt else None
    rep["1_primary_ext2_cited_vs_area_matched"] = s
    rep["2_heterogeneity_ext_vs_ext2_fisher_p"] = float(f"{fisher(102, 49, s['cited_more'], s['control_more']):.3g}")
    # 3. secondary: cited (main run) vs type-matched (concurrent run), feasible tiles only
    merged = dict(area)
    merged.update({k: v for k, v in typ.items() if k[1] == "type_matched"})
    rep["3_secondary_cited_vs_type_matched"] = summary(tiles_from(merged, "cited", "type_matched"))
    # 4. full-set summary
    def ext_tiles(path, ctrl="tissue_matched"):
        return tiles_from(load(path), "cited", ctrl)
    legacy = {}
    for p in ("masking_cte_p1_full_k3.jsonl", "masking_abl100_k3.jsonl"):
        for k, v in tiles_from(load(R(p)), "cited", "tissue_matched").items():
            legacy.setdefault(k, v)
    table = {
        "legacy_185_cerebras": summary(legacy),
        "legacy_185_friendli_cerebras_masks": summary(ext_tiles(R(f"masking_xover__{H}_k3.jsonl"))),
        "legacy_184_friendli_own_citations": summary(ext_tiles(R(f"masking_xover_own__{H}_k3.jsonl"))),
        "extension_319_friendli": summary(ext_tiles(R(f"masking_ext__{H}_k3.jsonl"))),
        "ext2_friendli": s,
    }
    rep["4_full_set_table"] = {k: {x: v[x] for x in ("n_tiles", "cited_more", "control_more", "tied", "sign_test_p",
                                                     "cited_flip_rate", "control_flip_rate")} for k, v in table.items()}
    # Friendli own-citation pool (pre-specified): extension + ext2 + legacy own-citation
    parts = [table["extension_319_friendli"], table["legacy_184_friendli_own_citations"], s]
    pos = sum(p["cited_more"] for p in parts)
    neg = sum(p["control_more"] for p in parts)
    tie = sum(p["tied"] for p in parts)
    # 3x2 chi-square for heterogeneity across the three Friendli tile sets
    tot_pos, tot_n = pos, pos + neg
    chi = sum((p["cited_more"] - (p["cited_more"] + p["control_more"]) * tot_pos / tot_n) ** 2 /
              max((p["cited_more"] + p["control_more"]) * tot_pos / tot_n, 1e-9) +
              (p["control_more"] - (p["cited_more"] + p["control_more"]) * (tot_n - tot_pos) / tot_n) ** 2 /
              max((p["cited_more"] + p["control_more"]) * (tot_n - tot_pos) / tot_n, 1e-9) for p in parts)
    rep["4b_friendli_own_citation_pool"] = {
        "tile_sets": "extension 319 + legacy 184 (own citations) + ext2",
        "n_tiles": sum(p["n_tiles"] for p in parts), "cited_more": pos, "control_more": neg, "tied": tie,
        "sign_test_p": float(f"{sign_p(pos, neg):.3g}"),
        "heterogeneity_chi2_df2": round(chi, 3), "heterogeneity_p": float(f"{math.exp(-chi / 2):.3g}"),
        "note": "report only together with the heterogeneity across tile sets"}
    # coverage of the 977-tile test set
    covered_friendli = set(table and ext_tiles(R(f"masking_ext__{H}_k3.jsonl"))) | set(t_area) | \
        set(ext_tiles(R(f"masking_xover_own__{H}_k3.jsonl")))
    rep["5_coverage"] = {"test_tiles": 977, "with_a_usable_friendli_own_citation_masking_result": len(covered_friendli)}
    # integrity
    provs = {(r.get("meta") or {}).get("provider") for r in list(area.values()) + list(typ.values())}
    bprovs = {(r.get("baseline") or {}).get("provider") for r in list(area.values()) + list(typ.values())}
    rep["integrity"] = {"masked_providers": sorted(map(str, provs)), "baseline_providers": sorted(map(str, bprovs)),
                        "area_records": len(area), "type_records": len(typ)}
    os.makedirs(R("masking_ext2"), exist_ok=True)
    json.dump(rep, open(R("masking_ext2", "RESULTS.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
