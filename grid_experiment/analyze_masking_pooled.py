"""Pooled, cluster-correct grid-masking test across sweeps, one unit per TILE.

Unit: a tile. Per tile, net = sum over the three occlusions of
(cited-arm flip - tissue-matched-control flip), where flip = masked label differs
from that tile's unmasked baseline label. Primary test: exact two-sided sign test
on tiles with net != 0. Also reported: mean net with a tile-bootstrap 95% CI, a
minimum-detectable-effect statement, per-sweep and per-host strata, and the
pseudo-replicated pair-level count for comparison with earlier reports.

Sweeps pooled (all use the model's own full-run citations and baselines):
  B    runs/masking_cte_p1_full_k3.jsonl          proportional, 100 tiles
  C    runs/masking_abl100_k3.jsonl               design-matched, 100 tiles
  ext  runs/masking_ext__<host>_k3.jsonl          extension tiles (any host)

Each tile is counted ONCE. C re-used B's records verbatim for 15 shared tiles.
Counting them in both sweeps -- as the earlier "61 vs 57" pooled figure did --
double-counts them. Precedence is B, then C, then ext. Any tile appearing in two
sweeps with DIFFERENT records is reported as a conflict, not silently merged.

The 20-tile pilot (runs/masking_cte_p1_k3.jsonl) used pilot-run citations, and
18 of its tiles are re-tested in C, so it is excluded from the primary pool and
reported as a sensitivity line only.

    python3 analyze_masking_pooled.py
        -> runs/masking_pooled_analysis.json          B + C + finished extensions
        -> runs/masking_pooled_legacy_analysis.json   B + C only (Cerebras), always rewritten,
                                                      so the legacy number stays citable
    python3 analyze_masking_pooled.py --include-unfinished
        -> runs/masking_pooled_analysis_INCLUDING_UNFINISHED.json  (never overwrites the above)
"""

import glob
import json
import math
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OCCS = ("mean", "blur", "black")
SEED = 20260929


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


LABELS = ("HP", "SSA")
LEGACY_HOST = "legacy-cerebras__gemma-4-31b"


def record_host(r):
    """Deployment id of a masked call: legacy sweeps predate base_url logging."""
    if not r.get("base_url"):
        return LEGACY_HOST
    return f"{r['base_url']} | {r.get('model')} | provider={(r.get('meta') or {}).get('provider')}"


def baseline_host(r):
    """Deployment id of the unmasked baseline a masked call is compared with."""
    b = r.get("baseline") or {}
    if not b.get("base_url"):
        return LEGACY_HOST if not r.get("base_url") else "unrecorded"
    return f"{b['base_url']} | {b.get('model')} | provider={b.get('provider')}"


def load_sweep(path, name):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    ok = [r for r in recs if not r.get("error") and (r.get("parsed") or {}).get("label") in LABELS
          and (r.get("baseline") or {}).get("label") in LABELS]
    cell = {(r["image"], r["arm"], r["occlusion"]): r for r in ok}
    tiles, incomplete, mixed = {}, [], []
    for img in sorted({r["image"] for r in recs}):
        pairs = []
        for occ in OCCS:
            a, b = cell.get((img, "cited", occ)), cell.get((img, "tissue_matched", occ))
            if not a or not b:
                break
            pairs.append((int(a["parsed"]["label"] != a["baseline"]["label"]),
                          int(b["parsed"]["label"] != b["baseline"]["label"])))
        if len(pairs) != 3:
            incomplete.append(img)
            continue
        hosts = {record_host(cell[k]) for k in cell if k[0] == img}
        # the flip is masked label vs baseline label, so both must come from one deployment
        hosts |= {baseline_host(cell[k]) for k in cell if k[0] == img}
        if len(hosts) != 1:
            mixed.append({"image": img, "hosts": sorted(hosts)})
            continue
        sig = tuple(sorted((k[1], k[2], cell[k]["raw_response"] or "") for k in cell if k[0] == img))
        tiles[img] = {"pairs": pairs, "net": sum(c - t for c, t in pairs), "sweep": name,
                      "host": hosts.pop(), "signature": hash(sig)}
    return tiles, {"records": len(recs), "errors": sum(1 for r in recs if r.get("error")),
                   "invalid_label": sum(1 for r in recs if not r.get("error") and r not in ok),
                   "incomplete_tiles": incomplete, "mixed_host_tiles": mixed}


def summarize(units, rng):
    if not units:
        return "EMPTY"
    nets = [u["net"] for u in units]
    pos = sum(n > 0 for n in nets)
    neg = sum(n < 0 for n in nets)
    pair_c = sum(c and not t for u in units for c, t in u["pairs"])
    pair_t = sum(t and not c for u in units for c, t in u["pairs"])
    boots = []
    for _ in range(10000):
        s = [nets[rng.randrange(len(nets))] for _ in nets]
        boots.append(sum(s) / len(s))
    boots.sort()
    return {
        "n_tiles": len(units), "tiles_cited_more": pos, "tiles_control_more": neg,
        "tiles_tied": len(units) - pos - neg, "sign_test_p": round(sign_p(pos, neg), 4),
        "mean_net_per_tile": round(sum(nets) / max(len(nets), 1), 4),
        "mean_net_bootstrap_95ci": [round(boots[250], 4), round(boots[9749], 4)],
        "pair_level_cited_only": pair_c, "pair_level_control_only": pair_t,
        "pair_level_note": "pseudo-replicated (3 occlusions per tile share cells and baseline); reference only",
    }


def min_detectable(n_nontied, alpha=0.05, power=0.80):
    """Smallest cited share of non-tied tiles detectable with the given power
    (two-sided exact sign test)."""
    if n_nontied == 0:
        return None
    crit = None
    for k in range(n_nontied + 1):  # upper rejection threshold
        if sign_p(k, n_nontied - k) <= alpha and k > n_nontied / 2:
            crit = k
            break
    if crit is None:
        return None
    for q in [x / 1000 for x in range(500, 1000)]:
        pw = sum(math.comb(n_nontied, i) * q ** i * (1 - q) ** (n_nontied - i)
                 for i in range(crit, n_nontied + 1))
        if pw >= power:
            return {"n_nontied": n_nontied, "min_cited_share": q,
                    "as_odds": f"{q/(1-q):.2f}:1 cited:control among non-tied tiles"}
    return None


def fisher_2x2(a, b, c, d):
    n = a + b + c + d
    def pr(x):
        return math.comb(a + b, x) * math.comb(c + d, a + c - x) / math.comb(n, a + c)
    obs = pr(a)
    lo, hi = max(0, a + c - (c + d)), min(a + b, a + c)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= obs * (1 + 1e-9)))


LEGACY_SOURCES = [("B", "runs/masking_cte_p1_full_k3.jsonl"), ("C", "runs/masking_abl100_k3.jsonl")]


def main():
    import sys
    include_unfinished = "--include-unfinished" in sys.argv
    sources, skipped_ext = list(LEGACY_SOURCES), []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "masking_ext__*_k3.jsonl"))):
        host = os.path.basename(p).replace("masking_ext__", "").replace("_k3.jsonl", "")
        # only extensions whose chain step finished; partial or abandoned hosts are listed, not pooled
        if os.path.exists(os.path.join(ROOT, "runs", "logs", f".done__{host}__ext_masked")) or include_unfinished:
            sources.append((f"ext:{host}", p))
        else:
            skipped_ext.append(os.path.relpath(p, ROOT))
    out_main = "masking_pooled_analysis_INCLUDING_UNFINISHED.json" if include_unfinished else "masking_pooled_analysis.json"
    legacy = analyse(LEGACY_SOURCES, [])
    legacy["scope"] = "legacy only: sweeps B + C, Cerebras gemma-4-31b (model since archived)"
    write(legacy, "masking_pooled_legacy_analysis.json", quiet=True)
    full = analyse(sources, skipped_ext)
    full["scope"] = ("B + C + " + (", ".join(n for n, _ in sources[2:]) or "no finished extensions")
                     + (" -- INCLUDES UNFINISHED EXTENSIONS, not for citation" if include_unfinished else ""))
    write(full, out_main)


def write(report, name, quiet=False):
    out = os.path.join(ROOT, "runs", name)
    json.dump(report, open(out, "w"), indent=2)
    if not quiet:
        show = ("scope", "integrity", "skipped_unfinished_extensions", "duplicates_identical_records_counted_once",
                "duplicate_conflicts_different_records", "primary_pooled_unique_tiles",
                "per_host_unique_tiles", "between_host_heterogeneity", "min_detectable_effect_80pct_power")
        print(json.dumps({k: report[k] for k in show if k in report}, indent=1))
    p = report["primary_pooled_unique_tiles"]
    print(f"-> {out}: {p['n_tiles']} tiles, {p['tiles_cited_more']} cited-more vs "
          f"{p['tiles_control_more']} control-more, {p['tiles_tied']} tied, sign test p = {p['sign_test_p']}")


def analyse(sources, skipped_ext):
    rng = random.Random(SEED)
    pooled, dup_identical, conflicts, per_sweep, integrity = {}, [], [], {}, {}
    for name, path in sources:
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        tiles, integ = load_sweep(full, name)
        integrity[name] = {**integ, "complete_tiles": len(tiles)}
        per_sweep[name] = list(tiles.values())
        for img, u in tiles.items():
            if img in pooled:
                (dup_identical if pooled[img]["signature"] == u["signature"] else conflicts).append(
                    {"image": img, "kept": pooled[img]["sweep"], "dropped": name})
                continue
            pooled[img] = u

    report = {
        "unit": "tile",
        "skipped_unfinished_extensions": skipped_ext,
        "sources": {n: os.path.relpath(p if os.path.isabs(p) else os.path.join(ROOT, p), ROOT) for n, p in sources},
        "integrity": integrity,
        "duplicates_identical_records_counted_once": len(dup_identical),
        "duplicate_conflicts_different_records": conflicts,
        "primary_pooled_unique_tiles": summarize(list(pooled.values()), rng),
        "per_sweep_as_run": {n: summarize(v, rng) for n, v in per_sweep.items() if v},
        "per_host_unique_tiles": {},
    }
    by_host = defaultdict(list)
    for u in pooled.values():
        by_host[u["host"]].append(u)
    report["per_host_unique_tiles"] = {h: summarize(v, random.Random(f"{SEED}:{h}")) for h, v in by_host.items()}
    hosts = sorted(by_host)
    if len(hosts) == 2:
        s1, s2 = (report["per_host_unique_tiles"][h] for h in hosts)
        p_het = fisher_2x2(s1["tiles_cited_more"], s1["tiles_control_more"],
                           s2["tiles_cited_more"], s2["tiles_control_more"])
        report["between_host_heterogeneity"] = {
            "test": "Fisher exact on host x (cited-more, control-more) tiles",
            "hosts": hosts, "p": round(p_het, 4),
            "reading": "pooled estimate mixes two deployments that DISAGREE (p < .05); report per host"
            if p_het < 0.05 else "no evidence the two deployments disagree; pooled estimate reported, per-host shown",
        }
    elif len(hosts) > 2:
        report["between_host_heterogeneity"] = "more than two hosts: report per host; pooled figure not interpreted"
    prim = report["primary_pooled_unique_tiles"]
    report["min_detectable_effect_80pct_power"] = min_detectable(prim["tiles_cited_more"] + prim["tiles_control_more"])

    pilot = os.path.join(ROOT, "runs", "masking_cte_p1_k3.jsonl")
    if os.path.exists(pilot):
        pt, _ = load_sweep(pilot, "pilot")
        extra = [u for img, u in pt.items() if img not in pooled]
        report["sensitivity_plus_pilot_tiles_not_otherwise_present"] = {
            "added_tiles": len(extra), **summarize(list(pooled.values()) + extra, rng)}
    return report


if __name__ == "__main__":
    main()
