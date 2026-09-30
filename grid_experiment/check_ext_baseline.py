"""Gate between the extension's classify-then-explain baseline and its masking calls.

Before ~1,900 masked calls are spent, check the new host's cte_p1 run on the
340 extension tiles (runs/cte_p1_ext__<host>.jsonl):

  completeness   every tile has an error-free replicate-1 record with an HP/SSA label,
                 except tiles the runner GAVE UP on (MAX_TRIES unusable responses),
                 tolerated up to 5% of the 340 and listed; unattempted tiles fail
  provider       one upstream provider across the file (router hosts)
  host agreement label agreement with the archived Cerebras run on the SAME tiles
                 (runs/cte_p1_full.jsonl), SSA-call rates, exact McNemar on discordant
                 pairs, and citation statistics (cells per response, empty-cell rate)

Exits 1 on incomplete or mixed-provider data, and 3 if label agreement with
Cerebras falls below 0.60. That threshold is written down before any data: two
independent temperature-1.0 samples on one host disagreed on about 5% of pilot
tiles and about 24% on deliberately unstable tiles. Below 0.60 the new host is
serving something materially different, and a human should decide before masking.
Writes runs/ext_baseline_check__<host>.json.

    python3 check_ext_baseline.py <host_tag>
"""

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MIN_AGREEMENT = 0.60
MAX_GAVE_UP_FRAC = 0.05


def mcnemar(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("replicate", 1) == 1 and not r.get("error") and (r.get("parsed") or {}).get("label") in ("HP", "SSA"):
            out[r["image"]] = r
    return out


def cite_stats(recs):
    cells = [{c for e in r["parsed"]["evidence"] for c in e.get("grid_cells_valid", [])} for r in recs]
    emp = sum(len(c & set(r["empty_cells"])) for c, r in zip(cells, recs))
    tot = sum(len(c) for c in cells)
    return {"mean_cells_per_response": round(tot / max(len(recs), 1), 2),
            "empty_cell_citation_rate": round(emp / max(tot, 1), 4),
            "responses_with_no_cells": sum(1 for c in cells if not c),
            # these tiles cannot be masked; listed so the attrition is traceable
            "tiles_with_no_cells": sorted(r["image"] for c, r in zip(cells, recs) if not c)}


def gave_up(path, images):
    """Tiles whose replicate-1 attempts hit the runner's unusable-response budget."""
    sys.path.insert(0, ROOT)
    import run_experiment as rx
    tries = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("replicate", 1) == 1 and r["image"] in images and rx.counts_as_try(r.get("error")):
            tries[r["image"]] = tries.get(r["image"], 0) + 1
    return sorted(t for t in images if tries.get(t, 0) >= rx.MAX_TRIES)


def main():
    host = sys.argv[1]
    tiles = json.load(open(os.path.join(ROOT, "runs", ".mask_ext_tiles.json")))["tiles"]
    new = load(os.path.join(ROOT, "runs", f"cte_p1_ext__{host}.jsonl"))
    old = load(os.path.join(ROOT, "runs", "cte_p1_full.jsonl"))
    missing = [t for t in tiles if t not in new]
    given_up = gave_up(os.path.join(ROOT, "runs", f"cte_p1_ext__{host}.jsonl"), set(missing))
    unattempted = [t for t in missing if t not in given_up]
    provs = {(r.get("meta") or {}).get("provider") for r in new.values()}
    router = any("openrouter" in (r.get("base_url") or "") for r in new.values())

    shared = [t for t in tiles if t in new and t in old]
    agree = sum(new[t]["parsed"]["label"] == old[t]["parsed"]["label"] for t in shared)
    h2s = sum(old[t]["parsed"]["label"] == "HP" and new[t]["parsed"]["label"] == "SSA" for t in shared)
    s2h = sum(old[t]["parsed"]["label"] == "SSA" and new[t]["parsed"]["label"] == "HP" for t in shared)
    rep = {
        "host": host, "n_planned": len(tiles), "n_valid": len(new), "missing": missing,
        "missing_given_up": given_up, "missing_unfinished": unattempted,
        "providers": sorted(map(str, provs)),
        "agreement_with_cerebras": round(agree / max(len(shared), 1), 4), "n_shared": len(shared),
        "ssa_rate_new": round(sum(new[t]["parsed"]["label"] == "SSA" for t in shared) / max(len(shared), 1), 4),
        "ssa_rate_cerebras": round(sum(old[t]["parsed"]["label"] == "SSA" for t in shared) / max(len(shared), 1), 4),
        "cerebras_HP_to_new_SSA": h2s, "cerebras_SSA_to_new_HP": s2h, "mcnemar_p": mcnemar(h2s, s2h),
        "accuracy_new": round(sum(new[t]["parsed"]["label"] == new[t]["label_true"] for t in shared) / max(len(shared), 1), 4),
        "accuracy_cerebras": round(sum(old[t]["parsed"]["label"] == old[t]["label_true"] for t in shared) / max(len(shared), 1), 4),
        "citations_new": cite_stats([new[t] for t in shared]),
        "citations_cerebras": cite_stats([old[t] for t in shared]),
    }
    json.dump(rep, open(os.path.join(ROOT, "runs", f"ext_baseline_check__{host}.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))
    if unattempted:
        print(f"GATE: {len(unattempted)} extension tiles lack a valid baseline and were not given up on; re-run")
        sys.exit(1)
    if len(given_up) > MAX_GAVE_UP_FRAC * len(tiles):
        print(f"GATE: {len(given_up)} tiles given up (> {MAX_GAVE_UP_FRAC:.0%}); the host is not answering "
              "the prompt reliably -- stop for review")
        sys.exit(1)
    if given_up:
        print(f"note: {len(given_up)} tiles given up and excluded from masking: {given_up}")
    if router and (len(provs) != 1 or None in provs):
        print(f"GATE: providers {provs} -- not one pinned upstream")
        sys.exit(1)
    if rep["agreement_with_cerebras"] < MIN_AGREEMENT:
        print(f"GATE: agreement {rep['agreement_with_cerebras']} < {MIN_AGREEMENT}; new host differs materially -- stop for review")
        sys.exit(3)
    print("GATE: pass")


if __name__ == "__main__":
    main()
