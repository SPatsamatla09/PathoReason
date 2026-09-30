"""Choose the tiles for the grid-masking extension (target: >= 400-500 unique tiles pooled).

Draws from the MHIST test split, excluding every tile already masked
(20-tile pilot, proportional sweep B, design-matched sweep C). Allocation is
proportional to the test split's label x agreement-band composition, as in
sweep B. It draws 340. A zero-call self-test using these tiles' existing
classify-then-explain citations kept 313 (8% cannot support a matched control),
which brings the pooled unique-tile count to about 498. A new host's citations
will differ slightly, so the final count will too.

Citations and baselines for these tiles are NOT taken from the old Cerebras
run: the chain re-runs classify-then-explain on whichever host executes the
extension. Each tile's citations, baseline and masked answers therefore all
come from one pinned provider (checked per tile by analyze_masking_pooled.py).

    python3 select_mask_extension.py   ->  runs/.mask_ext_tiles.json
"""

import csv
import json
import os
import random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 20260929
N_DRAW = 340


def band(v):
    return "unanimous" if v in (0, 7) else "borderline" if v in (3, 4) else "strong"


def used_tiles():
    used = set()
    for f in ("runs/masking_cte_p1_k3.jsonl", "runs/masking_cte_p1_full_k3.jsonl",
              "runs/masking_abl100_k3.jsonl"):
        for line in open(os.path.join(ROOT, f)):
            used.add(json.loads(line)["image"])
    return used


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "coverage_index.csv")))
            if r["partition"] == "test"]
    used = used_tiles()
    pool = [r for r in rows if r["image"] not in used]
    strata = defaultdict(list)
    for r in pool:
        strata[(r["label"], band(int(r["ssa_votes"])))].append(r["image"])
    test_mix = Counter((r["label"], band(int(r["ssa_votes"]))) for r in rows)

    rng = random.Random(SEED)
    alloc = {k: round(N_DRAW * v / len(rows)) for k, v in test_mix.items()}
    diff = N_DRAW - sum(alloc.values())
    for k in sorted(alloc, key=lambda k: -test_mix[k])[: abs(diff)]:
        alloc[k] += 1 if diff > 0 else -1
    chosen = []
    for k in sorted(strata):
        members = sorted(strata[k])
        take = min(alloc.get(k, 0), len(members))
        chosen += rng.sample(members, take)
    chosen.sort()

    out = {
        "seed": SEED,
        "n": len(chosen),
        "excluded_already_masked": len(used),
        "pool_size": len(pool),
        "allocation": {f"{k[0]}/{k[1]}": alloc[k] for k in sorted(alloc)},
        "tiles": chosen,
    }
    json.dump(out, open(os.path.join(ROOT, "runs", ".mask_ext_tiles.json"), "w"), indent=1)
    mix = Counter((r["label"], band(int(r["ssa_votes"]))) for r in rows if r["image"] in set(chosen))
    print(f"selected {len(chosen)} of {len(pool)} unused test tiles (excluded {len(used)} already masked)")
    print("mix:", {f"{a}/{b}": v for (a, b), v in sorted(mix.items())})


if __name__ == "__main__":
    main()
