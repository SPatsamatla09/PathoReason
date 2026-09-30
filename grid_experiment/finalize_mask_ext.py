"""Move extension tiles whose matched control is infeasible out of the masked dir.

mask.py writes a manifest for every tile, flagging `match_quality.usable=False`
when no disjoint 3-cell set matches the cited cells' tissue. run_masked.py runs
every tile directory it finds, so unusable tiles are moved to a sibling
`<dir>_unusable/` directory and listed, rather than silently masked against a
mismatched control.

    python3 finalize_mask_ext.py masked/ext__<host>_k3
"""

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    d = os.path.join(ROOT, sys.argv[1])
    aside = d.rstrip("/") + "_unusable"
    os.makedirs(aside, exist_ok=True)
    moved, kept = [], 0
    for t in sorted(os.listdir(d)):
        man = os.path.join(d, t, "manifest.json")
        if not os.path.isfile(man):
            continue
        if json.load(open(man))["match_quality"]["usable"]:
            kept += 1
        else:
            shutil.move(os.path.join(d, t), os.path.join(aside, t))
            moved.append(t)
    json.dump({"kept": kept, "moved_unusable": moved}, open(aside + ".json", "w"), indent=1)
    print(f"kept {kept} usable tiles; moved {len(moved)} unusable -> {os.path.relpath(aside, ROOT)}")


if __name__ == "__main__":
    main()
