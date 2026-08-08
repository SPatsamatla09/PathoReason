"""Compute per-cell tissue coverage for every MHIST tile.

Writes one JSON per tile into coverage/ plus a consolidated coverage_index.csv
used by the selection step. Run from grid_experiment/:

    python3 build_coverage.py
"""

import csv
import json
import os
import sys

import numpy as np

from grid import CELL, N, TILE, cell_box, cell_name
from tissue import GRAY_THRESH, SAT_THRESH, VAL_THRESH, gray_mask, load_rgb, tissue_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(ROOT)
IMAGES = os.path.join(DATA, "images")
ANNOTATIONS = os.path.join(DATA, "annotations.csv")
COVERAGE_DIR = os.path.join(ROOT, "coverage")
INDEX_CSV = os.path.join(ROOT, "coverage_index.csv")

EMPTY_PCT = 5.0  # a cell below this is treated as bare slide, i.e. a negative control

TISSUE_RULE = {
    "primary": "HSV: saturation > 0.08 OR value < 0.80",
    "sat_thresh": SAT_THRESH,
    "val_thresh": VAL_THRESH,
    "secondary": f"grayscale < {GRAY_THRESH}",
    "gray_thresh": GRAY_THRESH,
    "empty_cell_pct": EMPTY_PCT,
}

GRID_SPEC = {
    "rows": N,
    "cols": N,
    "cell_px": CELL,
    "tile_px": TILE,
    "naming": "row letter A-D top to bottom, column number 1-4 left to right; A1 = top-left",
}


def cell_stats(rgb, tmask, gmask):
    """Per-cell coverage record for one tile."""
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=2)

    cells = {}
    for r in range(N):
        for c in range(N):
            x0, y0, x1, y1 = cell_box(r, c)
            sl = (slice(y0, y1), slice(x0, x1))
            pct = float(tmask[sl].mean() * 100)
            cells[cell_name(r, c)] = {
                "row": r,
                "col": c,
                "bbox_xyxy": [x0, y0, x1, y1],
                "tissue_pct": round(pct, 2),
                "tissue_pct_gray_rule": round(float(gmask[sl].mean() * 100), 2),
                "mean_gray": round(float(gray[sl].mean()), 1),
                "mean_saturation": round(float(sat[sl].mean()), 4),
                "is_empty": pct < EMPTY_PCT,
                "touches_tile_border": r in (0, N - 1) or c in (0, N - 1),
            }
    return cells


def main():
    os.makedirs(COVERAGE_DIR, exist_ok=True)
    ann = {r["Image Name"]: r for r in csv.DictReader(open(ANNOTATIONS))}
    names = sorted(ann)

    index_rows = []
    for i, name in enumerate(names, 1):
        rgb = load_rgb(os.path.join(IMAGES, name))
        tmask, gmask = tissue_mask(rgb), gray_mask(rgb)
        cells = cell_stats(rgb, tmask, gmask)
        empty = sorted(k for k, v in cells.items() if v["is_empty"])
        meta = ann[name]
        votes = int(meta["Number of Annotators who Selected SSA (Out of 7)"])

        rec = {
            "image": name,
            "label": meta["Majority Vote Label"],
            "ssa_votes_out_of_7": votes,
            "partition": meta["Partition"],
            "grid": GRID_SPEC,
            "tissue_rule": TISSUE_RULE,
            "tile_tissue_pct": round(float(tmask.mean() * 100), 2),
            "n_empty_cells": len(empty),
            "empty_cells": empty,
            "cells": cells,
        }
        with open(os.path.join(COVERAGE_DIR, name.replace(".png", ".json")), "w") as fh:
            json.dump(rec, fh, indent=2)

        index_rows.append(
            {
                "image": name,
                "label": rec["label"],
                "ssa_votes": votes,
                "partition": rec["partition"],
                "tile_tissue_pct": rec["tile_tissue_pct"],
                "n_empty_cells": rec["n_empty_cells"],
                "min_cell_pct": round(min(c["tissue_pct"] for c in cells.values()), 2),
                "max_cell_pct": round(max(c["tissue_pct"] for c in cells.values()), 2),
            }
        )
        if i % 500 == 0:
            print(f"  {i}/{len(names)}", file=sys.stderr)

    with open(INDEX_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0]))
        w.writeheader()
        w.writerows(index_rows)

    print(f"wrote {len(index_rows)} JSONs to coverage/ and {INDEX_CSV}")


if __name__ == "__main__":
    main()
