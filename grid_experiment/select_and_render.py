"""Select 20 evaluation tiles and render the gridded / clean pair for each.

Stratified on annotator agreement (unanimous / strong / borderline) crossed with
the majority label, balanced 10 HP vs 10 SSA, and deliberately oversampled for
tiles carrying >=2 empty grid cells so the run has enough negative-control
trials -- cells that contain no tissue at all, where any named histological
feature is a demonstrable localisation failure.

Run from grid_experiment/ after build_coverage.py:

    python3 select_and_render.py
"""

import csv
import json
import os
import random
import shutil

import numpy as np
from PIL import Image

from grid import draw_grid
from tissue import load_rgb, tissue_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(ROOT)
IMAGES = os.path.join(DATA, "images")
INDEX_CSV = os.path.join(ROOT, "coverage_index.csv")
GRIDDED_DIR = os.path.join(ROOT, "gridded")
CLEAN_DIR = os.path.join(ROOT, "clean")

SEED = 20260806
PARTITION = "test"  # evaluate on the held-out split

# (band, label, n_total, n_with_2plus_empty). Rich picks are further split so
# roughly half carry 4+ empty cells rather than the bare minimum of 2.
ALLOCATION = [
    ("unanimous", "HP", 3, 2),
    ("unanimous", "SSA", 3, 2),
    ("strong", "HP", 4, 2),
    ("strong", "SSA", 3, 2),
    ("borderline", "HP", 3, 2),
    ("borderline", "SSA", 4, 2),
]


def band(votes):
    if votes in (0, 7):
        return "unanimous"
    if votes in (3, 4):
        return "borderline"
    return "strong"


def select(rows, rng):
    pool = [r for r in rows if r["partition"] == PARTITION]
    for r in pool:
        r["ssa_votes"] = int(r["ssa_votes"])
        r["n_empty_cells"] = int(r["n_empty_cells"])
        r["band"] = band(r["ssa_votes"])

    picked = []
    for b, label, n_total, n_rich in ALLOCATION:
        stratum = [r for r in pool if r["band"] == b and r["label"] == label]
        mid = sorted([r for r in stratum if 2 <= r["n_empty_cells"] <= 3], key=lambda r: r["image"])
        high = sorted([r for r in stratum if r["n_empty_cells"] >= 4], key=lambda r: r["image"])
        poor = sorted([r for r in stratum if r["n_empty_cells"] < 2], key=lambda r: r["image"])

        n_high = n_rich // 2 + (n_rich % 2)  # favour the stronger controls
        chosen = rng.sample(high, n_high) + rng.sample(mid, n_rich - n_high)
        chosen += rng.sample(poor, n_total - n_rich)
        for r in chosen:
            r["stratum"] = f"{b}/{label}"
            r["control_class"] = "negative-control-rich" if r["n_empty_cells"] >= 2 else "dense"
        picked += chosen

    picked.sort(key=lambda r: (r["band"], r["label"], -r["n_empty_cells"], r["image"]))
    return picked


def render_all():
    """Render the gridded/clean pair for every tile in the dataset, not just the 20."""
    os.makedirs(GRIDDED_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(IMAGES) if f.endswith(".png"))
    for i, name in enumerate(names, 1):
        src = os.path.join(IMAGES, name)
        draw_grid(Image.open(src).convert("RGB")).save(
            os.path.join(GRIDDED_DIR, name.replace(".png", "_grid.png"))
        )
        shutil.copyfile(src, os.path.join(CLEAN_DIR, name))
        if i % 500 == 0:
            print(f"  {i}/{len(names)}")
    print(f"rendered {len(names)} tiles")


def main():
    rng = random.Random(SEED)
    rows = list(csv.DictReader(open(INDEX_CSV)))
    picked = select(rows, rng)

    os.makedirs(GRIDDED_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    manifest = []
    for r in picked:
        name = r["image"]
        src = os.path.join(IMAGES, name)
        clean = Image.open(src).convert("RGB")
        gridded = draw_grid(clean)

        gridded.save(os.path.join(GRIDDED_DIR, name.replace(".png", "_grid.png")))
        shutil.copyfile(src, os.path.join(CLEAN_DIR, name))

        # how much of the overlay's ink actually lands on tissue rather than glass
        rgb = load_rgb(src)
        tmask = tissue_mask(rgb)
        changed = np.abs(
            np.asarray(clean, np.int16) - np.asarray(gridded, np.int16)
        ).sum(axis=2) > 0
        cov = json.load(open(os.path.join(ROOT, "coverage", name.replace(".png", ".json"))))

        manifest.append(
            {
                "image": name,
                "gridded": f"gridded/{name.replace('.png', '_grid.png')}",
                "clean": f"clean/{name}",
                "coverage_json": f"coverage/{name.replace('.png', '.json')}",
                "label": r["label"],
                "ssa_votes_out_of_7": r["ssa_votes"],
                "agreement_band": r["band"],
                "stratum": r["stratum"],
                "control_class": r["control_class"],
                "tile_tissue_pct": float(r["tile_tissue_pct"]),
                "n_empty_cells": r["n_empty_cells"],
                "empty_cells": cov["empty_cells"],
                "overlay_pct_of_tile": round(float(changed.mean()) * 100, 2),
                "overlay_pct_of_tissue_occluded": round(
                    float((changed & tmask).sum()) / max(int(tmask.sum()), 1) * 100, 2
                ),
            }
        )

    with open(os.path.join(ROOT, "selection_manifest.json"), "w") as fh:
        json.dump(
            {
                "seed": SEED,
                "partition": PARTITION,
                "n_tiles": len(manifest),
                "allocation": [
                    {"band": b, "label": l, "n": n, "n_rich": nr} for b, l, n, nr in ALLOCATION
                ],
                "grid": cov["grid"],
                "tissue_rule": cov["tissue_rule"],
                "tiles": manifest,
            },
            fh,
            indent=2,
        )

    with open(os.path.join(ROOT, "selection.csv"), "w", newline="") as fh:
        cols = [
            "image", "label", "ssa_votes_out_of_7", "agreement_band", "control_class",
            "tile_tissue_pct", "n_empty_cells", "overlay_pct_of_tile",
            "overlay_pct_of_tissue_occluded",
        ]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest)

    print(f"selected {len(manifest)} tiles -> gridded/, clean/, selection_manifest.json")
    return manifest


if __name__ == "__main__":
    import sys

    if "--all" in sys.argv:
        render_all()
    else:
        main()
