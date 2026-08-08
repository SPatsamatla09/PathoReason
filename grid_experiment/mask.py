"""Occlusion variants for grid cells, for testing whether cited evidence is causal.

Given a clean tile and a set of grid cells, produce three ablations of those
cells -- mean fill, Gaussian blur, black occlusion -- plus matched controls that
mask the same number of cells elsewhere. If the cells a model cited actually
drive its decision, masking them should move the label more than masking a
matched set it did not cite.

Masking is applied to the CLEAN tile and the grid is re-drawn afterwards. Doing
it the other way round would paint over the green lines and destroy the cell
labels inside the masked region, leaving the model unable to read the very
coordinate system it is being asked to use. Pass --no-regrid to skip the
re-overlay.

    python3 mask.py --image MHIST_ccn.png --cells B2,B3,C2,C3
    python3 mask.py --from-run runs/cte_p1.jsonl
"""

import argparse
import itertools
import json
import os
import random

import numpy as np
from PIL import Image, ImageFilter

from grid import N, ROWS, cell_box, draw_grid
from tissue import load_rgb, tissue_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(ROOT, "clean")
OUT = os.path.join(ROOT, "masked")
SEED = 20260806

# sigma=12: a Gaussian attenuates a sinusoid of wavelength L by exp(-2*pi^2*s^2/L^2),
# so killing the ~28px crypt period below 5% needs sigma > 10.9. What survives at
# 12 is coarse tissue-vs-background layout, not morphology -- which is the point:
# a blur should remove structure while leaving "there is tissue here" intact.
BLUR_SIGMA = 12.0

ALL_CELLS = [f"{r}{c}" for r in ROWS for c in range(1, 5)]


def parse_cells(text):
    out = []
    for tok in str(text).replace(";", ",").split(","):
        t = tok.strip().upper()
        if not t:
            continue
        if t not in ALL_CELLS:
            raise ValueError(f"{t!r} is not a valid cell (A1-D4)")
        out.append(t)
    return sorted(set(out))


def cell_pixel_mask(cells):
    """Boolean 224x224 mask covering the named cells."""
    m = np.zeros((224, 224), dtype=bool)
    for c in cells:
        x0, y0, x1, y1 = cell_box(ROWS.index(c[0]), int(c[1]) - 1)
        m[y0:y1, x0:x1] = True
    return m


def mask_mean(rgb, m):
    """Fill with the whole-tile mean colour: removes content, stays in-gamut."""
    out = rgb.copy()
    out[m] = rgb.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
    return out


def mask_blur(rgb, m, sigma=BLUR_SIGMA):
    """Composite a blur of the FULL tile.

    Blurring the isolated crop instead would clamp at the crop edge and leave a
    seam. The trade-off is that content bleeds in from neighbouring cells at the
    border; at sigma=12 the interior of a 56px cell is still dominated by its own
    (destroyed) content.
    """
    blurred = np.asarray(
        Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(sigma)), dtype=np.uint8
    )
    out = rgb.copy()
    out[m] = blurred[m]
    return out


def mask_black(rgb, m):
    """Hard occlusion. Strongest ablation, but pure black is out of distribution
    for H&E -- nothing on a real slide is (0,0,0) -- so it is also the most
    detectable as an artefact."""
    out = rgb.copy()
    out[m] = 0
    return out


MASKERS = {"mean": mask_mean, "blur": mask_blur, "black": mask_black}


def tissue_per_cell(rgb):
    t = tissue_mask(rgb)
    return {
        c: int(t[cell_box(ROWS.index(c[0]), int(c[1]) - 1)[1]: cell_box(ROWS.index(c[0]), int(c[1]) - 1)[3],
                 cell_box(ROWS.index(c[0]), int(c[1]) - 1)[0]: cell_box(ROWS.index(c[0]), int(c[1]) - 1)[2]].sum())
        for c in ALL_CELLS
    }


def control_random(cited, rng):
    """Same number of cells, drawn at random, avoiding the cited set when it can.

    With a model that cites 8-10 of 16 cells there are often too few uncited
    cells left, so overlap is recorded rather than silently permitted.
    """
    k = len(cited)
    pool = [c for c in ALL_CELLS if c not in set(cited)]
    if len(pool) >= k:
        return sorted(rng.sample(pool, k)), 0
    chosen = list(pool)
    overlap = k - len(chosen)
    chosen += rng.sample(sorted(set(cited)), overlap)
    return sorted(chosen), overlap


def control_tissue_matched(cited, rng, tissue):
    """Same number of cells AND closest achievable total tissue area.

    This is the control that matters. The uncited cells are systematically the
    empty border ones, so a plain random control masks mostly blank slide and
    will look like a weaker ablation for reasons that have nothing to do with
    whether the cited cells were causal. Exhaustive over C(16,k) <= 12870.

    Overlap is minimised BEFORE tissue gap. A set sharing cells with the cited
    one is not a control however well its area matches, so an exactly-matched
    set that reuses half the treatment loses to a disjoint near-match.
    """
    k = len(cited)
    target = sum(tissue[c] for c in cited)
    cited_set = set(cited)
    best = None
    for combo in itertools.combinations(ALL_CELLS, k):
        if set(combo) == cited_set:
            continue
        diff = abs(sum(tissue[c] for c in combo) - target)
        overlap = len(cited_set & set(combo))
        key = (overlap, diff)
        if best is None or key < best[0]:
            best = (key, [combo])
        elif key == best[0]:
            best[1].append(combo)
    (overlap, diff), ties = best
    return sorted(rng.choice(ties)), overlap, diff


def pick_subset(cited, k, tissue):
    """The k cited cells carrying the most tissue, deterministic on ties."""
    return sorted(sorted(cited, key=lambda c: (-tissue[c], c))[:k])


def build(image_name, cited, rng, regrid=True, out_root=OUT, subset=None):
    src = os.path.join(CLEAN, image_name)
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} -- mask the clean tile; run select_and_render.py first")
    rgb = load_rgb(src)
    tmask = tissue_mask(rgb)
    tissue = tissue_per_cell(rgb)

    cited_full = list(cited)
    if subset and subset < len(cited):
        cited = pick_subset(cited, subset, tissue)

    rand_cells, rand_overlap = control_random(cited, rng)
    tm_cells, tm_overlap, tm_diff = control_tissue_matched(cited, rng, tissue)
    cellsets = {
        "cited": {"cells": cited, "overlap_with_cited": len(cited)},
        "random": {"cells": rand_cells, "overlap_with_cited": rand_overlap},
        "tissue_matched": {
            "cells": tm_cells,
            "overlap_with_cited": tm_overlap,
            "tissue_px_gap_vs_cited": tm_diff,
        },
    }

    stem = image_name.replace(".png", "")
    tile_dir = os.path.join(out_root, stem)
    os.makedirs(tile_dir, exist_ok=True)

    total_px = rgb.shape[0] * rgb.shape[1]
    total_tissue = int(tmask.sum())
    written = []
    for set_name, info in cellsets.items():
        pm = cell_pixel_mask(info["cells"])
        info["n_cells"] = len(info["cells"])
        info["pct_of_tile_masked"] = round(float(100 * pm.sum() / total_px), 2)
        info["pct_of_tissue_masked"] = round(
            float(100 * (pm & tmask).sum() / max(total_tissue, 1)), 2
        )
        for method, fn in MASKERS.items():
            out_rgb = fn(rgb, pm)
            img = Image.fromarray(out_rgb)
            if regrid:
                img = draw_grid(img)
            path = os.path.join(tile_dir, f"{set_name}__{method}.png")
            img.save(path)
            written.append(os.path.relpath(path, ROOT))

    # A same-size disjoint control can only match the cited set's tissue area
    # when that area is near half the tile's total: the control is forced into
    # the complement, which holds whatever tissue the cited set left behind. Once
    # the cited set covers most of the tissue no matched control exists, and a
    # weaker effect from the control is then guaranteed by geometry rather than
    # by the cited cells being causal. Flag it rather than let it pass silently.
    gap_pp = (
        cellsets["cited"]["pct_of_tissue_masked"]
        - cellsets["tissue_matched"]["pct_of_tissue_masked"]
    )
    quality = {
        "tissue_gap_pp": round(float(gap_pp), 2),
        "control_overlap_cells": int(cellsets["tissue_matched"]["overlap_with_cited"]),
        "cited_pct_of_tissue": cellsets["cited"]["pct_of_tissue_masked"],
        "usable": bool(
            abs(gap_pp) <= 5.0 and cellsets["tissue_matched"]["overlap_with_cited"] == 0
        ),
    }
    if not quality["usable"]:
        quality["reason"] = (
            "control forced to overlap the cited set (too many cells cited)"
            if cellsets["tissue_matched"]["overlap_with_cited"]
            else f"no disjoint control matches the cited tissue area (off by {gap_pp:+.1f}pp)"
        )

    manifest = {
        "image": image_name,
        "source": os.path.relpath(src, ROOT),
        "regridded": regrid,
        "blur_sigma": BLUR_SIGMA,
        "seed": SEED,
        "cited_cells_full": sorted(cited_full),
        "subset_used": subset if subset and subset < len(cited_full) else None,
        "match_quality": quality,
        "tile_tissue_px": total_tissue,
        "tissue_px_per_cell": tissue,
        "cellsets": cellsets,
        "methods": sorted(MASKERS),
        "files": written,
    }
    with open(os.path.join(tile_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="tile filename, e.g. MHIST_ccn.png")
    ap.add_argument("--cells", help="comma-separated cells, e.g. B2,B3,C2,C3")
    ap.add_argument("--from-run", help="JSONL run; mask each tile's cited cells")
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument(
        "--subset",
        type=int,
        default=None,
        help="mask only the N highest-tissue cited cells. Needed when the model "
        "cites most of the tile: a same-size disjoint control cannot match a "
        "cited set covering far more than half the tissue. N=3 keeps controls "
        "feasible on every tile.",
    )
    ap.add_argument("--no-regrid", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rng = random.Random(SEED)
    jobs = []
    if args.from_run:
        from score import cited_cells

        for line in open(os.path.join(ROOT, args.from_run)):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("replicate") != args.replicate:
                continue
            cells = cited_cells(r)
            if cells:
                jobs.append((r["image"], cells))
    elif args.image and args.cells:
        jobs.append((args.image, parse_cells(args.cells)))
    else:
        ap.error("give --image with --cells, or --from-run")

    made = 0
    usable = 0
    for name, cells in jobs:
        m = build(
            name,
            cells,
            rng,
            regrid=not args.no_regrid,
            out_root=os.path.join(ROOT, args.out),
            subset=args.subset,
        )
        made += len(m["files"])
        q = m["match_quality"]
        usable += q["usable"]
        flag = "ok " if q["usable"] else "SKEW"
        print(
            f"{flag} {name}  masked={m['cellsets']['cited']['n_cells']} cells "
            f"({q['cited_pct_of_tissue']}% of tissue)  control gap={q['tissue_gap_pp']:+.1f}pp "
            f"overlap={q['control_overlap_cells']}"
        )
    print(f"\n{len(jobs)} tiles, {made} masked images -> {args.out}/")
    print(f"{usable}/{len(jobs)} tiles have a disjoint control matched within 5pp of tissue area.")
    if usable < len(jobs):
        print(
            "Tiles marked SKEW cannot support a fair ablation as-is: the control masks\n"
            "materially less tissue than the cited set, so it would show a smaller effect\n"
            "for geometric reasons alone."
        )
        if args.subset is None:
            print("Re-run with --subset 3 to make controls feasible on most tiles.")
        else:
            print(
                f"Already at --subset {args.subset}; these tiles have too little tissue\n"
                "for any matched control. Drop them from the ablation rather than\n"
                "reporting them alongside the usable ones."
            )


if __name__ == "__main__":
    main()
