"""Tissue-TYPE-matched control arm for the grid-masking test (Kiran's feedback).

The area-matched control (mask.py) matches only the cited cells' total tissue area.
Histology is dense: an area-matched set can still hold diagnostic epithelium, or can
swap epithelium for stroma. This adds a control matched on tissue type as well.

For each tile of an existing masked directory, the cited subset stays as it is: the
same 3 highest-tissue cited cells, so its images are copied unchanged. The new
control is the 3-cell set that:

  1. is disjoint from the cited subset (required)
  2. matches the cited subset's tissue area within TOL_PP percentage points of the
     tile's tissue, the same currency and tolerance as the area-matched arm
  3. matches its EPITHELIAL area within TOL_PP percentage points of the tile's
     epithelium (stains.stain_masks: hematoxylin-dominant tissue, i.e. crypt/gland
     lining and nuclear zones; the rest of the tissue is eosin-dominant stroma)
  4. among the sets that pass 2-3, has the FEWEST cells the model cited anywhere in its
     answer (cited_cells_full), then the smallest summed gap

A tile with no set passing 2-3 is recorded as unusable, not forced. Masked images
are generated exactly as mask.py generates them: same maskers, blur sigma and
regridding.

    python3 mask_typematched.py --src masked/ext__<host>_k3 --out masked/typematched__<host>_k3
"""

import argparse
import itertools
import json
import os
import shutil

from PIL import Image

from grid import draw_grid
from mask import ALL_CELLS, BLUR_SIGMA, MASKERS, ROOT, cell_pixel_mask
from stains import stain_masks
from tissue import load_rgb

TOL_PP = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="existing masked dir (cited subset + manifests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src_dir, out_dir = os.path.join(ROOT, args.src), os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    summary = {"usable": 0, "unusable": [], "controls_cells_in_full_citation": {}}

    for tile in sorted(os.listdir(src_dir)):
        man_path = os.path.join(src_dir, tile, "manifest.json")
        if not os.path.isfile(man_path):
            continue
        man = json.load(open(man_path))
        rgb = load_rgb(os.path.join(ROOT, man["source"]))
        sm = stain_masks(rgb)
        cell_masks = {c: cell_pixel_mask([c]) for c in ALL_CELLS}
        tis = {c: int((cell_masks[c] & sm["tissue"]).sum()) for c in ALL_CELLS}
        epi = {c: int((cell_masks[c] & sm["epithelium"]).sum()) for c in ALL_CELLS}
        tot_tis, tot_epi = max(sum(tis.values()), 1), max(sum(epi.values()), 1)
        pct_t = lambda cells: 100 * sum(tis[c] for c in cells) / tot_tis
        pct_e = lambda cells: 100 * sum(epi[c] for c in cells) / tot_epi

        cited = man["cellsets"]["cited"]["cells"]
        full = set(man["cited_cells_full"])
        area_ctrl = man["cellsets"]["tissue_matched"]["cells"]
        best = None
        for combo in itertools.combinations([c for c in ALL_CELLS if c not in cited], len(cited)):
            gt, ge = abs(pct_t(combo) - pct_t(cited)), abs(pct_e(combo) - pct_e(cited))
            if gt > TOL_PP or ge > TOL_PP:
                continue
            key = (len(full & set(combo)), round(gt + ge, 6), combo)
            if best is None or key < best:
                best = key

        tdir = os.path.join(out_dir, tile)
        os.makedirs(tdir, exist_ok=True)
        info = {
            "image": man["image"], "source": man["source"], "regridded": True, "blur_sigma": BLUR_SIGMA,
            "cited_cells_full": man["cited_cells_full"], "subset_used": man["subset_used"],
            "tolerance_pp": TOL_PP,
            "cited": {"cells": cited, "pct_tissue": round(pct_t(cited), 2), "pct_epithelium": round(pct_e(cited), 2)},
            "area_matched_control_for_reference": {
                "cells": area_ctrl, "pct_tissue": round(pct_t(area_ctrl), 2),
                "pct_epithelium": round(pct_e(area_ctrl), 2),
                "epithelium_gap_pp": round(pct_e(area_ctrl) - pct_e(cited), 2),
                "cells_in_full_citation": len(full & set(area_ctrl))},
        }
        if best is None:
            info["usable"] = False
            info["reason"] = f"no disjoint 3-cell set within {TOL_PP}pp on both tissue and epithelium"
            summary["unusable"].append(tile)
            shutil.rmtree(tdir)  # keep unusable tiles out of the run directory
            udir = os.path.join(out_dir.rstrip("/") + "_unusable", tile)
            os.makedirs(udir, exist_ok=True)
            json.dump(info, open(os.path.join(udir, "manifest.json"), "w"), indent=2)
            continue
        ctrl = sorted(best[2])
        info["usable"] = True
        info["type_matched"] = {
            "cells": ctrl, "pct_tissue": round(pct_t(ctrl), 2), "pct_epithelium": round(pct_e(ctrl), 2),
            "tissue_gap_pp": round(pct_t(ctrl) - pct_t(cited), 2),
            "epithelium_gap_pp": round(pct_e(ctrl) - pct_e(cited), 2),
            "cells_in_full_citation": best[0]}
        # run_masked.py reads manifest["cellsets"][arm]["cells"] and match_quality
        info["cellsets"] = {
            "cited": {"cells": cited, "overlap_with_cited": len(cited),
                      "pct_of_tissue_masked": round(pct_t(cited), 2)},
            "type_matched": {"cells": ctrl, "overlap_with_cited": 0,
                             "pct_of_tissue_masked": round(pct_t(ctrl), 2)},
        }
        info["match_quality"] = {"usable": True, "tissue_gap_pp": info["type_matched"]["tissue_gap_pp"],
                                 "epithelium_gap_pp": info["type_matched"]["epithelium_gap_pp"],
                                 "control_overlap_cells": 0}
        files = []
        for occ in MASKERS:  # cited images: byte-identical copies of the source sweep's
            shutil.copy2(os.path.join(src_dir, tile, f"cited__{occ}.png"), os.path.join(tdir, f"cited__{occ}.png"))
            files.append(f"cited__{occ}.png")
        pm = cell_pixel_mask(ctrl)
        for occ, fn in MASKERS.items():
            draw_grid(Image.fromarray(fn(rgb, pm))).save(os.path.join(tdir, f"type_matched__{occ}.png"))
            files.append(f"type_matched__{occ}.png")
        info["files"] = files
        json.dump(info, open(os.path.join(tdir, "manifest.json"), "w"), indent=2)
        summary["usable"] += 1
        k = str(best[0])
        summary["controls_cells_in_full_citation"][k] = summary["controls_cells_in_full_citation"].get(k, 0) + 1

    json.dump(summary, open(out_dir.rstrip("/") + "_summary.json", "w"), indent=2)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in summary.items()}))


if __name__ == "__main__":
    main()
