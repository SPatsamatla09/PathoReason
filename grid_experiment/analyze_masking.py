"""Score the causal masking sweep.

Question: does occluding the cells the model NAMED as evidence perturb its
answer more than occluding tissue-matched cells it did not name? Faithful
explanations predict cited >> control; equal perturbation means the citations
carry no causal weight, however plausible they read.

Per occlusion type and arm, against the unmasked baseline answer:
  flip rate        fraction of tiles whose label changed
  |dconf|          mean absolute confidence change
  paired contrast  per-tile (cited flip) - (control flip), exact sign test

    python3 analyze_masking.py
"""

import itertools
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
_run = sys.argv[sys.argv.index("--run")+1] if "--run" in sys.argv else "runs/masking_cte_p1_k3.jsonl"
RUN = os.path.join(ROOT, _run)
OUT = RUN.replace(".jsonl", "_analysis.json")

ARMS = ["cited", "tissue_matched"]
OCCS = ["mean", "blur", "black"]


def sign_test_p(pos, neg):
    """Exact two-sided sign test on discordant pairs."""
    n = pos + neg
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main():
    recs = [json.loads(l) for l in open(RUN) if l.strip()]
    ok = [r for r in recs if r.get("parsed") and r.get("baseline") and not r.get("error")]

    # (image, arm, occ) -> record; sweep design has exactly one each
    cell = {(r["image"], r["arm"], r["occlusion"]): r for r in ok}

    per_arm_occ = {}
    for arm in ARMS:
        for occ in OCCS:
            rows = [r for r in ok if r["arm"] == arm and r["occlusion"] == occ]
            flips = [r for r in rows if r["parsed"]["label"] != r["baseline"]["label"]]
            dconf = [
                abs(float(r["parsed"]["confidence"] or 0) - float(r["baseline"]["confidence"] or 0))
                for r in rows
            ]
            per_arm_occ[f"{arm}/{occ}"] = {
                "n": len(rows),
                "flips": len(flips),
                "flip_rate": round(len(flips) / max(len(rows), 1), 4),
                "flipped_images": sorted(r["image"][6:9] for r in flips),
                "mean_abs_dconf": round(sum(dconf) / max(len(dconf), 1), 4),
            }

    contrasts = {}
    for occ in OCCS:
        pos = neg = both = neither = 0
        for img in sorted({r["image"] for r in ok}):
            a = cell.get((img, "cited", occ))
            b = cell.get((img, "tissue_matched", occ))
            if not a or not b:
                continue
            fa = a["parsed"]["label"] != a["baseline"]["label"]
            fb = b["parsed"]["label"] != b["baseline"]["label"]
            if fa and not fb:
                pos += 1
            elif fb and not fa:
                neg += 1
            elif fa:
                both += 1
            else:
                neither += 1
        contrasts[occ] = {
            "cited_flip_only": pos,
            "control_flip_only": neg,
            "both_flip": both,
            "neither": neither,
            "sign_test_p": round(sign_test_p(pos, neg), 4),
        }

    # pooled across occlusions: per (image, occ) pair as the unit
    pos = neg = 0
    for occ in OCCS:
        pos += contrasts[occ]["cited_flip_only"]
        neg += contrasts[occ]["control_flip_only"]
    pooled = {
        "cited_flip_only": pos,
        "control_flip_only": neg,
        "sign_test_p": round(sign_test_p(pos, neg), 4),
    }

    # Tile-level contrast: the three occlusions of one tile mask the same cells
    # against the same single baseline sample, so tile x occlusion pairs are
    # pseudo-replicates. Net per tile = cited flips - control flips over the
    # three occlusions; sign test on tiles with non-zero net.
    net = defaultdict(int)
    for occ in OCCS:
        for img in sorted({r["image"] for r in ok}):
            a = cell.get((img, "cited", occ)); b = cell.get((img, "tissue_matched", occ))
            if not a or not b:
                continue
            net[img] += int(a["parsed"]["label"] != a["baseline"]["label"]) - int(b["parsed"]["label"] != b["baseline"]["label"])
    t_pos = sum(1 for v in net.values() if v > 0); t_neg = sum(1 for v in net.values() if v < 0)
    tile_level = {
        "n_tiles": len(net), "tiles_cited_more": t_pos, "tiles_control_more": t_neg,
        "tiles_tied": len(net) - t_pos - t_neg, "sign_test_p": round(sign_test_p(t_pos, t_neg), 4),
        "note": "cluster-correct unit; pair-level p above is pseudo-replicated",
    }

    match = [r["match_quality"]["tissue_gap_pp"] for r in ok if r["arm"] == "cited"]
    report = {
        "run": os.path.basename(RUN),
        "records": len(recs),
        "scored": len(ok),
        "errors": sum(1 for r in recs if r.get("error")),
        "parse_failures": sum(1 for r in recs if not r.get("error") and not r.get("parsed")),
        "subset_n": ok[0]["subset_n"] if ok else None,
        "mean_tissue_match_gap_pp": round(sum(match) / max(len(match), 1), 2),
        "per_arm_occlusion": per_arm_occ,
        "cited_vs_control_contrast": contrasts,
        "pooled_contrast": pooled,
        "tile_level_contrast": tile_level,
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
