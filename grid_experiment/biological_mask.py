"""Biologically-targeted masking: occlude structures, not grid squares.

For each tile, three structure masks:

  epithelium  hematoxylin-dominant tissue partition (stains.py)
  stroma      eosin-dominant tissue partition (stains.py)
  nuclei      CellPose 'nuclei' model on the hematoxylin channel, x4 upsample,
              diameter=10, labels downscaled to 224 and dilated 1px

and per structure a matched-area control: the structure mask cut into 8x8
blocks and shuffled uniformly over the tile's tissue-bearing blocks. Area is
preserved to the block quantum and fragment scale is similar; structural
alignment is destroyed. When a structure covers most of the tissue the control
necessarily re-overlaps it -- the overlap is recorded, not hidden.

Whether the model NAMED each structure on that tile comes from the cte_p1 run
via the vocabulary mapping below; the runner stores it per record so analysis
can split named vs unnamed structures.

Occlusion is mean-fill only (mask.py's mask_mean), matching the grid sweep's
'mean' arm: the grid results barely differed across occlusion types, and one
type keeps the sweep at 120 calls.

    python3 biological_mask.py            # builds masked_bio/<tile>/*.png
    python3 biological_mask.py --qa-only  # just re-render the QA sheet
"""

import argparse
import json
import os
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from grid import draw_grid
from mask import mask_mean
from stains import hed_channels, stain_masks

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "masked_bio")
NUCLEI_CACHE = os.path.join(ROOT, "biological", "nuclei")
BASELINE_RUN = os.path.join(ROOT, "runs", "cte_p1.jsonl")

SEED = 20260806
BLOCK = 8
UPSAMPLE = 4
DIAMETER = 10
SEGMENTATION_METHOD = (
    "cellpose-3.1.1.3 model_type=nuclei on H channel (rgb2hed), "
    f"x{UPSAMPLE} lanczos upsample, diameter={DIAMETER}, "
    "labels nearest-downscaled to 224, binary mask dilated 1px"
)

FEATURE_TO_SOURCE = {
    "nuclear atypia": "nuclei",
    "mitotic figures": "nuclei",
    "crypt/gland architecture": "epithelium",
    "serration": "epithelium",
    "surface epithelium": "epithelium",
    "stroma": "stroma",
    "inflammatory infiltrate": "stroma",
    # necrosis intentionally unmapped: not a stain compartment
}

SOURCES = ["epithelium", "stroma", "nuclei"]


def cited_features(run_path=BASELINE_RUN):
    """image -> list of features gemma cited (replicate 1)."""
    out = {}
    for line in open(run_path):
        r = json.loads(line)
        if r["replicate"] == 1 and r.get("parsed"):
            out[r["image"]] = [e["feature"] for e in r["parsed"]["evidence"]]
    return out


def nuclei_mask(rgb, model):
    h, _ = hed_channels(rgb)
    from tissue import tissue_mask

    tis = tissue_mask(rgb)
    v = h[tis]
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    hn = np.clip((h - lo) / (hi - lo + 1e-9), 0, 1)
    big = (
        np.asarray(
            Image.fromarray((hn * 255).astype(np.uint8)).resize(
                (224 * UPSAMPLE,) * 2, Image.LANCZOS
            )
        )
        / 255.0
    )
    labels = model.eval(big.astype(np.float32), diameter=DIAMETER, channels=[0, 0])[0]
    small = np.asarray(
        Image.fromarray(labels.astype(np.int32), mode="I").resize((224, 224), Image.NEAREST)
    )
    n = int(labels.max())
    m = ndi.binary_dilation(small > 0, np.ones((3, 3)))
    return m & tis, n


def shuffled_control(mask, tissue, rng):
    """Permute BLOCK x BLOCK chunks of `mask` over the tissue-bearing blocks."""
    nb = 224 // BLOCK
    src_blocks = []
    dst_slots = []
    for by in range(nb):
        for bx in range(nb):
            sl = np.s_[by * BLOCK : (by + 1) * BLOCK, bx * BLOCK : (bx + 1) * BLOCK]
            if mask[sl].any():
                src_blocks.append(mask[sl].copy())
            if tissue[sl].mean() > 0.3:
                dst_slots.append(sl)
    rng.shuffle(dst_slots)
    out = np.zeros_like(mask)
    for blk, sl in zip(src_blocks, dst_slots):
        out[sl] |= blk
    return out


def build_tile(name, feats, model, rng):
    rgb = np.asarray(Image.open(os.path.join(ROOT, "clean", name)).convert("RGB"))
    sm = stain_masks(rgb)
    tis = sm["tissue"]
    os.makedirs(NUCLEI_CACHE, exist_ok=True)
    cache = os.path.join(NUCLEI_CACHE, name)
    if os.path.exists(cache):
        nuc = np.asarray(Image.open(cache)) > 0
        n_nuc = json.load(open(cache + ".json"))["n_nuclei"]
    else:
        nuc, n_nuc = nuclei_mask(rgb, model)
        Image.fromarray((nuc * 255).astype(np.uint8)).save(cache)
        json.dump({"n_nuclei": n_nuc}, open(cache + ".json", "w"))

    masks = {"epithelium": sm["epithelium"], "stroma": sm["stroma"], "nuclei": nuc}
    named_sources = sorted({FEATURE_TO_SOURCE[f] for f in feats if f in FEATURE_TO_SOURCE})

    tdir = os.path.join(OUT, name.replace(".png", ""))
    os.makedirs(tdir, exist_ok=True)
    man = {
        "image": name,
        "seed": SEED,
        "occlusion": "mean",
        "block_px": BLOCK,
        "segmentation_method": SEGMENTATION_METHOD,
        "n_nuclei": n_nuc,
        "cited_features": feats,
        "named_sources": named_sources,
        "tile_tissue_px": int(tis.sum()),
        "sources": {},
    }
    for src in SOURCES:
        m = masks[src]
        ctrl = shuffled_control(m, tis, rng)
        for tag, mm in (("struct", m), ("control", ctrl)):
            out = mask_mean(rgb, mm)
            img = draw_grid(Image.fromarray(out))
            img.save(os.path.join(tdir, f"{src}__{tag}.png"))
        man["sources"][src] = {
            "named": src in named_sources,
            "mask_px": int(m.sum()),
            "pct_of_tile": round(float(m.mean()) * 100, 2),
            "pct_of_tissue": round(float(m.sum()) / max(int(tis.sum()), 1) * 100, 2),
            "control_px": int(ctrl.sum()),
            "control_pct_of_tissue": round(
                float(ctrl.sum()) / max(int(tis.sum()), 1) * 100, 2
            ),
            "control_overlap_with_struct_pct": round(
                float((ctrl & m).sum()) / max(int(ctrl.sum()), 1) * 100, 2
            ),
        }
    with open(os.path.join(tdir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-only", action="store_true")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    feats_by_img = cited_features()
    manifest = json.load(open(os.path.join(ROOT, "selection_manifest.json")))
    names = [t["image"] for t in manifest["tiles"]]

    if not args.qa_only:
        from cellpose import models

        model = models.CellposeModel(model_type="nuclei", gpu=False)
        rng = np.random.default_rng(SEED)
        rng_shuffle = __import__("random").Random(SEED)
        for i, name in enumerate(names, 1):
            man = build_tile(name, feats_by_img[name], model, rng_shuffle)
            s = man["sources"]
            print(
                f"[{i:2d}/20] {name[6:9]} named={','.join(man['named_sources']) or '-'} "
                f"epi {s['epithelium']['pct_of_tissue']:.0f}% "
                f"str {s['stroma']['pct_of_tissue']:.0f}% "
                f"nuc {s['nuclei']['pct_of_tissue']:.0f}% ({man['n_nuclei']} nuclei)",
                flush=True,
            )

    # QA sheet: first 4 tiles x all 6 variants
    from PIL import ImageDraw

    W = 224
    rows = names[:4]
    sheet = Image.new("RGB", (6 * (W + 4) + 4, len(rows) * (W + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for i, name in enumerate(rows):
        tdir = os.path.join(OUT, name.replace(".png", ""))
        y = i * (W + 16)
        for j, v in enumerate(
            f"{s}__{t}" for s in SOURCES for t in ("struct", "control")
        ):
            sheet.paste(Image.open(os.path.join(tdir, v + ".png")), (4 + j * (W + 4), y))
        d.text((4, y + W + 2), f"{name[6:9]}  [epi|epi-ctl|str|str-ctl|nuc|nuc-ctl]", fill="black")
    qa = os.path.join(OUT, "qa_sheet.png")
    sheet.save(qa)
    print(f"QA sheet -> {qa}")


if __name__ == "__main__":
    main()
