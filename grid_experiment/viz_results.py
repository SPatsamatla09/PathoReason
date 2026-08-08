"""Visual audit of a scored run.

Two figures:
  cited_vs_truth.png -- each tile with its cited cells outlined; cells the model
                        cited that contain no tissue are outlined in red, so
                        control failures are visible rather than tabulated.
  cell_heatmap.png   -- how often each of the 16 cells was cited across tiles,
                        beside how often each was actually available (non-empty).

    python3 viz_results.py --run runs/cte_p1.jsonl
"""

import argparse
import json
import os
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

from grid import CELL, ROWS, cell_box
from score import ALL_CELLS, cited_cells, load

ROOT = os.path.dirname(os.path.abspath(__file__))
FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11, index=1)
SMALL = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 9, index=0)


def annotate(rec, cov):
    """Gridded tile with cited cells outlined; red where the cell has no tissue."""
    img = Image.open(os.path.join(ROOT, "gridded", rec["image"].replace(".png", "_grid.png")))
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    empty = set(cov["empty_cells"])
    for c in cited_cells(rec):
        x0, y0, x1, y1 = cell_box(ROWS.index(c[0]), int(c[1]) - 1)
        colour = (255, 0, 0) if c in empty else (0, 90, 255)
        d.rectangle([x0 + 1, y0 + 1, x1 - 2, y1 - 2], outline=colour, width=2)
    return img


def fig_cited(recs, cov, out):
    firsts = [r for r in recs if r["replicate"] == 1]
    W, pad, cols = 224, 30, 5
    rows_n = (len(firsts) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (W + 8) + 8, rows_n * (W + pad) + 8), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(firsts):
        x, y = 8 + (i % cols) * (W + 8), 8 + (i // cols) * (W + pad)
        sheet.paste(annotate(r, cov[r["image"]]), (x, y))
        p = r.get("parsed") or {}
        hits = [c for c in cited_cells(r) if c in set(cov[r["image"]]["empty_cells"])]
        ok = "OK" if p.get("label") == r["label_true"] else "WRONG"
        d.text(
            (x, y + W + 3),
            f"{r['image'][6:9]} true={r['label_true']} pred={p.get('label')} {ok}",
            fill=(0, 0, 0) if ok == "OK" else (170, 0, 0),
            font=SMALL,
        )
        d.text(
            (x, y + W + 14),
            f"cited {len(cited_cells(r))}  on-empty {len(hits)}: {','.join(hits) if hits else '-'}",
            fill=(200, 0, 0) if hits else (60, 60, 60),
            font=SMALL,
        )
    sheet.save(out)
    return out


def fig_heatmap(recs, cov, out):
    firsts = [r for r in recs if r["replicate"] == 1]
    cited = Counter()
    available = Counter()
    for r in firsts:
        cited.update(cited_cells(r))
        empty = set(cov[r["image"]]["empty_cells"])
        for c in ALL_CELLS:
            if c not in empty:
                available[c] += 1

    S, gap = 64, 40
    img = Image.new("RGB", (2 * 4 * S + gap + 40, 4 * S + 56), "white")
    d = ImageDraw.Draw(img)
    panels = [("cited by model", cited, max(cited.values()) if cited else 1),
              ("tiles where cell has tissue", available, len(firsts))]
    for pi, (title, data, vmax) in enumerate(panels):
        ox = 20 + pi * (4 * S + gap)
        d.text((ox, 6), title, fill="black", font=FONT)
        for ri, rl in enumerate(ROWS):
            for ci in range(4):
                c = f"{rl}{ci + 1}"
                v = data[c]
                t = v / vmax if vmax else 0
                shade = int(255 - 175 * t)
                fill = (shade, shade, 255) if pi == 0 else (shade, 255, shade)
                x0, y0 = ox + ci * S, 24 + ri * S
                d.rectangle([x0, y0, x0 + S - 2, y0 + S - 2], fill=fill, outline=(180, 180, 180))
                d.text((x0 + 6, y0 + 6), c, fill=(0, 0, 0), font=SMALL)
                d.text((x0 + 6, y0 + 22), str(v), fill=(0, 0, 0), font=FONT)
    img.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/cte_p1.jsonl")
    args = ap.parse_args()
    recs, cov = load(os.path.join(ROOT, args.run))
    a = fig_cited(recs, cov, os.path.join(ROOT, "runs", "cited_vs_truth.png"))
    b = fig_heatmap(recs, cov, os.path.join(ROOT, "runs", "cell_heatmap.png"))
    print(a)
    print(b)


if __name__ == "__main__":
    main()
