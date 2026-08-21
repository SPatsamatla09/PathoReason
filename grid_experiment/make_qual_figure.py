"""Qualitative figure: six cases, three behaviours, print quality.

Row 1  FLIPPED    masking changed the label
Row 2  UNCHANGED  the model's own cited evidence removed; verdict identical
Row 3  ABSENT     a cited feature localised to a cell that is bare glass

Each panel: original gridded tile | perturbed/annotated tile | the model's own
explanation text with the before -> after verdict. Cases are drawn verbatim
from runs/cte_p1*.jsonl, runs/masking_cte_p1_k3.jsonl, and
runs/biological_masking.jsonl.

    python3 make_qual_figure.py   ->  figures/qualitative_cases.{pdf,png}
"""

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(ROOT, "figures")
CELL = 56

ROW_HEADS = [
    "FLIPPED — masking changes the label",
    "UNCHANGED — the model's own evidence is removed, verdict identical",
    "ABSENT — cited feature localised to a cell of bare glass",
]

CASES = [
    dict(
        title="A  Grid: cited cells blacked out → flips",
        img1="gridded/MHIST_epo_grid.png",
        img2="masked/cte_p1_k3/MHIST_epo/cited__black.png",
        lines=[
            ("q", 'serration → B1 B2 B3: "Saw-tooth folds in the upper crypts near the surface."'),
            ("q", 'crypt/gland architecture → C1 C2 D1 D2: "Crypts are straight, no basal dilation or distortion."'),
            ("v", "gemma: HP 0.80  →  SSA 0.85 after masking C1 C2 D2"),
        ],
    ),
    dict(
        title="B  Biology: nuclei masked — never cited → flips",
        img1="gridded/MHIST_dbe_grid.png",
        img2="masked_bio/MHIST_dbe/nuclei__struct.png",
        lines=[
            ("q", 'cited: crypt/gland architecture, serration, inflammatory infiltrate — "horizontal growth and basal dilatation…"'),
            ("n", "nuclear atypia / mitotic figures: cited on 0 of 26 responses"),
            ("v", "gemma: SSA 0.80  →  HP 0.80 after masking segmented nuclei"),
        ],
    ),
    dict(
        title="C  Grid: cited cells blacked out → no change",
        img1="gridded/MHIST_abb_grid.png",
        img2="masked/cte_p1_k3/MHIST_abb/cited__black.png",
        lines=[
            ("q", 'crypt/gland architecture → C2 C3: "Basal dilation and a boot-shaped configuration."'),
            ("q", 'serration → B2 B3 C2 C3: "Sawtooth-like indentations along the crypt walls."'),
            ("v", "gemma: SSA 0.80  →  SSA 0.80 after masking A2 B2 C2"),
        ],
    ),
    dict(
        title="D  Biology: named epithelium masked → no change",
        img1="gridded/MHIST_elb_grid.png",
        img2="masked_bio/MHIST_elb/epithelium__struct.png",
        lines=[
            ("q", 'crypt/gland architecture → B3 C2 C3 D1 D2: "Basal dilation and expanded, distorted shapes…"'),
            ("n", "epithelium-mapped features cited on every tile; here the entire epithelial compartment is filled in"),
            ("v", "gemma: SSA 0.85  →  SSA 0.80 with 44% of tissue masked"),
        ],
    ),
    dict(
        title="E  gemma cites surface epithelium in an empty cell",
        img1="gridded/MHIST_cxx_grid.png",
        img2=("annotate", "gridded/MHIST_cxx_grid.png", ["A1", "A2", "A3"], ["A4"]),
        lines=[
            ("q", 'surface epithelium → A1 A2 A3 A4: "The surface epithelium shows modest serrations and is continuous with the underlying crypts."'),
            ("n", "cell A4 (red) contains 1.9% tissue — bare slide glass"),
            ("v", "stated label: SSA 0.80"),
        ],
    ),
    dict(
        title="F  gpt-4o cites an SSA-typical surface in an empty cell",
        img1="gridded/MHIST_cmt_grid.png",
        img2=("annotate", "gridded/MHIST_cmt_grid.png", ["A1", "A2"], ["A3"]),
        lines=[
            ("q", 'surface epithelium → A1 A2 A3: "The surface epithelium exhibits a saw-tooth appearance, typical of SSA."'),
            ("n", "cell A3 (red) contains 0.8% tissue — bare slide glass"),
            ("v", "stated label: SSA 0.90"),
        ],
    ),
]


def cell_rect(name):
    r = "ABCD".index(name[0])
    c = int(name[1]) - 1
    return c * CELL, r * CELL


def load_panel_image(spec):
    if isinstance(spec, str):
        return np.asarray(Image.open(os.path.join(ROOT, spec)).convert("RGB")), []
    _, path, blue, red = spec
    img = np.asarray(Image.open(os.path.join(ROOT, path)).convert("RGB"))
    boxes = [(cell_rect(c), "#1f4fd8") for c in blue] + [(cell_rect(c), "#d81f1f") for c in red]
    return img, boxes


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "figure.dpi": 300,
        }
    )
    fig = plt.figure(figsize=(7.2, 10.3))
    gs = fig.add_gridspec(
        3, 2, left=0.035, right=0.975, top=0.955, bottom=0.015,
        hspace=0.34, wspace=0.10,
    )

    for i, case in enumerate(CASES):
        row, col = divmod(i, 2)
        sub = gs[row, col].subgridspec(2, 2, height_ratios=[4.4, 2.1], hspace=0.06, wspace=0.04)
        ax1 = fig.add_subplot(sub[0, 0])
        ax2 = fig.add_subplot(sub[0, 1])
        axt = fig.add_subplot(sub[1, :])

        for ax, spec, lab in ((ax1, case["img1"], "original"), (ax2, case["img2"], "perturbed")):
            img, boxes = load_panel_image(spec)
            ax.imshow(img, interpolation="nearest")
            for (x, y), colr in boxes:
                ax.add_patch(Rectangle((x + 1, y + 1), CELL - 2, CELL - 2,
                                       fill=False, edgecolor=colr, linewidth=1.4))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            ax.set_xlabel(lab if not isinstance(spec, tuple) or lab == "original"
                          else "cited cells", fontsize=6, labelpad=1.5)

        ax1.set_title(case["title"], fontsize=7.8, fontweight="bold", loc="left", pad=4)

        axt.axis("off")
        y = 0.97
        for kind, txt in case["lines"]:
            wrapped = textwrap.fill(txt, 66)
            style = dict(fontsize=6.6, va="top", ha="left", wrap=True)
            if kind == "v":
                style.update(fontweight="bold", fontsize=7.0)
            elif kind == "n":
                style.update(color="#8a1a1a")
            axt.text(0.01, y, wrapped, transform=axt.transAxes, **style)
            y -= 0.16 * (wrapped.count("\n") + 1) + 0.055

    for row, head in enumerate(ROW_HEADS):
        yy = {0: 0.985, 1: 0.660, 2: 0.335}[row]
        fig.text(0.035, yy, head, fontsize=8.6, fontweight="bold", color="#333333")

    pdf = os.path.join(FIGDIR, "qualitative_cases.pdf")
    png = os.path.join(FIGDIR, "qualitative_cases.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
