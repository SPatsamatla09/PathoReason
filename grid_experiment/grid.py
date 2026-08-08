"""4x4 labelled grid overlay for 224x224 MHIST tiles.

Cell naming: row letter A-D top to bottom, column number 1-4 left to right.
A1 is the top-left cell, A4 top-right, D1 bottom-left, D4 bottom-right.

The labels are burned into the pixels, so a model does not have to infer the
convention -- it can read it. That makes a systematically transposed answer
(naming B3 where C2 was meant) a detectable failure mode rather than an
ambiguity in the prompt.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE = 224
N = 4
CELL = TILE // N  # 56

GRID_RGB = (0, 255, 0)  # green: complementary to the pink/purple H&E gamut
LINE_W = 1
LABEL_PT = 12
LABEL_INSET = (3, 2)  # px from the cell's top-left corner

# Menlo Bold keeps its counters open at 12px once a 1px stroke is added, where
# Arial Bold fills them and "B1" degrades into a blob. The stroke guarantees
# contrast against both bare slide glass and dark hematoxylin.
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT_INDEX = 1  # 0=Regular 1=Bold 2=Italic 3=Bold Italic
STROKE_W = 1
STROKE_RGB = (0, 0, 0)

# The tile edge already bounds the outer cells, so an outer rectangle only adds
# ink and crowds the row-A / column-1 labels against the frame.
DRAW_OUTER_BORDER = False

ROWS = "ABCD"


def cell_name(r, c):
    """(row index, col index) -> 'A1' style label."""
    return f"{ROWS[r]}{c + 1}"


def cell_box(r, c):
    """(row, col) -> (left, top, right, bottom) pixel bounds, right/bottom exclusive."""
    return (c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL)


def _font():
    try:
        return ImageFont.truetype(FONT_PATH, LABEL_PT, index=FONT_INDEX)
    except OSError:
        return ImageFont.load_default(size=LABEL_PT)


def draw_grid(img, lines=True, labels=True):
    """Return a copy of `img` with the labelled grid drawn on top.

    `lines` / `labels` can be disabled independently to measure how much of the
    overlay's occlusion each component is responsible for.
    """
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)

    if lines:
        for i in range(1, N):
            p = i * CELL
            d.line([(p, 0), (p, TILE - 1)], fill=GRID_RGB, width=LINE_W)
            d.line([(0, p), (TILE - 1, p)], fill=GRID_RGB, width=LINE_W)
        if DRAW_OUTER_BORDER:
            d.rectangle([0, 0, TILE - 1, TILE - 1], outline=GRID_RGB, width=LINE_W)

    if labels:
        font = _font()
        for r in range(N):
            for c in range(N):
                x, y, _, _ = cell_box(r, c)
                d.text(
                    (x + LABEL_INSET[0], y + LABEL_INSET[1]),
                    cell_name(r, c),
                    font=font,
                    fill=GRID_RGB,
                    stroke_width=STROKE_W,
                    stroke_fill=STROKE_RGB,
                )
    return out


def occlusion_fraction(clean, gridded):
    """Fraction of tile pixels the overlay changed. Sanity metric for the writeup."""
    a = np.asarray(clean.convert("RGB"), dtype=np.int16)
    b = np.asarray(gridded.convert("RGB"), dtype=np.int16)
    return float((np.abs(a - b).sum(axis=2) > 0).mean())
