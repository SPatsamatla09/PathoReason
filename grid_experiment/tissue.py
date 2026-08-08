"""Tissue detection for MHIST H&E tiles.

Background in these tiles is bare slide glass: bright and nearly colorless.
Tissue is chromatic (eosin pink / hematoxylin purple), so a pixel counts as
tissue when it carries stain saturation OR is dark enough that it cannot be
glass. The OR term catches pale washed-out tissue that a plain grayscale
threshold misses.
"""

import numpy as np
from PIL import Image

SAT_THRESH = 0.08
VAL_THRESH = 0.80
GRAY_THRESH = 220  # secondary rule, kept for comparison


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def tissue_mask(rgb, sat_thresh=SAT_THRESH, val_thresh=VAL_THRESH):
    """Boolean HxW mask, True where the pixel is tissue rather than slide glass."""
    a = rgb.astype(np.float32) / 255.0
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (sat > sat_thresh) | (mx < val_thresh)


def gray_mask(rgb, thresh=GRAY_THRESH):
    g = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
    return g < thresh
