"""Stain deconvolution for MHIST H&E tiles: hematoxylin vs eosin masks.

rgb2hed (Ruifrok & Johnston deconvolution) separates the optical-density
contributions of hematoxylin (chromatin: nuclei, and at this magnification the
nuclei-dense crypt epithelium) from eosin (cytoplasm, collagen, stroma).

Masks returned, all restricted to the HSV tissue mask from tissue.py:

  epithelium  hematoxylin-dominant tissue -- crypt/gland lining, nuclear zones
  stroma      eosin-dominant tissue without hematoxylin dominance -- lamina
              propria, collagen, cytoplasm-rich regions

Dominance is defined on per-tile z-scored channels so a lightly-stained tile is
not swallowed by a fixed threshold; ties go to the stronger z-score.
"""

import numpy as np
from skimage.color import rgb2hed
from scipy import ndimage as ndi

from tissue import tissue_mask


def hed_channels(rgb):
    """Return (hematoxylin, eosin) optical-density images, float64 HxW."""
    hed = rgb2hed(rgb)
    return hed[..., 0], hed[..., 1]


def _z(x, mask):
    """z-score x over the masked pixels only."""
    v = x[mask]
    mu, sd = float(v.mean()), float(v.std() + 1e-9)
    return (x - mu) / sd


def stain_masks(rgb, sigma=2.0, min_blob_px=30):
    """Return dict of boolean masks: tissue, epithelium, stroma.

    Epithelium and stroma PARTITION the tissue mask: every tissue pixel is
    assigned to whichever smoothed, per-tile z-scored stain channel dominates.
    Visual QA across tiles shows hematoxylin-dominance tracking crypt
    epithelium / nuclear zones and the eosin side tracking lamina propria,
    collagen, and hemorrhagic regions. The known error mode is pale epithelial
    cytoplasm reading as stroma; boundaries carry a few px of noise, removed as
    sub-min_blob_px speckle. The two masks are disjoint by construction but do
    not sum to tissue exactly after speckle cleaning.
    """
    tis = tissue_mask(rgb)
    h, e = hed_channels(rgb)
    hs = ndi.gaussian_filter(h, sigma)
    es = ndi.gaussian_filter(e, sigma)
    hz, ez = _z(hs, tis), _z(es, tis)

    epi = tis & (hz > ez)
    str_ = tis & ~epi

    def clean(m):
        m = ndi.binary_opening(m, structure=np.ones((3, 3)))
        lab, n = ndi.label(m)
        if n:
            sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sizes >= min_blob_px
            m = keep[lab]
        return m

    return {"tissue": tis, "epithelium": clean(epi), "stroma": clean(str_)}
