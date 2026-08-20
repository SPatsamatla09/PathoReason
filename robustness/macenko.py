"""Macenko stain normalization for RGB H&E histopathology images."""

from __future__ import annotations

import numpy as np
from PIL import Image


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))

    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    return image.astype(np.uint8)


def macenko_normalize(
    image: Image.Image | np.ndarray,
    target_stain_matrix: np.ndarray | None = None,
    target_concentrations: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.15,
) -> Image.Image:
    """Normalize an H&E image using the Macenko optical-density method."""

    rgb = _as_rgb_array(image)

    # RGB -> optical density.
    optical_density = -np.log((rgb.astype(np.float64) + 1.0) / 256.0)
    flat_od = optical_density.reshape((-1, 3))

    # Remove nearly transparent/background pixels.
    tissue_od = flat_od[np.all(flat_od > beta, axis=1)]

    if tissue_od.shape[0] < 3:
        return Image.fromarray(rgb)

    # Principal stain plane.
    covariance = np.cov(tissue_od, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    plane = eigenvectors[:, np.argsort(eigenvalues)[-2:]]

    projected = tissue_od @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])

    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100.0 - alpha)

    stain_1 = plane @ np.array([np.cos(min_angle), np.sin(min_angle)])
    stain_2 = plane @ np.array([np.cos(max_angle), np.sin(max_angle)])

    # Conventionally order hematoxylin before eosin.
    if stain_1[0] < stain_2[0]:
        stain_matrix = np.column_stack((stain_1, stain_2))
    else:
        stain_matrix = np.column_stack((stain_2, stain_1))

    concentrations, *_ = np.linalg.lstsq(
        stain_matrix,
        flat_od.T,
        rcond=None,
    )

    source_max = np.percentile(concentrations, 99, axis=1)
    source_max = np.maximum(source_max, 1e-8)

    if target_stain_matrix is None:
        target_stain_matrix = np.array(
            [
                [0.650, 0.072],
                [0.704, 0.990],
                [0.286, 0.105],
            ],
            dtype=np.float64,
        )

    if target_concentrations is None:
        target_concentrations = np.array([1.9705, 1.0308])

    normalized_concentrations = (
        concentrations
        * (target_concentrations / source_max)[:, None]
    )

    reconstructed_od = target_stain_matrix @ normalized_concentrations
    reconstructed_od = np.clip(reconstructed_od, 0.0, 20.0)
    normalized = 255.0 * np.exp(-reconstructed_od)
    normalized = normalized.T.reshape(rgb.shape)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return Image.fromarray(normalized, mode="RGB")
