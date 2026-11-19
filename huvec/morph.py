"""Mask clean-up helpers that work across scikit-image versions."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion
from skimage import morphology as _m


def remove_small(mask: np.ndarray, size: int) -> np.ndarray:
    """Drop connected regions with <= ``size`` pixels."""
    if size <= 0 or not mask.any():
        return mask
    try:
        return _m.remove_small_objects(mask, max_size=int(size))
    except TypeError:  # scikit-image < 0.26
        return _m.remove_small_objects(mask, min_size=int(size) + 1)


def fill_small_holes(mask: np.ndarray, size: int) -> np.ndarray:
    """Fill holes with <= ``size`` pixels."""
    if size <= 0 or mask.all() or not mask.any():
        return mask
    try:
        return _m.remove_small_holes(mask, max_size=int(size))
    except TypeError:
        return _m.remove_small_holes(mask, area_threshold=int(size) + 1)


def outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Inner boundary band of a mask, ``width`` pixels thick."""
    if not mask.any():
        return mask
    inner = binary_erosion(mask, iterations=width, border_value=0)
    return mask & ~inner
