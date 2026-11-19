"""Small black specks (possibly balled-up nuclei of unhealthy cells).

A speck is a small, compact blob that is much darker than normal nuclei.
Images with many specks get flagged for follow-up; the specks themselves do
not change the damage %.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import label, regionprops

from .imageio import normalized_gray
from .morph import remove_small


@dataclass
class SpeckResult:
    centroids: list[tuple[float, float]] = field(default_factory=list)  # (x, y)
    areas: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.centroids)


def detect_specks(img: np.ndarray, cfg: dict, region: np.ndarray | None = None,
                  dark_level: float = 0.12) -> SpeckResult:
    g = normalized_gray(img)
    max_area = int(cfg.get("max_area", 80))
    min_area = int(cfg.get("min_area", 4))
    min_solidity = float(cfg.get("min_solidity", 0.8))

    # Large dark areas (black corners, out-of-field) plus a margin are excluded.
    big_dark = remove_small(g < dark_level, max(max_area * 10, 500))
    exclude = binary_dilation(big_dark, iterations=6) if big_dark.any() else big_dark

    dark = g < float(cfg.get("max_intensity", 0.18))
    lab = label(dark, connectivity=2)
    out = SpeckResult()
    for rp in regionprops(lab):
        if not (min_area <= rp.area <= max_area):
            continue
        if rp.solidity < min_solidity:
            continue
        r, c = rp.centroid
        ri, ci = int(round(r)), int(round(c))
        if exclude[ri, ci]:
            continue
        if region is not None and not region[ri, ci]:
            continue
        out.centroids.append((float(c), float(r)))
        out.areas.append(int(rp.area))
    return out
