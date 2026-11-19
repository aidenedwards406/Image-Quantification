"""How much of an image is in focus, from pixel-to-pixel intensity change.

The curved silicone puts parts of each image out of focus.  Blur removes
*fine* intensity changes (sharp nucleus edges) first, while *coarse*
structure (soft blobs where nuclei were) survives much longer.  For every
block of the image (32 x 32 px by default) the detector measures how much
intensity variation there is at three scales:

    detail    = variation between 1 and 2 px      (sharp nucleus edges)
    structure = variation between 2 and 8 px      (nuclei / blurred nuclei)

and decides:

* DARK         block is very dark (black corners, outside the field of view)
* FEATURELESS  almost no structure above camera noise: bare silicone, i.e.
               typically *damage*, which has no nuclei.  It can't be judged,
               so it counts as neither sharp nor blurry; damaged areas are
               never mistaken for blur.
* SHARP        local detail / structure (upper quartile in the block) >= ``sharp_ratio``
* BLURRY       otherwise

Energies are measured relative to the image's own noise level (estimated with
Immerkaer's method), so the thresholds don't depend on exposure.  Defaults
were calibrated on synthetic images where the true blur is known (sharp
cells: ratio ~0.37, blur sigma 1-2 px: ~0.22, sigma 2-3.5 px: ~0.12, damage:
structure ~0.01; damage edges are not mistaken for blur).  Check them on real images on the "Tune focus & specks"
page.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve, gaussian_filter, median_filter

from .imageio import normalized_gray

DARK, FEATURELESS, BLURRY, SHARP = 0, 1, 2, 3
FOCUS_NAMES = {DARK: "dark", FEATURELESS: "featureless", BLURRY: "blurry", SHARP: "in focus"}
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def _block_mean(a: np.ndarray, b: int) -> np.ndarray:
    h, w = a.shape
    ph, pw = (-h) % b, (-w) % b
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), mode="edge")
    H, W = a.shape
    return a.reshape(H // b, b, W // b, b).mean(axis=(1, 3))


def noise_sigma(g: np.ndarray) -> float:
    """Immerkaer (1996) fast noise estimate of a grayscale image."""
    resp = np.abs(convolve(g.astype(np.float32), _IMMERKAER, mode="reflect"))[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) / 6 * resp.mean())


@dataclass
class FocusResult:
    blocks: np.ndarray  # (hb, wb) uint8 class per block
    ratio: np.ndarray  # (hb, wb) detail / structure
    structure: np.ndarray  # (hb, wb) structure energy in units of noise variance
    block: int
    height: int
    width: int
    noise: float

    def pixel_map(self) -> np.ndarray:
        """Per-pixel class map at full resolution."""
        m = np.repeat(np.repeat(self.blocks, self.block, axis=0), self.block, axis=1)
        return m[: self.height, : self.width]

    def fraction(self, cls: int) -> float:
        return float((self.blocks == cls).mean())

    @property
    def in_focus_pct(self) -> float:
        """% of judgeable (structured) image area that is sharp."""
        sharp = int((self.blocks == SHARP).sum())
        blurry = int((self.blocks == BLURRY).sum())
        return 100.0 * sharp / (sharp + blurry) if (sharp + blurry) else float("nan")

    def blurry_pct_within(self, region: np.ndarray) -> float:
        """% of ``region`` (e.g. the usable area) covered by blurry blocks."""
        n = int(region.sum())
        if not n:
            return float("nan")
        return 100.0 * float(((self.pixel_map() == BLURRY) & region).sum()) / n


def _block_quantile(a: np.ndarray, b: int, q: float) -> np.ndarray:
    h, w = a.shape
    ph, pw = (-h) % b, (-w) % b
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), mode="edge")
    H, W = a.shape
    blocks = a.reshape(H // b, b, W // b, b).transpose(0, 2, 1, 3).reshape(H // b, W // b, b * b)
    return np.quantile(blocks, q, axis=2)


def compute_focus(img: np.ndarray, cfg: dict) -> FocusResult:
    g = normalized_gray(img).astype(np.float32)
    s = float(cfg.get("scale", 1.0))
    b = int(cfg.get("block", 32))
    g1, g2, g4, g8 = (gaussian_filter(g, s * k) for k in (1, 2, 4, 8))
    d = (g1 - g2) ** 2
    st = (g2 - g4) ** 2 + (g4 - g8) ** 2
    mean = _block_mean(g, b)
    sn = max(noise_sigma(g), 0.002)
    structure = _block_mean(st, b) / sn**2
    # Sharpness is judged locally (~8 px neighbourhoods) and each block takes
    # its upper quartile, so a block that is half cells / half bare silicone
    # (a damage edge) is judged by its cells instead of the soft edge.
    local = gaussian_filter(d, 4 * s) / (gaussian_filter(st, 4 * s) + 0.05 * sn**2)
    ratio = _block_quantile(local, b, 0.75)

    blocks = np.full(mean.shape, FEATURELESS, dtype=np.uint8)
    has_structure = structure >= float(cfg.get("min_structure", 0.25))
    sharp = ratio >= float(cfg.get("sharp_ratio", 0.25))
    blocks[has_structure & sharp] = SHARP
    blocks[has_structure & ~sharp] = BLURRY
    blocks[mean < float(cfg.get("dark_level", 0.12))] = DARK
    if min(blocks.shape) >= 3:
        # Smooth isolated decisions (classes are ordered dark < featureless < blurry < sharp).
        blocks = median_filter(blocks, size=3, mode="nearest")
    return FocusResult(blocks=blocks, ratio=ratio.astype(np.float32), structure=structure.astype(np.float32),
                       block=b, height=img.shape[0], width=img.shape[1], noise=sn)
