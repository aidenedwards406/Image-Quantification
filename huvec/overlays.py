"""Pictures for QC review, Excel and the 3D viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import DAMAGED, IGNORE
from .focus import BLURRY, SHARP
from .imageio import save_jpeg
from .morph import outline

YELLOW = np.array([255, 212, 0], np.float32)  # usable-area outline (like ImageJ's default ROI colour)
RED = np.array([255, 40, 40], np.float32)  # damage
GREEN = np.array([40, 200, 90], np.float32)
ORANGE = np.array([255, 140, 0], np.float32)
CYAN = (0, 230, 255)


def _blend(base: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if mask.any():
        base[mask] = base[mask] * (1 - alpha) + color * alpha


def tua_overlay(img: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Outside the usable area darkened; usable area outlined in yellow."""
    out = img.astype(np.float32)
    outside = label == IGNORE
    out[outside] = out[outside] * 0.35 + np.array([10, 20, 60], np.float32) * 0.4
    _blend(out, outline(~outside, 3), YELLOW, 1.0)
    return out.clip(0, 255).astype(np.uint8)


def damage_overlay(img: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Damage filled red with a solid edge; outside usable area dimmed."""
    out = img.astype(np.float32)
    outside = label == IGNORE
    dmg = label == DAMAGED
    out[outside] *= 0.55
    _blend(out, dmg, RED, 0.42)
    _blend(out, outline(~outside, 2), YELLOW, 0.9)
    _blend(out, outline(dmg, 2), RED, 1.0)
    return out.clip(0, 255).astype(np.uint8)


def focus_overlay(img: np.ndarray, focus_map: np.ndarray, label: np.ndarray | None = None) -> np.ndarray:
    out = img.astype(np.float32)
    _blend(out, focus_map == SHARP, GREEN, 0.30)
    _blend(out, focus_map == BLURRY, ORANGE, 0.50)
    if label is not None:
        _blend(out, outline(label != IGNORE, 2), YELLOW, 0.9)
    return out.clip(0, 255).astype(np.uint8)


def specks_overlay(img: np.ndarray, centroids: list[tuple[float, float]], radius: int = 9) -> np.ndarray:
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im)
    for x, y in centroids:
        d.ellipse((x - radius, y - radius, x + radius, y + radius), outline=CYAN, width=2)
    return np.asarray(im)


def trace_png(label: np.ndarray) -> Image.Image:
    """Transparent PNG of the damage for the 3D Vessel Model viewer.

    Saved as ``<image>_trace.png`` it is recognised as the traced outline of
    ``<image>``.  Damage is semi-transparent red with an opaque edge, so the
    viewer's "As drawn" and "Mask" modes both work.
    """
    h, w = label.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    dmg = label == DAMAGED
    rgba[dmg] = (255, 40, 40, 110)
    rgba[outline(dmg, 2)] = (255, 40, 40, 255)
    return Image.fromarray(rgba, "RGBA")


RENDER_NAMES = ("original", "tua", "damage", "focus", "specks")


def render_all(img: np.ndarray, label: np.ndarray, focus_map: np.ndarray, speck_xy, out_dir: Path,
               thumb: int = 300) -> None:
    """Write the QC pictures for one image into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    views = {
        "original": img,
        "tua": tua_overlay(img, label),
        "damage": damage_overlay(img, label),
        "focus": focus_overlay(img, focus_map, label),
        "specks": specks_overlay(img, speck_xy),
    }
    for name, arr in views.items():
        save_jpeg(arr, out_dir / f"{name}.jpg", max_side=1600, quality=86)
        if name in ("original", "tua", "damage"):
            save_jpeg(arr, out_dir / f"thumb_{name}.jpg", max_side=thumb, quality=82)
