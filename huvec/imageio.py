"""Reading microscope images into a predictable format: RGB uint8 (H, W, 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if np.issubdtype(arr.dtype, np.integer):
        mx = int(arr.max()) if arr.size else 0
        # Guess the camera bit depth from the data (12-bit cameras are common).
        for bits in (8, 10, 12, 14, 16):
            if mx < 2**bits:
                break
        return np.clip(arr.astype(np.float32) * (255.0 / (2**bits - 1)), 0, 255).astype(np.uint8)
    arr = arr.astype(np.float32)
    mx = float(np.nanmax(arr)) if arr.size else 1.0
    scale = 255.0 if mx <= 1.0 else 255.0 / mx
    return np.clip(np.nan_to_num(arr) * scale, 0, 255).astype(np.uint8)


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(arr)
    while arr.ndim > 3:  # multi-page / z / t stacks: take the first plane
        arr = arr[0]
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] in (3, 4):
        return arr[..., :3]
    if arr.shape[0] in (3, 4):
        return np.moveaxis(arr, 0, -1)[..., :3]
    if arr.shape[-1] == 2:
        return np.repeat(arr[..., :1], 3, axis=2)
    return np.repeat(arr[0][..., None], 3, axis=2)


def read_image(path: str | Path) -> np.ndarray:
    """Return an RGB uint8 array of shape (H, W, 3)."""
    path = Path(path)
    arr = None
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        try:
            arr = tifffile.imread(path)
        except Exception:  # some writers produce TIFFs tifffile dislikes; PIL may cope
            arr = None
    if arr is None:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(_to_uint8(_to_rgb(np.asarray(arr))))


def image_size(path: str | Path) -> tuple[int, int] | None:
    """(width, height) without decoding pixel data, when possible."""
    path = Path(path)
    try:
        if path.suffix.lower() in (".tif", ".tiff"):
            import tifffile

            with tifffile.TiffFile(path) as tf:
                shape = tf.pages[0].shape
                axes = tf.pages[0].axes
            if "X" in axes and "Y" in axes:
                return int(shape[axes.index("X")]), int(shape[axes.index("Y")])
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def gray(img: np.ndarray) -> np.ndarray:
    """Luminance in [0, 1] (float32)."""
    f = img.astype(np.float32) / 255.0
    return f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114


def brightness_reference(g: np.ndarray) -> float:
    """Typical background brightness: 90th percentile of luminance."""
    sample = g[::4, ::4]
    return float(max(np.percentile(sample, 90), 1e-3))


def normalized_gray(img: np.ndarray) -> np.ndarray:
    """Luminance scaled so the bright background sits near 1.0.

    Makes intensity thresholds (specks, black corners) robust to lamp/exposure
    differences between imaging sessions.
    """
    g = gray(img)
    return np.clip(g / brightness_reference(g), 0, 1.5)


def save_jpeg(arr: np.ndarray, path: Path, max_side: int | None = None, quality: int = 88) -> None:
    im = Image.fromarray(arr)
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=quality, optimize=True)
