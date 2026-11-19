"""Synthetic look-alike data for demos, practice and tests.

Creates images resembling 4x brightfield HUVEC images (pink cell layer, dark
nuclei, nucleus-free damage patches, blurry curved edges, black corners,
specks) together with ImageJ ROI files that follow the lab's naming and
tracing conventions.  Nothing here is real data, so the demo can be shared,
screenshotted and used to train new lab members without touching NDA images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse, polygon as draw_polygon
from skimage.measure import approximate_polygon, find_contours

POSITIONS = ["P", "M", "D", "M1", "M2"]


def _blob_polygon(rng, cx, cy, r0, n=48):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones_like(th)
    for k in (2, 3, 4, 5):
        r += rng.uniform(0, 0.16) * np.sin(k * th + rng.uniform(0, 2 * np.pi))
    r *= r0
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1)


def _poly_mask(poly, H, W):
    m = np.zeros((H, W), bool)
    rr, cc = draw_polygon(poly[:, 1] - 0.5, poly[:, 0] - 0.5, shape=(H, W))
    m[rr, cc] = True
    return m


def make_image(rng: np.random.Generator, H: int = 1040, W: int = 1392, *, n_damage: int = 1,
               blur_edges: int = 1, black_corners: bool = False, outside_vessel: bool = False,
               n_specks: int = 0, return_sigma: bool = False):
    """Return (rgb uint8, tua bool mask, list of damage polygons (x, y)) [+ blur sigma map]."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # --- defocus map: sigma grows toward curved edges -----------------------
    sigma = np.zeros((H, W), np.float32)
    edges = rng.choice(["top", "bottom", "left", "right"], size=blur_edges, replace=False) if blur_edges else []
    for e in edges:
        along, across, L = (xx, yy, H) if e in ("top", "bottom") else (yy, xx, W)
        span = W if e in ("top", "bottom") else H
        depth = rng.uniform(0.08, 0.22) * L
        curve = depth + rng.uniform(0.02, 0.07) * L * np.sin(2 * np.pi * along / span * rng.uniform(0.5, 1.2) + rng.uniform(0, 6.28))
        dist = (across if e in ("top", "left") else (L - 1 - across)) - curve  # <0 beyond the boundary
        sigma = np.maximum(sigma, np.clip(2.0 - dist / 25.0, 0, 7.0) * (dist < 50))

    # --- what is "vessel" (cells can grow) vs. in-focus non-vessel ------------
    vessel = np.ones((H, W), bool)
    if outside_vessel:
        side = rng.choice(["left", "right"])
        x0 = rng.uniform(0.08, 0.2) * W
        bx = x0 + 0.03 * W * np.sin(2 * np.pi * yy / H * rng.uniform(0.4, 1.0) + rng.uniform(0, 6.28))
        vessel = xx > bx if side == "left" else xx < (W - 1 - bx)

    # --- damage polygons -----------------------------------------------------
    polys, damage = [], np.zeros((H, W), bool)
    for _ in range(n_damage):
        r0 = rng.uniform(0.05, 0.15) * min(H, W)
        cx, cy = rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H
        poly = _blob_polygon(rng, cx, cy, r0)
        polys.append(poly)
        damage |= _poly_mask(poly, H, W)

    # --- compose the sharp scene --------------------------------------------
    tint = rng.uniform(0.96, 1.04, 3)
    bg = np.array([0.90, 0.82, 0.87], np.float32) * tint
    img = np.ones((H, W, 3), np.float32) * bg
    mottle = gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), 6)
    mottle /= mottle.std() + 1e-6
    cells = vessel & ~damage
    body = np.array([0.93, 0.86, 0.92], np.float32)
    img[cells] *= body + 0.015 * mottle[cells][:, None]
    img[damage & vessel] *= np.array([1.02, 1.03, 1.02], np.float32)
    img[~vessel] = np.array([0.62, 0.60, 0.63], np.float32) * (1 + 0.04 * mottle[~vessel][:, None])

    nuc = np.zeros((H, W), np.float32)
    spacing = 22
    for gy in range(spacing // 2, H, spacing):
        for gx in range(spacing // 2, W, spacing):
            cy, cx = gy + rng.uniform(-7, 7), gx + rng.uniform(-7, 7)
            iy, ix = int(cy), int(cx)
            if not (0 <= iy < H and 0 <= ix < W) or not cells[iy, ix]:
                continue
            rr, cc = ellipse(cy, cx, rng.uniform(3.0, 4.4), rng.uniform(2.4, 3.4), shape=(H, W),
                             rotation=rng.uniform(0, np.pi))
            nuc[rr, cc] = 1.0
    alpha = gaussian_filter(nuc, 0.7) * 0.92
    nuc_col = np.array([0.36, 0.26, 0.42], np.float32)
    img = img * (1 - alpha[..., None]) + nuc_col * alpha[..., None]

    if n_specks:
        sp = np.zeros((H, W), np.float32)
        cand = np.argwhere(cells & (sigma < 1.0))
        for i in rng.choice(len(cand), size=min(n_specks, len(cand)), replace=False) if len(cand) else []:
            cy, cx = cand[i]
            rr, cc = ellipse(cy, cx, rng.uniform(1.6, 2.4), rng.uniform(1.6, 2.4), shape=(H, W))
            sp[rr, cc] = 1.0
        a2 = gaussian_filter(sp, 0.5)
        img = img * (1 - a2[..., None]) + np.array([0.05, 0.04, 0.06], np.float32) * a2[..., None]

    # --- defocus: blend a stack of blurred versions -------------------------
    if sigma.max() > 0:
        levels = [0.0, 1.0, 2.0, 3.5, 5.0, 7.0]
        stack = [img] + [np.stack([gaussian_filter(img[..., c], s) for c in range(3)], -1) for s in levels[1:]]
        idx = np.clip(np.searchsorted(levels, sigma) - 1, 0, len(levels) - 2)
        lo = np.take(levels, idx)
        hi = np.take(levels, idx + 1)
        t = np.clip((sigma - lo) / (hi - lo), 0, 1)[..., None]
        out = np.zeros_like(img)
        for k in range(len(levels) - 1):
            sel = idx == k
            if sel.any():
                out[sel] = stack[k][sel] * (1 - t[sel]) + stack[k + 1][sel] * t[sel]
        img = out

    # --- illumination, black corners, noise ----------------------------------
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    img *= (1 - 0.08 * r2)[..., None]
    dark = np.zeros((H, W), bool)
    if black_corners:
        rr = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        R = rng.uniform(0.9, 0.97) * np.sqrt((W / 2) ** 2 + (H / 2) ** 2)
        fall = np.clip((rr - R) / 6.0, 0, 1)
        img *= (1 - 0.96 * fall)[..., None]
        dark = rr > R - 6
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    rgb = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)

    tua = vessel & (sigma < 2.0) & ~dark
    if return_sigma:
        return rgb, tua, polys, sigma
    return rgb, tua, polys


def mask_to_polygons(mask: np.ndarray, min_frac: float = 0.002) -> list[np.ndarray]:
    """Outlines of a mask as ImageJ-style polygons (x, y in pixel-corner coordinates)."""
    padded = np.pad(mask.astype(np.float32), 1)
    out = []
    for c in find_contours(padded, 0.5):
        c = approximate_polygon(c, tolerance=0.8)
        ring = c - 1.0  # undo padding
        area = 0.5 * abs(np.dot(ring[:, 0], np.roll(ring[:, 1], 1)) - np.dot(ring[:, 1], np.roll(ring[:, 0], 1)))
        if area < min_frac * mask.size:
            continue
        out.append(np.stack([ring[:, 1] + 0.5, ring[:, 0] + 0.5], 1))
    return out


def _roi(points, name):
    from roifile import ImagejRoi

    roi = ImagejRoi.frompoints(np.asarray(points, np.float32), name=name)
    return roi


def write_rois(stem: str, folder: Path, tua_polys, damage_polys, rng) -> str | None:
    """Write ROI file(s) following the lab conventions; returns the file name."""
    from roifile import roiwrite

    whole_usable = not tua_polys
    if whole_usable and not damage_polys:
        return None  # no annotation file = whole image usable, 0 % damage
    if whole_usable and len(damage_polys) == 1:
        name = f"{stem}{rng.choice(['.roi', '_ROI.roi'])}"
        _roi(damage_polys[0], "").tofile(folder / name)  # single unnamed .roi = damage only
        return name
    rois = [_roi(p, "TUA") for p in tua_polys]
    rois += [_roi(p, f"D{i + 1}") for i, p in enumerate(damage_polys)]
    name = f"{stem}{rng.choice(['.tif.zip', '_ROI.roi.zip'])}"
    roiwrite(folder / name, rois)
    return name


def make_dataset(out_dir: str | Path, n_vessels: int = 5, per_vessel: int = 8, size=(1040, 1392),
                 seed: int = 0, vessel_prefix: str = "BR-DEMO", with_rois: bool = True,
                 date: str = "261101", first_vessel: int = 1) -> list[dict]:
    """Write a folder of synthetic images (+ ROI files). Returns per-image truth."""
    import tifffile

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    H, W = size
    truth = []
    for v in range(first_vessel, first_vessel + n_vessels):
        vessel = f"{vessel_prefix}-{v}"
        for k in range(per_vessel):
            section = 1 + k % 4
            side = "T" if (k // 4) % 2 == 0 else "B"
            pos = POSITIONS[(k // 8) % len(POSITIONS)]
            stem = f"{date}_{vessel}_4x_{section}_{side}_{pos}"
            has_damage = rng.random() < 0.67  # about 1/3 of images undamaged
            img, tua, polys = make_image(
                rng, H, W,
                n_damage=int(rng.integers(1, 4)) if has_damage else 0,
                blur_edges=int(rng.choice([0, 1, 1, 2])) if section != 4 else 2,
                black_corners=bool(rng.random() < 0.3),
                outside_vessel=bool(section == 4 and rng.random() < 0.6),
                n_specks=int(rng.choice([0, 0, 0, 4, 25])),
            )
            tifffile.imwrite(out / f"{stem}.tif", img, photometric="rgb")
            tua_polys = [] if tua.all() else mask_to_polygons(tua)
            roi_name = write_rois(stem, out, tua_polys, polys, rng) if with_rois else None
            truth.append({"name": f"{stem}.tif", "vessel": vessel, "section": section, "roi": roi_name})
    return truth


def make_demo(root: str | Path, size=(1040, 1392), per_vessel: int = 10, seed: int = 7) -> dict:
    """The demo study used by the app's "Create demo data" button."""
    from .rois import load_annotation

    root = Path(root)
    train = make_dataset(root / "traced_images", n_vessels=5, per_vessel=per_vessel, size=size, seed=seed)
    bench = make_dataset(root / "percent_only_images", n_vessels=3, per_vessel=per_vessel, size=size,
                         seed=seed + 1, first_vessel=6)
    # Hand-measured % for the benchmark: computed from (unsaved) ROIs plus a
    # little human variation, then the ROIs are deleted - like the lab's ~500
    # images that only have a % recorded.
    import pandas as pd

    rng = np.random.default_rng(seed + 2)
    rows = []
    for t in bench:
        p = root / "percent_only_images"
        roi_files = [p / t["roi"]] if t["roi"] else []
        from .imageio import image_size

        w, h = image_size(p / t["name"])
        pct = load_annotation(roi_files, w, h).pct_damage
        rows.append({"Image": t["name"], "% Damage": round(max(0.0, pct + (rng.normal(0, 1.2) if pct > 0 else 0)), 2)})
        for f in roi_files:
            f.unlink()
    pd.DataFrame(rows).to_excel(root / "percent_only_manual_damage.xlsx", index=False)
    new = make_dataset(root / "new_images_to_analyze", n_vessels=2, per_vessel=6, size=size, seed=seed + 3,
                       first_vessel=9, with_rois=False)
    return {"training_folder": str(root / "traced_images"), "benchmark_folder": str(root / "percent_only_images"),
            "benchmark_table": str(root / "percent_only_manual_damage.xlsx"),
            "new_images_folder": str(root / "new_images_to_analyze"),
            "n_train": len(train), "n_bench": len(bench), "n_new": len(new)}
