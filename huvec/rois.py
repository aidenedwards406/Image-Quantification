"""ImageJ ROI files -> masks, following the lab's tracing rules.

Rules (see docs/TRAINING_GUIDE.md):

* ``TUA`` (total usable area) ROIs: union = usable area.  If none was traced,
  the whole image is usable.
* ``D1``, ``D2`` ... ROIs: union = damage, clipped to the usable area.
* A single unnamed ``.roi``: damage only, whole image usable.
* No ROI file at all: whole image usable, 0 % damage.

Pixels are rasterised the way ImageJ does it: a pixel belongs to a ROI when
its *centre* lies inside the outline.  This keeps pixel counts within a
fraction of a percent of ImageJ's "Measure" numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from skimage.draw import polygon as _draw_polygon

from . import DAMAGED, HEALTHY, IGNORE

_AUTO_NAME = re.compile(r"^\d{4}(-\d{4}){1,2}(-\d+)?$")  # ImageJ default names like 0123-0456


def classify_name(name: str) -> str:
    n = (name or "").strip().lower().replace(" ", "").replace("_", "")
    if n.startswith("tua") or n in ("usable", "usablearea", "totalusablearea"):
        return "tua"
    if re.match(r"^d\d+", n) or n.startswith("damage") or n == "d":
        return "damage"
    return "unknown"


@dataclass
class RoiShape:
    name: str
    role: str  # "tua" | "damage" | "unknown"
    polygons: list[np.ndarray]  # each (N, 2) array of x, y in image pixels
    composite: bool = False
    source_file: str = ""


def read_roi_file(path: str | Path) -> tuple[list[RoiShape], list[str]]:
    """Read a ``.roi`` or a ROI-set ``.zip``.  Returns (shapes, warnings)."""
    from roifile import ROI_TYPE, roiread

    path = Path(path)
    warnings: list[str] = []
    try:
        rois = roiread(path)
    except Exception as exc:  # corrupt / not an ImageJ ROI
        return [], [f"could not read ROI file {path.name}: {exc}"]
    if not isinstance(rois, list):
        rois = [rois]
    line_types = {ROI_TYPE.LINE, ROI_TYPE.FREELINE, ROI_TYPE.POLYLINE, ROI_TYPE.ANGLE, ROI_TYPE.POINT, ROI_TYPE.NOROI}
    shapes: list[RoiShape] = []
    for r in rois:
        name = (r.name or "").strip()
        label = name or "(unnamed)"
        composite = r.multi_coordinates is not None
        if r.roitype in line_types and not composite:
            warnings.append(f"ROI {label} in {path.name} is a line/point, not an area - ignored")
            continue
        try:
            polys = r.coordinates(multi=True)
        except Exception as exc:
            warnings.append(f"ROI {label} in {path.name} could not be decoded ({exc}) - ignored")
            continue
        polys = [np.asarray(p, dtype=np.float64) for p in polys if p is not None and len(p) >= 3]
        if not polys:
            warnings.append(f"ROI {label} in {path.name} has no area - ignored")
            continue
        shapes.append(RoiShape(name=name, role=classify_name(name), polygons=polys,
                               composite=composite, source_file=path.name))
    return shapes, warnings


def rasterize(shape: RoiShape, height: int, width: int, scale: float = 1.0) -> np.ndarray:
    """Boolean mask of a ROI at ``scale`` x the original resolution (pixel-centre rule)."""
    mask = np.zeros((height, width), dtype=bool)
    for poly in shape.polygons:
        rr, cc = _draw_polygon(poly[:, 1] * scale - 0.5, poly[:, 0] * scale - 0.5, shape=(height, width))
        if shape.composite:  # composite shapes use the even-odd rule (holes)
            sub = np.zeros_like(mask)
            sub[rr, cc] = True
            mask ^= sub
        else:
            mask[rr, cc] = True
    return mask


@dataclass
class Annotation:
    tua: np.ndarray  # bool (H, W)
    damage: np.ndarray  # bool (H, W), already clipped to tua
    source: str  # "none" | "roi"
    roi_files: list[str] = field(default_factory=list)
    roi_names: list[str] = field(default_factory=list)
    has_tua_roi: bool = False
    n_damage_rois: int = 0
    warnings: list[str] = field(default_factory=list)  # real problems worth checking
    notes: list[str] = field(default_factory=list)  # informational (rules applied as expected)

    @property
    def tua_px(self) -> int:
        return int(self.tua.sum())

    @property
    def damage_px(self) -> int:
        return int(self.damage.sum())

    @property
    def pct_damage(self) -> float:
        t = self.tua_px
        return 100.0 * self.damage_px / t if t else float("nan")

    def label(self) -> np.ndarray:
        """uint8 label image: 0 = ignore (outside TUA), 1 = healthy, 2 = damaged."""
        lab = np.full(self.tua.shape, IGNORE, dtype=np.uint8)
        lab[self.tua] = HEALTHY
        lab[self.damage] = DAMAGED
        return lab


def load_annotation(roi_files: list[str | Path], width: int, height: int, scale: float = 1.0) -> Annotation:
    """Build the 3-class annotation for one image of size ``width`` x ``height``.

    With ``scale`` != 1 the masks are produced directly at the scaled size
    (used for training at reduced resolution).
    """
    h = max(1, int(round(height * scale)))
    w = max(1, int(round(width * scale)))
    warnings: list[str] = []
    notes: list[str] = []
    roi_files = [Path(p) for p in roi_files]
    if not roi_files:
        full = np.ones((h, w), dtype=bool)
        return Annotation(tua=full, damage=np.zeros_like(full), source="none")

    if len(roi_files) > 1:
        warnings.append("several ROI files match this image (" + ", ".join(p.name for p in roi_files)
                        + ") - all of them were combined")
    shapes: list[RoiShape] = []
    for p in roi_files:
        s, wn = read_roi_file(p)
        shapes += s
        warnings += wn

    tua_shapes = [s for s in shapes if s.role == "tua"]
    dmg_shapes = [s for s in shapes if s.role != "tua"]
    single_unnamed = len(shapes) == 1 and shapes[0].role == "unknown"
    for s in dmg_shapes:
        if s.role == "unknown" and not single_unnamed:
            shown = s.name or "(unnamed)"
            if s.name and not _AUTO_NAME.match(s.name):
                warnings.append(f"ROI '{shown}' is not named TUA or D# - counted as damage")
            else:
                warnings.append(f"ROI {shown} has no TUA/D# name - counted as damage")

    if tua_shapes:
        tua = np.zeros((h, w), dtype=bool)
        for s in tua_shapes:
            tua |= rasterize(s, h, w, scale)
        if tua.mean() < 0.05:
            warnings.append(f"usable area (TUA) covers only {100 * tua.mean():.1f}% of the image")
    else:
        tua = np.ones((h, w), dtype=bool)
        if shapes and not single_unnamed and len(shapes) > 1:
            notes.append("no TUA ROI in the set - whole image counted as usable")

    damage_raw = np.zeros((h, w), dtype=bool)
    for s in dmg_shapes:
        m = rasterize(s, h, w, scale)
        if not m.any():
            warnings.append(f"damage ROI '{s.name or '(unnamed)'}' lies outside the image")
        damage_raw |= m
    damage = damage_raw & tua
    outside = int(damage_raw.sum() - damage.sum())
    if outside and outside > 0.02 * max(1, int(damage_raw.sum())):
        notes.append(f"{outside:,} px of traced damage lie outside the usable area (clipped, as the rules say)")

    if not shapes:
        warnings.append("ROI file(s) contained no usable shapes - treated as 0% damage")

    return Annotation(
        tua=tua,
        damage=damage,
        source="roi",
        roi_files=[p.name for p in roi_files],
        roi_names=[s.name or "(unnamed)" for s in shapes],
        has_tua_roi=bool(tua_shapes),
        n_damage_rois=len(dmg_shapes),
        warnings=warnings,
        notes=notes,
    )
