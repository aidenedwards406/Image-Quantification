"""File-name conventions.

Image names look like ``260919_BR-S24-9_4x_2_T_P.tif``::

    260919      date (YYMMDD)
    BR-S24-9    bioreactor name + number = exactly one physical vessel
    4x          magnification
    2           vessel section (1-4)
    T           Top / Bottom
    P           position within the section (D, M, M1, M2, P)

ROI files share the image's base name.  Seen in the wild: ``.tif.zip``,
``.roi``, ``_ROI.roi`` and ``_ROI.roi.zip``.  Rather than listing every
variant, ROI names are reduced to a *key* by repeatedly stripping known
suffixes, and matched to the image with the same key.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

IMAGE_EXTS = (".tif", ".tiff")
ROI_EXTS = (".roi", ".zip")

# Parsing mirrors the 3D Vessel Model viewer (src/lib/naming.ts): the
# magnification token ("4x") anchors the name, everything before it is the
# vessel (after an optional YYMMDD/YYYYMMDD date), and section / side /
# position follow it.
_MAG_RE = re.compile(r"^\d+(?:\.\d+)?\s*x$", re.IGNORECASE)
_DATE_RE = re.compile(r"^(\d{6}|\d{8})$")
_SECTION_RE = re.compile(r"^(?:s|sec|section)?0*(\d{1,2})$", re.IGNORECASE)


def _read_side(tok: str) -> str | None:
    t = tok.lower()
    if t in ("t", "top"):
        return "T"
    if t in ("b", "bot", "bottom", "btm"):
        return "B"
    return None


def _read_position(tok: str) -> str | None:
    t = tok.lower()
    if t in ("p", "prox", "proximal"):
        return "P"
    if t in ("d", "dist", "distal"):
        return "D"
    m = re.match(r"^(?:m|med|medial|mid)-?0*(\d{0,2})$", t)
    if m:
        return "M" + m.group(1) if m.group(1) else "M"
    return None


_ROI_SUFFIXES = (".zip", ".roi", "_rois", "_roi", "-roi", " roi", "_roiset", "roiset", ".tiff", ".tif")


@dataclass
class ImageName:
    filename: str
    stem: str
    date: str = ""
    vessel: str = ""
    magnification: str = ""
    section: str = ""
    side: str = ""
    subsection: str = ""
    follows_convention: bool = False
    problem: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def image_stem(filename: str) -> str:
    name = Path(filename).name
    lower = name.lower()
    for ext in IMAGE_EXTS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


EXPECTED = "date_Vessel_4x_section_T/B_position, e.g. 260919_BR-S24-9_4x_2_T_P"


def parse_image_name(filename: str) -> ImageName:
    stem = image_stem(filename)
    info = ImageName(filename=Path(filename).name, stem=stem)
    tokens = [t for t in re.split(r"[_\s]+", stem.strip()) if t]
    mag_idx = next((k for k in range(1, len(tokens)) if _MAG_RE.match(tokens[k])), None)
    if mag_idx is None:
        if len(tokens) >= 2:
            info.vessel = tokens[1] if _DATE_RE.match(tokens[0]) else tokens[0]
            info.problem = f"no magnification (e.g. 4x) in the name - guessed vessel '{info.vessel}'. Expected {EXPECTED}"
        else:
            info.problem = f"name has too few parts - vessel unknown. Expected {EXPECTED}"
        return info

    head = tokens[:mag_idx]
    if len(head) >= 2 and _DATE_RE.match(head[0]):
        info.date = head[0]
        head = head[1:]
    info.vessel = "_".join(head)
    info.magnification = tokens[mag_idx].lower().replace(" ", "")
    rest = tokens[mag_idx + 1:]
    problems = []
    sec = _SECTION_RE.match(rest[0]) if len(rest) > 0 else None
    side = _read_side(rest[1]) if len(rest) > 1 else None
    pos = _read_position(rest[2]) if len(rest) > 2 else None
    if sec:
        info.section = str(int(sec.group(1)))
    else:
        problems.append("section number missing after the magnification")
    if side:
        info.side = side
    else:
        problems.append("side (T or B) missing after the section")
    if pos:
        info.subsection = pos
    else:
        problems.append("position (P, M, M1, M2 ... or D) missing after the side")
    if len(rest) > 3:
        problems.append(f"extra text '{'_'.join(rest[3:])}' after the position")
    info.follows_convention = not problems or (len(problems) == 1 and problems[0].startswith("extra text"))
    info.problem = "; ".join(problems)
    return info


def roi_key(filename: str) -> str:
    """``X.tif.zip`` / ``X_ROI.roi.zip`` / ``X.roi`` ... -> ``x``."""
    name = Path(filename).name.lower().strip()
    changed = True
    while changed:
        changed = False
        for suf in _ROI_SUFFIXES:
            if name.endswith(suf) and len(name) > len(suf):
                name = name[: -len(suf)].rstrip()
                changed = True
    return name


def image_key(filename: str) -> str:
    return image_stem(filename).lower().strip()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._")


def is_roi(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ROI_EXTS and not path.name.startswith("._")


@dataclass
class FolderMatch:
    images: list[Path] = field(default_factory=list)
    rois_for: dict[str, list[Path]] = field(default_factory=dict)  # image key -> roi files
    orphan_rois: list[Path] = field(default_factory=list)
    duplicate_images: list[str] = field(default_factory=list)


def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted((p for p in it if is_image(p)), key=lambda p: p.name.lower())


def match_folder(folder: str | Path, recursive: bool = False) -> FolderMatch:
    folder = Path(folder)
    it = list(folder.rglob("*") if recursive else folder.iterdir())
    images = sorted((p for p in it if is_image(p)), key=lambda p: p.name.lower())
    rois = sorted((p for p in it if is_roi(p)), key=lambda p: p.name.lower())
    out = FolderMatch(images=images)
    seen: dict[str, Path] = {}
    for img in images:
        k = image_key(img.name)
        if k in seen:
            out.duplicate_images.append(img.name)
        seen[k] = img
        out.rois_for.setdefault(k, [])
    for r in rois:
        k = roi_key(r.name)
        if k in out.rois_for:
            out.rois_for[k].append(r)
        else:
            out.orphan_rois.append(r)
    return out
