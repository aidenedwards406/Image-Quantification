"""Training data library: folders of traced images (TIFF + ImageJ ROIs).

A *training set* points at a folder; images stay where they are.  Sets can be
added over the years by different people ("Fall 2026, traced by ...").
Training combines any sets you tick.  Vessels can be marked as controls
(always left out of training).
"""

from __future__ import annotations

import math
import shutil
from collections import Counter
from pathlib import Path

from . import workspace
from .imageio import image_size
from .naming import image_key, match_folder, parse_image_name
from .rois import load_annotation
from .utils import now_iso, read_json, slugify, stamp, write_json


def _set_path(set_id: str) -> Path:
    return workspace.library_dir() / f"{set_id}.json"


def _scan_path(set_id: str) -> Path:
    return workspace.library_dir() / f"{set_id}.scan.json"


def list_sets() -> list[dict]:
    out = []
    for p in sorted(workspace.library_dir().glob("*.json")):
        if p.name.endswith(".scan.json"):
            continue
        d = read_json(p, {}) or {}
        if d:
            scan = read_json(_scan_path(d["id"]), {}) or {}
            d["summary"] = scan.get("summary", {})
            out.append(d)
    return out


def get_set(set_id: str) -> dict:
    return read_json(_set_path(set_id), {}) or {}


def add_set(name: str, folder: str, added_by: str = "", traced_by: str = "", notes: str = "") -> str:
    folder_p = Path(folder)
    if not folder_p.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    set_id = f"{stamp()}_{slugify(name, 32)}"
    write_json(_set_path(set_id), {
        "id": set_id, "name": name, "folder": str(folder_p.resolve()), "added": now_iso(),
        "added_by": added_by, "traced_by": traced_by, "notes": notes,
        "control_vessels": [], "excluded_images": [],
    })
    return set_id


def update_set(set_id: str, **fields) -> None:
    d = get_set(set_id)
    d.update(fields)
    write_json(_set_path(set_id), d)


def remove_set(set_id: str) -> None:
    _set_path(set_id).unlink(missing_ok=True)
    _scan_path(set_id).unlink(missing_ok=True)


def scan_folder(folder: str | Path, progress=None) -> dict:
    """Check every image in a folder: name, ROI match, readable ROIs, % damage."""
    fm = match_folder(folder)
    rows = []
    sizes = Counter()
    n = len(fm.images)
    for i, img in enumerate(fm.images):
        if progress:
            progress(i, n, img.name)
        info = parse_image_name(img.name)
        rois = fm.rois_for.get(image_key(img.name), [])
        row = {
            "name": img.name, "path": str(img), "stem": info.stem, "vessel": info.vessel, "date": info.date,
            "section": info.section, "side": info.side, "position": info.subsection,
            "naming_ok": info.follows_convention, "roi_files": [r.name for r in rois],
            "status": "ok", "messages": [],
        }
        if info.problem:
            row["messages"].append("Name: " + info.problem)
            if not info.vessel:
                row["status"] = "error"
            elif not info.follows_convention:
                row["status"] = "warning"
        size = image_size(img)
        if size is None:
            row["status"] = "error"
            row["messages"].append("Image could not be read")
            rows.append(row)
            continue
        w, h = size
        row["width"], row["height"] = w, h
        sizes[(w, h)] += 1
        try:
            ann = load_annotation(rois, w, h)
        except Exception as exc:
            row["status"] = "error"
            row["messages"].append(f"ROI problem: {exc}")
            rows.append(row)
            continue
        row.update(
            tua_px=ann.tua_px, damage_px=ann.damage_px, pct_damage=ann.pct_damage,
            has_tua_roi=ann.has_tua_roi, n_damage_rois=ann.n_damage_rois, roi_names=ann.roi_names,
            annotation=("No ROI file: whole image usable, 0% damage" if ann.source == "none"
                        else f"{'TUA + ' if ann.has_tua_roi else 'whole image usable + '}{ann.n_damage_rois} damage ROI(s)"),
        )
        if ann.warnings:
            row["messages"] += ann.warnings
            if row["status"] == "ok":
                row["status"] = "warning"
        row["notes"] = ann.notes
        if ann.tua_px == 0:
            row["status"] = "error"
            row["messages"].append("Usable area is empty")
        rows.append(row)

    if sizes:
        common = sizes.most_common(1)[0][0]
        for row in rows:
            if "width" in row and (row["width"], row["height"]) != common:
                row["messages"].append(f"Image size {row['width']}x{row['height']} differs from most images ({common[0]}x{common[1]})")
                if row["status"] == "ok":
                    row["status"] = "warning"
    for r in fm.orphan_rois:
        rows.append({"name": r.name, "path": str(r), "stem": "", "vessel": "", "status": "warning",
                     "messages": ["ROI file has no matching image (check the spelling of the name)"],
                     "orphan_roi": True})
    for d in fm.duplicate_images:
        for row in rows:
            if row["name"] == d:
                row["status"] = "error"
                row["messages"].append("Two images share this name (e.g. .tif and .tiff)")

    images = [r for r in rows if not r.get("orphan_roi")]
    vessels = Counter(r["vessel"] for r in images if r.get("vessel"))
    pcts = [r["pct_damage"] for r in images if isinstance(r.get("pct_damage"), float) and not math.isnan(r["pct_damage"])]
    summary = {
        "images": len(images),
        "with_rois": sum(1 for r in images if r.get("roi_files")),
        "without_rois": sum(1 for r in images if not r.get("roi_files")),
        "with_damage": sum(1 for p in pcts if p > 0),
        "errors": sum(1 for r in rows if r["status"] == "error"),
        "warnings": sum(1 for r in rows if r["status"] == "warning"),
        "orphan_rois": len(fm.orphan_rois),
        "vessels": dict(sorted(vessels.items())),
        "mean_pct_damage": (sum(pcts) / len(pcts)) if pcts else None,
        "image_size": f"{sizes.most_common(1)[0][0][0]}x{sizes.most_common(1)[0][0][1]}" if sizes else None,
    }
    return {"rows": rows, "summary": summary, "scanned": now_iso(), "folder": str(folder)}


def scan_set(set_id: str, progress=None) -> dict:
    s = get_set(set_id)
    result = scan_folder(s["folder"], progress=progress)
    write_json(_scan_path(set_id), result)
    return result


def cached_scan(set_id: str) -> dict | None:
    return read_json(_scan_path(set_id), None)


def training_rows(set_ids: list[str]) -> list[dict]:
    """Usable training images from the chosen sets (errors and exclusions removed)."""
    out = []
    for sid in set_ids:
        s = get_set(sid)
        scan = cached_scan(sid) or scan_set(sid)
        controls = set(s.get("control_vessels", []))
        excluded = set(s.get("excluded_images", []))
        for r in scan["rows"]:
            if r.get("orphan_roi") or r["status"] == "error" or not r.get("vessel"):
                continue
            if r["vessel"] in controls or r["name"] in excluded:
                continue
            folder = Path(s["folder"])
            out.append({**r, "set_id": sid, "roi_paths": [str(folder / f) for f in r.get("roi_files", [])]})
    return out


def copy_set_to_workspace(set_id: str) -> None:
    """Optional: copy a set's images + ROIs into the workspace (e.g. off a USB drive)."""
    s = get_set(set_id)
    dest = workspace.sub("library_data") / set_id
    dest.mkdir(parents=True, exist_ok=True)
    for p in Path(s["folder"]).iterdir():
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".roi", ".zip"):
            shutil.copy2(p, dest / p.name)
    update_set(set_id, folder=str(dest), copied_from=s["folder"])
