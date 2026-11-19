"""Analysis batches: a set of images analysed together, then QC-reviewed.

``<workspace>/batches/<batch_id>/``::

    batch.json          name, who, when, model used, settings snapshot, image list
    inputs/             uploaded images (folder-mode batches read images in place)
    results/<stem>.json numbers + flags per image
    renders/<stem>/     original / overlays / thumbnails (JPEG)
    masks/<stem>.png    predicted label (0 ignore, 1 healthy, 2 damaged)
    review.json         QC decisions
    exports/            Excel files
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from . import registry, workspace
from . import settings as settings_mod
from .naming import image_stem, list_images, parse_image_name
from .utils import now_iso, read_json, slugify, stamp, write_json

APPROVED, REJECTED = "approved", "rejected"
REJECT_REASONS = ["Usable area is wrong", "Missed damage", "Marked damage that isn't there",
                  "Image too blurry to judge", "Other"]


def batch_dir(batch_id: str) -> Path:
    return workspace.batches_dir() / batch_id


def create_batch(name: str, created_by: str, notes: str, image_paths: list[Path], source: dict,
                 model_id: str, tta: bool = False) -> str:
    batch_id = f"{stamp()}_{slugify(name, 40)}"
    d = batch_dir(batch_id)
    d.mkdir(parents=True)
    images, seen = [], set()
    for p in image_paths:
        stem = image_stem(p.name)
        key = stem.lower()
        if key in seen:
            continue
        seen.add(key)
        info = parse_image_name(p.name)
        images.append({"name": p.name, "stem": stem, "path": str(p), "vessel": info.vessel,
                       "section": info.section, "side": info.side, "position": info.subsection,
                       "naming_ok": info.follows_convention})
    write_json(d / "batch.json", {
        "id": batch_id, "name": name, "created": now_iso(), "created_by": created_by, "notes": notes,
        "source": source, "model_id": model_id, "model_name": registry.display_name(model_id),
        "model_card": _model_summary(model_id),
        "settings": settings_mod.load(), "tta": tta, "images": images,
    })
    return batch_id


def _model_summary(model_id: str) -> dict:
    c = registry.card(model_id)
    out = {k: v for k, v in c.items() if k not in ("evaluation", "training")}
    out["evaluation_summary"] = {k: v for k, v in (c.get("evaluation") or {}).items() if k != "per_image"}
    return out


def save_uploads(batch_name: str, files: list[tuple[str, bytes]]) -> Path:
    """Write browser uploads to a fresh folder in the workspace; returns it."""
    d = workspace.sub("uploads") / f"{stamp()}_{slugify(batch_name, 40)}"
    d.mkdir(parents=True, exist_ok=True)
    for name, data in files:
        (d / Path(name).name).write_bytes(data)
    return d


def images_in_folder(folder: str | Path) -> list[Path]:
    return list_images(Path(folder))


def load(batch_id: str) -> dict:
    return read_json(batch_dir(batch_id) / "batch.json", {}) or {}


def list_batches() -> list[dict]:
    out = []
    for d in sorted(workspace.batches_dir().iterdir(), reverse=True):
        if d.is_dir() and (d / "batch.json").exists():
            b = read_json(d / "batch.json", {}) or {}
            b["status"] = summary(d.name, b)
            out.append(b)
    return out


def result(batch_id: str, stem: str) -> dict | None:
    return read_json(batch_dir(batch_id) / "results" / f"{stem}.json", None)


def results(batch_id: str) -> dict[str, dict]:
    d = batch_dir(batch_id) / "results"
    if not d.exists():
        return {}
    return {p.stem: r for p in d.glob("*.json") if (r := read_json(p, None))}


def render_path(batch_id: str, stem: str, view: str) -> Path:
    return batch_dir(batch_id) / "renders" / stem / f"{view}.jpg"


def review(batch_id: str) -> dict[str, dict]:
    return read_json(batch_dir(batch_id) / "review.json", {}) or {}


def set_decision(batch_id: str, stem: str, decision: str | None, reasons: list[str] | None = None,
                 note: str = "", by: str = "") -> None:
    path = batch_dir(batch_id) / "review.json"
    data = read_json(path, {}) or {}
    if decision is None:
        data.pop(stem, None)
    else:
        data[stem] = {"decision": decision, "reasons": reasons or [], "note": note, "by": by, "at": now_iso()}
    write_json(path, data)


def summary(batch_id: str, b: dict | None = None) -> dict:
    b = b or load(batch_id)
    res = results(batch_id)
    rev = review(batch_id)
    n = len(b.get("images", []))
    analysed = sum(1 for im in b.get("images", []) if im["stem"] in res)
    errors = sum(1 for r in res.values() if r.get("error"))
    approved = sum(1 for v in rev.values() if v.get("decision") == APPROVED)
    rejected = sum(1 for v in rev.values() if v.get("decision") == REJECTED)
    flagged = sum(1 for r in res.values() if r.get("flags"))
    return {"images": n, "analysed": analysed, "errors": errors, "approved": approved, "rejected": rejected,
            "reviewed": approved + rejected, "to_review": max(0, analysed - errors - approved - rejected),
            "flagged": flagged}


def delete_batch(batch_id: str) -> None:
    b = load(batch_id)
    src = b.get("source", {})
    shutil.rmtree(batch_dir(batch_id), ignore_errors=True)
    if src.get("type") == "upload" and src.get("folder"):
        up = Path(src["folder"])
        if up.is_dir() and workspace.root() in up.resolve().parents:
            shutil.rmtree(up, ignore_errors=True)


def fmt_pct(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if math.isnan(x) else f"{x:.1f}%"
