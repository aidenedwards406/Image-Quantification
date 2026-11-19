"""Trained models: list, activate, import/export.

Each model lives in ``<workspace>/models/<model_id>/`` as ``model.pth`` plus
``card.json`` (what it was trained on, how well it did).  Exactly one model is
*active*: new analysis batches use it.  Models are never overwritten; a
retrained model is a new entry, and older ones can be re-activated.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import workspace
from .utils import now_iso, read_json, slugify, stamp, write_json


def _dir(model_id: str) -> Path:
    return workspace.models_dir() / model_id


def model_path(model_id: str) -> Path:
    return _dir(model_id) / "model.pth"


def card(model_id: str) -> dict:
    return read_json(_dir(model_id) / "card.json", {}) or {}


def save_card(model_id: str, data: dict) -> None:
    write_json(_dir(model_id) / "card.json", data)


def list_models() -> list[dict]:
    out = []
    for d in sorted(workspace.models_dir().iterdir(), reverse=True):
        if d.is_dir() and (d / "model.pth").exists():
            c = card(d.name)
            c.setdefault("id", d.name)
            c.setdefault("name", d.name)
            out.append(c)
    return out


def active_id() -> str | None:
    info = read_json(workspace.models_dir() / "active.json", {}) or {}
    mid = info.get("model_id")
    return mid if mid and model_path(mid).exists() else None


def active_card() -> dict | None:
    mid = active_id()
    return card(mid) if mid else None


def set_active(model_id: str, by: str = "", reason: str = "") -> None:
    if not model_path(model_id).exists():
        raise FileNotFoundError(model_id)
    path = workspace.models_dir() / "active.json"
    info = read_json(path, {}) or {}
    history = info.get("history", [])
    history.append({"model_id": model_id, "at": now_iso(), "by": by, "reason": reason})
    write_json(path, {"model_id": model_id, "activated": now_iso(), "by": by, "history": history[-50:]})


def new_id(name: str) -> str:
    return f"{stamp()}_{slugify(name, 32)}"


def register(src_pth: Path, card_data: dict, move: bool = False) -> str:
    model_id = card_data.get("id") or new_id(card_data.get("name", "model"))
    d = _dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    (shutil.move if move else shutil.copy2)(str(src_pth), str(d / "model.pth"))
    card_data = dict(card_data, id=model_id)
    card_data.setdefault("registered", now_iso())
    save_card(model_id, card_data)
    if active_id() is None:  # the very first model becomes active automatically
        set_active(model_id, by=card_data.get("trained_by", ""), reason="first model in this workspace")
    return model_id


def import_model(data: bytes, filename: str, by: str = "") -> str:
    """Add a ``.pth`` shared by someone else (e.g. the lab's current model)."""
    from .model import read_meta

    tmp = workspace.models_dir() / f".import_{stamp()}.pth"
    tmp.write_bytes(data)
    try:
        meta = read_meta(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    c = dict(meta.get("card") or {})
    c.setdefault("name", Path(filename).stem)
    c["imported"] = now_iso()
    c["imported_by"] = by
    c["imported_from_file"] = filename
    original = c.get("id")
    c["id"] = original if original and not _dir(original).exists() else new_id(c["name"])
    return register(tmp, c, move=True)


def delete_model(model_id: str) -> None:
    if model_id == active_id():
        raise ValueError("The active model can't be deleted. Activate another model first.")
    shutil.rmtree(_dir(model_id), ignore_errors=True)


def load(model_id: str):
    """(torch model, meta) - meta includes the card and the damage threshold."""
    from .model import load_checkpoint

    model, meta = load_checkpoint(model_path(model_id))
    c = card(model_id)
    # The card can override the threshold (calibration creates a new model id,
    # but keep this as the single source of truth for the running pipeline).
    if "damage_threshold" in c:
        meta["damage_threshold"] = c["damage_threshold"]
    meta["model_id"] = model_id
    return model, meta


def display_name(model_id: str | None) -> str:
    if not model_id:
        return "—"
    c = card(model_id)
    return c.get("name") or model_id
