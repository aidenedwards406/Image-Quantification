"""Small helpers shared by the library, the background jobs and the app."""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def stamp() -> str:
    """Sortable timestamp for folder names, e.g. 20261119-142501."""
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def pretty_time(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return _dt.datetime.fromisoformat(iso).strftime("%b %d, %Y %H:%M")
    except ValueError:
        return iso


def slugify(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-.")
    return (s or "item")[:max_len]


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: str | os.PathLike, data: Any) -> None:
    """Atomic JSON write (temp file + replace) so readers never see half a file.

    On Windows ``os.replace`` fails if another process has the target open,
    so retry briefly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=_json_default)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    for attempt in range(3):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.05 * (attempt + 1))
    return default


def finite_or_none(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "—"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


def clean_path_text(text: str) -> str:
    """Tidy a path pasted from Windows Explorer ("C:\\...") or a terminal."""
    return text.strip().strip('"').strip("'").strip()
