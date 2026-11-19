"""Where the app keeps its data.

The *workspace* is one folder that holds everything the app creates: models,
analysis batches, QC decisions, training runs and benchmark results.  Point
several lab computers at the same workspace on a shared lab drive and they
share models and batches.  The workspace may contain NDA material (copies of
uploaded images, renders), so keep it on storage approved for that data.

Resolution order:
1. ``HUVEC_WORKSPACE`` environment variable
2. ``local_settings.json`` next to the app (written by the Help & settings page)
3. ``<app folder>/workspace``
"""

from __future__ import annotations

import os
from pathlib import Path

from .utils import read_json, write_json

APP_ROOT = Path(__file__).resolve().parent.parent
LOCAL_SETTINGS = APP_ROOT / "local_settings.json"


def local_settings() -> dict:
    return read_json(LOCAL_SETTINGS, {}) or {}


def save_local_setting(key: str, value) -> None:
    data = local_settings()
    data[key] = value
    write_json(LOCAL_SETTINGS, data)


def root() -> Path:
    env = os.environ.get("HUVEC_WORKSPACE")
    if env:
        p = Path(env)
    else:
        p = Path(local_settings().get("workspace") or (APP_ROOT / "workspace"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def sub(name: str) -> Path:
    p = root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    return sub("models")


def batches_dir() -> Path:
    return sub("batches")


def jobs_dir() -> Path:
    return sub("jobs")


def library_dir() -> Path:
    return sub("library")


def benchmarks_dir() -> Path:
    return sub("benchmarks")


def cache_dir() -> Path:
    return sub("cache")


def demo_dir() -> Path:
    return sub("demo")
