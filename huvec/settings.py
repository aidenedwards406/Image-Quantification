"""Lab-wide detector and flag settings (focus map, specks, QC flags).

Defaults live here; the "Tune focus & specks" page writes overrides to
``<workspace>/settings.yaml``.  Every analysis batch stores a snapshot of the
settings it used, so old results stay reproducible after settings change.
"""

from __future__ import annotations

import copy

import yaml

from . import workspace

DEFAULTS: dict = {
    "focus": {
        # Analysis grid: the image is judged in square blocks of this size (px).
        "block": 32,
        # Block is "in focus" if (1-2 px detail) / (2-8 px structure) >= this.
        "sharp_ratio": 0.25,
        # Blocks with less 2-8 px structure than this (x camera noise level)
        # are "featureless" (bare silicone / damage) - neither sharp nor blurry.
        "min_structure": 0.25,
        # Blocks darker than this (0-1, background = 1) are black corners.
        "dark_level": 0.12,
        # Multiplies all length scales; raise for higher magnification / smaller pixels.
        "scale": 1.0,
    },
    "specks": {
        # A speck is a small, round, very dark blob (balled-up nucleus).
        "max_intensity": 0.18,  # 0 = black, 1 = background brightness
        "min_area": 4,  # px at full resolution
        "max_area": 80,
        "min_solidity": 0.8,
        "flag_count": 10,  # flag the image at this many specks
    },
    "flags": {
        "max_blurry_in_tua_pct": 15.0,
        "min_tua_pct": 20.0,
        "max_uncertain_pct": 15.0,
    },
    "postprocess": {
        # Remove predicted damage islands smaller than this (px, full res).
        "min_damage_area": 120,
        # Fill holes inside damage smaller than this.
        "min_damage_hole": 120,
        # Same for the usable area (TUA).
        "min_tua_area": 3000,
        "min_tua_hole": 3000,
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif k in out:
            out[k] = v
    return out


def settings_path():
    return workspace.root() / "settings.yaml"


def load() -> dict:
    path = settings_path()
    over = {}
    if path.exists():
        try:
            over = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            over = {}
    return _merge(DEFAULTS, over)


def save(settings: dict) -> None:
    clean = _merge(DEFAULTS, settings)
    settings_path().write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")


def defaults() -> dict:
    return copy.deepcopy(DEFAULTS)
