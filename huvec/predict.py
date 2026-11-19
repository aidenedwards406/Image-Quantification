"""One image -> % damage, usable-area pixels, focus, specks and QC flags.

``analyze_probs`` is shared by batch analysis, training evaluation and the
benchmark, so every number in the app is produced the same way.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from . import DAMAGED, HEALTHY, IGNORE
from .focus import FocusResult, compute_focus
from .morph import fill_small_holes, remove_small
from .specks import SpeckResult, detect_specks

UNSURE_BAND = 0.15  # probabilities within 0.5 +/- this count as "model unsure"


@dataclass
class Analysis:
    numbers: dict  # JSON-friendly results (see ``analyze_probs``)
    label: np.ndarray  # uint8 (H, W): 0 ignore, 1 healthy, 2 damaged
    focus: FocusResult
    specks: SpeckResult


def label_from_probs(probs: np.ndarray, damage_threshold: float, pp: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tua, damage, ratio) from class probabilities, with clean-up."""
    p_ign, p_h, p_d = probs[0], probs[1], probs[2]
    tua = p_ign < 0.5
    tua = fill_small_holes(remove_small(tua, int(pp.get("min_tua_area", 0))), int(pp.get("min_tua_hole", 0)))
    ratio = p_d / (p_h + p_d + 1e-6)
    damage = tua & (ratio >= damage_threshold)
    damage = remove_small(damage, int(pp.get("min_damage_area", 0)))
    damage = fill_small_holes(damage, int(pp.get("min_damage_hole", 0))) & tua
    return tua, damage, ratio


def make_flags(n: dict, flag_cfg: dict, speck_cfg: dict) -> list[dict]:
    flags = []
    if n["tua_px"] == 0:
        flags.append({"code": "NO_USABLE_AREA", "text": "No usable area found"})
    elif n["tua_pct_of_image"] < float(flag_cfg["min_tua_pct"]):
        flags.append({"code": "SMALL_USABLE_AREA", "text": f"Usable area is only {n['tua_pct_of_image']:.0f}% of the image"})
    b = n.get("blurry_in_tua_pct")
    if b is not None and not math.isnan(b) and b > float(flag_cfg["max_blurry_in_tua_pct"]):
        flags.append({"code": "BLURRY", "text": f"{b:.0f}% of the usable area looks blurry"})
    u = n.get("uncertain_pct")
    if u is not None and not math.isnan(u) and u > float(flag_cfg["max_uncertain_pct"]):
        flags.append({"code": "UNSURE", "text": f"Model unsure on {u:.0f}% of the usable area"})
    if n["specks"] >= int(speck_cfg["flag_count"]):
        flags.append({"code": "SPECKS", "text": f"{n['specks']} black specks (possible balled-up nuclei)"})
    return flags


def analyze_probs(img: np.ndarray, probs: np.ndarray, meta: dict, settings: dict) -> Analysis:
    H, W = img.shape[:2]
    thr = float(meta.get("damage_threshold", 0.5))
    tua, damage, ratio = label_from_probs(probs, thr, settings["postprocess"])
    label = np.full((H, W), IGNORE, dtype=np.uint8)
    label[tua] = HEALTHY
    label[damage] = DAMAGED

    tua_px = int(tua.sum())
    damage_px = int(damage.sum())
    p_ign = probs[0]
    unsure = (tua & (np.abs(ratio - 0.5) < UNSURE_BAND)) | (np.abs(p_ign - 0.5) < UNSURE_BAND)
    hist = np.histogram(ratio[tua], bins=100, range=(0.0, 1.0))[0] if tua_px else np.zeros(100, int)

    focus = compute_focus(img, settings["focus"])
    specks = detect_specks(img, settings["specks"], region=tua if tua_px else None,
                           dark_level=float(settings["focus"]["dark_level"]))

    numbers = {
        "width": W,
        "height": H,
        "tua_px": tua_px,
        "damage_px": damage_px,
        "pct_damage": (100.0 * damage_px / tua_px) if tua_px else float("nan"),
        "tua_pct_of_image": 100.0 * tua_px / (H * W),
        "in_focus_pct": focus.in_focus_pct,
        "blurry_in_tua_pct": focus.blurry_pct_within(tua) if tua_px else float("nan"),
        "uncertain_pct": 100.0 * float(unsure.sum()) / max(tua_px, 1),
        "specks": specks.count,
        "specks_per_mpx": specks.count / max(tua_px / 1e6, 1e-6) if tua_px else 0.0,
        "damage_threshold": thr,
        # Histogram of P(damage | usable) inside the usable area, used to
        # re-compute % damage at other thresholds (calibration) without
        # re-running the model.
        "ratio_hist": hist.astype(int).tolist(),
    }
    numbers["flags"] = make_flags(numbers, settings["flags"], settings["specks"])
    return Analysis(numbers=numbers, label=label, focus=focus, specks=specks)


def analyze_image(img: np.ndarray, model, meta: dict, settings: dict, tta: bool = False) -> Analysis:
    from .model import predict_probs

    t0 = time.time()
    probs = predict_probs(model, img, meta, tta=tta)
    an = analyze_probs(img, probs, meta, settings)
    an.numbers["seconds"] = round(time.time() - t0, 2)
    return an


def pct_at_threshold(ratio_hist: list[int], tua_px: int, threshold: float) -> float:
    """% damage that a different damage threshold would have given (approximate:
    ignores the small-island clean-up)."""
    if not tua_px:
        return float("nan")
    k = int(round(threshold * 100))
    return 100.0 * float(sum(ratio_hist[k:])) / tua_px
