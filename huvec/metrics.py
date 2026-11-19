"""How well model numbers agree with hand tracing.

Two kinds of agreement:

* **Overlap** (Dice / IoU): do the outlines match pixel for pixel?
* **% damage agreement**: the lab's real target ("within about +/-5 percentage
  points of the manual number").  Reported as mean absolute error, bias
  (does the model systematically over/under-call?), Bland-Altman limits of
  agreement, share of images within +/-5 points, and Lin's concordance
  correlation coefficient (CCC; 1 = perfect agreement).
"""

from __future__ import annotations

import math

import numpy as np

from . import DAMAGED, IGNORE


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else 2.0 * int((a & b).sum()) / s


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return 1.0 if u == 0 else int((a & b).sum()) / u


def overlap_counts(pred_label: np.ndarray, gt_label: np.ndarray) -> dict:
    """Pixel counts for pooled (whole-dataset) Dice."""
    pd, gd = pred_label == DAMAGED, gt_label == DAMAGED
    pt, gt = pred_label != IGNORE, gt_label != IGNORE
    return {"dmg_inter": int((pd & gd).sum()), "dmg_pred": int(pd.sum()), "dmg_gt": int(gd.sum()),
            "tua_inter": int((pt & gt).sum()), "tua_pred": int(pt.sum()), "tua_gt": int(gt.sum())}


def image_scores(pred_label: np.ndarray, gt_label: np.ndarray) -> dict:
    pd, gd = pred_label == DAMAGED, gt_label == DAMAGED
    pt, gt = pred_label != IGNORE, gt_label != IGNORE
    tp, tg = int(pt.sum()), int(gt.sum())
    return {
        "dice_damage": dice(pd, gd), "iou_damage": iou(pd, gd),
        "dice_tua": dice(pt, gt), "iou_tua": iou(pt, gt),
        "pct_pred": 100.0 * int(pd.sum()) / tp if tp else float("nan"),
        "pct_true": 100.0 * int(gd.sum()) / tg if tg else float("nan"),
        "tua_px_pred": tp, "tua_px_true": tg,
        **overlap_counts(pred_label, gt_label),
    }


def pooled_dice(rows: list[dict]) -> dict:
    s = {k: sum(r.get(k, 0) for r in rows) for k in ("dmg_inter", "dmg_pred", "dmg_gt", "tua_inter", "tua_pred", "tua_gt")}
    dd = s["dmg_pred"] + s["dmg_gt"]
    dt = s["tua_pred"] + s["tua_gt"]
    return {"dice_damage_pooled": (2 * s["dmg_inter"] / dd) if dd else float("nan"),
            "dice_tua_pooled": (2 * s["tua_inter"] / dt) if dt else float("nan")}


def agreement(pred: list[float], true: list[float], tolerance: float = 5.0) -> dict:
    """Agreement between model and manual % damage (percentage points)."""
    p = np.asarray(pred, float)
    t = np.asarray(true, float)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    n = int(len(p))
    if n == 0:
        return {"n": 0}
    d = p - t
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    out = {
        "n": n,
        "mae": float(np.abs(d).mean()),
        "rmse": float(math.sqrt((d ** 2).mean())),
        "bias": float(d.mean()),
        "sd_diff": sd,
        "loa_low": float(d.mean() - 1.96 * sd),
        "loa_high": float(d.mean() + 1.96 * sd),
        "within_tol": float((np.abs(d) <= tolerance).mean()),
        "tolerance": tolerance,
        "max_abs": float(np.abs(d).max()),
        "mean_true": float(t.mean()),
        "mean_pred": float(p.mean()),
    }
    if n > 2 and t.std() > 0 and p.std() > 0:
        out["pearson_r"] = float(np.corrcoef(p, t)[0, 1])
        cov = float(((p - p.mean()) * (t - t.mean())).mean())
        out["ccc"] = float(2 * cov / (p.var() + t.var() + (p.mean() - t.mean()) ** 2))
    return out


def describe(a: dict) -> str:
    """One-line plain-language summary of ``agreement``."""
    if not a or not a.get("n"):
        return "No images to compare."
    s = (f"On {a['n']} images the model is off by {a['mae']:.1f} percentage points on average; "
         f"{100 * a['within_tol']:.0f}% of images are within ±{a['tolerance']:.0f} points.")
    if abs(a["bias"]) >= 0.5:
        s += f" It tends to {'over' if a['bias'] > 0 else 'under'}-call damage by {abs(a['bias']):.1f} points."
    return s
