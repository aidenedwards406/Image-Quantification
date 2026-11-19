"""Benchmarks: images that only have a hand-measured % damage (no ROIs).

The lab's ~500 images traced for % damage but without saved ROIs can't
teach the model *where* damage is, but they are the best possible test of
the lab's real goal: does the model's % match the hand-measured %?

* Run any model on a benchmark -> agreement statistics, by section and vessel.
* Compare models on the same benchmark before activating a new one.
* "Trace next" list: the images the model gets most wrong.  Tracing those
  with ROIs and adding them to the training data helps the model most.
* Threshold calibration: tracers differ in how generously they outline
  damage.  The damage threshold can be tuned so the model's % matches the
  lab's style.  The honest error is estimated by calibrating on half the
  vessels and testing on the other half.

Job entry: ``python -m huvec.benchmark --job-dir DIR``.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np

from . import registry, workspace
from . import settings as settings_mod
from .imageio import read_image, save_jpeg
from .jobs import Job, run_worker
from .metrics import agreement
from .model import configure_threads, predict_probs
from .naming import image_key, list_images, parse_image_name
from .overlays import damage_overlay
from .predict import analyze_probs, pct_at_threshold
from .utils import now_iso, read_json, slugify, stamp, write_json

NAME_HINT = ("image", "file", "name", "picture")
PCT_HINT = ("damage", "%", "percent", "pct", "injur")


# ── creating a benchmark ───────────────────────────────────────────────────

def read_table(data: bytes, filename: str, sheet: str | int | None = 0):
    import io

    import pandas as pd

    if filename.lower().endswith((".csv", ".txt", ".tsv")):
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet if sheet is not None else 0)


def sheet_names(data: bytes, filename: str) -> list[str]:
    import io

    import pandas as pd

    if filename.lower().endswith((".csv", ".txt", ".tsv")):
        return []
    return list(pd.ExcelFile(io.BytesIO(data)).sheet_names)


def guess_columns(df) -> tuple[str | None, str | None]:
    name_col, best = None, -1
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(300)
        hits = sum(1 for v in vals if parse_image_name(v).vessel and "_" in v)
        if any(h in str(c).lower() for h in NAME_HINT):
            hits += 0.5
        if hits > best:
            name_col, best = c, hits
    pct_col, best = None, -1.0
    for c in df.columns:
        if c == name_col:
            continue
        nums = _to_number(df[c])
        filled = df[c].notna().sum()
        if not filled:
            continue
        frac = nums.notna().sum() / filled
        if frac < 0.6:
            continue
        score = float(((nums >= 0) & (nums <= 100)).mean())
        if any(h in str(c).lower() for h in PCT_HINT):
            score += 2
        if score > best:
            pct_col, best = c, score
    return name_col, pct_col


def _to_number(series):
    import pandas as pd

    s = series.astype(str).str.strip().str.rstrip("%").str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def match_table(folder: str | Path, df, name_col, pct_col) -> dict:
    images = {image_key(p.name): p for p in list_images(Path(folder))}
    nums = _to_number(df[pct_col])
    matched, unmatched, seen = [], [], set()
    for name, val in zip(df[name_col], nums):
        if name is None or (isinstance(name, float) and math.isnan(name)):
            continue
        key = image_key(str(name).strip())
        p = images.get(key)
        if p is None or val is None or (isinstance(val, float) and math.isnan(val)):
            unmatched.append({"name": str(name), "value": None if val is None or math.isnan(val) else float(val),
                              "reason": "no image with this name in the folder" if p is None else "no % value"})
            continue
        if key in seen:
            continue
        seen.add(key)
        info = parse_image_name(p.name)
        matched.append({"name": p.name, "stem": info.stem, "path": str(p), "manual_pct": float(val),
                        "vessel": info.vessel, "section": info.section, "side": info.side, "position": info.subsection})
    vals = [m["manual_pct"] for m in matched]
    looks_fraction = bool(vals) and max(vals) <= 1.0 and any(v > 0 for v in vals)
    images_without = sorted(p.name for k, p in images.items() if k not in seen)
    return {"matched": matched, "unmatched": unmatched, "images_without_value": images_without,
            "looks_like_fraction": looks_fraction}


def create(name: str, folder: str, matched: list[dict], by: str = "", notes: str = "", table_name: str = "",
           multiply_by_100: bool = False) -> str:
    bench_id = f"{stamp()}_{slugify(name, 40)}"
    if multiply_by_100:
        matched = [dict(m, manual_pct=m["manual_pct"] * 100) for m in matched]
    write_json(_dir(bench_id) / "benchmark.json", {
        "id": bench_id, "name": name, "folder": str(folder), "created": now_iso(), "created_by": by,
        "notes": notes, "table": table_name, "images": matched})
    return bench_id


def _dir(bench_id: str) -> Path:
    d = workspace.benchmarks_dir() / bench_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_all() -> list[dict]:
    out = []
    for d in sorted(workspace.benchmarks_dir().iterdir(), reverse=True):
        b = read_json(d / "benchmark.json", None)
        if b:
            b["models_run"] = sorted(p.stem for p in (d / "results").glob("*.json")) if (d / "results").exists() else []
            out.append(b)
    return out


def load(bench_id: str) -> dict:
    return read_json(_dir(bench_id) / "benchmark.json", {}) or {}


def delete(bench_id: str) -> None:
    shutil.rmtree(workspace.benchmarks_dir() / bench_id, ignore_errors=True)


def results(bench_id: str, model_id: str) -> dict | None:
    return read_json(_dir(bench_id) / "results" / f"{model_id}.json", None)


def overlay_path(bench_id: str, model_id: str, stem: str) -> Path:
    return _dir(bench_id) / "renders" / model_id / f"{stem}.jpg"


def leakage(bench: dict, model_card: dict) -> list[str]:
    """Benchmark vessels that the model was trained on (makes scores optimistic)."""
    tv = model_card.get("training_data", {}).get("vessels", {})
    trained = set(tv.get("train", [])) | set(tv.get("val", []))
    return sorted({im["vessel"] for im in bench.get("images", [])} & trained)


# ── analysis of results ────────────────────────────────────────────────────

def summarize(rows: list[dict], key: str = "pred_pct") -> dict:
    ok = [r for r in rows if r.get(key) is not None and not math.isnan(r[key])]
    out = {"overall": agreement([r[key] for r in ok], [r["manual_pct"] for r in ok])}
    for group in ("section", "vessel"):
        g = {}
        for r in ok:
            g.setdefault(r.get(group) or "?", []).append(r)
        out[f"by_{group}"] = {k: agreement([r[key] for r in v], [r["manual_pct"] for r in v]) for k, v in sorted(g.items())}
    return out


def calibrate(rows: list[dict], current: float = 0.5) -> dict:
    """Best damage threshold for this benchmark + an honest (vessel-split) error estimate."""
    grid = np.round(np.arange(0.20, 0.81, 0.01), 2)
    ok = [r for r in rows if r.get("tua_px")]
    if len(ok) < 5:
        return {}

    def mae_at(t, subset):
        d = [abs(pct_at_threshold(r["ratio_hist"], r["tua_px"], t) - r["manual_pct"]) for r in subset]
        return float(np.mean(d)) if d else float("nan")

    curve = [mae_at(t, ok) for t in grid]
    best_i = int(np.nanargmin(curve))
    vessels = sorted({r.get("vessel", "") for r in ok})
    honest = None
    if len(vessels) >= 2:
        halves = [set(vessels[0::2]), set(vessels[1::2])]
        errs = []
        for k in (0, 1):
            fit = [r for r in ok if r.get("vessel", "") in halves[k]]
            test = [r for r in ok if r.get("vessel", "") in halves[1 - k]]
            if fit and test:
                t_fit = grid[int(np.nanargmin([mae_at(t, fit) for t in grid]))]
                errs += [abs(pct_at_threshold(r["ratio_hist"], r["tua_px"], t_fit) - r["manual_pct"]) for r in test]
        honest = float(np.mean(errs)) if errs else None
    return {"grid": grid.tolist(), "mae": curve, "best_threshold": float(grid[best_i]), "best_mae": curve[best_i],
            "current_threshold": current, "current_mae": mae_at(current, ok), "honest_mae": honest,
            "n": len(ok)}


def make_calibrated_model(model_id: str, threshold: float, by: str, bench_name: str) -> str:
    c = registry.card(model_id)
    new = dict(c)
    new["name"] = f"{c.get('name', model_id)} · calibrated {threshold:.2f}"
    new["id"] = registry.new_id(new["name"])
    new["damage_threshold"] = float(threshold)
    new["calibrated_from"] = model_id
    new["calibration"] = {"benchmark": bench_name, "threshold": float(threshold), "by": by, "at": now_iso()}
    new.pop("benchmarks", None)
    new.pop("registered", None)
    return registry.register(registry.model_path(model_id), new)


# ── job ────────────────────────────────────────────────────────────────────

def main(job: Job) -> None:
    bench_id, model_id = job.args["bench_id"], job.args["model_id"]
    bench = load(bench_id)
    configure_threads()
    model, meta = registry.load(model_id)
    settings = settings_mod.load()
    rows = []
    n = len(bench["images"])
    out_dir = _dir(bench_id)
    for i, im in enumerate(bench["images"]):
        if job.should_stop():
            break
        job.update(message=f"Analysing {im['name']}", done=i, total=n)
        row = dict(im)
        try:
            img = read_image(im["path"])
            an = analyze_probs(img, predict_probs(model, img, meta), meta, settings)
            row.update(pred_pct=an.numbers["pct_damage"], tua_px=an.numbers["tua_px"],
                       ratio_hist=an.numbers["ratio_hist"], flags=[f["code"] for f in an.numbers["flags"]])
            save_jpeg(damage_overlay(img, an.label), overlay_path(bench_id, model_id, im["stem"]), max_side=900, quality=82)
        except Exception as exc:
            row.update(pred_pct=float("nan"), error=str(exc))
        rows.append(row)
    summary = summarize(rows)
    cal = calibrate(rows, float(meta.get("damage_threshold", 0.5)))
    write_json(out_dir / "results" / f"{model_id}.json", {
        "bench_id": bench_id, "model_id": model_id, "run": now_iso(), "rows": rows, "summary": summary,
        "calibration": cal, "complete": len(rows) == n})
    c = registry.card(model_id)
    a = summary["overall"]
    c.setdefault("benchmarks", {})[bench_id] = {
        "name": bench["name"], "n": a.get("n", 0), "mae": a.get("mae"), "bias": a.get("bias"),
        "within_tol": a.get("within_tol"), "ccc": a.get("ccc"), "run": now_iso(),
        "leakage": leakage(bench, c), "complete": len(rows) == n}
    registry.save_card(model_id, c)
    job.finish("stopped" if job.should_stop() else "done", done=len(rows), total=n,
               message=f"Done: mean error {a.get('mae', float('nan')):.2f} points on {a.get('n', 0)} images.")


if __name__ == "__main__":
    run_worker(main)
