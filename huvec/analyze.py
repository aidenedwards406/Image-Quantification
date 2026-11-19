"""Batch analysis job: ``python -m huvec.analyze --job-dir DIR``.

Runs the batch's model on every image that has no result yet, so a batch
interrupted by a shutdown can simply be resumed.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from . import batches, registry
from .imageio import read_image
from .jobs import Job, run_worker
from .model import configure_threads
from .overlays import render_all
from .predict import analyze_image
from .utils import now_iso, write_json


def process_one(img_path: str, stem: str, bdir: Path, model, meta: dict, settings: dict, tta: bool) -> dict:
    img = read_image(img_path)
    an = analyze_image(img, model, meta, settings, tta=tta)
    render_all(img, an.label, an.focus.pixel_map(), an.specks.centroids, bdir / "renders" / stem)
    (bdir / "masks").mkdir(exist_ok=True)
    Image.fromarray(an.label).save(bdir / "masks" / f"{stem}.png", optimize=True)
    res = dict(an.numbers)
    res["speck_xy"] = [[round(x, 1), round(y, 1)] for x, y in an.specks.centroids]
    return res


def main(job: Job) -> None:
    batch_id = job.args["batch_id"]
    bdir = batches.batch_dir(batch_id)
    b = batches.load(batch_id)
    configure_threads()
    job.update(force=True, message="Loading the model...", done=0, total=len(b["images"]))
    model, meta = registry.load(b["model_id"])
    settings = b["settings"]
    (bdir / "results").mkdir(exist_ok=True)
    todo = [im for im in b["images"] if not (bdir / "results" / f"{im['stem']}.json").exists()]
    done0 = len(b["images"]) - len(todo)
    t0 = time.time()
    for i, im in enumerate(todo):
        if job.should_stop():
            job.log("Stopped by user.")
            break
        job.update(message=f"Analysing {im['name']}", current=im["name"], done=done0 + i, total=len(b["images"]),
                   eta_s=((time.time() - t0) / i * (len(todo) - i)) if i else None)
        try:
            res = process_one(im["path"], im["stem"], bdir, model, meta, settings, bool(b.get("tta")))
        except Exception as exc:
            job.log(f"Could not analyse {im['name']}: {exc}")
            res = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:],
                   "flags": [{"code": "ERROR", "text": f"Could not be analysed: {exc}"}],
                   "pct_damage": float("nan"), "tua_px": 0, "damage_px": 0}
        res.update(name=im["name"], stem=im["stem"], model_id=b["model_id"], analysed=now_iso())
        write_json(bdir / "results" / f"{im['stem']}.json", res)
    job.finish("stopped" if job.should_stop() else "done", done=len(batches.results(batch_id)),
               total=len(b["images"]), message="Analysis finished.", eta_s=0)


if __name__ == "__main__":
    run_worker(main)
