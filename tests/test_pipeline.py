"""End to end on tiny synthetic data: train -> analyze -> QC -> Excel -> benchmark.

Also checks that the exported workbook is what the 3D Vessel Model viewer
expects (its sheet/column auto-detection is re-implemented below).
"""

import math
import re
import time

import numpy as np
import openpyxl
import pytest
from PIL import Image

from huvec import batches, benchmark, excel, jobs, library, registry
from huvec.jobs import Job
from huvec.naming import parse_image_name
from huvec.utils import write_json


def make_job(ws, kind, args):
    d = ws / "jobs" / f"test_{kind}_{time.time_ns()}"
    d.mkdir(parents=True)
    write_json(d / "job.json", {"id": d.name, "kind": kind, "args": args})
    return Job(d)


# --- the 3D viewer's damage-sheet detection (src/lib/importers.ts), simplified port ---
DAMAGE_HEADER = re.compile(r"damage|injur|%|percent|pct|area|lesion", re.I)
PENALTY = re.compile(r"mag|section|position|row|#|count|n$", re.I)


def viewer_pick(wb):
    best, best_hits = None, -1
    for ws in wb.worksheets:
        rows = [[c for c in r] for r in ws.iter_rows(values_only=True)]
        cols = max((len(r) for r in rows), default=0)
        hits = max((sum(1 for r in rows if c < len(r) and isinstance(r[c], str) and parse_image_name(r[c]).follows_convention)
                    for c in range(cols)), default=0)
        if hits > best_hits:
            best, best_hits = ws, hits
    rows = [list(r) for r in best.iter_rows(values_only=True)]
    header, data = rows[0], rows[1:]
    name_col = max(range(len(header)), key=lambda c: sum(1 for r in data if isinstance(r[c], str) and parse_image_name(r[c]).follows_convention))
    best_col, best_score = None, -9
    for c in range(len(header)):
        if c == name_col:
            continue
        vals = [r[c] for r in data if r[c] not in (None, "")]
        if not vals:
            continue
        nums = [v for v in vals if isinstance(v, (int, float))]
        if len(nums) / len(vals) < 0.6:
            continue
        score = sum(1 for v in nums if 0 <= v <= 100) / len(vals)
        if DAMAGE_HEADER.search(str(header[c] or "")):
            score += 2
        if PENALTY.search(str(header[c] or "")):
            score -= 1
        if score > best_score:
            best_col, best_score = c, score
    return best.title, header[name_col], header[best_col]



def test_end_to_end(ws):
    from huvec import analyze, train
    from huvec.synthetic import make_dataset

    data = ws / "data"
    make_dataset(data, n_vessels=3, per_vessel=3, size=(260, 348), seed=1)
    sid = library.add_set("tiny", str(data), added_by="test")
    scan = library.scan_set(sid)
    assert scan["summary"]["images"] == 9 and scan["summary"]["errors"] == 0
    vessels = sorted(scan["summary"]["vessels"])

    # --- train (in-process, a few steps) ---
    job = make_job(ws, "train", {"name": "tiny model", "trained_by": "pytest", "set_ids": [sid], "mode": "split",
                                 "roles": {vessels[0]: "train", vessels[1]: "val", vessels[2]: "test"}, "preset": "quick",
                                 "overrides": {"steps": 8, "eval_every": 4, "tile": 128, "batch": 2, "pretrained": False}})
    train.main(job)
    models = registry.list_models()
    assert len(models) == 1
    mid = models[0]["id"]
    assert registry.active_id() == mid  # first model becomes active
    ev = registry.card(mid)["evaluation"]
    assert len(ev["per_image"]) == 3 and ev["agreement"]["n"] == 3

    # --- analyze a batch through the real background-process path ---
    imgs = batches.images_in_folder(data)[:4]
    bid = batches.create_batch("pytest batch", "tester", "", imgs, {"type": "folder", "folder": str(data)}, mid)
    jid = jobs.start_job("analyze", {"batch_id": bid}, "pytest batch")
    t0 = time.time()
    while jobs.status(jid)["status"] in jobs.ACTIVE and time.time() - t0 < 300:
        time.sleep(1)
    assert jobs.status(jid)["status"] == "done", jobs.log_tail(jid)
    res = batches.results(bid)
    assert len(res) == 4
    for stem, r in res.items():
        assert r["tua_px"] > 0 and 0 <= r["pct_damage"] <= 100
        assert len(r["ratio_hist"]) == 100 and sum(r["ratio_hist"]) <= r["tua_px"]
        assert batches.render_path(bid, stem, "thumb_damage").exists()

    # --- QC decisions ---
    stems = sorted(res)
    batches.set_decision(bid, stems[0], batches.APPROVED, by="tester")
    batches.set_decision(bid, stems[1], batches.APPROVED, by="tester")
    batches.set_decision(bid, stems[2], batches.REJECTED, ["Missed damage"], "edge", by="tester")
    s = batches.summary(bid)
    assert (s["approved"], s["rejected"], s["to_review"]) == (2, 1, 1)

    # --- Excel ---
    out = excel.export_batch(bid, ws / "out.xlsx")
    wb = openpyxl.load_workbook(out, data_only=True)
    assert wb.sheetnames == ["Approved", "Needs tracing", "All images", "Per vessel", "Details", "About"]
    ap = list(wb["Approved"].iter_rows(values_only=True))
    assert ap[0] == ("Image name", "Total usable area (px)", "% damage")
    names = {res[x]["name"]: res[x] for x in stems[:2]}
    for name, tua, pct in ap[1:]:
        assert tua == names[name]["tua_px"]
        assert pct == pytest.approx(names[name]["pct_damage"])
    tr = list(wb["Needs tracing"].iter_rows(values_only=True))
    assert tr[0] == ("Image name", "Total usable area (px)", "% damage")
    assert len(tr) == 2 and tr[1][0] == res[stems[2]]["name"] and tr[1][1] is None and tr[1][2] is None
    # All images: cached values present for approved rows, blank otherwise.
    allrows = {r[0]: r for r in wb["All images"].iter_rows(min_row=2, values_only=True)}
    assert len(allrows) == 4
    for name, r in allrows.items():
        if name in names:
            assert r[8] == pytest.approx(names[name]["pct_damage"]) and r[6] == "Model (QC approved)"
        else:
            assert r[8] in (None, "")
    # Formulas (not just values) are there, so Excel recalculates as students type.
    wbf = openpyxl.load_workbook(out)
    assert str(wbf["All images"]["I2"].value).startswith("=IFERROR(INDEX('Approved'!$C:$C")
    # Heat map on the % damage column + validation on the tracing sheet.
    cf = wbf["Approved"].conditional_formatting
    assert any("C2:C3" in str(rng.sqref) for rng in cf)
    assert len(wbf["Needs tracing"].data_validations.dataValidation) == 2
    # Thumbnails embedded in Details (3 per analysed image that isn't an error).
    assert len(wbf["Details"]._images) == 12
    # The 3D Vessel Model viewer picks the All images sheet and its % damage column.
    sheet, name_col, dmg_col = viewer_pick(wb)
    assert (sheet, name_col, dmg_col) == ("All images", "Image name", "% damage")

    # --- damage outlines for the 3D viewer ---
    files = excel.export_traces(bid, ws / "traces")
    assert len(files) == 2 and files[0].name.endswith("_trace.png")
    im = Image.open(files[0])
    assert im.mode == "RGBA" and im.size == (348, 260)

    # --- benchmark against "hand-measured" % ---
    rows = [r for r in scan["rows"] if not r.get("orphan_roi")]
    matched = [{"name": r["name"], "stem": r["stem"], "path": r["path"], "manual_pct": r["pct_damage"],
                "vessel": r["vessel"], "section": r["section"]} for r in rows]
    bench_id = benchmark.create("pytest bench", str(data), matched, "tester")
    benchmark.main(make_job(ws, "benchmark", {"bench_id": bench_id, "model_id": mid}))
    br = benchmark.results(bench_id, mid)
    assert br["complete"] and br["summary"]["overall"]["n"] == 9
    assert set(br["summary"]["by_vessel"]) == set(vessels)
    assert br["calibration"]["best_threshold"] >= 0.2
    assert registry.card(mid)["benchmarks"][bench_id]["leakage"]  # trained on these vessels -> warned
    cal_id = benchmark.make_calibrated_model(mid, 0.4, "tester", "pytest bench")
    assert registry.card(cal_id)["damage_threshold"] == 0.4
    _, meta = registry.load(cal_id)
    assert meta["damage_threshold"] == 0.4


def test_table_matching_for_benchmarks(ws, tmp_path):
    import pandas as pd

    from huvec.synthetic import make_dataset

    make_dataset(tmp_path / "imgs", n_vessels=1, per_vessel=3, size=(64, 80), seed=2, with_rois=False)
    names = sorted(p.name for p in (tmp_path / "imgs").glob("*.tif"))
    df = pd.DataFrame({"File": [names[0], names[1].replace(".tif", ""), "missing_4x_1_T_P"], "% Damage": [0.1, 0.25, 0.3]})
    g_name, g_pct = benchmark.guess_columns(df)
    assert (g_name, g_pct) == ("File", "% Damage")
    m = benchmark.match_table(tmp_path / "imgs", df, "File", "% Damage")
    assert len(m["matched"]) == 2 and len(m["unmatched"]) == 1 and m["looks_like_fraction"]
    assert m["images_without_value"] == [names[2]]


def test_cross_validation_training(ws):
    """Leave-one-vessel-out: one fold per vessel + a final model on all vessels."""
    from huvec import train
    from huvec.synthetic import make_dataset

    data = ws / "cv"
    make_dataset(data, n_vessels=3, per_vessel=2, size=(200, 268), seed=5)
    sid = library.add_set("cv", str(data))
    library.scan_set(sid)
    job = make_job(ws, "train", {"name": "cv model", "set_ids": [sid], "mode": "cv", "roles": {}, "preset": "quick",
                                 "overrides": {"steps": 6, "eval_every": 3, "tile": 96, "batch": 2, "pretrained": False}})
    train.main(job)
    (mid,) = [m["id"] for m in registry.list_models()]
    ev = registry.card(mid)["evaluation"]
    assert ev["mode"] == "cv"
    assert len(ev["per_fold"]) == 3                      # every vessel tested once
    assert len(ev["per_image"]) == 6                     # every image predicted by a model that never saw its vessel
    assert {r["vessel"] for r in ev["per_image"]} == {f["test"][0] for f in ev["per_fold"]}
    assert sorted(registry.card(mid)["training_data"]["vessels"]["train"]) == sorted({r["vessel"] for r in ev["per_image"]})
    assert job.progress["status"] == "done" and job.progress["model_id"] == mid
