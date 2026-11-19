"""The QC'd results workbook.

Sheets (in this order):

1. **Approved**       QC "yes": Image name | Total usable area (px) | % damage
2. **Needs tracing**  QC "no":  same three columns, left blank (yellow) for
                      the student to fill in after tracing by hand in ImageJ.
3. **All images**     every image with its final numbers: the model's number
                      when approved, the hand-traced number once typed in.
                      Formulas look values up *by name*, so sorting the other
                      sheets can't mix them up.  This is the sheet the 3D
                      Vessel Model viewer picks up automatically.
4. **Per vessel**     image counts and mean % damage per vessel, plus a
                      Treatment column to fill in (the 3D viewer reads it via
                      the All images sheet).
5. **Details**        thumbnails (original, usable area, damage), pixel
                      counts, focus, specks, flags and the QC decision.
6. **About**          model, settings and how to use the file.

The "% damage" column is always last and carries a heat-map colour scale.
Formula cells are written with their current values (XlsxWriter) so programs
that read cached values - like the 3D viewer - see the numbers immediately.
"""

from __future__ import annotations

import math
from pathlib import Path

import xlsxwriter

from . import __version__, batches
from .naming import parse_image_name
from .utils import now_iso, pretty_time

HEAT = {"type": "3_color_scale", "min_type": "num", "min_value": 0, "min_color": "#63BE7B",
        "mid_type": "percentile", "mid_value": 50, "mid_color": "#FFEB84", "max_type": "max", "max_color": "#F8696B"}
S_APPROVED, S_TRACE, S_ALL, S_VESSEL, S_DETAILS, S_ABOUT = (
    "Approved", "Needs tracing", "All images", "Per vessel", "Details", "About")


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _q(sheet: str) -> str:
    return "'" + sheet.replace("'", "''") + "'"


def _lookup(sheet: str, col: str, row: int) -> str:
    """Value from ``sheet`` column ``col`` for the image named in A{row}; "" if blank/missing."""
    idx = f"INDEX({_q(sheet)}!${col}:${col},MATCH($A{row},{_q(sheet)}!$A:$A,0))"
    return f'IFERROR(IF({idx}="","",{idx}),"")'


def build_rows(batch_id: str) -> dict:
    b = batches.load(batch_id)
    res = batches.results(batch_id)
    rev = batches.review(batch_id)
    rows = []
    for im in sorted(b["images"], key=lambda r: r["name"].lower()):
        r = res.get(im["stem"])
        d = rev.get(im["stem"], {})
        decision = d.get("decision")
        if r is None:
            status = "not analysed"
        elif r.get("error"):
            status = batches.REJECTED  # could not be analysed: must be traced by hand
        else:
            status = decision or "not reviewed"
        info = parse_image_name(im["name"])
        rows.append({"im": im, "res": r or {}, "rev": d, "status": status, "info": info})
    return {"batch": b, "rows": rows}


def export_batch(batch_id: str, out_path: str | Path, include_images: bool = True) -> Path:
    data = build_rows(batch_id)
    b, rows = data["batch"], data["rows"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(out_path), {"nan_inf_to_errors": True})
    wb.set_calc_mode("auto")
    wb.set_properties({"title": f"HUVEC damage results - {b.get('name', '')}", "comments": f"HUVEC Damage Analyzer {__version__}"})

    hdr = wb.add_format({"bold": True, "bg_color": "#E8EEF4", "bottom": 1, "text_wrap": True, "valign": "top"})
    txt = wb.add_format({"valign": "vcenter"})
    px = wb.add_format({"num_format": "#,##0", "valign": "vcenter"})
    pct = wb.add_format({"num_format": "0.00", "valign": "vcenter"})
    todo_px = wb.add_format({"num_format": "#,##0", "bg_color": "#FFF7D6", "border": 1, "border_color": "#E6D8A3"})
    todo_pct = wb.add_format({"num_format": "0.00", "bg_color": "#FFF7D6", "border": 1, "border_color": "#E6D8A3"})
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    title = wb.add_format({"bold": True, "font_size": 16})
    bold = wb.add_format({"bold": True})

    approved = [r for r in rows if r["status"] == batches.APPROVED]
    to_trace = [r for r in rows if r["status"] == batches.REJECTED]

    # 1. Approved ---------------------------------------------------------------
    ws = wb.add_worksheet(S_APPROVED)
    ws.write_row(0, 0, ["Image name", "Total usable area (px)", "% damage"], hdr)
    for i, r in enumerate(approved, start=1):
        ws.write_string(i, 0, r["im"]["name"], txt)
        ws.write_number(i, 1, int(r["res"].get("tua_px", 0)), px)
        v = _num(r["res"].get("pct_damage"))
        ws.write_number(i, 2, v, pct) if v is not None else ws.write_blank(i, 2, None, pct)
    _finish_simple(ws, len(approved))

    # 2. Needs tracing ---------------------------------------------------------------
    ws = wb.add_worksheet(S_TRACE)
    ws.write_row(0, 0, ["Image name", "Total usable area (px)", "% damage"], hdr)
    for i, r in enumerate(to_trace, start=1):
        ws.write_string(i, 0, r["im"]["name"], txt)
        ws.write_blank(i, 1, None, todo_px)
        ws.write_blank(i, 2, None, todo_pct)
    n = len(to_trace)
    if n:
        ws.data_validation(1, 1, n, 1, {"validate": "integer", "criteria": ">=", "value": 0,
                                        "input_title": "Usable area (px)",
                                        "input_message": "Pixel area of the TUA you traced (ImageJ: Measure > Area in pixels).",
                                        "error_title": "Not a pixel count", "error_message": "Enter a whole number of pixels."})
        ws.data_validation(1, 2, n, 2, {"validate": "decimal", "criteria": "between", "minimum": 0, "maximum": 100,
                                        "input_title": "% damage", "input_message": "Hand-traced % damage, 0-100 (e.g. 12.5).",
                                        "error_title": "Out of range", "error_message": "Enter a percentage between 0 and 100 (e.g. 12.5, not 0.125)."})
    _finish_simple(ws, n)

    # 3. All images (formulas with cached values) ---------------------------------
    ws = wb.add_worksheet(S_ALL)
    # "TUA (px)" rather than "... area ...": the 3D viewer's column guesser gives
    # headers containing "area" the same score as "% damage" (whose "damage"
    # contains "mag" and gets penalised as if it were a magnification column).
    cols = ["Image name", "Vessel", "Treatment", "Section", "Side", "Position", "Result from",
            "TUA (px)", "% damage"]
    ws.write_row(0, 0, cols, hdr)
    vessels = sorted({r["info"].vessel for r in rows if r["info"].vessel})
    for i, r in enumerate(rows, start=1):
        x = i + 1  # Excel row number
        info = r["info"]
        ws.write_string(i, 0, r["im"]["name"], txt)
        ws.write_string(i, 1, info.vessel or "", txt)
        tr = f"INDEX({_q(S_VESSEL)}!$B:$B,MATCH($B{x},{_q(S_VESSEL)}!$A:$A,0))"
        ws.write_formula(i, 2, f'=IFERROR(IF({tr}="","",{tr}),"")', txt, "")
        if info.section:
            ws.write_number(i, 3, int(info.section), txt)
        ws.write_string(i, 4, {"T": "Top", "B": "Bottom"}.get(info.side, info.side or ""), txt)
        ws.write_string(i, 5, info.subsection or "", txt)
        is_ok = r["status"] == batches.APPROVED
        tua_v = int(r["res"].get("tua_px", 0)) if is_ok else ""
        pct_v = _num(r["res"].get("pct_damage")) if is_ok else ""
        pct_v = "" if pct_v is None else pct_v
        src_v = {"approved": "Model (QC approved)", "rejected": "Waiting for hand tracing"}.get(r["status"], "Not reviewed")
        ws.write_formula(i, 6, (f'=IF(ISNUMBER(MATCH($A{x},{_q(S_APPROVED)}!$A:$A,0)),"Model (QC approved)",'
                                f'IF(ISNUMBER(MATCH($A{x},{_q(S_TRACE)}!$A:$A,0)),IF($I{x}="","Waiting for hand tracing","Hand traced"),"Not reviewed"))'),
                         txt, src_v)
        ws.write_formula(i, 7, f"=IFERROR(INDEX({_q(S_APPROVED)}!$B:$B,MATCH($A{x},{_q(S_APPROVED)}!$A:$A,0)),{_lookup(S_TRACE, 'B', x)})", px, tua_v)
        ws.write_formula(i, 8, f"=IFERROR(INDEX({_q(S_APPROVED)}!$C:$C,MATCH($A{x},{_q(S_APPROVED)}!$A:$A,0)),{_lookup(S_TRACE, 'C', x)})", pct, pct_v)
    ws.set_column(0, 0, 38)
    ws.set_column(1, 2, 16)
    ws.set_column(3, 5, 9)
    ws.set_column(6, 6, 24)
    ws.set_column(7, 7, 14)
    ws.set_column(8, 8, 11)
    ws.freeze_panes(1, 1)
    if rows:
        ws.autofilter(0, 0, len(rows), len(cols) - 1)
        ws.conditional_format(1, 8, len(rows), 8, HEAT)

    # 4. Per vessel ---------------------------------------------------------------
    ws = wb.add_worksheet(S_VESSEL)
    ws.write_row(0, 0, ["Vessel", "Treatment (type in)", "Images", "Images with a result", "Mean % damage"], hdr)
    for i, v in enumerate(vessels, start=1):
        x = i + 1
        vr = [r for r in rows if r["info"].vessel == v]
        with_res = [r for r in vr if r["status"] == batches.APPROVED and _num(r["res"].get("pct_damage")) is not None]
        mean_v = (sum(_num(r["res"]["pct_damage"]) for r in with_res) / len(with_res)) if with_res else ""
        ws.write_string(i, 0, v, txt)
        ws.write_blank(i, 1, None, wb.add_format({"bg_color": "#FFF7D6", "border": 1, "border_color": "#E6D8A3"}))
        allq = _q(S_ALL)
        ws.write_formula(i, 2, f"=COUNTIF({allq}!$B:$B,$A{x})", txt, len(vr))
        ws.write_formula(i, 3, f'=COUNTIFS({allq}!$B:$B,$A{x},{allq}!$I:$I,">=0")', txt, len(with_res))
        ws.write_formula(i, 4, f'=IFERROR(AVERAGEIFS({allq}!$I:$I,{allq}!$B:$B,$A{x},{allq}!$I:$I,">=0"),"")', pct, mean_v)
    ws.set_column(0, 0, 18)
    ws.set_column(1, 1, 22)
    ws.set_column(2, 4, 16)
    ws.freeze_panes(1, 0)
    if vessels:
        ws.conditional_format(1, 4, len(vessels), 4, HEAT)

    # 5. Details ---------------------------------------------------------------
    ws = wb.add_worksheet(S_DETAILS)
    dcols = ["Image name", "Original", "Usable area (TUA)", "Damage", "Damage (px)", "Total usable area (px)",
             "% damage (model)", "In focus (% of image)", "Blurry inside usable area (%)", "Specks",
             "Model unsure (% of usable area)", "Flags", "QC decision", "Reason(s)", "Note", "Reviewed by",
             "Reviewed at", "Vessel", "Section", "Side", "Position", "Analysed"]
    ws.write_row(0, 0, dcols, hdr)
    thumb_w, thumb_h = 0, 0
    for i, r in enumerate(rows, start=1):
        res, d = r["res"], r["rev"]
        ws.write_string(i, 0, r["im"]["name"], wrap)
        if include_images and res and not res.get("error"):
            for c, view in ((1, "thumb_original"), (2, "thumb_tua"), (3, "thumb_damage")):
                p = batches.render_path(batch_id, r["im"]["stem"], view)
                if p.exists():
                    if not thumb_w:
                        from PIL import Image

                        with Image.open(p) as im:
                            thumb_w, thumb_h = im.size
                    ws.insert_image(i, c, str(p), {"x_offset": 3, "y_offset": 3, "object_position": 1})
        for c, key, fmt in ((4, "damage_px", px), (5, "tua_px", px), (6, "pct_damage", pct), (7, "in_focus_pct", pct),
                            (8, "blurry_in_tua_pct", pct), (10, "uncertain_pct", pct)):
            v = _num(res.get(key))
            if v is not None:
                ws.write_number(i, c, v, fmt)
        if res:
            ws.write_number(i, 9, int(res.get("specks", 0) or 0), txt)
        ws.write_string(i, 11, "; ".join(f["text"] for f in res.get("flags", [])), wrap)
        decision = {"approved": "Yes - approved", "rejected": "No - needs tracing"}.get(r["status"], r["status"].capitalize())
        if res.get("error"):
            decision = "Could not be analysed - needs tracing"
        ws.write_string(i, 12, decision, wrap)
        ws.write_string(i, 13, ", ".join(d.get("reasons", [])), wrap)
        ws.write_string(i, 14, d.get("note", ""), wrap)
        ws.write_string(i, 15, d.get("by", ""), wrap)
        ws.write_string(i, 16, pretty_time(d.get("at")), wrap)
        info = r["info"]
        ws.write_string(i, 17, info.vessel or "", wrap)
        ws.write_string(i, 18, info.section or "", wrap)
        ws.write_string(i, 19, info.side or "", wrap)
        ws.write_string(i, 20, info.subsection or "", wrap)
        ws.write_string(i, 21, pretty_time(res.get("analysed")), wrap)
    if include_images and thumb_w:
        ws.set_column_pixels(1, 3, thumb_w + 8)
        for i in range(1, len(rows) + 1):
            ws.set_row_pixels(i, thumb_h + 8)
    ws.set_column(0, 0, 30)
    ws.set_column(4, 10, 13)
    ws.set_column(11, 11, 34)
    ws.set_column(12, 16, 16)
    ws.freeze_panes(1, 1)
    if rows:
        ws.autofilter(0, 0, len(rows), len(dcols) - 1)
        ws.conditional_format(1, 6, len(rows), 6, HEAT)

    # 6. About ---------------------------------------------------------------
    ws = wb.add_worksheet(S_ABOUT)
    ws.set_column(0, 0, 30)
    ws.set_column(1, 1, 100)
    ws.write(0, 0, "HUVEC damage results", title)
    card = b.get("model_card", {})
    ev = card.get("evaluation_summary") or {}
    lines = [
        ("Batch", b.get("name", "")),
        ("Analysed", f"{pretty_time(b.get('created'))} by {b.get('created_by') or '—'}"),
        ("Workbook created", pretty_time(now_iso())),
        ("Images", f"{len(rows)} total · {len(approved)} approved · {len(to_trace)} need tracing · "
                   f"{sum(1 for r in rows if r['status'] in ('not reviewed', 'not analysed'))} not reviewed"),
        ("Model", f"{b.get('model_name', '')} ({b.get('model_id', '')})"),
        ("Model trained", f"{pretty_time(card.get('created'))} by {card.get('trained_by') or '—'}"),
        ("Damage threshold", str(card.get("damage_threshold", 0.5))),
        ("App version", __version__),
        ("", ""),
        ("How % damage is defined", "Damaged usable area ÷ total usable area × 100. Usable area (TUA) excludes out-of-focus "
                                    "areas, areas outside the vessel and black corners. Damage = areas with no nuclei."),
        ("Approved", "QC reviewer agreed with the model. Numbers are the model's."),
        ("Needs tracing", "QC reviewer rejected the model's result (or it could not be analysed). Trace the image by hand "
                          "in ImageJ (TUA + D1, D2 ...), then type the usable area (px) and % damage into the yellow cells."),
        ("All images", "Final numbers for every image. Updates automatically as you fill in 'Needs tracing'. "
                       "Load this workbook into the 3D Vessel Model viewer (step 4: damage data) - it picks this sheet."),
        ("Per vessel", "Mean % damage per vessel. Type each vessel's treatment (e.g. Control) and the 3D viewer "
                       "will group vessels by it."),
        ("Details", "Thumbnails, focus, specks, QC flags and the reviewer's reasons."),
        ("", ""),
        ("Flags", "Blurry: more than {max_blurry_in_tua_pct}% of the usable area looks out of focus. "
                  "Small usable area: under {min_tua_pct}% of the image. Model unsure: over {max_uncertain_pct}% of "
                  "the usable area. Specks: {flag_count}+ small black specks.".format(
                      **b.get("settings", {}).get("flags", {}), **b.get("settings", {}).get("specks", {}))),
    ]
    if ev.get("agreement", {}).get("n"):
        a = ev["agreement"]
        lines.insert(7, ("Model test result", f"Off by {a['mae']:.1f} percentage points on average on {a['n']} held-out "
                                              f"images; {100 * a['within_tol']:.0f}% within ±{a['tolerance']:.0f} points "
                                              f"(bias {a['bias']:+.1f})."))
    for i, (k, v) in enumerate(lines, start=2):
        ws.write(i, 0, k, bold)
        ws.write(i, 1, v, wrap)

    wb.worksheets()[0].activate()
    wb.close()
    return out_path


def _finish_simple(ws, n: int) -> None:
    ws.set_column(0, 0, 40)
    ws.set_column(1, 1, 22)
    ws.set_column(2, 2, 12)
    ws.freeze_panes(1, 0)
    if n:
        ws.autofilter(0, 0, n, 2)
        ws.conditional_format(1, 2, n, 2, HEAT)


def export_traces(batch_id: str, out_dir: str | Path) -> list[Path]:
    """``<image>_trace.png`` damage outlines (approved images only) for the 3D viewer."""
    import numpy as np
    from PIL import Image

    from .overlays import trace_png

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rev = batches.review(batch_id)
    written = []
    for stem, d in rev.items():
        if d.get("decision") != batches.APPROVED:
            continue
        mask = batches.batch_dir(batch_id) / "masks" / f"{stem}.png"
        if mask.exists():
            label = np.asarray(Image.open(mask))
            p = out_dir / f"{stem}_trace.png"
            trace_png(label).save(p, optimize=True)
            written.append(p)
    return written
