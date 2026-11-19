import math
import os

import pandas as pd
import streamlit as st

from huvec import library
from huvec.utils import pretty_time
from ui import common

common.header("Training data",
              "Folders of traced images: TIFFs plus ImageJ ROI files (TUA, D1, D2 …). Each folder is checked against the "
              "naming and tracing rules before it's used for training.")

with st.expander("Tracing rules the app applies", icon=":material/rule:"):
    st.markdown("""
- **ROI file** shares the image's name: `X.tif.zip`, `X.roi`, `X_ROI.roi` or `X_ROI.roi.zip` are all recognised.
- **`TUA`** = total usable area (one or more ROIs, combined). **`D1`, `D2`, …** = damage (combined, then clipped to the TUA).
- **A single unnamed `.roi`** = damage only; the whole image counts as usable.
- **No ROI file** = whole image usable, **0 % damage** (only for images someone has checked).
- **% damage** = damaged usable area ÷ total usable area × 100, with pixels counted the way ImageJ counts them.
- Images are split into train/test **by vessel**, which is read from the name (`260919_BR-S24-9_4x_2_T_P` → `BR-S24-9`).
""")

with st.container(border=True):
    st.markdown("##### Add a folder of traced images")
    c1, c2 = st.columns(2)
    name = c1.text_input("Name", placeholder="e.g. Fall 2026 - 5 vessels")
    traced_by = c2.text_input("Traced by", placeholder="Who drew the ROIs")
    folder = common.folder_input("Folder with the TIFFs and ROI files", key="lib_folder")
    notes = st.text_input("Notes (optional)", key="lib_notes")
    if st.button("Add and check", type="primary", icon=":material/fact_check:",
                 disabled=not (name and common.path_exists(folder))):
        sid = library.add_set(name, folder, common.user(), traced_by, notes)
        bar = st.progress(0.0, text="Checking images…")
        library.scan_set(sid, progress=lambda i, n, f: bar.progress((i + 1) / max(n, 1), text=f"Checking {f}"))
        st.session_state["open_set"] = sid
        st.rerun()

sets = library.list_sets()
if not sets:
    st.caption("No training data yet.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=24)
def preview(path: str, rois: tuple, mtime: float):
    from huvec.imageio import read_image
    from huvec.overlays import damage_overlay, tua_overlay
    from huvec.rois import load_annotation

    img = read_image(path)
    ann = load_annotation(list(rois), img.shape[1], img.shape[0])
    lab = ann.label()
    return img, tua_overlay(img, lab), damage_overlay(img, lab), ann.tua_px, ann.damage_px, ann.pct_damage


STATUS = {"ok": "✅ OK", "warning": "⚠️ Check", "error": "⛔ Problem"}

for sset in sets:
    sid = sset["id"]
    sm = sset.get("summary", {})
    title = f"**{sset['name']}** · {sm.get('images', '?')} images · {len(sm.get('vessels', {}))} vessels"
    if sm.get("errors"):
        title += f" · :red[{sm['errors']} problem(s)]"
    elif sm.get("warnings"):
        title += f" · :orange[{sm['warnings']} to check]"
    with st.expander(title, expanded=st.session_state.get("open_set") == sid):
        st.caption(f"Folder: `{sset['folder']}` · added {pretty_time(sset.get('added'))} by {sset.get('added_by') or '—'} · "
                   f"traced by {sset.get('traced_by') or '—'}")
        scan = library.cached_scan(sid)
        if not scan:
            st.info("Not checked yet.")
        else:
            k = st.columns(6)
            k[0].metric("Images", sm.get("images", 0))
            k[1].metric("With ROI file", sm.get("with_rois", 0))
            k[2].metric("No ROI (0 %)", sm.get("without_rois", 0))
            k[3].metric("With damage", sm.get("with_damage", 0))
            k[4].metric("Mean % damage", common.num(sm.get("mean_pct_damage"), "{:.1f}%"))
            k[5].metric("Image size", sm.get("image_size") or "—")

            vessels = list(sm.get("vessels", {}).keys())
            ctrl = st.multiselect("Control vessels (never used for training)", vessels,
                                  default=[v for v in sset.get("control_vessels", []) if v in vessels], key=f"ctrl_{sid}",
                                  help="Controls are left out of training, as planned. Run the model on them afterwards instead.")
            if ctrl != sset.get("control_vessels", []):
                library.update_set(sid, control_vessels=ctrl)
                st.toast("Saved control vessels.")

            rows = scan["rows"]
            only_problems = st.toggle("Show only images that need a look", key=f"prob_{sid}",
                                      value=bool(sm.get("errors") or sm.get("warnings")))
            df = pd.DataFrame([{
                "Status": STATUS.get(r["status"], r["status"]), "Image": r["name"], "Vessel": r.get("vessel", ""),
                "Sec": r.get("section", ""), "Side": r.get("side", ""), "Pos": r.get("position", ""),
                "ROI file": ", ".join(r.get("roi_files", [])) or ("—" if not r.get("orphan_roi") else ""),
                "Annotation": r.get("annotation", ""),
                "Usable px": r.get("tua_px"), "Damage px": r.get("damage_px"),
                "% damage": None if r.get("pct_damage") is None or (isinstance(r.get("pct_damage"), float) and math.isnan(r["pct_damage"])) else r["pct_damage"],
                "Messages": " · ".join(r.get("messages", [])),
                "Notes": " · ".join(r.get("notes", [])),
            } for r in rows if not only_problems or r["status"] != "ok"])
            excluded = set(sset.get("excluded_images", []))
            if len(df):
                st.dataframe(df, hide_index=True, height=min(420, 38 + 35 * len(df)), column_config={
                    "Usable px": st.column_config.NumberColumn(format="localized"),
                    "Damage px": st.column_config.NumberColumn(format="localized"),
                    "% damage": st.column_config.NumberColumn(format="%.2f"),
                    "Messages": st.column_config.TextColumn(width="large"),
                })
            else:
                st.success("Every image passed the checks.")

            names = [r["name"] for r in rows if not r.get("orphan_roi")]
            ex = st.multiselect("Leave these images out of training", names, default=[n for n in names if n in excluded],
                                key=f"ex_{sid}")
            if set(ex) != excluded:
                library.update_set(sid, excluded_images=ex)

            st.markdown("###### Look at an image with its ROIs")
            pick = st.selectbox("Image", names, key=f"pv_{sid}", index=None, placeholder="Choose an image to preview")
            if pick:
                r = next(x for x in rows if x["name"] == pick)
                folder_ = sset["folder"]
                rois = tuple(os.path.join(folder_, f) for f in r.get("roi_files", []))
                try:
                    img, tua_v, dmg_v, tpx, dpx, pct = preview(r["path"], rois, os.path.getmtime(r["path"]))
                    c = st.columns(3)
                    c[0].image(img, caption="Original", width="stretch")
                    c[1].image(tua_v, caption="Usable area from the TUA ROI", width="stretch")
                    c[2].image(dmg_v, caption="Damage from the D ROIs", width="stretch")
                    st.markdown(f"Usable area **{tpx:,} px** · damage **{dpx:,} px** · **{common.num(pct, '{:.2f}%')}** damage. "
                                "Compare with your ImageJ numbers: they should agree to within a fraction of a percent.")
                except Exception as exc:
                    st.error(f"Could not preview: {exc}")

        a, b_, c_ = st.columns([1, 1, 4])
        if a.button("Check again", key=f"rescan_{sid}", icon=":material/refresh:"):
            bar = st.progress(0.0)
            library.scan_set(sid, progress=lambda i, n, f: bar.progress((i + 1) / max(n, 1), text=f"Checking {f}"))
            st.rerun()
        if b_.button("Remove", key=f"rm_{sid}", icon=":material/delete:", help="Removes it from this list only; files stay where they are."):
            library.remove_set(sid)
            st.rerun()
