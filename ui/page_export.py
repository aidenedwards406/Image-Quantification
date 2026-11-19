import io
import zipfile

import pandas as pd
import streamlit as st

from huvec import batches, excel
from huvec.utils import pretty_time, slugify, stamp
from ui import common

common.header("3 · Export to Excel",
              "One workbook per batch. **Approved** holds the model's numbers. **Needs tracing** is the list to trace by hand: "
              "type the usable area (px) and % damage into its yellow cells. **All images** combines both and updates by itself.")

bid = common.batch_selector()
if not bid:
    st.info("No batches yet.")
    st.stop()

b = batches.load(bid)
s = batches.summary(bid, b)
data = excel.build_rows(bid)
rows = data["rows"]
approved = [r for r in rows if r["status"] == batches.APPROVED]
trace = [r for r in rows if r["status"] == batches.REJECTED]
unreviewed = [r for r in rows if r["status"] in ("not reviewed", "not analysed")]

c = st.columns(4)
c[0].metric("Approved", len(approved))
c[1].metric("Needs tracing", len(trace))
c[2].metric("Not reviewed yet", len(unreviewed))
c[3].metric("Images", len(rows))
if unreviewed:
    st.warning(f"{len(unreviewed)} image(s) haven't been reviewed. They will only appear in 'All images' and 'Details' "
               "as *Not reviewed*. You can export now and export again later.", icon=":material/warning:")
    st.page_link("ui/page_review.py", label="Finish QC review", icon=":material/fact_check:")

with st.container(border=True):
    st.markdown("##### Create the workbook")
    o1, o2 = st.columns(2)
    with_images = o1.checkbox("Include thumbnails in the Details sheet", value=True,
                              help="Original, usable-area and damage pictures for every image. Makes the file larger "
                                   "(about 40 KB per image).")
    traces = o2.checkbox("Also export damage outlines for the 3D Vessel Model viewer", value=False,
                         help="A '<image>_trace.png' per approved image. Drop them in with the images in the 3D viewer "
                              "to see the model's outline over each image.")
    fname = f"{slugify(b['name'], 60)}_damage_{stamp()[:8]}.xlsx"
    if st.button("Create Excel workbook", type="primary", icon=":material/table_view:"):
        out = batches.batch_dir(bid) / "exports" / fname
        with st.spinner("Writing the workbook…"):
            excel.export_batch(bid, out, include_images=with_images)
        st.session_state[f"export_{bid}"] = str(out)
        if traces:
            with st.spinner("Writing damage outlines…"):
                tdir = batches.batch_dir(bid) / "exports" / f"traces_{stamp()}"
                files = excel.export_traces(bid, tdir)
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in files:
                        z.write(f, f.name)
                st.session_state[f"traces_{bid}"] = (buf.getvalue(), len(files))
    path = st.session_state.get(f"export_{bid}")
    if path:
        from pathlib import Path

        p = Path(path)
        if p.exists():
            st.success(f"Saved **{p.name}**. Download it below. A copy is also kept in the batch folder.",
                       icon=":material/check_circle:")
            st.download_button("Download Excel workbook", p.read_bytes(), file_name=p.name, type="primary",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               icon=":material/download:")
    tz = st.session_state.get(f"traces_{bid}")
    if tz:
        st.download_button(f"Download {tz[1]} damage outlines (.zip)", tz[0],
                           file_name=f"{slugify(b['name'], 60)}_traces.zip", mime="application/zip",
                           icon=":material/download:")

st.markdown("##### Preview")
t1, t2 = st.tabs([f"Approved ({len(approved)})", f"Needs tracing ({len(trace)})"])
with t1:
    df = pd.DataFrame([{"Image name": r["im"]["name"], "Total usable area (px)": r["res"].get("tua_px"),
                        "% damage": r["res"].get("pct_damage")} for r in approved])
    if len(df):
        st.dataframe(df, hide_index=True, column_config={
            "Total usable area (px)": st.column_config.NumberColumn(format="localized"),
            "% damage": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=max(1.0, float(df["% damage"].max())))})
    else:
        st.caption("Nothing approved yet.")
with t2:
    df = pd.DataFrame([{"Image name": r["im"]["name"],
                        "Why": ", ".join(r["rev"].get("reasons", [])) or ("could not be analysed" if r["res"].get("error") else ""),
                        "Note": r["rev"].get("note", "")} for r in trace])
    if len(df):
        st.dataframe(df, hide_index=True)
        st.caption("In Excel, these rows have empty yellow cells for the usable area (px) and % damage from the hand tracing.")
    else:
        st.caption("Nothing needs tracing.")

exports = sorted((batches.batch_dir(bid) / "exports").glob("*.xlsx"), reverse=True) if (batches.batch_dir(bid) / "exports").exists() else []
if exports:
    with st.expander(f"Earlier exports of this batch ({len(exports)})"):
        for p in exports[:20]:
            cc = st.columns([5, 1], vertical_alignment="center")
            cc[0].markdown(f"`{p.name}`")
            cc[1].download_button("Download", p.read_bytes(), file_name=p.name, key=f"dl_{p.name}",
                                  mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
