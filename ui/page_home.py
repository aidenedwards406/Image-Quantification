import streamlit as st

from huvec import batches, registry
from huvec.utils import pretty_time
from ui import common

common.header("HUVEC Damage Analyzer",
              "Outline endothelial damage on 4x brightfield images, check every result, and get % damage per image in Excel. "
              "Everything runs on this computer, so images never leave it.")

card = registry.active_card()
if not card:
    st.warning("**No model yet.** Ask your lab for the current model file (.pth) and import it on the "
               "**Models** page, or train one from traced images. To practise first, create the demo data under "
               "**Help & settings**.", icon=":material/info:")

st.markdown("#### Analyzing new images")
c1, c2, c3 = st.columns(3)
steps = [
    (c1, "Step 1", "Analyze images", "Upload a whole folder of TIFFs at once. The model outlines the usable area and the "
     "damage, and measures focus and specks.", "ui/page_analyze.py", ":material/upload_file:", "Add images"),
    (c2, "Step 2", "QC review", "Look at every image. Press **Y** if the outline is right, **N** if it isn't. "
     "Rejected images go to the tracing list.", "ui/page_review.py", ":material/fact_check:", "Review images"),
    (c3, "Step 3", "Export to Excel", "One workbook: approved results, a tracing list to fill in by hand, and a "
     "combined sheet for the 3D Vessel Model viewer.", "ui/page_export.py", ":material/table_view:", "Get the Excel file"),
]
for col, step, title, text, page, icon, link in steps:
    with col.container(border=True, height="stretch"):
        st.markdown(f"<div class='huvec-step'>{step}</div>", unsafe_allow_html=True)
        st.markdown(f"**{title}**")
        st.markdown(text)
        st.page_link(page, label=link, icon=icon)

st.markdown("#### Improving the model")
r1, r2, r3, r4 = st.columns(4)
for col, title, text, page, icon in [
    (r1, "Training data", "Add folders of TIFFs + ImageJ ROIs. Each one is checked against the naming and tracing rules.",
     "ui/page_library.py", ":material/folder_open:"),
    (r2, "Train", "Overnight on a laptop. Train and test vessels are kept separate, so the scores are honest.",
     "ui/page_train.py", ":material/model_training:"),
    (r3, "Check vs hand-measured %", "Score a model against images that only have a recorded % damage.",
     "ui/page_benchmark.py", ":material/rule:"),
    (r4, "Models", "Compare models, activate the best one, and share it with the lab.", "ui/page_models.py",
     ":material/deployed_code:"),
]:
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{title}**")
        st.caption(text)
        st.page_link(page, label="Open", icon=icon)

st.markdown("#### Recent batches")
bl = batches.list_batches()[:8]
if not bl:
    st.caption("No batches yet. Start with **1 · Analyze images**.")
else:
    for b in bl:
        s = b["status"]
        with st.container(border=True):
            a, bcol, c, d = st.columns([4, 3, 2, 2], vertical_alignment="center")
            a.markdown(f"**{common.esc(b['name'])}**  \n<span class='huvec-muted'>{pretty_time(b['created'])} · "
                       f"{common.esc(b.get('created_by') or '—')} · model: {common.esc(b.get('model_name', ''))}</span>",
                       unsafe_allow_html=True)
            bcol.progress(s["reviewed"] / s["analysed"] if s["analysed"] else 0.0,
                          text=f"{s['reviewed']}/{s['analysed']} reviewed · {s['approved']} ✓ · {s['rejected']} ✗")
            if c.button("Review", key=f"rev_{b['id']}", icon=":material/fact_check:", width="stretch"):
                st.session_state["batch_id"] = b["id"]
                st.switch_page("ui/page_review.py")
            if d.button("Export", key=f"exp_{b['id']}", icon=":material/table_view:", width="stretch"):
                st.session_state["batch_id"] = b["id"]
                st.switch_page("ui/page_export.py")
