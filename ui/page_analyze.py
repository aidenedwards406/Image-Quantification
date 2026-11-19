import datetime as dt
from pathlib import Path

import streamlit as st

from huvec import batches, jobs, registry
from huvec.utils import pretty_time
from ui import common

common.header("1 · Analyze images",
              "Add a batch of 4x brightfield TIFFs. The model outlines the usable area (TUA) and the damage in each image, "
              "then you check every result in QC review.")

mid = registry.active_id()
if not mid:
    st.error("There's no active model yet. Import the lab's model file or train one on the **Models** page.",
             icon=":material/block:")
    st.page_link("ui/page_models.py", label="Go to Models", icon=":material/arrow_forward:")
    st.stop()

card = registry.card(mid)
ag = (card.get("evaluation") or {}).get("agreement") or {}
model_line = f"Model: **{card.get('name', mid)}**, trained {pretty_time(card.get('created'))}"
if ag.get("n"):
    model_line += f". On held-out images it was off by **{ag['mae']:.1f} points** on average."
st.caption(model_line)

running = jobs.running_jobs("analyze")
with st.container(border=True):
    st.markdown("##### New batch")
    c1, c2 = st.columns([2, 3])
    name = c1.text_input("Batch name", value=f"Batch {dt.date.today():%b %d, %Y}",
                         help="For example the study and vessels, like 'Study 12 - BR-S24-9 to 12'.")
    notes = c2.text_input("Notes (optional)", placeholder="Anything worth remembering about these images")
    tab_up, tab_folder = st.tabs([":material/upload: Upload files", ":material/folder: A folder on this computer"])
    with tab_up:
        files = st.file_uploader("Drop all the TIFF images here, or click to choose them", type=["tif", "tiff"],
                                 accept_multiple_files=True, key=f"uploads_{st.session_state.get('upload_round', 0)}")
        st.caption("Uploads only go to this app on your own computer. You can select hundreds of files at once.")
    with tab_folder:
        folder = common.folder_input("Image folder", key="analyze_folder")
        found = batches.images_in_folder(folder) if common.path_exists(folder) else []
        if folder and not common.path_exists(folder):
            st.warning("That folder doesn't exist or can't be opened.")
        elif folder:
            st.caption(f"{len(found)} TIFF image(s) found in this folder (sub-folders are not searched).")
    with st.expander("Options"):
        tta = st.checkbox("Extra-careful mode", value=False,
                          help="Runs the model on 4 flipped copies of each image and averages them. "
                               "Slightly more accurate and stable, about 4 times slower.")
    use_upload = bool(files)
    n_images = len(files) if use_upload else len(found)
    start = st.button(f"Analyze {n_images} image(s)" if n_images else "Analyze", type="primary",
                      disabled=n_images == 0, icon=":material/play_arrow:")
    if start:
        if use_upload:
            with st.spinner("Saving uploaded images…"):
                up = batches.save_uploads(name, [(f.name, f.getvalue()) for f in files])
            paths, source = batches.images_in_folder(up), {"type": "upload", "folder": str(up)}
        else:
            paths, source = found, {"type": "folder", "folder": folder}
        bid = batches.create_batch(name, common.user(), notes, paths, source, mid, tta=tta)
        jid = jobs.start_job("analyze", {"batch_id": bid}, name)
        st.session_state["batch_id"] = bid
        st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1  # empties the uploader
        st.toast("Analysis started. You can start reviewing as soon as the first images are done.")
        st.rerun()

st.markdown("##### Batches")
bl = batches.list_batches()
if not bl:
    st.caption("No batches yet.")
for b in bl[:15]:
    s = b["status"]
    job = common.active_job_for("analyze", {"batch_id": b["id"]})
    with st.container(border=True):
        top = st.columns([5, 2, 2, 2], vertical_alignment="center")
        top[0].markdown(f"**{common.esc(b['name'])}**  \n<span class='huvec-muted'>{pretty_time(b['created'])} · "
                        f"{common.esc(b.get('created_by') or '—')} · {s['images']} images · "
                        f"model: {common.esc(b.get('model_name', ''))}</span>", unsafe_allow_html=True)
        if job:
            with top[1]:
                st.caption("Analyzing…")
        else:
            top[1].markdown(f"{s['analysed']}/{s['images']} analysed" + (f"  \n:orange[{s['errors']} failed]" if s["errors"] else ""))
        if top[2].button("QC review", key=f"go_{b['id']}", icon=":material/fact_check:", width="stretch",
                         disabled=s["analysed"] == 0):
            st.session_state["batch_id"] = b["id"]
            st.switch_page("ui/page_review.py")
        with top[3].popover("More", icon=":material/more_horiz:", width="stretch"):
            if not job and s["analysed"] < s["images"]:
                if st.button("Resume analysis", key=f"res_{b['id']}", icon=":material/play_arrow:"):
                    jobs.start_job("analyze", {"batch_id": b["id"]}, b["name"])
                    st.rerun()
            st.caption(f"Source: {b.get('source', {}).get('folder', '')}")
            if st.button("Delete batch", key=f"del_{b['id']}", icon=":material/delete:", disabled=bool(job)):
                st.session_state["confirm_delete"] = b["id"]
        if st.session_state.get("confirm_delete") == b["id"]:
            st.warning(f"Delete **{b['name']}** with its results and QC decisions? Exported Excel files you "
                       "saved elsewhere are not affected. Images in a folder you pointed to are not deleted.")
            cc = st.columns([1, 1, 6])
            if cc[0].button("Delete", key=f"yes_{b['id']}", type="primary"):
                batches.delete_batch(b["id"])
                st.session_state.pop("confirm_delete", None)
                st.rerun()
            if cc[1].button("Cancel", key=f"no_{b['id']}"):
                st.session_state.pop("confirm_delete", None)
                st.rerun()
        if job:
            common.job_panel(job["id"])
