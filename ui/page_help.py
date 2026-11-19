import streamlit as st

from huvec import __version__, library, workspace
from huvec.utils import clean_path_text
from ui import common

common.header("Help & settings")

t1, t2, t3, t4 = st.tabs(["Quick start", "Definitions", "Questions", "Settings & demo"])

with t1:
    st.markdown("""
### For lab members analyzing images
1. **Put your name in the sidebar.** It's saved with your QC decisions.
2. **1 · Analyze images**: drop in all the TIFFs of a batch (or point to their folder) and press *Analyze*.
   About 3–5 seconds per image on a laptop. You can start reviewing straight away.
3. **2 · QC review**: for every image, compare the original with the outlines.
   - **Yes (Y)**: the yellow usable-area outline and the red damage look right → the model's numbers are kept.
   - **No (N)**: something's wrong → the image goes on the **Needs tracing** list. Picking a reason is optional but helps improve the model.
   - Pay extra attention to images with orange **flags** (blurry, small usable area, model unsure, specks).
4. **3 · Export to Excel**: download the workbook. On the **Needs tracing** sheet, trace each image in ImageJ
   (TUA + D1, D2 …) and type its **usable area (px)** and **% damage** into the yellow cells. The **All images**
   sheet updates by itself.
5. Optional: load the workbook into the **3D Vessel Model** viewer (step 4) to see the damage on the vessel.

### For researchers improving the model
- **Training data**: add folders of traced images. Fix anything marked ⛔/⚠️. Mark control vessels.
- **Train a model**: cross-validation for an honest score; Standard length; usually overnight.
- **Check vs hand-measured %**: run the new model on the images that only have a hand-measured %.
- **Models**: compare with the active model and activate the better one. Share the `.pth` file with the lab.
""")

with t2:
    st.markdown("""
| Term | Meaning |
|---|---|
| **Usable area (TUA)** | Total usable area: in focus and inside the vessel. Excludes blurry edges, areas outside the vessel and black corners. Slightly blurry areas you're sure about can stay in. |
| **Damage** | A region with no nuclei. Blurry nuclei don't count as missing. |
| **% damage** | damaged usable area ÷ total usable area × 100 |
| **Approved** | QC reviewer agreed with the model's outlines |
| **Needs tracing** | QC reviewer rejected the outlines; the image is traced by hand instead |
| **In focus %** | Share of the image (with cell texture) that is sharp, from pixel-to-pixel intensity change |
| **Blurry inside usable area** | Part of the model's usable area that looks out of focus. High values mean lower confidence |
| **Specks** | Small, very dark, round blobs (possibly balled-up nuclei of unhealthy cells), flagged for follow-up |
| **Model unsure** | Part of the usable area where the model was close to 50/50 |
| **Dice** | Pixel overlap between two outlines: 1 = identical, 0 = no overlap |
| **Bias** | Average (model − hand) difference. Positive = model calls more damage |
""")

with t3:
    st.markdown("""
**Do images leave my computer?** No. The app runs locally (address `localhost`), usage statistics are switched off,
and nothing is uploaded anywhere. The workspace folder holds copies of uploaded images and pictures of them, so keep
it on storage approved for NDA data.

**The model is wrong on a type of image again and again.** Reject those images with a reason, trace them, save the ROIs,
add them as a new training set and retrain. That's how the model gets better.

**Can I change a number in the app?** No, on purpose. Numbers are only changed in the Excel workbook.

**Training stopped / the laptop restarted.** Start it again. Image preparation is cached, so it restarts faster.
An interrupted batch analysis can be resumed (Analyze page → *More* → *Resume*).

**Where are my files?** In the workspace folder shown in the sidebar. Excel exports are also in each batch's `exports` folder.

**Different microscope or magnification?** Retrain with traced images from it. If pixel sizes differ, adjust the focus
*scale* in `settings.yaml`.
""")

with t4:
    st.markdown("##### Workspace folder")
    st.caption("Where models, batches and QC decisions are stored. Point several computers at the same folder on a lab "
               "drive to share them (one person reviewing a batch at a time).")
    new = st.text_input("Workspace folder", value=str(workspace.root()))
    if st.button("Use this folder"):
        p = clean_path_text(new)
        workspace.save_local_setting("workspace", p)
        st.success("Saved. Restart the app (close the window, double-click start again) to switch.")

    st.markdown("##### Training cache")
    cache = workspace.cache_dir()
    size_mb = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()) / 1e6
    st.caption(f"Decoded training images are kept so retraining starts faster: currently {size_mb:,.0f} MB in `{cache}`. "
               "Safe to clear; it is rebuilt when needed.")
    if st.button("Clear training cache", disabled=size_mb == 0):
        import shutil

        shutil.rmtree(cache / "train", ignore_errors=True)
        st.success("Cleared.")

    st.markdown("##### Practice data")
    st.caption("Creates synthetic images (fake, so safe to share) with ROI files, a hand-measured % spreadsheet and new "
               "images to analyze. Good for learning the workflow and for testing after updates. Takes about 2 minutes.")
    if st.button("Create demo data", icon=":material/science:"):
        from huvec.synthetic import make_demo

        with st.spinner("Generating synthetic images…"):
            info = make_demo(workspace.demo_dir() / "study")
        sid = library.add_set("Demo traced images", info["training_folder"], common.user(), "synthetic generator",
                              "Synthetic practice data")
        library.scan_set(sid)
        st.success("Demo data created. Next:")
        st.markdown(f"""
1. **Train a model**: the *Demo traced images* set is ready (try *Quick check*).
2. **Check vs hand-measured %**: images `{info['benchmark_folder']}` with the spreadsheet `{info['benchmark_table']}`.
3. **Analyze images**: use the folder `{info['new_images_folder']}`.
""")
    st.caption(f"HUVEC Damage Analyzer {__version__}")
