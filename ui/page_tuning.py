import copy
import io

import numpy as np
import streamlit as st

from huvec import batches, library, registry
from huvec import settings as settings_mod
from huvec.focus import BLURRY, DARK, FEATURELESS, SHARP, compute_focus
from huvec.imageio import read_image
from huvec.overlays import focus_overlay, specks_overlay
from huvec.specks import detect_specks
from ui import common

common.header("Tune focus & specks",
              "The focus and speck detectors are simple measurements, not AI, so you can set them by eye. Pick a few "
              "typical images (sharp, blurry edges, specks) and adjust until the colours match what you see. "
              "Saved settings apply to new batches.")

cfg = st.session_state.setdefault("tune_cfg", settings_mod.load())

# ── choose an image ────────────────────────────────────────────────────────
src = st.segmented_control("Image from", ["Training data", "A batch", "Upload"], default="Training data", key="tune_src")
img_path, img_bytes = None, None
if src == "Training data":
    opts = []
    for s in library.list_sets():
        scan = library.cached_scan(s["id"]) or {}
        opts += [(r["name"], r["path"]) for r in scan.get("rows", []) if not r.get("orphan_roi") and r["status"] != "error"]
    if opts:
        pick = st.selectbox("Image", range(len(opts)), format_func=lambda i: opts[i][0])
        img_path = opts[pick][1]
    else:
        st.caption("No training data added yet.")
elif src == "A batch":
    bl = batches.list_batches()
    if bl:
        bsel = st.selectbox("Batch", bl, format_func=lambda b: b["name"])
        ims = bsel["images"]
        pick = st.selectbox("Image", range(len(ims)), format_func=lambda i: ims[i]["name"])
        img_path = ims[pick]["path"]
    else:
        st.caption("No batches yet.")
else:
    up = st.file_uploader("One TIFF", type=["tif", "tiff"], key="tune_up")
    if up:
        img_bytes = up.getvalue()

if not img_path and not img_bytes:
    st.stop()


@st.cache_data(show_spinner="Loading image…", max_entries=8)
def load(path, data):
    if data is not None:
        import tifffile

        from huvec.imageio import _to_rgb, _to_uint8

        return np.ascontiguousarray(_to_uint8(_to_rgb(tifffile.imread(io.BytesIO(data)))))
    return read_image(path)


img = load(img_path, img_bytes)


@st.cache_data(show_spinner="Finding the usable area with the active model…", max_entries=8)
def usable_area(path, data, model_id):
    from huvec.model import predict_probs
    from huvec.predict import label_from_probs

    model, meta = registry.load(model_id)
    probs = predict_probs(model, img, meta)
    tua, _, _ = label_from_probs(probs, float(meta.get("damage_threshold", 0.5)), settings_mod.DEFAULTS["postprocess"])
    return tua


mid = registry.active_id()
tua = None
if mid and st.toggle("Use the active model's usable area (for 'blurry inside usable area' and specks)", value=True):
    tua = usable_area(img_path, img_bytes, mid)

# ── controls ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 2])
with left:
    f, s, fl = cfg["focus"], cfg["specks"], cfg["flags"]
    st.markdown("##### Focus")
    f["sharp_ratio"] = st.slider("Sharpness needed to count as in focus", 0.05, 0.60, float(f["sharp_ratio"]), 0.01,
                                 help="Higher = stricter. Raise it if slightly soft areas show as green.")
    f["min_structure"] = st.slider("Minimum texture to judge (× camera noise)", 0.05, 5.0, float(f["min_structure"]), 0.05,
                                   help="Areas with less texture (bare silicone, damage) are left uncoloured.")
    f["block"] = st.select_slider("Block size (px)", [16, 24, 32, 48, 64], value=int(f["block"]))
    f["dark_level"] = st.slider("Black-corner darkness", 0.02, 0.40, float(f["dark_level"]), 0.01)
    st.markdown("##### Specks")
    s["max_intensity"] = st.slider("How dark a speck must be (0 = black, 1 = background)", 0.02, 0.50,
                                   float(s["max_intensity"]), 0.01)
    s["min_area"], s["max_area"] = st.slider("Speck size (px)", 1, 300, (int(s["min_area"]), int(s["max_area"])))
    s["min_solidity"] = st.slider("How round/solid", 0.3, 1.0, float(s["min_solidity"]), 0.05)
    s["flag_count"] = st.number_input("Flag an image at this many specks", 1, 500, int(s["flag_count"]))
    st.markdown("##### Flags")
    fl["max_blurry_in_tua_pct"] = st.slider("Flag 'blurry' above (% of usable area)", 1.0, 60.0,
                                            float(fl["max_blurry_in_tua_pct"]), 1.0)
    fl["min_tua_pct"] = st.slider("Flag 'small usable area' below (% of image)", 1.0, 80.0, float(fl["min_tua_pct"]), 1.0)
    fl["max_uncertain_pct"] = st.slider("Flag 'model unsure' above (% of usable area)", 1.0, 60.0,
                                        float(fl["max_uncertain_pct"]), 1.0)

with right:
    fr = compute_focus(img, f)
    pm = fr.pixel_map()
    sp = detect_specks(img, s, region=tua, dark_level=float(f["dark_level"]))
    lab = None
    if tua is not None:
        lab = tua.astype(np.uint8)
    k = st.columns(4)
    k[0].metric("In focus", common.num(fr.in_focus_pct, "{:.0f}%"))
    k[1].metric("Blurry inside usable area", common.num(fr.blurry_pct_within(tua), "{:.0f}%") if tua is not None else "—")
    k[2].metric("Specks", sp.count, delta="flagged" if sp.count >= s["flag_count"] else None, delta_color="off")
    k[3].metric("Noise level", f"{fr.noise:.4f}")
    view = st.segmented_control("Show", ["Focus map", "Specks", "Original"], default="Focus map", key="tune_view") or "Focus map"
    if view == "Focus map":
        st.image(focus_overlay(img, pm, lab), width="stretch",
                 caption="Green = in focus · orange = blurry · no tint = featureless (bare silicone / damage) or dark")
        shares = {n: 100 * fr.fraction(c) for n, c in (("in focus", SHARP), ("blurry", BLURRY), ("featureless", FEATURELESS), ("dark", DARK))}
        st.caption(" · ".join(f"{k_}: {v:.0f}%" for k_, v in shares.items()))
    elif view == "Specks":
        st.image(specks_overlay(img, sp.centroids), width="stretch", caption="Cyan circles = detected specks")
    else:
        st.image(img, width="stretch")

c1, c2, _ = st.columns([2, 2, 4])
if c1.button("Save as lab settings", type="primary", icon=":material/save:"):
    settings_mod.save(cfg)
    st.success("Saved. New batches will use these settings (finished batches keep theirs).")
if c2.button("Reset to defaults", icon=":material/restart_alt:"):
    st.session_state["tune_cfg"] = copy.deepcopy(settings_mod.DEFAULTS)
    st.rerun()
