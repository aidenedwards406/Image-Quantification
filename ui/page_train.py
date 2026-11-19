import altair as alt
import pandas as pd
import streamlit as st

from huvec import jobs, library, registry
from huvec import settings as settings_mod
from huvec.metrics import describe
from huvec.train import PRESETS, build_config
from huvec.utils import fmt_duration, pretty_time
from ui import common

common.header("Train a model",
              "Teach the model from traced images. Training runs in the background (overnight on a laptop is normal). "
              "You can close this tab, and on Windows the computer is kept awake until it finishes.")

ROLE_LABELS = ["Train", "Validation", "Test", "Leave out"]
ROLE_KEYS = {"Train": "train", "Validation": "val", "Test": "test", "Leave out": "exclude"}


def charts(p: dict) -> None:
    losses = pd.DataFrame(p.get("losses") or [])
    evals = pd.DataFrame(p.get("evals") or [])
    c1, c2 = st.columns(2)
    if len(losses):
        losses["run"] = losses["run"].replace("", "training")
        c1.altair_chart(alt.Chart(losses).mark_line().encode(
            x=alt.X("step:Q", title="Step"), y=alt.Y("loss:Q", title="Training loss (lower is better)"),
            color=alt.Color("run:N", title=None, legend=alt.Legend(orient="bottom"))).properties(height=230), width="stretch")
    if len(evals):
        evals["run"] = evals["run"].replace("", "validation")
        c2.altair_chart(alt.Chart(evals).mark_line(point=True).encode(
            x=alt.X("step:Q", title="Step"), y=alt.Y("mae:Q", title="Validation: avg % damage error (points)"),
            color=alt.Color("run:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=["run", "step", alt.Tooltip("mae:Q", format=".2f"), alt.Tooltip("dice_damage_pooled:Q", format=".3f"),
                     alt.Tooltip("dice_tua_pooled:Q", format=".3f")]).properties(height=230), width="stretch")


@st.fragment(run_every=5.0)
def live_charts(job_id: str) -> None:
    s = jobs.status(job_id)
    charts(s)
    if s.get("status") not in jobs.ACTIVE:
        st.rerun()


running = jobs.running_jobs("train")
if running:
    j = running[0]
    st.markdown(f"##### Training now: {j.get('title')}")
    with st.container(border=True):
        common.job_panel(j["id"], show_log=True)
        live_charts(j["id"])
    st.caption("You can leave this page. Training continues, and the finished model appears on the **Models** page.")
    st.stop()

# Show the most recent finished training, if any.
recent = jobs.list_jobs("train", limit=1)
if recent and recent[0].get("status") in ("done", "stopped", "failed", "crashed") and not st.session_state.get("hide_last_train"):
    j = recent[0]
    with st.container(border=True):
        st.markdown(f"##### Last training: {j.get('title')} · {pretty_time(j.get('created'))}")
        common.job_panel(j["id"])
        mid = j.get("model_id")
        if mid and registry.model_path(mid).exists():
            ev = registry.card(mid).get("evaluation", {})
            st.markdown(describe(ev.get("agreement", {})) if ev.get("agreement") else (ev.get("note") or ""))
            charts(j)
            st.page_link("ui/page_models.py", label="See the full report and activate it on the Models page",
                         icon=":material/arrow_forward:")
        if st.button("Hide", type="tertiary"):
            st.session_state["hide_last_train"] = True
            st.rerun()

sets = library.list_sets()
if not sets:
    st.info("Add traced images on the **Training data** page first.", icon=":material/info:")
    st.page_link("ui/page_library.py", label="Training data", icon=":material/folder_open:")
    st.stop()

st.markdown("##### 1 · Which training data")
chosen = st.multiselect("Training sets", [s["id"] for s in sets], default=[s["id"] for s in sets],
                        format_func=lambda i: next(s["name"] for s in sets if s["id"] == i))
if not chosen:
    st.stop()
rows = library.training_rows(chosen)
controls = set()
for s in sets:
    if s["id"] in chosen:
        controls |= set(s.get("control_vessels", []))
vessels = sorted({r["vessel"] for r in rows})
st.caption(f"{len(rows)} usable images from {len(vessels)} vessels"
           + (f" (control vessels already left out: {', '.join(sorted(controls))})" if controls else "")
           + ". Images with problems or marked 'leave out' are not counted.")
if len(vessels) < 2:
    st.warning("Training needs images from at least 2 vessels (one to learn from, one to check on).")
    st.stop()

st.markdown("##### 2 · How to test it honestly")
mode = st.radio("Evaluation", ["cv", "split"], horizontal=False, format_func=lambda m: {
    "cv": f"Leave-one-vessel-out cross-validation (most reliable): trains {len(vessels) + 1} times, "
          "each vessel is the test vessel once, then a final model learns from all vessels",
    "split": "Choose train / validation / test vessels myself: trains once, faster"}[m],
    index=1 if len(vessels) < 3 else 0)

counts = pd.DataFrame(rows).groupby("vessel").agg(images=("name", "count"),
                                                   damaged=("pct_damage", lambda x: int((x > 0).sum())),
                                                   mean_pct=("pct_damage", "mean")).reset_index()
default_roles = {}
for i, v in enumerate(vessels):
    if mode == "split":
        default_roles[v] = "Test" if i == len(vessels) - 1 else ("Validation" if i == len(vessels) - 2 and len(vessels) >= 3 else "Train")
    else:
        default_roles[v] = "Train"
counts["Role"] = counts["vessel"].map(default_roles)
counts = counts.rename(columns={"vessel": "Vessel", "images": "Images", "damaged": "With damage", "mean_pct": "Mean % damage"})
edited = st.data_editor(counts, hide_index=True, disabled=["Vessel", "Images", "With damage", "Mean % damage"],
                        key=f"roles_{mode}_{'-'.join(chosen)}",
                        column_config={"Role": st.column_config.SelectboxColumn(
                            options=ROLE_LABELS if mode == "split" else ["Train", "Leave out"], required=True),
                            "Mean % damage": st.column_config.NumberColumn(format="%.1f")})
roles = {r["Vessel"]: ROLE_KEYS[r["Role"]] for _, r in edited.iterrows()}
active_vessels = [v for v, r in roles.items() if r != "exclude"]
problems = []
if mode == "split":
    if not any(r == "train" for r in roles.values()):
        problems.append("Choose at least one **Train** vessel.")
    if not any(r == "test" for r in roles.values()):
        st.info("No **Test** vessel: the model will have no honest score. Fine for a final model, "
                "but run cross-validation at least once to know how good it is.", icon=":material/info:")
    if not any(r == "val" for r in roles.values()):
        st.caption("No validation vessel: training runs the full length instead of stopping at its best point.")
elif len(active_vessels) < 3:
    problems.append("Cross-validation needs at least 3 vessels.")

st.markdown("##### 3 · How long to train")
preset = st.segmented_control("Length", list(PRESETS), default="standard", format_func=lambda p: PRESETS[p]["label"],
                              key="preset") or "standard"
st.caption({"quick": "A short run to check that everything works. Not accurate enough to use.",
            "standard": "The normal choice. On a laptop it usually takes a few hours per training run.",
            "thorough": "About twice as long as Standard. Use it for the final model."}[preset])
c1, c2 = st.columns(2)
scale = c1.radio("Detail level", [1.0, 0.5], format_func=lambda s: {1.0: "Full resolution (recommended)",
                                                                    0.5: "Half resolution (about 3× faster, less detail)"}[s])
with c2.expander("Advanced"):
    steps = st.number_input("Training steps", 100, 50000, PRESETS[preset]["steps"], step=100)
    tile = st.select_slider("Tile size (px)", [256, 320, 384, 448, 512], value=384)
    batch = st.select_slider("Tiles per step", [2, 4, 6, 8], value=4)
    lr = st.select_slider("Learning rate", [1e-4, 2e-4, 3e-4, 5e-4, 1e-3], value=3e-4)
    pretrained = st.checkbox("Start from ImageNet weights (downloaded once, ~45 MB)", value=True)
overrides = {"steps": int(steps), "tile": int(tile), "batch": int(batch), "lr": float(lr), "scale": float(scale),
             "pretrained": bool(pretrained)}
cfg = build_config(preset, overrides)
runs = len(active_vessels) + 1 if mode == "cv" else 1
n_train = sum(1 for r in rows if roles.get(r["vessel"]) == "train")
n_val = sum(1 for r in rows if roles.get(r["vessel"]) == "val") if mode == "split" else len(rows) // max(1, len(active_vessels))
n_test = sum(1 for r in rows if roles.get(r["vessel"]) == "test") if mode == "split" else len(rows) // max(1, len(active_vessels))
if st.button("Estimate time on this computer", icon=":material/timer:"):
    from huvec.train import estimate_seconds

    with st.spinner("Timing a few training steps…"):
        est = estimate_seconds(cfg, n_train, n_val, n_test, runs)
    st.session_state["estimate"] = (est, runs)
if st.session_state.get("estimate"):
    est, r_ = st.session_state["estimate"]
    st.info(f"About **{fmt_duration(est['total_s'])}** in total ({r_} training run{'s' if r_ > 1 else ''}, "
            f"{est['step_s']:.2f} s per step, {est['image_s']:.1f} s per test image). Early stopping often finishes sooner.",
            icon=":material/timer:")

st.markdown("##### 4 · Name it and start")
c1, c2 = st.columns([2, 3])
name = c1.text_input("Model name", value=f"Model {pd.Timestamp.now():%b %d}")
notes = c2.text_input("Notes", placeholder="What's different about this model?")
for p in problems:
    st.error(p)
st.caption(":material/power: Plug the laptop in. You can keep using it, but training will slow other work down.")
if st.button("Start training", type="primary", icon=":material/model_training:", disabled=bool(problems)):
    jobs.start_job("train", {"name": name, "notes": notes, "trained_by": common.user(), "set_ids": chosen,
                             "roles": roles, "mode": mode, "preset": preset, "overrides": overrides,
                             "settings": settings_mod.load()}, name)
    st.session_state.pop("hide_last_train", None)
    st.rerun()
