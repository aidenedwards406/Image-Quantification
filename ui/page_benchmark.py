import pandas as pd
import streamlit as st

from huvec import benchmark, jobs, registry
from huvec.metrics import describe
from huvec.utils import pretty_time
from ui import charts, common

common.header("Check vs hand-measured %",
              "Images with only a recorded % damage (no ROIs) can't teach the model, but they're the best test of the real "
              "goal: does the model's % match the lab's? Use them to compare models, pick a damage threshold, and "
              "choose which images to trace next.")

with st.expander("Add a benchmark: a folder of images + a spreadsheet of their % damage", icon=":material/add:",
                 expanded=not benchmark.list_all()):
    name = st.text_input("Benchmark name", placeholder="e.g. 2025-26 hand-measured images")
    folder = common.folder_input("Image folder", key="bm_folder")
    table = st.file_uploader("Spreadsheet with image names and % damage (.xlsx or .csv)", type=["xlsx", "xlsm", "csv"],
                             key="bm_table")
    if table is not None:
        data = table.getvalue()
        sheets = benchmark.sheet_names(data, table.name)
        sheet = st.selectbox("Sheet", sheets) if len(sheets) > 1 else (sheets[0] if sheets else None)
        try:
            df = benchmark.read_table(data, table.name, sheet)
        except Exception as exc:
            st.error(f"Couldn't read the spreadsheet: {exc}")
            st.stop()
        g_name, g_pct = benchmark.guess_columns(df)
        cols = list(df.columns)
        c1, c2 = st.columns(2)
        name_col = c1.selectbox("Column with image names", cols, index=cols.index(g_name) if g_name in cols else 0)
        pct_col = c2.selectbox("Column with % damage", cols, index=cols.index(g_pct) if g_pct in cols else min(1, len(cols) - 1))
        st.dataframe(df.head(8), hide_index=True)
        if common.path_exists(folder):
            m = benchmark.match_table(folder, df, name_col, pct_col)
            st.markdown(f"**{len(m['matched'])}** images matched to a % value · {len(m['unmatched'])} spreadsheet rows "
                        f"without a matching image · {len(m['images_without_value'])} images without a value.")
            x100 = False
            if m["looks_like_fraction"]:
                x100 = st.checkbox("The values look like fractions (0.125 instead of 12.5%). Multiply by 100?", value=True)
            if m["unmatched"]:
                with st.expander("Rows that didn't match"):
                    st.dataframe(pd.DataFrame(m["unmatched"]), hide_index=True)
            if st.button("Create benchmark", type="primary", disabled=not (name and m["matched"])):
                bid = benchmark.create(name, folder, m["matched"], common.user(), table_name=table.name,
                                       multiply_by_100=x100)
                st.session_state["bench_id"] = bid
                st.rerun()
        elif folder:
            st.warning("That image folder doesn't exist.")

bl = benchmark.list_all()
if not bl:
    st.stop()
ids = [b["id"] for b in bl]
cur = st.session_state.get("bench_id")
bid = st.selectbox("Benchmark", ids, index=ids.index(cur) if cur in ids else 0,
                   format_func=lambda i: next(f"{b['name']} · {len(b['images'])} images" for b in bl if b["id"] == i))
st.session_state["bench_id"] = bid
bench = benchmark.load(bid)

models = registry.list_models()
if not models:
    st.info("No models to check yet.")
    st.stop()
mids = [m["id"] for m in models]
act = registry.active_id()
mid = st.selectbox("Model", mids, index=mids.index(act) if act in mids else 0,
                   format_func=lambda i: ("★ " if i == act else "") + registry.display_name(i))

leak = benchmark.leakage(bench, registry.card(mid))
if leak:
    st.warning(f"This model was trained on some of the same vessels ({', '.join(leak)}), so its score here will look "
               "better than it really is.", icon=":material/warning:")

job = common.active_job_for("benchmark", {"bench_id": bid, "model_id": mid})
res = benchmark.results(bid, mid)
if job:
    common.job_panel(job["id"])
    st.stop()
label = "Run again" if res else f"Run the model on these {len(bench['images'])} images"
if st.button(label, type="primary" if not res else "secondary", icon=":material/play_arrow:"):
    jobs.start_job("benchmark", {"bench_id": bid, "model_id": mid}, f"Benchmark {bench['name']}")
    st.rerun()
if not res:
    st.stop()

ov = res["summary"]["overall"]
st.caption(f"Run {pretty_time(res.get('run'))}" + ("" if res.get("complete") else " · stopped before finishing"))
st.markdown(describe(ov))
k = st.columns(5)
k[0].metric("Images", ov.get("n", 0))
k[1].metric("Avg error", f"{ov.get('mae', float('nan')):.2f} pts")
k[2].metric("Within ±5 points", f"{100 * ov.get('within_tol', float('nan')):.0f}%")
k[3].metric("Bias", f"{ov.get('bias', float('nan')):+.2f} pts")
k[4].metric("Agreement (CCC)", f"{ov.get('ccc', float('nan')):.3f}" if ov.get("ccc") is not None else "—",
            help="Lin's concordance correlation: 1 = perfect agreement, above about 0.9 is good.")

df = pd.DataFrame([r for r in res["rows"] if r.get("pred_pct") is not None]).rename(
    columns={"manual_pct": "manual", "pred_pct": "model"})
df = df[df["model"].notna()]
if df.empty:
    st.warning("The model produced no usable results on this benchmark. Check the log of the run.")
    st.stop()
g1, g2 = st.columns(2)
g1.altair_chart(charts.scatter(df, color="section"), width="stretch")
g2.altair_chart(charts.bland_altman(df, stats=ov), width="stretch")


def group_table(d: dict, label: str):
    return pd.DataFrame([{label: k, "Images": v.get("n"), "Avg error (pts)": round(v.get("mae", float("nan")), 2),
                          "Within ±5": f"{100 * v.get('within_tol', float('nan')):.0f}%", "Bias": round(v.get("bias", float("nan")), 2)}
                         for k, v in d.items() if v.get("n")])


t1, t2 = st.columns(2)
t1.markdown("###### By section")
t1.dataframe(group_table(res["summary"]["by_section"], "Section"), hide_index=True)
t2.markdown("###### By vessel")
t2.dataframe(group_table(res["summary"]["by_vessel"], "Vessel"), hide_index=True)

st.markdown("#### Trace these next")
st.caption("The images where the model and the hand measurement disagree most. Tracing them with ROIs (TUA + D#) and "
           "adding them to the training data helps the model most.")
df["error"] = df["model"] - df["manual"]
worst = df.reindex(df["error"].abs().sort_values(ascending=False).index).head(12)
st.download_button("Download the full list (.csv)", df.sort_values("error", key=abs, ascending=False)[
    ["name", "vessel", "section", "manual", "model", "error"]].to_csv(index=False).encode(), file_name="trace_next.csv",
                   icon=":material/download:")
for i in range(0, len(worst), 4):
    cols = st.columns(4)
    for col, (_, r) in zip(cols, worst.iloc[i:i + 4].iterrows()):
        with col.container(border=True):
            p = benchmark.overlay_path(bid, mid, r["stem"])
            if p.exists():
                st.image(str(p), width="stretch")
            st.markdown(f"<span class='huvec-muted'>{common.esc(r['name'])}</span>  \nHand **{r['manual']:.1f}%** · model "
                        f"**{r['model']:.1f}%** ({r['error']:+.1f})", unsafe_allow_html=True)

cal = res.get("calibration") or {}
if cal:
    st.markdown("#### Match the lab's tracing style (damage threshold)")
    st.caption("The model scores each usable pixel from 0 to 1 for 'damaged'. The threshold decides where damage starts. "
               "Tracers differ in how generously they outline damage, so a slightly different threshold can match "
               "your lab better. This changes no outlines in finished batches.")
    a, b = st.columns([3, 2])
    a.altair_chart(charts.calibration_curve(cal["grid"], cal["mae"], cal["best_threshold"], cal["current_threshold"]),
                   width="stretch")
    with b:
        st.metric("Current threshold", f"{cal['current_threshold']:.2f}", help="0.50 is the default.")
        st.metric("Best threshold here", f"{cal['best_threshold']:.2f}",
                  delta=f"{cal['best_mae'] - cal['current_mae']:+.2f} pts avg error", delta_color="inverse")
        if cal.get("honest_mae") is not None:
            st.caption(f"Honest estimate (choose the threshold on half the vessels, test on the other half): "
                       f"**{cal['honest_mae']:.2f} points** average error. Base the decision on this number.")
        gain = cal["current_mae"] - (cal.get("honest_mae") or cal["best_mae"])
        if gain > 0.2 and abs(cal["best_threshold"] - cal["current_threshold"]) >= 0.02:
            if st.button("Create a calibrated copy of this model", icon=":material/tune:"):
                new = benchmark.make_calibrated_model(mid, cal["best_threshold"], common.user(), bench["name"])
                st.success(f"Created **{registry.display_name(new)}**. Compare and activate it on the Models page.")
        else:
            st.caption("The current threshold is already about as good as it gets on this benchmark.")

others = [m for m in models if bid in (m.get("benchmarks") or {})]
if len(others) > 1:
    st.markdown("#### All models on this benchmark")
    st.dataframe(pd.DataFrame([{
        "Model": ("★ " if m["id"] == act else "") + m.get("name", m["id"]),
        "Avg error (pts)": round(m["benchmarks"][bid]["mae"], 2),
        "Within ±5": f"{100 * (m['benchmarks'][bid].get('within_tol') or 0):.0f}%",
        "Bias": round(m["benchmarks"][bid].get("bias") or 0, 2),
        "Trained on these vessels": ", ".join(m["benchmarks"][bid].get("leakage") or []) or "no"}
        for m in others]).sort_values("Avg error (pts)"), hide_index=True)
