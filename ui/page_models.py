import math

import pandas as pd
import streamlit as st

from huvec import registry
from huvec.metrics import describe
from huvec.utils import pretty_time
from ui import charts, common

common.header("Models",
              "Every trained model is kept. The **active** model is the one new batches use. Compare models on their test "
              "results and on the hand-measured benchmarks before switching.")

models = registry.list_models()
active = registry.active_id()

with st.expander("Import a model file (.pth) shared by your lab", icon=":material/upload:", expanded=not models):
    up = st.file_uploader("Model file", type=["pth"], key="model_upload")
    if up and st.button("Import", type="primary"):
        try:
            mid = registry.import_model(up.getvalue(), up.name, common.user())
            st.success(f"Imported **{registry.display_name(mid)}**.")
            st.rerun()
        except Exception as exc:
            st.error(f"That file couldn't be imported: {exc}")

if not models:
    st.info("No models yet. Train one or import the lab's model file above.", icon=":material/info:")
    st.stop()


def fmt(x, f="{:.1f}"):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if math.isnan(x) else f.format(x)


table = []
for c in models:
    ag = (c.get("evaluation") or {}).get("agreement") or {}
    bms = c.get("benchmarks") or {}
    best_bm = min((b for b in bms.values() if b.get("mae") is not None), key=lambda b: b["mae"], default=None)
    table.append({
        "Active": "★" if c["id"] == active else "", "Name": c.get("name"), "Trained": pretty_time(c.get("created") or c.get("imported")),
        "By": c.get("trained_by") or c.get("imported_by") or "",
        "Test: avg error (pts)": fmt(ag.get("mae"), "{:.2f}") if ag.get("n") else "—",
        "Test: within ±5": fmt(100 * ag["within_tol"], "{:.0f}%") if ag.get("n") else "—",
        "Benchmark: avg error": (f"{best_bm['mae']:.2f} ({best_bm['name']})" if best_bm else "—"),
        "Images": (c.get("training_data") or {}).get("n_images", ""),
        "id": c["id"],
    })
st.dataframe(pd.DataFrame(table), hide_index=True, column_config={"id": None})

ids = [c["id"] for c in models]
pick = st.selectbox("Model details", ids, index=ids.index(active) if active in ids else 0,
                    format_func=lambda i: ("★ " if i == active else "") + registry.display_name(i))
c = registry.card(pick)
ev = c.get("evaluation") or {}
ag = ev.get("agreement") or {}

with st.container(border=True):
    top = st.columns([4, 2], vertical_alignment="center")
    top[0].markdown(f"### {c.get('name')}" + ("  :green-badge[Active]" if pick == active else ""))
    td = c.get("training_data", {})
    vs = td.get("vessels", {})
    top[0].caption(
        f"Trained {pretty_time(c.get('created'))} by {c.get('trained_by') or '—'} · {td.get('n_images', '?')} images · "
        f"train {', '.join(vs.get('train', [])) or '—'} · validation {', '.join(vs.get('val', [])) or '—'} · "
        f"test {', '.join(vs.get('test', [])) or ('every vessel in turn (cross-validation)' if ev.get('mode') == 'cv' else '—')}"
        + (f" · damage threshold {c.get('damage_threshold')}" if c.get("damage_threshold", 0.5) != 0.5 else ""))
    if c.get("notes"):
        top[0].markdown(f"_{c['notes']}_")

    with top[1]:
        if pick != active:
            # Guardrail: compare with the active model on shared benchmarks.
            act = registry.card(active) if active else {}
            worse = []
            for bid_, bm in (c.get("benchmarks") or {}).items():
                ab = (act.get("benchmarks") or {}).get(bid_)
                if ab and bm.get("mae") is not None and ab.get("mae") is not None and bm["mae"] > ab["mae"] + 0.25:
                    worse.append(f"{bm['name']}: {bm['mae']:.2f} vs {ab['mae']:.2f} points")
            ok = True
            if worse:
                st.warning("This model does **worse** than the active one on: " + "; ".join(worse))
                ok = st.checkbox("Activate anyway", key=f"force_{pick}")
            elif active and not c.get("benchmarks"):
                st.caption("Tip: run it on a hand-measured benchmark first to compare it with the active model.")
            if st.button("Make this the active model", type="primary", icon=":material/star:", disabled=not ok,
                         width="stretch"):
                registry.set_active(pick, by=common.user(), reason="activated on Models page")
                st.success("Activated. New batches will use this model.")
                st.rerun()
        st.download_button("Download model file", registry.model_path(pick).read_bytes(),
                           file_name=f"{c.get('name', pick).replace(' ', '_')}.pth", icon=":material/download:",
                           width="stretch", help="Share it with lab members, who import it on this page. It is "
                                                 "derived from NDA images, so share it only within the lab.")

    if ag.get("n"):
        st.markdown("#### How well it did on held-out images")
        st.markdown(describe(ag))
        k = st.columns(5)
        k[0].metric("Avg error", f"{ag['mae']:.2f} pts", help="Mean absolute difference from the hand-traced % damage.")
        k[1].metric("Within ±5 points", f"{100 * ag['within_tol']:.0f}%")
        k[2].metric("Bias", f"{ag['bias']:+.2f} pts", help="Positive = the model calls more damage than the tracer.")
        k[3].metric("Damage overlap (Dice)", fmt(ev.get("dice_damage_pooled"), "{:.2f}"),
                    help="How well the outlines match, pixel for pixel (1 = perfect).")
        k[4].metric("Usable-area overlap", fmt(ev.get("dice_tua_pooled"), "{:.2f}"))
        df = pd.DataFrame(ev.get("per_image", []))
        if len(df):
            df = df.rename(columns={"pct_true": "manual", "pct_pred": "model"})
            g1, g2 = st.columns(2)
            g1.altair_chart(charts.scatter(df, color="vessel", x_title="Hand-traced % damage (from ROIs)"), width="stretch")
            g2.altair_chart(charts.bland_altman(df, stats=ag), width="stretch")
            if ev.get("per_fold"):
                st.markdown("###### Per test vessel")
                st.dataframe(pd.DataFrame([{
                    "Run": f["run"], "Test vessel": ", ".join(f.get("test", [])),
                    "Images": f.get("agreement", {}).get("n"),
                    "Avg error (pts)": fmt(f.get("agreement", {}).get("mae"), "{:.2f}"),
                    "Within ±5": fmt(100 * f.get("agreement", {}).get("within_tol", float("nan")), "{:.0f}%"),
                    "Bias": fmt(f.get("agreement", {}).get("bias"), "{:+.2f}"),
                    "Damage Dice": fmt(f.get("dice_damage_pooled"), "{:.2f}")} for f in ev["per_fold"]]), hide_index=True)
            with st.expander("Every test image"):
                show = df[["name", "vessel", "section", "manual", "model", "dice_damage", "dice_tua"]].copy()
                show["error"] = show["model"] - show["manual"]
                st.dataframe(show.sort_values("error", key=abs, ascending=False), hide_index=True, column_config={
                    "manual": st.column_config.NumberColumn("Manual %", format="%.2f"),
                    "model": st.column_config.NumberColumn("Model %", format="%.2f"),
                    "error": st.column_config.NumberColumn("Model − manual", format="%+.2f"),
                    "dice_damage": st.column_config.NumberColumn("Damage Dice", format="%.2f"),
                    "dice_tua": st.column_config.NumberColumn("Usable-area Dice", format="%.2f")})
    else:
        st.info(ev.get("note") or "This model has no held-out test results. Check it against hand-measured % on the "
                "benchmark page.", icon=":material/info:")

    bms = c.get("benchmarks") or {}
    if bms:
        st.markdown("#### Hand-measured benchmarks")
        st.dataframe(pd.DataFrame([{
            "Benchmark": b["name"], "Images": b.get("n"), "Avg error (pts)": fmt(b.get("mae"), "{:.2f}"),
            "Within ±5": fmt(100 * (b.get("within_tol") or float("nan")), "{:.0f}%"), "Bias": fmt(b.get("bias"), "{:+.2f}"),
            "Run": pretty_time(b.get("run")),
            "Note": ("trained on some of these vessels: " + ", ".join(b["leakage"])) if b.get("leakage") else ""}
            for b in bms.values()]), hide_index=True)

    with st.expander("Rename, add notes or delete"):
        new_name = st.text_input("Name", value=c.get("name", ""), key=f"name_{pick}")
        new_notes = st.text_area("Notes", value=c.get("notes", ""), key=f"notes_{pick}")
        if st.button("Save", key=f"save_{pick}"):
            c["name"], c["notes"] = new_name, new_notes
            registry.save_card(pick, c)
            st.rerun()
        if pick != active:
            if st.checkbox("I want to delete this model permanently", key=f"delok_{pick}"):
                if st.button("Delete model", icon=":material/delete:", key=f"del_{pick}"):
                    registry.delete_model(pick)
                    st.rerun()
        else:
            st.caption("The active model can't be deleted.")
