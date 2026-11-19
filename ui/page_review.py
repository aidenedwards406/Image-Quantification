import math

import streamlit as st

from huvec import batches
from huvec.batches import APPROVED, REJECTED, REJECT_REASONS
from ui import common

common.header("2 · QC review",
              "Check every image. **Yes** keeps the model's numbers. **No** sends the image to the tracing list, "
              "to be traced by hand in ImageJ and typed into Excel.")

bid = common.batch_selector()
if not bid:
    st.info("No batches yet. Analyze some images first.", icon=":material/info:")
    st.page_link("ui/page_analyze.py", label="1 · Analyze images", icon=":material/upload_file:")
    st.stop()

b = batches.load(bid)
res = batches.results(bid)
rev = batches.review(bid)
s = batches.summary(bid, b)
images = sorted(b["images"], key=lambda im: im["name"].lower())
by_stem = {im["stem"]: im for im in images}

job = common.active_job_for("analyze", {"batch_id": bid})
if job:
    with st.container(border=True):
        st.caption("Still analyzing. You can review the finished images now.")
        common.job_panel(job["id"])

m = st.columns(5)
m[0].metric("To review", s["to_review"])
m[1].metric("Approved", s["approved"])
m[2].metric("Needs tracing", s["rejected"])
m[3].metric("Flagged by the app", s["flagged"], help="Blurry, small usable area, model unsure or specks. Look at these carefully.")
m[4].metric("Analysed", f"{s['analysed']}/{s['images']}")
if s["analysed"]:
    st.progress(s["reviewed"] / max(1, s["analysed"] - s["errors"]) if s["analysed"] > s["errors"] else 1.0)

if not common.user():
    st.caption(":material/person: Add your name in the sidebar so your decisions are recorded under it.")


# ── which images to show ───────────────────────────────────────────────────
def status_of(stem: str) -> str:
    r = res.get(stem)
    if r is None:
        return "pending"
    if r.get("error"):
        return "error"
    return rev.get(stem, {}).get("decision") or "todo"


cur_key = f"cur_{bid}"


def _restart() -> None:
    st.session_state.pop(cur_key, None)


fc = st.columns([3, 2], vertical_alignment="bottom")
show = fc[0].segmented_control("Show", ["To review", "Flagged", "Approved", "Needs tracing", "All"],
                               default="To review", key=f"filter_{bid}", on_change=_restart) or "To review"
order = fc[1].selectbox("Order", ["File name", "Flagged first", "Highest % damage first", "Lowest % damage first"],
                        key=f"order_{bid}", on_change=_restart)


def keep(stem: str) -> bool:
    st_ = status_of(stem)
    if st_ == "pending":
        return False
    if show == "To review":
        return st_ == "todo"
    if show == "Flagged":
        return bool(res[stem].get("flags"))
    if show == "Approved":
        return st_ == APPROVED
    if show == "Needs tracing":
        return st_ in (REJECTED, "error")
    return True


stems = [im["stem"] for im in images if keep(im["stem"])]


def pct_of(stem):
    v = res.get(stem, {}).get("pct_damage")
    return -1 if v is None or (isinstance(v, float) and math.isnan(v)) else v


if order == "Flagged first":
    stems.sort(key=lambda x: (not res[x].get("flags"),))
elif order == "Highest % damage first":
    stems.sort(key=lambda x: -pct_of(x))
elif order == "Lowest % damage first":
    stems.sort(key=pct_of)

if not stems:
    if show == "To review" and s["analysed"] and not s["to_review"]:
        st.success(f"All {s['analysed']} analysed images are reviewed: {s['approved']} approved, "
                   f"{s['rejected']} need{'s' if s['rejected'] == 1 else ''} tracing.", icon=":material/task_alt:")
        st.page_link("ui/page_export.py", label="3 · Export to Excel", icon=":material/table_view:")
    else:
        st.info("No images match this filter.")
    st.stop()
if st.session_state.get(cur_key) not in stems:
    st.session_state[cur_key] = stems[0]


# ── actions (callbacks run before the page redraws) ───────────────────────
def _neighbour(stem: str, step: int) -> str:
    i = stems.index(stem)
    return stems[min(max(i + step, 0), len(stems) - 1)]


def decide(stem: str, decision: str | None) -> None:
    reasons = st.session_state.get(f"reasons_{bid}_{stem}") or []
    note = st.session_state.get(f"note_{bid}_{stem}", "")
    batches.set_decision(bid, stem, decision, reasons if decision == REJECTED else [], note, common.user())
    if decision is not None:
        i = stems.index(stem)
        rest = [x for x in stems[i + 1:] + stems[:i] if status_of(x) == "todo" and x != stem]
        st.session_state[cur_key] = rest[0] if rest else _neighbour(stem, 1)


def go(stem: str, step: int) -> None:
    st.session_state[cur_key] = _neighbour(stem, step)


mode = st.segmented_control("Mode", ["One at a time", "Overview grid"], default="One at a time", key="review_mode",
                            label_visibility="collapsed") or "One at a time"

if mode == "One at a time":
    stem = st.session_state[cur_key]
    im, r, d = by_stem[stem], res[stem], rev.get(stem, {})
    pos = stems.index(stem) + 1
    head = st.columns([6, 2], vertical_alignment="center")
    head[0].markdown(f"### {im['name']}")
    head[1].markdown(f"<div style='text-align:right' class='huvec-muted'>{pos} of {len(stems)} ({show.lower()})</div>",
                     unsafe_allow_html=True)
    if d.get("decision") == APPROVED:
        st.markdown(common.chips([f"Approved by {d.get('by') or '—'}"], "ok"), unsafe_allow_html=True)
    elif d.get("decision") == REJECTED:
        why = f": {', '.join(d['reasons'])}" if d.get("reasons") else ""
        st.markdown(common.chips([f"Needs tracing{why}"], "no"), unsafe_allow_html=True)
    common.flag_chips(r.get("flags", []))

    if r.get("error"):
        st.error(f"This image couldn't be analysed ({r['error']}). It goes to the tracing list automatically.")
    else:
        view = st.segmented_control("View", ["Side by side", "Original", "Usable area", "Damage", "Focus map", "Specks"],
                                    default="Side by side", key="view") or "Side by side"
        path = lambda v: str(batches.render_path(bid, stem, v))  # noqa: E731
        if view == "Side by side":
            c = st.columns(3)
            c[0].image(path("original"), caption="Original", width="stretch")
            c[1].image(path("tua"), caption="Usable area (yellow outline; dark = ignored)", width="stretch")
            c[2].image(path("damage"), caption="Damage (red)", width="stretch")
        else:
            key = {"Original": "original", "Usable area": "tua", "Damage": "damage", "Focus map": "focus",
                   "Specks": "specks"}[view]
            caption = {"focus": "Green = in focus, orange = blurry, no tint = bare/featureless or dark",
                       "specks": "Cyan circles = small black specks"}.get(key, "")
            st.image(path(key), caption=caption or None, width="stretch")

        k = st.columns(7)
        k[0].markdown(f"<div class='huvec-step'>% damage</div><div class='huvec-big'>{common.num(r.get('pct_damage'), '{:.1f}%')}</div>",
                      unsafe_allow_html=True)
        k[1].metric("Usable area (px)", common.num(r.get("tua_px")))
        k[2].metric("Damaged (px)", common.num(r.get("damage_px")))
        k[3].metric("In focus", common.num(r.get("in_focus_pct"), "{:.0f}%"),
                    help="Share of the image with cell texture that is sharp (not blurry).")
        k[4].metric("Blurry in usable area", common.num(r.get("blurry_in_tua_pct"), "{:.0f}%"))
        k[5].metric("Specks", common.num(r.get("specks")))
        k[6].metric("Model unsure", common.num(r.get("uncertain_pct"), "{:.0f}%"),
                    help="Share of the usable area where the model was close to 50/50.")

    st.markdown("")
    btn = st.columns([3, 3, 1.3, 1.3], vertical_alignment="center")
    btn[0].button("Yes, looks right", type="primary", icon=":material/check:", shortcut="Y", width="stretch",
                  on_click=decide, args=(stem, APPROVED), key="yes", disabled=bool(r.get("error")))
    btn[1].button("No, trace by hand", icon=":material/close:", shortcut="N", width="stretch",
                  on_click=decide, args=(stem, REJECTED), key="no")
    btn[2].button("Previous", icon=":material/chevron_left:", shortcut="Left", width="stretch",
                  on_click=go, args=(stem, -1), key="prev", disabled=pos == 1)
    btn[3].button("Next", icon=":material/chevron_right:", shortcut="Right", width="stretch",
                  on_click=go, args=(stem, 1), key="next", disabled=pos == len(stems))
    with st.expander("If No: why? (optional, choose before pressing No)", expanded=bool(d.get("reasons"))):
        st.pills("Reasons", REJECT_REASONS, selection_mode="multi", key=f"reasons_{bid}_{stem}",
                 default=d.get("reasons") or None, label_visibility="collapsed")
        st.text_input("Note", key=f"note_{bid}_{stem}", value=d.get("note", ""), placeholder="Optional note")
    if d.get("decision"):
        st.button("Undo my decision for this image", icon=":material/undo:", on_click=decide, args=(stem, None),
                  type="tertiary")
    st.caption("Keyboard: **Y** = yes, **N** = no, **←/→** = previous/next. Pictures open full screen from "
               "the icon in their corner.")

else:
    per_page = 24
    pages = max(1, math.ceil(len(stems) / per_page))
    page = st.number_input("Page", 1, pages, 1, key=f"grid_page_{bid}") if pages > 1 else 1
    chunk = stems[(page - 1) * per_page: page * per_page]
    for row in range(0, len(chunk), 4):
        cols = st.columns(4)
        for col, x in zip(cols, chunk[row:row + 4]):
            r = res[x]
            with col.container(border=True):
                p = batches.render_path(bid, x, "thumb_damage")
                if p.exists():
                    st.image(str(p), width="stretch")
                stt = status_of(x)
                badge = {"approved": ":green[✓ approved]", "rejected": ":red[✗ needs tracing]", "error": ":red[failed]"}.get(stt, "")
                st.markdown(f"<span class='huvec-muted'>{common.esc(by_stem[x]['name'])}</span>", unsafe_allow_html=True)
                st.markdown(f"**{common.num(r.get('pct_damage'), '{:.1f}%')}** damage {badge}"
                            + (" · :orange[flagged]" if r.get("flags") else ""))
                st.button("Open", key=f"open_{x}", type="tertiary", icon=":material/open_in_full:",
                          on_click=lambda x=x: st.session_state.update({cur_key: x, "review_mode": "One at a time"}))
