"""Shared pieces of the Streamlit app: sidebar, folder picker, job progress."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import streamlit as st

from huvec import batches, jobs, registry, workspace
from huvec.utils import clean_path_text, fmt_duration, pretty_time

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
.huvec-sub {color: var(--text-color); opacity: .72; margin-top: -.6rem; margin-bottom: 1.1rem; font-size: 1.02rem;}
.huvec-chip {display:inline-block; padding:2px 10px; margin:2px 6px 2px 0; border-radius:999px; font-size:.82rem;
             background: rgba(255,140,0,.14); border: 1px solid rgba(255,140,0,.45);}
.huvec-chip.ok {background: rgba(40,170,90,.12); border-color: rgba(40,170,90,.45);}
.huvec-chip.no {background: rgba(220,50,50,.12); border-color: rgba(220,50,50,.45);}
.huvec-chip.info {background: rgba(80,120,220,.12); border-color: rgba(80,120,220,.45);}
.huvec-step {font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; opacity: .6; margin-bottom: .1rem;}
.huvec-big {font-size: 2.4rem; font-weight: 700; line-height: 1.1;}
.huvec-muted {opacity: .65; font-size: .9rem;}
div[data-testid="stImage"] img {border-radius: 6px;}
div[data-testid="stMetricValue"] {font-size: 1.65rem;}
</style>
"""


def init_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if "user" not in st.session_state:
        st.session_state["user"] = workspace.local_settings().get("user", "")


def header(title: str, subtitle: str = "") -> None:
    import re

    st.title(title)
    if subtitle:
        html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", subtitle)
        st.markdown(f"<div class='huvec-sub'>{html}</div>", unsafe_allow_html=True)


def user() -> str:
    return (st.session_state.get("user") or "").strip()


def _save_user() -> None:
    workspace.save_local_setting("user", st.session_state.get("user", "").strip())


def sidebar() -> None:
    with st.sidebar:
        st.text_input("Your name", key="user", on_change=_save_user, placeholder="e.g. Jordan K.",
                      help="Saved with your QC decisions and training runs.")
        card = registry.active_card()
        with st.container(border=True):
            st.caption("ACTIVE MODEL")
            if card:
                st.markdown(f"**{card.get('name', card.get('id'))}**")
                ev = (card.get("evaluation") or {}).get("agreement") or {}
                parts = [f"trained {pretty_time(card.get('created'))[:12]}"]
                if ev.get("n"):
                    parts.append(f"±{ev['mae']:.1f} pts avg error")
                st.caption(" · ".join(parts))
            else:
                st.markdown("No model yet")
                st.page_link("ui/page_models.py", label="Import or train one", icon=":material/arrow_forward:")
        running = jobs.running_jobs()
        if running:
            st.caption("RUNNING IN THE BACKGROUND")
            for j in running[:4]:
                frac = _job_fraction(j)
                st.progress(frac, text=f"{j.get('title', j['kind'])}")
        st.caption(f"Workspace: `{workspace.root()}`")


def _job_fraction(j: dict) -> float:
    if j.get("kind") == "train":
        return float(min(1.0, max(0.0, j.get("overall") or 0.0)))
    total = j.get("total") or 0
    return float(min(1.0, (j.get("done") or 0) / total)) if total else 0.0


# ── folders ────────────────────────────────────────────────────────────────

def _pick_folder_dialog(initial: str = "") -> str | None:
    """Open the operating system's folder picker (works because the app runs locally)."""
    code = (
        "import tkinter as tk\nfrom tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw()\n"
        "try:\n    r.wm_attributes('-topmost', 1)\nexcept Exception:\n    pass\n"
        f"p = filedialog.askdirectory(initialdir={initial!r} or None, title='Choose a folder')\n"
        "print(p or '')\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or ""


def folder_input(label: str, key: str, help: str | None = None, placeholder: str = r"e.g. C:\Lab\huvec_inputs") -> str:
    def _browse():
        p = _pick_folder_dialog(st.session_state.get(key, ""))
        if p is None:
            st.session_state[f"{key}__err"] = True
        elif p:
            st.session_state[key] = p

    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    with c1:
        st.text_input(label, key=key, help=help, placeholder=placeholder)
    with c2:
        st.button("Browse…", key=f"{key}__browse", on_click=_browse, width="stretch")
    if st.session_state.pop(f"{key}__err", False):
        st.info("The folder picker isn't available here. Paste the folder path instead "
                "(in Windows Explorer: click the address bar, copy, paste).")
    return clean_path_text(st.session_state.get(key, "") or "")


# ── job progress ───────────────────────────────────────────────────────────

def _render_job(s: dict, show_log: bool) -> None:
    status = s.get("status")
    frac = _job_fraction(s)
    eta = s.get("eta_s")
    label = s.get("message") or ""
    if status in jobs.ACTIVE:
        extra = f" · about {fmt_duration(eta)} left" if eta else ""
        st.progress(frac, text=f"{label}{extra}")
        if st.button("Stop", key=f"stop_{s['id']}", icon=":material/stop_circle:"):
            jobs.request_stop(s["id"])
            st.toast("Stopping after the current step…")
    elif status == "done":
        st.success(label or "Finished.", icon=":material/check_circle:")
    elif status == "stopped":
        st.warning(label or "Stopped.", icon=":material/pause_circle:")
    else:
        st.error(f"{label}", icon=":material/error:")
        if s.get("traceback"):
            with st.expander("Technical details (send these to whoever maintains the app)"):
                st.code(s["traceback"])
    if show_log:
        with st.expander("Log"):
            st.code(jobs.log_tail(s["id"], 60) or "(empty)")


@st.fragment(run_every=2.0)
def _live_job(job_id: str, show_log: bool) -> None:
    s = jobs.status(job_id)
    _render_job(s, show_log)
    if s.get("status") not in jobs.ACTIVE:
        st.rerun()


def job_panel(job_id: str, show_log: bool = False) -> dict:
    """Progress for a background job; refreshes itself while the job runs."""
    s = jobs.status(job_id)
    if s.get("status") in jobs.ACTIVE:
        _live_job(job_id, show_log)
    else:
        _render_job(s, show_log)
    return s


# ── small renderers ────────────────────────────────────────────────────────

def esc(text) -> str:
    import html

    return html.escape(str(text or ""))


def chips(items: list[str], kind: str = "") -> str:
    return "".join(f"<span class='huvec-chip {kind}'>{esc(t)}</span>" for t in items)


def flag_chips(flags: list[dict]) -> None:
    if flags:
        st.markdown(chips([f["text"] for f in flags]), unsafe_allow_html=True)


def num(x, fmt: str = "{:,.0f}", none: str = "—") -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return none
    return none if math.isnan(x) else fmt.format(x)


def batch_selector(key: str = "batch_id", label: str = "Batch") -> str | None:
    bl = batches.list_batches()
    if not bl:
        return None
    ids = [b["id"] for b in bl]
    names = {b["id"]: f"{b['name']} · {b['status']['images']} images · {pretty_time(b['created'])}" for b in bl}
    current = st.session_state.get(key)
    idx = ids.index(current) if current in ids else 0
    choice = st.selectbox(label, ids, index=idx, format_func=lambda i: names[i], key=f"{key}__select")
    st.session_state[key] = choice
    return choice


def active_job_for(kind: str, match: dict) -> dict | None:
    for j in jobs.running_jobs(kind):
        if all(j.get("args", {}).get(k) == v for k, v in match.items()):
            return j
    return None


def path_exists(p: str) -> bool:
    try:
        return bool(p) and Path(p).is_dir()
    except OSError:
        return False
