"""Background jobs (training, batch analysis, benchmark runs).

Long work runs in its own Python process, so it keeps going when the browser
tab is closed or the app window is shut, and the app stays responsive.  Each
job is a folder in ``<workspace>/jobs``:

    job.json        what to do (kind, arguments) + process id
    progress.json   written by the worker: status, progress, messages
    log.txt         everything the worker printed
    STOP            created by the app to ask the worker to stop
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import workspace
from .utils import now_iso, read_json, slugify, stamp, write_json

MODULES = {"train": "huvec.train", "analyze": "huvec.analyze", "benchmark": "huvec.benchmark"}
ACTIVE = ("starting", "running", "stopping")


def _pid_alive(pid: int | None, create_time: float | None = None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        p = psutil.Process(pid)
        if create_time and abs(p.create_time() - create_time) > 2:
            return False  # pid reused by another program
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def start_job(kind: str, args: dict, title: str) -> str:
    job_id = f"{stamp()}_{kind}_{slugify(title, 24)}"
    job_dir = workspace.jobs_dir() / job_id
    job_dir.mkdir(parents=True)
    write_json(job_dir / "job.json", {"id": job_id, "kind": kind, "title": title, "args": args, "created": now_iso()})
    write_json(job_dir / "progress.json", {"status": "starting", "updated": time.time(), "message": "Starting..."})
    env = dict(os.environ, PYTHONUNBUFFERED="1", HUVEC_WORKSPACE=str(workspace.root()))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log = open(job_dir / "log.txt", "a", encoding="utf-8")
    kw: dict = {}
    if os.name == "nt":
        # Own (hidden) console: closing the app's window doesn't kill the job.
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen([sys.executable, "-m", MODULES[kind], "--job-dir", str(job_dir)],
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            cwd=str(workspace.APP_ROOT), env=env, **kw)
    log.close()  # the child process keeps its own handle
    ct = None
    try:
        import psutil

        ct = psutil.Process(proc.pid).create_time()
    except Exception:
        pass
    info = read_json(job_dir / "job.json", {})
    info.update(pid=proc.pid, pid_create_time=ct)
    write_json(job_dir / "job.json", info)
    return job_id


def job_dir(job_id: str) -> Path:
    return workspace.jobs_dir() / job_id


def status(job_id: str) -> dict:
    d = job_dir(job_id)
    info = read_json(d / "job.json", {}) or {}
    prog = read_json(d / "progress.json", {}) or {}
    out = {**prog, "id": job_id, "kind": info.get("kind"), "title": info.get("title"),
           "created": info.get("created"), "args": info.get("args", {})}
    st = prog.get("status", "starting")
    if st in ACTIVE and not _pid_alive(info.get("pid"), info.get("pid_create_time")):
        # Give a just-launched process a moment to write its first progress.
        if time.time() - float(prog.get("updated", 0)) > 20:
            out["status"] = "crashed"
            out["message"] = "The job stopped unexpectedly (computer restarted, went to sleep or the process was closed)."
    return out


def list_jobs(kind: str | None = None, limit: int = 50) -> list[dict]:
    dirs = sorted((p for p in workspace.jobs_dir().iterdir() if p.is_dir()), reverse=True)
    out = []
    for d in dirs:
        info = read_json(d / "job.json", {}) or {}
        if kind and info.get("kind") != kind:
            continue
        out.append(status(d.name))
        if len(out) >= limit:
            break
    return out


def running_jobs(kind: str | None = None) -> list[dict]:
    return [j for j in list_jobs(kind, limit=30) if j.get("status") in ACTIVE]


def request_stop(job_id: str) -> None:
    (job_dir(job_id) / "STOP").write_text("stop requested " + now_iso())


def log_tail(job_id: str, n: int = 40) -> str:
    p = job_dir(job_id) / "log.txt"
    if not p.exists():
        return ""
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


# ── used inside worker processes ──────────────────────────────────────────

class Job:
    def __init__(self, job_dir: str | Path):
        self.dir = Path(job_dir)
        self.info = read_json(self.dir / "job.json", {}) or {}
        self.args = self.info.get("args", {})
        self.progress = read_json(self.dir / "progress.json", {}) or {}
        self._last_write = 0.0

    def update(self, force: bool = False, **fields) -> None:
        self.progress.update(fields)
        self.progress["updated"] = time.time()
        now = time.time()
        if force or now - self._last_write > 1.0:
            write_json(self.dir / "progress.json", self.progress)
            self._last_write = now

    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def should_stop(self) -> bool:
        return (self.dir / "STOP").exists()

    def finish(self, status: str = "done", **fields) -> None:
        self.update(force=True, status=status, finished=now_iso(), **fields)


def keep_awake(on: bool = True) -> None:
    """Ask Windows not to sleep while a long job runs (no effect elsewhere)."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def run_worker(main) -> None:
    """Entry point helper for ``python -m huvec.<job> --job-dir DIR``."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    a = ap.parse_args()
    job = Job(a.job_dir)
    job.update(force=True, status="running", started=now_iso(), message="Starting...")
    keep_awake(True)
    try:
        main(job)
        if job.progress.get("status") in ACTIVE:
            job.finish("stopped" if job.should_stop() else "done")
    except Exception as exc:  # report the error to the app instead of dying silently
        traceback.print_exc()
        job.finish("failed", message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
    finally:
        keep_awake(False)
