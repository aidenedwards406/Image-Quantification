"""Training job: ``python -m huvec.train --job-dir DIR`` (started by the app).

Two evaluation modes, both splitting by *physical vessel* (never by image):

* ``split``: you choose train / validation / test vessels.  The model is
  trained on the train vessels, the validation vessel picks the best moment
  to stop, and the test vessel gives the honest score.
* ``cv``: leave-one-vessel-out cross-validation.  With only ~5 vessels this is
  the most trustworthy estimate: every vessel is the test vessel once
  (K trainings).  A final model is then trained on all vessels.  It takes
  about K+1 times as long, so it suits an overnight or weekend run.

The finished model is registered in the workspace (not activated: you compare
and activate it on the Models page).
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import __version__, library, registry, workspace
from . import settings as settings_mod
from .data import TileDataset, VOID, class_weights, prepare, soft_dice_loss
from .imageio import read_image
from .jobs import Job, run_worker
from .metrics import agreement, image_scores, pooled_dice
from .model import DamageUNet, _forward_padded, configure_threads, predict_probs, save_checkpoint, to_tensor
from .predict import analyze_probs
from .rois import load_annotation
from .utils import now_iso

PRESETS = {
    "quick": {"steps": 1200, "eval_every": 150, "patience": 0, "label": "Quick check"},
    "standard": {"steps": 4000, "eval_every": 250, "patience": 6, "label": "Standard"},
    "thorough": {"steps": 8000, "eval_every": 400, "patience": 8, "label": "Thorough"},
}
DEFAULTS = {"tile": 384, "batch": 4, "lr": 3e-4, "weight_decay": 1e-4, "scale": 1.0,
            "pretrained": True, "augment": True, "seed": 0}


def build_config(preset: str, overrides: dict | None = None) -> dict:
    cfg = {**DEFAULTS, **PRESETS[preset], "preset": preset}
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


# ── evaluation ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_cached(model, items, threshold: float = 0.5) -> dict:
    """Quick validation at training scale (used to pick the best checkpoint)."""
    model.eval()
    rows = []
    for it in items:
        img = np.load(it.image_npy)
        gt = np.load(it.label_npy)
        logits = _forward_padded(model, to_tensor(img, it.factor)[None])
        p = torch.softmax(logits, 1)[0].numpy()
        tua = p[0] < 0.5
        ratio = p[2] / (p[1] + p[2] + 1e-6)
        pred = np.where(tua, np.where(ratio >= threshold, 2, 1), 0).astype(np.uint8)
        rows.append(image_scores(pred, gt))
    ag = agreement([r["pct_pred"] for r in rows], [r["pct_true"] for r in rows])
    pooled = pooled_dice(rows)
    return {"mae": ag.get("mae", float("nan")), "bias": ag.get("bias", float("nan")),
            "within5": ag.get("within_tol", float("nan")), **pooled}


def evaluate_full(model, meta: dict, rows: list[dict], settings: dict, job: Job | None = None,
                  label: str = "") -> list[dict]:
    """Test-set evaluation with the exact production pipeline (full resolution,
    clean-up, thresholds) so reported numbers match what users will see."""
    out = []
    for i, r in enumerate(rows):
        if job and job.should_stop():
            break
        if job:
            job.update(message=f"{label}Testing on held-out image {i + 1}/{len(rows)}: {r['name']}")
        img = read_image(r["path"])
        H, W = img.shape[:2]
        probs = predict_probs(model, img, meta)
        an = analyze_probs(img, probs, meta, settings)
        gt = load_annotation(r.get("roi_paths", []), W, H).label()
        s = image_scores(an.label, gt)
        out.append({"name": r["name"], "vessel": r.get("vessel", ""), "section": str(r.get("section", "")),
                    "pct_true": s["pct_true"], "pct_pred": an.numbers["pct_damage"],
                    "tua_px_true": s["tua_px_true"], "tua_px_pred": s["tua_px_pred"],
                    "dice_damage": s["dice_damage"], "dice_tua": s["dice_tua"],
                    "dmg_inter": s["dmg_inter"], "dmg_pred": s["dmg_pred"], "dmg_gt": s["dmg_gt"],
                    "tua_inter": s["tua_inter"], "tua_pred": s["tua_pred"], "tua_gt": s["tua_gt"],
                    "flags": [f["code"] for f in an.numbers["flags"]]})
    return out


def summarize(per_image: list[dict]) -> dict:
    if not per_image:
        return {}
    ag = agreement([r["pct_pred"] for r in per_image], [r["pct_true"] for r in per_image])
    dmg_imgs = [r["dice_damage"] for r in per_image if r["dmg_gt"] or r["dmg_pred"]]
    return {"agreement": ag, **pooled_dice(per_image),
            "mean_dice_damage": float(np.mean(dmg_imgs)) if dmg_imgs else float("nan"),
            "mean_dice_tua": float(np.mean([r["dice_tua"] for r in per_image]))}


# ── one training run ───────────────────────────────────────────────────────

def _thin(points: list, n: int = 300) -> list:
    if len(points) <= n:
        return points
    step = math.ceil(len(points) / n)
    return points[::step] + [points[-1]]


class _Clock:
    """Tracks progress across several trainings for one overall ETA."""

    def __init__(self, planned_steps: int):
        self.planned = max(1, planned_steps)
        self.done = 0
        self.t0 = time.time()

    def eta(self) -> float | None:
        if self.done < 5:
            return None
        rate = (time.time() - self.t0) / self.done
        return rate * max(0, self.planned - self.done)


def train_one(job: Job, cfg: dict, train_items, val_items, steps: int, clock: _Clock, label: str):
    torch.manual_seed(cfg["seed"])
    model = DamageUNet(pretrained=cfg["pretrained"])
    if cfg["pretrained"] and not model.pretrained_loaded:
        job.log("ImageNet weights could not be downloaded - training from scratch (works, may need more steps).")
    weights = class_weights(train_items)
    ce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=VOID)
    opt = torch.optim.AdamW([
        {"params": list(model.encoder_parameters()), "lr": cfg["lr"] * 0.5},
        {"params": list(model.decoder_parameters()), "lr": cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])
    warm = max(1, int(0.04 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    ds = TileDataset(train_items, cfg["tile"], steps * cfg["batch"], augment=cfg["augment"], seed=cfg["seed"])
    dl = DataLoader(ds, batch_size=cfg["batch"], shuffle=False, num_workers=0, drop_last=True)

    losses, evals = [], []
    best_score, best_state, best_step, stale = math.inf, None, 0, 0
    ema = None
    step = 0
    for x, y in dl:
        model.train()
        logits = model(x)
        loss = ce(logits, y) + 0.5 * soft_dice_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        clock.done += 1
        lv = float(loss.detach())
        ema = lv if ema is None else 0.95 * ema + 0.05 * lv
        if step % 10 == 0 or step == 1:
            losses.append({"step": step, "loss": round(ema, 4), "run": label})
        job.update(phase="training", run=label, step=step, steps=steps, loss=round(ema, 4),
                   overall=clock.done / clock.planned, eta_s=clock.eta(),
                   message=f"{label}Training step {step}/{steps}",
                   losses=job.progress.get("losses_prev", []) + _thin(losses), evals=job.progress.get("evals_prev", []) + evals)
        if job.should_stop():
            job.log("Stop requested - keeping the best model so far.")
            break
        if val_items and (step % cfg["eval_every"] == 0 or step == steps):
            job.update(force=True, message=f"{label}Checking validation images (step {step})")
            ev = evaluate_cached(model, val_items)
            score = ev["mae"] + 2.0 * (1.0 - (ev["dice_damage_pooled"] if np.isfinite(ev["dice_damage_pooled"]) else 0.0))
            ev.update(step=step, run=label, score=score)
            evals.append(ev)
            job.log(f"{label}step {step}: val % damage error {ev['mae']:.2f} pts, damage Dice {ev['dice_damage_pooled']:.3f}, "
                    f"usable-area Dice {ev['dice_tua_pooled']:.3f}")
            if score < best_score - 1e-6:
                best_score, best_state, best_step, stale = score, copy.deepcopy(model.state_dict()), step, 0
            else:
                stale += 1
            if cfg["patience"] and stale >= cfg["patience"]:
                job.log(f"{label}No improvement for {stale} checks - stopping early at step {step} (best: step {best_step}).")
                clock.planned -= steps - step
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_step = step
    job.progress["losses_prev"] = job.progress.get("losses_prev", []) + _thin(losses)
    job.progress["evals_prev"] = job.progress.get("evals_prev", []) + evals
    return model, {"best_step": best_step, "steps_run": step, "evals": evals, "losses": losses}


# ── job entry ──────────────────────────────────────────────────────────────

def main(job: Job) -> None:
    a = job.args
    cfg = build_config(a.get("preset", "standard"), a.get("overrides"))
    threads = configure_threads()
    job.log(f"Training with {threads} CPU threads; config: {cfg}")
    settings = a.get("settings") or settings_mod.load()

    rows = library.training_rows(a["set_ids"])
    roles = a.get("roles", {})
    rows = [r for r in rows if roles.get(r["vessel"], "train") != "exclude"]
    if not rows:
        raise RuntimeError("No usable training images were found in the chosen training sets.")

    job.update(force=True, phase="preparing", message="Preparing images (first time only)...")
    prepared = {}
    for i, r in enumerate(rows):
        job.update(message=f"Preparing image {i + 1}/{len(rows)}: {r['name']}", overall=0.0)
        prepared[r["name"] + "|" + r["set_id"]] = prepare(r, cfg["scale"])
        if job.should_stop():
            return

    def items_for(subset):
        return [prepared[r["name"] + "|" + r["set_id"]] for r in subset]

    mode = a.get("mode", "split")
    vessels = sorted({r["vessel"] for r in rows})
    plan = []
    if mode == "cv":
        if len(vessels) < 3:
            raise RuntimeError("Cross-validation needs at least 3 vessels.")
        for k, test_v in enumerate(vessels):
            val_v = vessels[(k + 1) % len(vessels)]
            plan.append({"label": f"Fold {k + 1}/{len(vessels)} (test {test_v}) · ", "test": [test_v], "val": [val_v],
                         "train": [v for v in vessels if v not in (test_v, val_v)]})
        plan.append({"label": "Final model (all vessels) · ", "test": [], "val": [], "train": vessels, "final": True})
    else:
        plan.append({"label": "", "train": [v for v in vessels if roles.get(v, "train") == "train"],
                     "val": [v for v in vessels if roles.get(v) == "val"],
                     "test": [v for v in vessels if roles.get(v) == "test"]})
    if not plan[0]["train"]:
        raise RuntimeError("No vessels are assigned to training.")

    clock = _Clock(cfg["steps"] * len(plan))
    job.update(force=True, phase="training", plan=[{k: v for k, v in p.items()} for p in plan], overall=0.0)
    per_image, per_fold, curves = [], [], []
    final_model, final_info = None, None
    for p in plan:
        if job.should_stop():
            break
        steps = cfg["steps"]
        if p.get("final"):
            best_steps = [f["best_step"] for f in per_fold if f.get("best_step")]
            steps = int(max(300, np.median(best_steps) * 1.1)) if best_steps else cfg["steps"]
            clock.planned += steps - cfg["steps"]
            job.log(f"Final model: training on all {len(vessels)} vessels for {steps} steps "
                    f"(median best step of the folds).")
        tr = items_for([r for r in rows if r["vessel"] in p["train"]])
        va = items_for([r for r in rows if r["vessel"] in p["val"]])
        job.log(f"{p['label']}train vessels {p['train']} ({len(tr)} images), validation {p['val']} ({len(va)}), test {p['test']}")
        model, info = train_one(job, cfg, tr, va, steps, clock, p["label"])
        curves.append({"run": p["label"], **{k: info[k] for k in ("best_step", "steps_run")}})
        meta = {"scale": cfg["scale"], "damage_threshold": 0.5}
        test_rows = [r for r in rows if r["vessel"] in p["test"]]
        if test_rows:
            res = evaluate_full(model, meta, test_rows, settings, job, p["label"])
            for r in res:
                r["fold"] = p["label"].strip(" ·") or "test"
            per_image += res
            per_fold.append({"run": p["label"].strip(" ·") or "Held-out test vessel", "test": p["test"],
                             "best_step": info["best_step"], **summarize(res)})
        if p.get("final") or mode != "cv":
            final_model, final_info = model, info
        if job.should_stop() and not p.get("final") and mode == "cv":
            break
    if final_model is None:
        job.finish("stopped", message="Stopped before a model was finished - nothing was saved.")
        return

    evaluation = {"mode": mode, "per_image": per_image, "per_fold": per_fold, **summarize(per_image)}
    if not per_image:
        evaluation["note"] = "No test vessel was held out, so this model has no honest test score."
    name = a.get("name") or "Damage model"
    model_id = registry.new_id(name)
    card = {
        "id": model_id, "name": name, "created": now_iso(), "trained_by": a.get("trained_by", ""),
        "notes": a.get("notes", ""), "arch": DamageUNet.arch, "app_version": __version__,
        "scale": cfg["scale"], "tile": cfg["tile"], "preset": cfg["preset"], "config": cfg,
        "pretrained_imagenet": bool(getattr(final_model, "pretrained_loaded", False)),
        "damage_threshold": 0.5,
        "training_data": {
            "sets": [{"id": s, "name": library.get_set(s).get("name", s)} for s in a["set_ids"]],
            "vessels": {"train": plan[-1]["train"] if mode == "cv" else plan[0]["train"],
                        "val": plan[0]["val"] if mode != "cv" else [],
                        "test": plan[0]["test"] if mode != "cv" else [],
                        "excluded": sorted(v for v, role in roles.items() if role == "exclude")},
            "n_images": len(rows),
        },
        "evaluation": evaluation,
        "training": {"runs": curves, "steps_final": final_info["steps_run"] if final_info else None},
        "stopped_early_by_user": job.should_stop(),
    }
    meta = {"scale": cfg["scale"], "damage_threshold": 0.5, "card": card}
    tmp = workspace.jobs_dir() / job.dir.name / "model.pth"
    save_checkpoint(final_model, tmp, meta)
    registry.register(tmp, card, move=True)
    job.log(f"Saved model {model_id}")
    job.finish("stopped" if job.should_stop() else "done", model_id=model_id, overall=1.0, eta_s=0,
               message=f"Finished - model '{name}' saved.")


# ── time estimate for the Train page ───────────────────────────────────────

def estimate_seconds(cfg: dict, n_train: int, n_val: int, n_test: int, runs: int = 1,
                     image_hw: tuple[int, int] = (1040, 1392)) -> dict:
    """Measure this computer's speed on a few dummy steps and extrapolate."""
    configure_threads()
    model = DamageUNet(pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(cfg["batch"], 3, cfg["tile"], cfg["tile"])
    y = torch.randint(0, 3, (cfg["batch"], cfg["tile"], cfg["tile"]))
    times = []
    for i in range(3):
        t = time.time()
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i:
            times.append(time.time() - t)
    step_s = float(np.mean(times))
    h, w = (round(image_hw[0] * cfg["scale"]), round(image_hw[1] * cfg["scale"]))
    model.eval()
    with torch.inference_mode():
        t = time.time()
        _forward_padded(model, torch.randn(1, 3, h, w))
        img_s = time.time() - t
    with torch.inference_mode():
        t = time.time()
        _forward_padded(model, torch.randn(1, 3, image_hw[0], image_hw[1]))
        full_s = time.time() - t + 0.6  # + focus/specks/clean-up
    n_evals = cfg["steps"] / cfg["eval_every"]
    per_run = cfg["steps"] * step_s + n_evals * n_val * img_s + n_test * full_s
    return {"step_s": step_s, "image_s": full_s, "per_run_s": per_run, "total_s": per_run * runs}


if __name__ == "__main__":
    run_worker(main)
