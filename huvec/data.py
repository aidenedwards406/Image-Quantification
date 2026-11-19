"""Training data: cached arrays and random augmented tiles.

Each traced image is decoded once, resized to the training scale, turned into
a 3-class label image (via ``huvec.rois``) and cached as ``.npy`` files in
``<workspace>/cache``.  Training then samples random square tiles, with extra
weight on damaged areas and usable-area edges, and applies augmentation:

* flips / 90-degree rotations (orientation is not meaningful),
* small zoom changes,
* brightness, contrast, gamma, stain colour (per-channel gain, saturation)
  and camera noise, so the model tolerates the variation between imaging
  sessions and stain batches.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import DAMAGED, IGNORE, workspace
from .imageio import read_image
from .model import IMAGENET_MEAN, IMAGENET_STD, brightness_factor, resize_image
from .morph import outline
from .rois import load_annotation

CACHE_VERSION = 2
VOID = 255  # label value excluded from the loss (padding)


@dataclass
class Prepared:
    name: str
    vessel: str
    section: str
    image_npy: str
    label_npy: str
    coords_npz: str
    factor: float
    pct_damage: float
    height: int
    width: int
    path: str
    roi_paths: list

    def as_dict(self) -> dict:
        return asdict(self)


def _cache_key(row: dict, scale: float) -> str:
    parts = [str(CACHE_VERSION), f"{scale:.4f}"]
    for p in [row["path"], *row.get("roi_paths", [])]:
        try:
            st = os.stat(p)
            parts.append(f"{p}|{st.st_size}|{int(st.st_mtime)}")
        except OSError:
            parts.append(p)
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()


def prepare(row: dict, scale: float) -> Prepared:
    key = _cache_key(row, scale)
    d = workspace.cache_dir() / "train" / key[:2] / key
    meta_path = d / "meta.json"
    if meta_path.exists():
        try:
            return Prepared(**json.loads(meta_path.read_text()))
        except Exception:
            pass
    d.mkdir(parents=True, exist_ok=True)
    img = read_image(row["path"])
    H, W = img.shape[:2]
    ann = load_annotation(row.get("roi_paths", []), W, H, scale)
    small = resize_image(img, scale)
    lab = ann.label()
    if small.shape[:2] != lab.shape:  # rounding guard
        lab = F.interpolate(torch.from_numpy(lab)[None, None].float(), size=small.shape[:2], mode="nearest")[0, 0].byte().numpy()
    rng = np.random.default_rng(0)
    dmg = np.argwhere(lab == DAMAGED)
    usable = lab != IGNORE
    edge = outline(usable, 3) & ~outline(np.ones_like(usable), 3)  # usable-area border, not the image border
    edge_pts = np.argwhere(edge)
    if len(dmg) > 4000:
        dmg = dmg[rng.choice(len(dmg), 4000, replace=False)]
    if len(edge_pts) > 4000:
        edge_pts = edge_pts[rng.choice(len(edge_pts), 4000, replace=False)]
    np.save(d / "image.npy", small)
    np.save(d / "label.npy", lab)
    np.savez(d / "coords.npz", damage=dmg.astype(np.int32), edge=edge_pts.astype(np.int32))
    p = Prepared(
        name=row["name"], vessel=row.get("vessel", ""), section=str(row.get("section", "")),
        image_npy=str(d / "image.npy"), label_npy=str(d / "label.npy"), coords_npz=str(d / "coords.npz"),
        factor=brightness_factor(img), pct_damage=float(ann.pct_damage), height=int(lab.shape[0]),
        width=int(lab.shape[1]), path=row["path"], roi_paths=list(row.get("roi_paths", [])),
    )
    meta_path.write_text(json.dumps(p.as_dict()))
    return p


def class_weights(items: list[Prepared]) -> torch.Tensor:
    counts = np.zeros(3, np.float64)
    for it in items:
        lab = np.load(it.label_npy, mmap_mode="r")
        counts += np.bincount(np.asarray(lab[::4, ::4]).ravel(), minlength=256)[:3]
    freq = counts / max(counts.sum(), 1)
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-4))
    w = w / w.mean()
    return torch.tensor(np.clip(w, 0.3, 5.0), dtype=torch.float32)


class TileDataset(Dataset):
    def __init__(self, items: list[Prepared], tile: int, length: int, augment: bool = True,
                 seed: int = 0, p_damage: float = 0.4, p_edge: float = 0.2):
        self.items, self.tile, self.length = items, tile, length
        self.augment, self.seed = augment, seed
        self.p_damage, self.p_edge = p_damage, p_edge
        self._cache: dict[int, tuple] = {}

    def __len__(self) -> int:
        return self.length

    def _arrays(self, i: int):
        if i not in self._cache:
            it = self.items[i]
            c = np.load(it.coords_npz)
            self._cache[i] = (np.load(it.image_npy, mmap_mode="r"), np.load(it.label_npy, mmap_mode="r"),
                              c["damage"], c["edge"])
        return self._cache[i]

    def __getitem__(self, idx: int):
        rng = np.random.default_rng([self.seed, idx])
        i = int(rng.integers(len(self.items)))
        it = self.items[i]
        img, lab, dmg, edge = self._arrays(i)
        h, w = lab.shape
        zoom = float(rng.uniform(0.85, 1.18)) if self.augment else 1.0
        crop = max(32, int(round(self.tile * zoom)))
        u = rng.random()
        if u < self.p_damage and len(dmg):
            cy, cx = dmg[rng.integers(len(dmg))]
        elif u < self.p_damage + self.p_edge and len(edge):
            cy, cx = edge[rng.integers(len(edge))]
        else:
            cy, cx = rng.integers(h), rng.integers(w)
        jitter = crop // 3
        cy = int(cy) + int(rng.integers(-jitter, jitter + 1))
        cx = int(cx) + int(rng.integers(-jitter, jitter + 1))
        y0 = int(np.clip(cy - crop // 2, 0, max(h - crop, 0)))
        x0 = int(np.clip(cx - crop // 2, 0, max(w - crop, 0)))
        patch = np.array(img[y0:y0 + crop, x0:x0 + crop], dtype=np.uint8)
        lpatch = np.array(lab[y0:y0 + crop, x0:x0 + crop], dtype=np.uint8)
        ph, pw = patch.shape[:2]
        if ph < crop or pw < crop:  # image smaller than the tile
            patch = np.pad(patch, ((0, crop - ph), (0, crop - pw), (0, 0)), mode="edge")
            lpatch = np.pad(lpatch, ((0, crop - ph), (0, crop - pw)), constant_values=VOID)

        x = torch.from_numpy(patch).permute(2, 0, 1).float().div_(255.0) * it.factor
        y = torch.from_numpy(lpatch.astype(np.int64))
        if crop != self.tile:
            x = F.interpolate(x[None], size=(self.tile, self.tile), mode="bilinear", align_corners=False, antialias=crop > self.tile)[0]
            y = F.interpolate(y[None, None].float(), size=(self.tile, self.tile), mode="nearest")[0, 0].long()
        if self.augment:
            x, y = self._geometric(x, y, rng)
            x = self._photometric(x, rng)
        x = x.clamp(0, 1)
        return (x - IMAGENET_MEAN) / IMAGENET_STD, y

    @staticmethod
    def _geometric(x, y, rng):
        k = int(rng.integers(4))
        if k:
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (0, 1))
        if rng.random() < 0.5:
            x, y = torch.flip(x, (2,)), torch.flip(y, (1,))
        return x.contiguous(), y.contiguous()

    @staticmethod
    def _photometric(x, rng):
        x = x * float(rng.uniform(0.85, 1.15))  # brightness
        m = x.mean()
        x = (x - m) * float(rng.uniform(0.85, 1.15)) + m  # contrast
        x = x * torch.tensor(rng.uniform(0.93, 1.07, 3), dtype=torch.float32).view(3, 1, 1)  # stain colour
        lum = (x * torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0, keepdim=True)
        x = lum + (x - lum) * float(rng.uniform(0.75, 1.25))  # saturation
        x = x.clamp(1e-4, 1.0) ** float(rng.uniform(0.85, 1.15))  # gamma
        sigma = float(rng.uniform(0.0, 0.015))
        if sigma > 0:
            x = x + torch.randn(x.shape, generator=torch.Generator().manual_seed(int(rng.integers(1 << 31)))) * sigma
        return x


def soft_dice_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    valid = (y != VOID)
    p = torch.softmax(logits, 1) * valid[:, None]
    yc = y.clone()
    yc[~valid] = 0
    onehot = F.one_hot(yc, 3).permute(0, 3, 1, 2).float() * valid[:, None]
    inter = (p * onehot).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (denom + 1)).mean()


def load_prepared_image(it: Prepared) -> tuple[np.ndarray, np.ndarray]:
    return np.load(it.image_npy), np.load(it.label_npy)


def cached_rows_dir() -> Path:
    return workspace.cache_dir() / "train"
