"""The segmentation network and how images are fed to it.

Architecture: U-Net with a ResNet-18 encoder (torchvision).  It's small
enough to train overnight on a laptop CPU, and it starts from ImageNet weights
when they can be downloaded, which helps a lot with ~200 training images.
The decoder stops at 1/2 resolution and the class scores are upsampled.
Damage and usable-area borders are smooth, so this costs no accuracy and
halves the compute.

Output: 3 classes per pixel - 0 ignore (outside usable area), 1 healthy,
2 damaged.

Model files (.pth) are self-contained: weights (float16, ~28 MB) plus a
``meta`` dict with everything needed to run them the same way they were
trained (input scale, damage threshold, training summary).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "huvec-damage-model"
FORMAT_VERSION = 1
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
BACKGROUND_TARGET = 0.85  # images are scaled so the bright background sits here


class _Block(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class DamageUNet(nn.Module):
    arch = "unet-resnet18"

    def __init__(self, n_classes: int = 3, pretrained: bool = False):
        super().__init__()
        from torchvision.models import resnet18

        self.pretrained_loaded = False
        enc = None
        if pretrained:
            try:
                from torchvision.models import ResNet18_Weights

                enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.pretrained_loaded = True
            except Exception:  # offline / blocked download: train from scratch
                enc = None
        if enc is None:
            enc = resnet18(weights=None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # 1/2, 64
        self.pool = enc.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.dec4 = _Block(512 + 256, 256)
        self.dec3 = _Block(256 + 128, 128)
        self.dec2 = _Block(128 + 64, 64)
        self.dec1 = _Block(64 + 64, 64)
        self.head = nn.Conv2d(64, n_classes, 1)

    def encoder_parameters(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def decoder_parameters(self):
        for m in (self.dec4, self.dec3, self.dec2, self.dec1, self.head):
            yield from m.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        s1 = self.stem(x)
        s2 = self.layer1(self.pool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        d = self.dec4(torch.cat([_up(s5, s4), s4], 1))
        d = self.dec3(torch.cat([_up(d, s3), s3], 1))
        d = self.dec2(torch.cat([_up(d, s2), s2], 1))
        d = self.dec1(torch.cat([_up(d, s1), s1], 1))
        return F.interpolate(self.head(d), size=(h, w), mode="bilinear", align_corners=False)


# ── input preparation ──────────────────────────────────────────────────────

def brightness_factor(img_uint8: np.ndarray) -> float:
    """Multiplier that puts the image's bright background at BACKGROUND_TARGET."""
    small = img_uint8[::4, ::4].astype(np.float32) / 255.0
    lum = small[..., 0] * 0.299 + small[..., 1] * 0.587 + small[..., 2] * 0.114
    ref = float(np.percentile(lum, 90))
    return float(np.clip(BACKGROUND_TARGET / max(ref, 1e-3), 0.5, 2.5))


def to_tensor(img_uint8: np.ndarray, factor: float | None = None) -> torch.Tensor:
    """RGB uint8 (H, W, 3) -> normalised float tensor (3, H, W)."""
    if factor is None:
        factor = brightness_factor(img_uint8)
    x = torch.from_numpy(np.ascontiguousarray(img_uint8)).permute(2, 0, 1).float().div_(255.0)
    x = (x * factor).clamp_(0, 1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def resize_image(img_uint8: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img_uint8
    from PIL import Image

    h, w = img_uint8.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(Image.fromarray(img_uint8).resize(size, Image.Resampling.BILINEAR if scale > 1 else Image.Resampling.BOX))


# ── inference ──────────────────────────────────────────────────────────────

def _forward_padded(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % 32, (-w) % 32
    if ph or pw:
        mode = "reflect" if (ph < h and pw < w) else "replicate"
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return model(x)[..., :h, :w]


def _forward_tiled(model: nn.Module, x: torch.Tensor, tile: int = 1600, overlap: int = 128) -> torch.Tensor:
    """Sliding window for images too large to process in one pass."""
    _, _, h, w = x.shape
    if h <= tile + overlap and w <= tile + overlap:
        return _forward_padded(model, x)
    out = None
    weight = torch.zeros((1, 1, h, w))
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    ramp = torch.ones((1, 1, min(tile, h), min(tile, w)))
    for y in ys:
        for x0 in xs:
            patch = x[..., y:y + tile, x0:x0 + tile]
            logits = _forward_padded(model, patch)
            if out is None:
                out = torch.zeros((1, logits.shape[1], h, w))
            ph, pw = patch.shape[-2:]
            out[..., y:y + ph, x0:x0 + pw] += torch.softmax(logits, 1) * ramp[..., :ph, :pw]
            weight[..., y:y + ph, x0:x0 + pw] += ramp[..., :ph, :pw]
    return torch.log((out / weight).clamp_min(1e-8))  # log-probs behave like logits under softmax


@torch.inference_mode()
def predict_probs(model: nn.Module, img_uint8: np.ndarray, meta: dict, tta: bool = False) -> np.ndarray:
    """Class probabilities (3, H, W) float32 at the image's full resolution."""
    model.eval()
    H, W = img_uint8.shape[:2]
    scale = float(meta.get("scale", 1.0))
    factor = brightness_factor(img_uint8)
    small = resize_image(img_uint8, scale)
    x = to_tensor(small, factor)[None]
    flips = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
    acc = None
    for dims in flips:
        xi = torch.flip(x, dims) if dims else x
        p = torch.softmax(_forward_tiled(model, xi), 1)
        if dims:
            p = torch.flip(p, dims)
        acc = p if acc is None else acc + p
    probs = acc / len(flips)
    if probs.shape[-2:] != (H, W):
        probs = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)
    return probs[0].float().numpy()


# ── saving / loading ───────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path, meta: dict) -> None:
    state = {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
             for k, v in model.state_dict().items()}
    torch.save({"format": FORMAT, "format_version": FORMAT_VERSION, "arch": DamageUNet.arch,
                "n_classes": 3, "state_dict": state, "meta": meta}, path)


def load_checkpoint(path, device: str = "cpu") -> tuple[DamageUNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != FORMAT:
        raise ValueError("This file is not a HUVEC damage model (.pth made by this app).")
    if ckpt.get("arch") != DamageUNet.arch:
        raise ValueError(f"Unknown model architecture {ckpt.get('arch')!r}.")
    model = DamageUNet(n_classes=int(ckpt.get("n_classes", 3)), pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    model.eval()
    return model, dict(ckpt.get("meta") or {})


def read_meta(path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != FORMAT:
        raise ValueError("This file is not a HUVEC damage model (.pth made by this app).")
    return dict(ckpt.get("meta") or {})


def configure_threads() -> int:
    """Use all but one CPU core so the laptop stays responsive."""
    import os

    n = max(1, (os.cpu_count() or 2) - 1)
    torch.set_num_threads(n)
    return n
