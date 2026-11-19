import numpy as np
import pytest

from huvec import settings
from huvec.focus import BLURRY, SHARP, compute_focus
from huvec.metrics import agreement, dice, iou
from huvec.specks import detect_specks
from huvec.synthetic import _poly_mask, make_image


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(3)
    img, tua, polys, sigma = make_image(rng, 520, 696, n_damage=2, blur_edges=1, n_specks=12, return_sigma=True)
    dmg = np.zeros_like(tua)
    for p in polys:
        dmg |= _poly_mask(p, *tua.shape)
    return img, tua, dmg, sigma


def test_focus_separates_sharp_and_blurred_cells(scene):
    img, tua, dmg, sigma = scene
    pm = compute_focus(img, settings.DEFAULTS["focus"]).pixel_map()
    sharp_cells = (sigma < 0.5) & ~dmg
    blurred_cells = (sigma >= 3) & ~dmg
    assert (pm[sharp_cells] == SHARP).mean() > 0.9
    assert (pm[blurred_cells] == BLURRY).mean() > 0.85


def test_damage_is_not_called_blurry(scene):
    img, tua, dmg, sigma = scene
    pm = compute_focus(img, settings.DEFAULTS["focus"]).pixel_map()
    assert (pm[dmg & (sigma < 0.5)] == BLURRY).mean() < 0.15


def test_specks_found(scene):
    img, tua, dmg, sigma = scene
    found = detect_specks(img, settings.DEFAULTS["specks"]).count
    assert 8 <= found <= 14  # 12 planted (a few may overlap or sit on nuclei)


def test_no_specks_on_clean_image():
    rng = np.random.default_rng(4)
    img, *_ = make_image(rng, 400, 500, n_damage=1, blur_edges=0, n_specks=0, black_corners=True)
    assert detect_specks(img, settings.DEFAULTS["specks"]).count <= 1


def test_dice_iou():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[:5] = True
    b[:5, :5] = True
    assert dice(a, b) == pytest.approx(2 * 25 / 75)
    assert iou(a, b) == pytest.approx(25 / 50)
    assert dice(np.zeros_like(a), np.zeros_like(b)) == 1.0


def test_agreement():
    manual = [0, 10, 20, 30]
    model = [1, 12, 19, 37]
    a = agreement(model, manual)
    assert a["n"] == 4
    assert a["bias"] == pytest.approx(np.mean([1, 2, -1, 7]))
    assert a["mae"] == pytest.approx(np.mean([1, 2, 1, 7]))
    assert a["within_tol"] == pytest.approx(0.75)
    perfect = agreement(manual, manual)
    assert perfect["ccc"] == pytest.approx(1.0) and perfect["mae"] == 0


def test_tiled_inference_matches_shape_and_is_a_probability():
    import torch

    from huvec.model import DamageUNet, _forward_tiled

    torch.manual_seed(0)
    model = DamageUNet(pretrained=False).eval()
    x = torch.randn(1, 3, 300, 410)
    with torch.inference_mode():
        out = _forward_tiled(model, x, tile=128, overlap=32)  # forces several overlapping tiles
        p = torch.softmax(out, 1)
    assert out.shape == (1, 3, 300, 410)
    assert torch.allclose(p.sum(1), torch.ones(1, 300, 410), atol=1e-4)
