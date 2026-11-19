import numpy as np
import pytest

from huvec import DAMAGED, HEALTHY, IGNORE
from huvec.rois import RoiShape, classify_name, load_annotation, rasterize
from tests.conftest import write_polygon_roi

W, H = 200, 100


def square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_pixel_centre_rule_counts_like_imagej():
    # A 10x10 px square drawn on pixel borders covers exactly 100 pixels.
    m = rasterize(RoiShape("x", "damage", [np.array(square(10, 10, 20, 20), float)]), H, W)
    assert m.sum() == 100
    assert m[10:20, 10:20].all()


def test_no_roi_file_means_whole_image_zero_damage():
    a = load_annotation([], W, H)
    assert a.tua_px == W * H and a.damage_px == 0 and a.pct_damage == 0.0
    assert (a.label() == HEALTHY).all()


def test_tua_and_damage_set(tmp_path):
    p = tmp_path / "img.tif.zip"
    write_polygon_roi(p, [("TUA", square(0, 0, 100, 100)),            # 10,000 px usable
                          ("D1", square(10, 10, 30, 30)),              # 400 px
                          ("D2", square(20, 20, 40, 40)),              # overlaps D1 by 100 px
                          ("D3", square(90, 0, 120, 10))])             # 300 px, only 100 inside TUA
    a = load_annotation([p], W, H)
    assert a.has_tua_roi and a.n_damage_rois == 3
    assert a.tua_px == 10_000
    assert a.damage_px == 400 + 400 - 100 + 100
    assert a.pct_damage == pytest.approx(100 * 800 / 10_000)
    lab = a.label()
    assert lab[50, 150] == IGNORE and lab[50, 50] == HEALTHY and lab[15, 15] == DAMAGED
    assert any("outside the usable area" in n for n in a.notes)
    assert not a.warnings


def test_single_unnamed_roi_is_damage_on_whole_image(tmp_path):
    p = tmp_path / "img_ROI.roi"
    write_polygon_roi(p, [("", square(0, 0, 20, 10))])
    a = load_annotation([p], W, H)
    assert a.tua_px == W * H and a.damage_px == 200 and not a.warnings


def test_tua_only_means_zero_damage(tmp_path):
    p = tmp_path / "img.tif.zip"
    write_polygon_roi(p, [("TUA", square(0, 0, 100, 100))])
    a = load_annotation([p], W, H)
    assert a.tua_px == 10_000 and a.damage_px == 0


def test_damage_set_without_tua_uses_whole_image(tmp_path):
    p = tmp_path / "img.tif.zip"
    write_polygon_roi(p, [("D1", square(0, 0, 10, 10)), ("D2", square(50, 50, 60, 60))])
    a = load_annotation([p], W, H)
    assert a.tua_px == W * H and a.damage_px == 200
    assert any("no TUA ROI" in n for n in a.notes)


def test_unknown_names_are_counted_as_damage_with_a_warning(tmp_path):
    p = tmp_path / "img.tif.zip"
    write_polygon_roi(p, [("TUA", square(0, 0, 100, 100)), ("scratch", square(0, 0, 10, 10))])
    a = load_annotation([p], W, H)
    assert a.damage_px == 100
    assert any("scratch" in w for w in a.warnings)


def test_scaled_masks_match_full_resolution(tmp_path):
    p = tmp_path / "img.tif.zip"
    write_polygon_roi(p, [("TUA", square(0, 0, 160, 80)), ("D1", square(20, 20, 60, 60))])
    full = load_annotation([p], W, H)
    half = load_annotation([p], W, H, scale=0.5)
    assert half.tua.shape == (50, 100)
    assert half.pct_damage == pytest.approx(full.pct_damage, rel=0.02)


def test_composite_roi_with_hole_uses_even_odd():
    outer = np.array(square(0, 0, 50, 50), float)
    hole = np.array(square(10, 10, 20, 20), float)
    m = rasterize(RoiShape("TUA", "tua", [outer, hole], composite=True), H, W)
    assert m.sum() == 2500 - 100 and not m[15, 15]


def test_rectangle_and_oval_rois(tmp_path):
    from roifile import ROI_TYPE, ImagejRoi, roiwrite

    rect = ImagejRoi(roitype=ROI_TYPE.RECT, top=0, left=0, bottom=40, right=80, name="TUA")
    oval = ImagejRoi(roitype=ROI_TYPE.OVAL, top=10, left=10, bottom=30, right=30, name="D1")
    p = tmp_path / "img.tif.zip"
    roiwrite(p, [rect, oval])
    a = load_annotation([p], W, H)
    assert a.tua_px == 3200
    assert a.damage_px == pytest.approx(np.pi * 10 * 10, rel=0.06)


@pytest.mark.parametrize("name,role", [("TUA", "tua"), ("tua 2", "tua"), ("D1", "damage"), ("d 12", "damage"),
                                       ("Damage3", "damage"), ("0012-0034", "unknown"), ("", "unknown")])
def test_classify_name(name, role):
    assert classify_name(name) == role
