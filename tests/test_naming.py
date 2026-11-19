from huvec.naming import image_key, match_folder, parse_image_name, roi_key


def test_lab_name_with_date():
    n = parse_image_name("260919_BR-S24-9_4x_2_T_P.tif")
    assert n.follows_convention
    assert (n.date, n.vessel, n.magnification, n.section, n.side, n.subsection) == ("260919", "BR-S24-9", "4x", "2", "T", "P")


def test_name_without_date_like_3d_viewer():
    n = parse_image_name("BRS24-10_4x_2_T_M2.tiff")
    assert n.follows_convention and n.vessel == "BRS24-10" and n.subsection == "M2" and n.date == ""


def test_forgiving_words_and_bare_m():
    n = parse_image_name("260919_BR-S24-9_4x_3_Bottom_M.tif")
    assert n.follows_convention and n.side == "B" and n.subsection == "M"


def test_bad_name_still_guesses_vessel():
    n = parse_image_name("260919_BR-S24-9_image7.tif")
    assert not n.follows_convention
    assert n.vessel == "BR-S24-9"
    assert "magnification" in n.problem


def test_extra_suffix_is_noted_but_ok():
    n = parse_image_name("260919_BR-S24-9_4x_2_T_P_2.tif")
    assert n.follows_convention and "extra text" in n.problem


def test_roi_name_variants_all_match_the_image():
    img = image_key("260919_BR-S24-9_4x_2_T_P.tif")
    for v in ["260919_BR-S24-9_4x_2_T_P.tif.zip", "260919_BR-S24-9_4x_2_T_P.roi",
              "260919_BR-S24-9_4x_2_T_P_ROI.roi", "260919_BR-S24-9_4x_2_T_P_ROI.roi.zip",
              "260919_br-s24-9_4x_2_t_p.ZIP"]:
        assert roi_key(v) == img, v


def test_match_folder(tmp_path):
    for f in ["a_V1_4x_1_T_P.tif", "a_V1_4x_1_T_P.tif.zip", "a_V1_4x_2_T_P.tif", "a_V1_4x_2_T_P_ROI.roi",
              "a_V1_4x_3_T_P.tif", "orphan_4x_1_T_P.roi"]:
        (tmp_path / f).write_bytes(b"x")
    m = match_folder(tmp_path)
    assert [p.name for p in m.images] == ["a_V1_4x_1_T_P.tif", "a_V1_4x_2_T_P.tif", "a_V1_4x_3_T_P.tif"]
    assert [p.name for p in m.rois_for["a_v1_4x_1_t_p"]] == ["a_V1_4x_1_T_P.tif.zip"]
    assert [p.name for p in m.rois_for["a_v1_4x_2_t_p"]] == ["a_V1_4x_2_T_P_ROI.roi"]
    assert m.rois_for["a_v1_4x_3_t_p"] == []
    assert [p.name for p in m.orphan_rois] == ["orphan_4x_1_T_P.roi"]
