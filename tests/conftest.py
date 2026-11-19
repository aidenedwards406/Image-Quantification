import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    """A fresh, empty workspace for each test."""
    monkeypatch.setenv("HUVEC_WORKSPACE", str(tmp_path / "workspace"))
    return tmp_path / "workspace"


def write_polygon_roi(path, polys_named):
    """Write ImageJ ROIs: list of (name, [(x, y), ...]). One ROI -> .roi, several -> .zip."""
    import numpy as np
    from roifile import ImagejRoi, roiwrite

    rois = [ImagejRoi.frompoints(np.asarray(pts, np.float32), name=name) for name, pts in polys_named]
    if str(path).endswith(".roi"):
        assert len(rois) == 1
        rois[0].tofile(path)
    else:
        roiwrite(path, rois)
