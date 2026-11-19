"""HUVEC damage analysis: segmentation of endothelial damage on 4x brightfield images.

The package is split so that everything except ``huvec.model``/``huvec.train``
works without PyTorch:

    naming      file-name conventions and ROI-file matching
    imageio     reading TIFFs into RGB uint8 arrays
    rois        ImageJ ROI files -> TUA / damage masks (the 3-class label)
    focus       pixel-to-pixel sharpness map ("how much is in focus")
    specks      small black speck detection
    predict     model output -> numbers, flags and overlays for one image
    excel       the QC'd Excel workbook
"""

__version__ = "1.0.0"

# Mask class values used everywhere (label images, model outputs, metrics).
IGNORE, HEALTHY, DAMAGED = 0, 1, 2
CLASS_NAMES = ("ignore", "healthy", "damaged")
