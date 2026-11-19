# How it works

Technical notes for whoever maintains the app next.

## Pipeline

```
TIFF ──► model (U-Net) ──► P(ignore), P(healthy), P(damaged) per pixel
                              │
                              ├─► usable area  = P(ignore) < 0.5          (+ small islands/holes removed)
                              ├─► damage       = usable & P(damaged | usable) ≥ threshold (default 0.5)
                              │                  (+ islands < 120 px removed, holes < 120 px filled)
                              ├─► % damage     = damage px / usable px × 100   (full resolution)
                              ├─► model unsure = usable px with probabilities within 0.5 ± 0.15
                              └─► ratio histogram (100 bins) → % at any other threshold (calibration)
TIFF ──► focus map (sharp / blurry / featureless / dark per 32 px block) ──► in-focus %, blurry inside usable area
TIFF ──► speck detector (small, dark, solid blobs inside the usable area) ──► count, positions
                              └─► flags → QC review → Excel
```

The same function (`huvec.predict.analyze_probs`) produces the numbers for batches, test-set evaluation and benchmarks, so reported accuracy is the accuracy users get.

## Labels from ImageJ ROIs (`huvec/rois.py`)

3 classes: `0 ignore` (outside the TUA), `1 healthy` (TUA minus damage), `2 damaged` (union of D ROIs ∩ TUA). Rules for missing or unnamed ROIs are in the [researcher guide](RESEARCHER_GUIDE.md#tracing-conventions-what-the-app-expects). ROIs are read with [`roifile`](https://pypi.org/project/roifile/). Pixels are filled by the pixel-centre rule, as ImageJ does, so a 10×10 px square drawn on pixel borders counts exactly 100 px (tested). Composite ROIs use the even-odd rule, so holes work. For training at reduced scale, polygons are scaled and rasterised directly at the target size rather than by resizing masks.

## Model (`huvec/model.py`)

- **U-Net with a ResNet-18 encoder** (torchvision), 14 M parameters. The decoder stops at 1/2 resolution and the logits are upsampled bilinearly. Damage and usable-area borders are smooth, so this costs nothing measurable and halves the compute.
- **Input:** RGB, brightness-normalised so the image's 90th-percentile luminance (the background) sits at 0.85, then ImageNet mean/std. This makes the model insensitive to lamp and exposure differences.
- **Images of any size:** padded to a multiple of 32; images larger than ~1700 px per side at the working scale go through overlapping tiles with blending. 1392×1040 runs in a single pass: about 1.7 s per image for model + focus + specks on a 4-core CPU.
- **Saved as** a self-contained `.pth`: float16 weights (~28 MB) and a `meta` dict (scale, damage threshold, training report). Loaded with `torch.load(weights_only=True)`, so a shared model file can't run code.

## Training (`huvec/train.py`, `huvec/data.py`)

- Images and labels are decoded once and cached as `.npy` in `<workspace>/cache`; tiles are read through memory maps.
- **Tiles:** 384×384 at full resolution by default, batch 4. 40% are centred on damage, 20% on usable-area edges, 40% random. That balances the classes without changing the label statistics.
- **Augmentation:** 90° rotations and flips (orientation carries no meaning, and images are independent), zoom 0.85–1.18, brightness ±15%, contrast ±15%, per-channel gain ±7% (stain colour), saturation ±25%, gamma 0.85–1.15, camera noise. Blur is deliberately *not* used: it would change what counts as usable.
- **Loss:** cross-entropy with class weights (1/√frequency) + 0.5 × soft Dice.
- **Optimiser:** AdamW, lr 3e-4 (encoder at half), weight decay 1e-4, 4% warm-up then cosine decay.
- **Checkpoint selection:** on the validation vessel every `eval_every` steps, by `validation % MAE + 2 × (1 − pooled damage Dice)`. That is the lab's real target, with outline quality as a tie-breaker. Early stopping after `patience` evaluations without improvement.
- **Presets:** Quick 1,200 steps, Standard 4,000 (patience 6), Thorough 8,000 (patience 8).
- **Cross-validation:** fold *k* tests vessel *k* and validates on vessel *k+1*. The final model trains on all vessels for 1.1 × the median best step of the folds.
- The test-set evaluation runs the full production pipeline at full resolution.

## Metrics (`huvec/metrics.py`)

Per image: Dice and IoU for damage and for the usable area, and model vs traced % damage. Across images: MAE, RMSE, bias, SD of differences, Bland–Altman 95% limits of agreement, share within ±5 percentage points, Pearson r and Lin's concordance correlation coefficient (CCC). Dice is reported *pooled* (sum of intersections over sum of areas across images) and as a per-image mean over images that have damage.

## Focus map (`huvec/focus.py`)

Blur removes fine intensity changes first and coarse structure later. On luminance normalised to the background:

- `detail = (G₁ − G₂)²` (1–2 px band), `structure = (G₂ − G₄)² + (G₄ − G₈)²` (2–8 px band), where Gₛ is a Gaussian blur of σ = s px.
- Camera noise σₙ is estimated per image (Immerkær 1996), so thresholds are in units of noise.
- Per 32 px block: **dark** if mean < 0.12; **featureless** if structure/σₙ² < 0.25 (bare silicone, i.e. damage, isn't mistaken for blur); otherwise **sharp** if the block's 75th percentile of the local (8 px) detail/structure ratio ≥ 0.25, else **blurry**. Using the upper quartile stops a block that's half cells, half bare silicone (a damage edge) from reading as blurry. A 3×3 median filter smooths the block decisions.
- Calibrated on synthetic images with a known blur map: sharp cells 99% "in focus", blur σ ≥ 2 px 99% "blurry", in-focus damage 4% "blurry". It takes ~0.25 s per image.
- Outputs: **in focus %** = sharp / (sharp + blurry) blocks; **blurry inside usable area %** = blurry px ∩ TUA / TUA px.

## Specks (`huvec/specks.py`)

Connected components of luminance < 0.18 × background, 4–80 px, solidity ≥ 0.8, centred inside the usable area. Large dark regions (black corners) plus a 6 px margin are excluded. In synthetic tests it finds 18–20 of 20 planted specks with no false positives on clean images. Thresholds are on the Tune page.

## Excel and the 3D Vessel Model viewer (`huvec/excel.py`)

Written with XlsxWriter, so formula cells carry *cached values*. The 3D viewer's built-in reader (and pandas) read cached values, not formulas, so the numbers appear without opening the file in Excel first. Lookups use `INDEX/MATCH` on the image name, so they survive sorting and work in Excel, LibreOffice and Google Sheets. The workbook was checked by recalculating it in LibreOffice after a simulated student filled in the tracing sheet.

The viewer picks the sheet with the most image names (*All images*, ahead of *Details*) and guesses the damage column from the headers. Its header test (`src/lib/importers.ts`) subtracts a point for `/mag|section|…/`, which also matches "da**mag**e". So any column with "area" in its header ties with "% damage". *All images* therefore names its pixel column **"TUA (px)"**, and a test re-implements the viewer's logic to keep it that way. A one-word fix in the viewer (`/\bmag|section|…/`) would remove the quirk.

Also note: the viewer treats everything before `4x` as the bioreactor ID, so `260919_BR-S24-9` (date included) is the vessel ID there. A separate treatment key uploaded to the viewer must use the same form. The Treatment column on *All images* avoids the issue.

`<image>_trace.png` exports are RGBA: semi-transparent red damage with an opaque edge. That works in the viewer's "As drawn" and "Mask" modes.

## Background jobs (`huvec/jobs.py`)

Training, batch analysis and benchmarks run as separate processes (`python -m huvec.train|analyze|benchmark --job-dir …`). On Windows they start with their own hidden console, so they survive the app window closing. Each job folder has `job.json`, `progress.json` (status, progress, ETA, curves), `log.txt` and an optional `STOP` file. Processes are checked with `psutil`, and a job whose process vanished shows as *crashed*. On Windows, `SetThreadExecutionState` keeps the computer awake while a job runs. Batch analysis skips images that already have results, so it resumes.

## Workspace layout

```
<workspace>/
  settings.yaml                 focus / speck / flag settings (Tune page)
  models/<id>/model.pth, card.json ; models/active.json
  library/<set>.json, <set>.scan.json          training sets (images stay in their folder)
  batches/<id>/batch.json, results/, renders/, masks/, review.json, exports/
  uploads/<id>/                 images uploaded through the browser
  benchmarks/<id>/benchmark.json, results/<model>.json, renders/
  jobs/<id>/                    background job folders
  cache/train/                  decoded training arrays
  demo/                         synthetic demo data
```

## Development

```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt   # Windows; .venv/bin/python on macOS/Linux
python -m pytest                                              # 34 tests, about a minute on CPU
python -m streamlit run app.py
```

- `huvec/` has no Streamlit imports, so everything works from a notebook: `from huvec.rois import load_annotation`, `from huvec import registry; model, meta = registry.load(registry.active_id())`, `from huvec.predict import analyze_image`.
- `huvec.synthetic.make_dataset()` makes look-alike images with ROI files in every naming variant. Use it for tests and demos; never commit real images.
- Keep the Excel layout test (`tests/test_pipeline.py::viewer_pick`) passing whenever the workbook changes.

## Known limitations

- Developed without access to the real (NDA) images. The ROI conversion, focus and speck defaults, and training settings were verified on synthetic look-alikes only. Real-image checks are listed in the researcher guide.
- The usable area is partly a judgment call, so expect it to be the main source of disagreement with tracers, especially in section 4.
- One person should review a given batch at a time; decisions are saved per image (last write wins).
- The Windows installer scripts haven't been run on a Windows machine yet.
