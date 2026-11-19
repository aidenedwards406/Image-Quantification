# Researcher guide: training data, training and models

For whoever maintains the model: adding traced images, training, checking a model against hand measurements, and sharing it with the lab.

## Tracing conventions (what the app expects)

| Item | Rule |
|---|---|
| ROI file name | Same base name as the TIFF. `X.tif.zip`, `X.roi`, `X_ROI.roi` and `X_ROI.roi.zip` are all recognised (and `X.zip`, any capitalisation). |
| `TUA` | Total usable area. Several TUA ROIs (e.g. `TUA`, `TUA2`) are combined. If none is traced, the whole image is usable. |
| `D1`, `D2`, … | Damage regions. Combined, then clipped to the TUA, so overlaps and damage drawn past the TUA edge are fine. |
| Single unnamed `.roi` | Damage only, whole image usable. |
| No ROI file | Whole image usable, 0% damage. **Only for images someone actually checked.** |
| Other names | Counted as damage, with a warning in the checker. |
| Shapes | Polygon, freehand, traced (wand), rectangle, oval and composite ROIs (with holes) all work. Lines and points are ignored with a warning. |

**Damage** = no nuclei; blurry nuclei don't count as missing. **TUA** excludes out-of-focus areas, in-focus areas outside the vessel, and black corners. Slightly blurry areas you're sure about can stay in. The usable area is the most subjective part, so writing down a few examples of edge cases helps keep tracers consistent.

Pixels are counted the way ImageJ counts them (a pixel belongs to a ROI if its centre is inside). The app's % damage for a traced image should match ImageJ's to within a fraction of a percent. Check this on a few images first (below).

## What to do first with real data

1. **Training data → Add a folder** pointing at `huvec_inputs`. Fix everything marked ⛔ and look at every ⚠️.
2. **Check the conversion:** in *Look at an image with its ROIs*, open 3–4 images you remember. The usable-area px, damage px and % shown should match your ImageJ numbers. If they don't, stop and report it, since everything else builds on this.
3. **Mark control vessels** in the set (they are never used for training).
4. **Tune focus & specks** on 5–10 typical images: a sharp one, one with blurry edges, a section-4 image, one with specks. Save as lab settings.
5. **Train → cross-validation, Standard length**, overnight. Use *Estimate time on this computer* first. With 5 vessels this means 6 training runs.
6. **Check vs hand-measured %:** add the ~500 images that only have a recorded % (folder + spreadsheet). Run the model. This is the real test of the ±5% target.
7. If the agreement is good, the model is ready: **Models → Download model file** and share it with the lab.

## Training

**Evaluation mode** (always split by *vessel*, never by image, so images from one vessel never sit in both train and test):

- **Leave-one-vessel-out cross-validation (recommended):** each vessel is the test vessel once; the remaining vessels train the model, and one of them is used for validation. That gives an honest score for every vessel. A final model is then trained on all vessels, for the median number of steps that worked best in the folds. The final model is the one you use. It costs K+1 training runs.
- **Choose vessels yourself:** one run. Train on some vessels, stop at the best point on the **validation** vessel, score on the **test** vessel. Faster, but the score comes from a single vessel.

**Length:** *Quick check* only proves the pipeline runs. Use *Standard* normally and *Thorough* for a final model. *Full resolution* is recommended because nuclei are only ~6–9 px wide at 4x. *Half resolution* is about 3× faster but loses detail.

**Where to train:** a laptop CPU is fine overnight. The training process sits on a hidden console and on Windows keeps the computer awake, but plug the laptop in. If a GPU is available (and allowed for these images), the code uses it automatically. Install the CUDA version of PyTorch on that computer.

**Pretrained weights:** training starts from ImageNet weights for the encoder (downloaded once, ~45 MB, from download.pytorch.org). That helps a lot with ~200 images. If the download is blocked, it trains from scratch and says so in the log.

## Reading a model report (Models page)

- **Avg error (pts):** mean |model % − traced %|, in percentage points. The lab target is about ±5.
- **Within ±5 points:** share of test images inside the target.
- **Bias:** average (model − traced). +2 means it calls 2 points more damage than the tracer on average.
- **Bland–Altman plot:** difference vs average. Look for a trend, e.g. over-calling only on heavily damaged images.
- **Damage / usable-area Dice:** how well the outlines overlap pixel for pixel (1 = identical). Useful for spotting *why* a % is off.
- **Per test vessel:** consistency between vessels. Consistency between *images* (the scatter) matters most.

## Hand-measured benchmarks (the ~500 images)

These images have a recorded % but no ROIs, so they can't train the model. They are the best test of the actual goal.

1. **Check vs hand-measured % → Add a benchmark:** image folder + spreadsheet (any columns; the app guesses which is the name and which is the %, and you can change it). Fractions (0.125) are detected and can be converted.
2. **Run** a model on it. You get agreement statistics overall and by section and vessel.
3. **Trace these next:** the images where model and hand disagree most. Tracing them with ROIs and adding them as training data improves the model fastest. Expect many from section 4.
4. **Calibration:** the model scores each usable pixel 0–1 for damage. The *threshold* (default 0.5) decides where damage starts. Tracers vary in how generously they outline damage, so a slightly different threshold can match the lab's style. The app picks the best threshold and reports an **honest estimate**: it chooses the threshold on half the vessels and tests on the other half. If the gain is real, *Create a calibrated copy* makes a new model entry with that threshold, which you then activate.
5. If the model was trained on some of the same vessels, the benchmark score is optimistic and the page warns you.

## Activating and sharing models

- Models are never overwritten. Each training run adds a new entry.
- **Make this the active model** decides what new batches use. Finished batches keep the model they were analysed with, which is recorded in their Excel file.
- If the new model does worse than the active one on a shared benchmark, you get a warning and must confirm.
- **Download model file** gives one `.pth` (~28 MB) containing the weights and its training report. Lab members import it on the Models page. It was derived from NDA images, so share it only inside the lab, not on GitHub (`.gitignore` blocks `*.pth`).

## Retraining in future years

The loop: QC rejects → traced by hand → ROIs saved → added as a new training set → retrain (cross-validation) → check on the benchmark → activate if better.

- Add each new batch of traced images as its **own training set** ("Fall 2027 - traced by …"), so you can see what each set adds.
- Different tracers trace a bit differently, and the model follows whoever traced most. Short written tracing guidelines with example images help. So does keeping a fixed benchmark to compare every new model against.
- **New microscope, magnification, stain or cell type:** retrain with traced images from the new setup, and check it on a hand-measured benchmark from the same setup. Stain and brightness variation is partly covered by augmentation (colour, brightness, contrast, gamma, noise). For a different pixel size, also adjust the focus `scale` in `settings.yaml`.

## Tuning focus & specks

These detectors are measurements, not AI, so you set them by eye on the **Tune focus & specks** page. The defaults were calibrated on synthetic images where the true blur is known; real images may need adjusting.

- **Sharpness needed:** raise it if slightly soft areas show green, lower it if sharp cells show orange.
- **Minimum texture:** areas with less texture (bare silicone, damage) are left uncoloured so damage isn't mistaken for blur.
- **Speck darkness / size / roundness:** fit them so the cyan circles mark balled-up nuclei but not normal nuclei.
- **Flag thresholds:** when an image gets a flag in QC.

Saved settings apply to new batches. Each batch keeps a snapshot of the settings it used.

## Workspace and several computers

The workspace folder (shown in the sidebar) holds models, batches, QC decisions, benchmarks and training runs. Change it under **Help & settings**, or set the `HUVEC_WORKSPACE` environment variable. Several computers can point to the same folder on a lab drive. Avoid two people reviewing the *same* batch at the same time. It contains copies of uploaded images, so keep it on NDA-approved storage.
