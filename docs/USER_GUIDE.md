# User guide: analyzing images

For lab members turning a batch of 4x brightfield images into % damage per image.

## Before you start

- The app is installed (see the [README](../README.md#quick-start-windows)) and the lab's model has been imported on the **Models** page. The sidebar shows the **Active model**.
- Image names follow the lab convention: `260919_BR-S24-9_4x_2_T_P.tif` (date, vessel, magnification, section, Top/Bottom, position). The vessel is read from the name, so typos matter.
- Type **your name** in the sidebar. QC decisions are saved under it.

## 1 · Analyze images

1. Open **1 · Analyze images**.
2. Name the batch (for example the study and vessels).
3. Either **Upload files** (select or drop all the TIFFs at once) or choose **A folder on this computer** (paste the path, or click *Browse…*).
4. Press **Analyze N image(s)**.

Each image takes about 2–4 seconds on a laptop. You can start reviewing as soon as the first images are done. If the laptop shuts down mid-batch, open the batch's **More** menu and press **Resume analysis**.

*Extra-careful mode* (under Options) runs each image 4 times, flipped, and averages the results. It is slightly more stable but 4× slower.

## 2 · QC review

For every image you see three pictures:

| Picture | What to check |
|---|---|
| **Original** | The image as it came off the microscope. |
| **Usable area** | The yellow line is the TUA. Everything darkened outside it is ignored (out of focus, outside the vessel, black corners). |
| **Damage** | Red = areas the model calls damage (no nuclei). |

Click a picture's corner icon to open it full screen. Other views:
- **Focus map**: green = in focus, orange = blurry, no tint = bare silicone or dark.
- **Specks**: cyan circles = small black specks.

Then decide:

- **Yes (press Y)**: the usable area and damage look right. The model's numbers are kept.
- **No (press N)**: something's clearly off, such as damage missed or invented, or a wrong usable area. The image goes on the **Needs tracing** list. Before pressing N you can pick a reason under *If No: why?*. It's optional, but it helps whoever improves the model.
- **← / →** move between images. **Undo** clears a decision.

Look harder at images with orange **flags**:

| Flag | Meaning |
|---|---|
| Blurry | A large part of the usable area looks out of focus, so the outlines are less trustworthy. |
| Small usable area | Little of the image could be used, so the % is based on few pixels. |
| Model unsure | The model was close to 50/50 on a large part of the image. |
| Specks | Many small black specks, possibly balled-up nuclei of unhealthy cells. They don't change the %, but they're worth a note. |

**What counts as right?** Use the same judgment you'd use when tracing. Tracing edges are a little subjective, so don't reject an image over a few pixels. Reject it when the % damage would clearly change, roughly by more than a couple of percentage points.

Use **Show** to switch between *To review*, *Flagged*, *Approved*, *Needs tracing* and *All*. **Overview grid** shows many images at once.

## 3 · Export to Excel

Open **3 · Export to Excel**, press **Create Excel workbook**, then **Download**. A copy also stays in the batch folder.

- **Approved**: done, nothing to do.
- **Needs tracing**: for each image, trace it in ImageJ as usual (TUA + D1, D2 …). Measure, then type the **usable area (px)** and **% damage** into the yellow cells. Type 12.5 for 12.5%, not 0.125.
- **All images**: fills itself in as you type. This is the final list.
- **Per vessel**: type each vessel's treatment (e.g. *Control*) once.

Please save your ImageJ ROIs for the traced images in the usual way (same name as the image). Those traced images are the best new training data for the model.

## 4 · (Optional) 3D Vessel Model viewer

Open the 3D Vessel Model viewer, load the vessel model and images, and drop the workbook into step 4 (damage data). It picks the **All images** sheet and its % damage column automatically, and the **Treatment** column if filled in. To see the model's outlines over the images in the viewer, tick *Also export damage outlines* on the Export page. Then add the `_trace.png` files together with the images.

## Problems

| Problem | What to do |
|---|---|
| The browser doesn't open | Go to http://localhost:8501 while the black window is open. |
| "No active model" | Import the lab's `.pth` on the **Models** page. |
| An image failed to analyse | It goes to *Needs tracing* automatically. Check the file opens in ImageJ. |
| The app seems stuck | Close the black window and start it again. Batches and decisions are saved. |
| Anything else | Take a screenshot (including any *Technical details* box) and send it to whoever maintains the app. |
