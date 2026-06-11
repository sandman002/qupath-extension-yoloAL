# QuPath YOLO Active Learning — Setup & Usage Guide

## Prerequisites

| Requirement | Version |
|---|---|
| QuPath | 0.6.x |
| Java JDK | 21+ (for building) |
| Python | 3.10 – 3.12 |
| GPU (optional) | CUDA 11.8+ recommended |

---

## One-Time Setup

### macOS / Linux
```bash
git clone <this-repo>
cd qupath-yolo-al
bash setup.sh
```

### Windows
```bat
git clone <this-repo>
cd qupath-yolo-al
setup.bat
```

The script:
1. Copies `yolo_server.py` to `~/.qupath/yolo-al/` (macOS/Linux) or `%APPDATA%\QuPath\yolo-al\` (Windows)
2. Creates a Python venv `yolo-venv` in that directory
3. Installs Ultralytics, FastAPI, uvicorn, PyTorch into the venv
4. Builds the extension JAR and copies it to the QuPath extensions directory

---

## First Launch

1. **Start QuPath 0.6** — the extension loads automatically.
2. Open your WSI project.
3. **Extensions → YOLO Active Learning → Open Panel…**

The panel floats alongside the QuPath viewer.

---

## Workflow

### Step 1 — Annotate Train Slides

- In the **Project panel**, select a slide and open it.
- Use QuPath's **Rectangle tool** (shortcut `R`) to draw bounding boxes around objects of interest.
- **Assign a PathClass** to each annotation: right-click the annotation → Set class, or use the class list panel. Every annotation needs a class or it will be skipped.
- Mark the slide as **train** or **test**: right-click the slide in the Project panel → **Edit metadata** → add key `role` with value `train` or `test`. Slides with no metadata are treated as `train`.

### Step 2 — Extract Patches (Dataset tab)

| Setting | Meaning |
|---|---|
| Patch size | Pixel size of the square crop (e.g. 640) |
| Max random offset | Max px shift of patch centre from bbox centre (data augmentation) |
| Augments per bbox | How many random crops to generate per bbox (1 = clean only) |
| Val fraction | Fraction of train-role slides automatically held out for validation |

Click **Extract Patches from All Project Slides**.  
Patches are written to `<dataset dir>/images/{train,val,test}/` with matching YOLO `.txt` labels.  
`dataset.yaml` is generated automatically.

### Step 3 — Start Python Server

Click **Start Python Server** (top bar).  
Status changes to `Server: running at http://localhost:5005` when ready.  
This only needs to be done once per QuPath session.

### Step 4 — Train (Train tab)

| Setting | Guidance |
|---|---|
| Model | `yolov8s.pt` is a good start; use `m`/`l` for better accuracy |
| Epochs | 30–100 for initial training; more for refinement |
| Batch | 8–16 for 640px patches on a GPU; reduce if OOM |
| Image size | Match patch size from Step 2 |
| LR0 | 0.01 default; lower (0.001) for fine-tuning |

Click **▶ Start Training**.  
Loss curves and mAP@50 update live every 2 seconds.  
Final P/R/mAP metrics appear below the charts when training finishes.

### Step 5 — Infer on Test Slides (Infer tab)

Click **▶ Run Inference on Test Slides**.  
The extension:
1. Collects patches from `images/test/`
2. Sends them to the server
3. Overlays detections as QuPath detection objects on the corresponding slides

**Colour coding:**
- Green detection = confidence ≥ 0.5 (model is sure)
- Red detection = confidence < 0.5 (model is uncertain → good candidate for annotation)

### Step 6 — Active Learning Loop

1. **Review detections in the viewer** — red ones are uncertain.
2. **Accept**: leave the detection as-is (it becomes a confirmed annotation automatically).
3. **Reject**: delete the detection object (select + Delete key).
4. For uncertain regions with no detection, **draw a new bbox** and classify it.
5. Go back to Step 2 — re-extract (new annotations are included) → re-train.

Repeat until performance plateaus.

---

## Dataset Directory Layout

After extraction:
```
my_dataset/
  dataset.yaml
  images/
    train/   slide1_box0_aug0.png  slide1_box0_aug1.png  …
    val/
    test/
  labels/
    train/   slide1_box0_aug0.txt  …
    val/
    test/
```

YOLO `.txt` label format: `<class_idx> <cx> <cy> <w> <h>` (normalised 0–1).

---

## Troubleshooting

**Server fails to start**
- Check the QuPath console (Help → Show Console) for Python errors.
- Verify the venv was created: `~/.qupath/yolo-al/yolo-venv/` should exist.
- Run manually: `~/.qupath/yolo-al/yolo-venv/bin/python ~/.qupath/yolo-al/yolo_server.py`

**No annotations found during extraction**
- Every annotation must have a PathClass set (not just drawn).
- Check with Script Editor: `print(getCurrentHierarchy().getAnnotationObjects())`

**Out of memory during training**
- Reduce Batch size.
- Use a smaller model (`yolov8n.pt`).
- Reduce Image size to 320.

**JAR not loading in QuPath**
- Confirm the JAR is in QuPath's extensions directory.
- Check QuPath 0.6 — older versions use a different extension API.

**Detections appear at wrong location on slide**
- The current release places detections in patch-local coordinates. Full slide coordinate mapping requires patch-origin metadata — see the "Advanced: Patch Origin Tracking" section below.

---

## Advanced: Patch Origin Tracking

To map detections back to exact slide coordinates, `PatchExtractor` needs to write a sidecar JSON with the patch origin `(x0, y0)` for each patch file. The inference code in `YoloALPanel` can then transform `[x1,y1,x2,y2]` patch coords → slide coords before creating QuPath `DetectionObject`s.

This is a straightforward enhancement — add to `PatchExtractor.extract()`:
```groovy
def meta = [x0: x0, y0: y0, downsample: downsample, slideId: slideId]
lblDir.resolve("${stem}.json").toFile().text = JsonOutput.toJson(meta)
```
And read in `YoloALPanel.runInference()`:
```groovy
def meta = new JsonSlurper().parse(new File(labelDir, "${stem}.json"))
double slideX1 = meta.x0 + det.x1 * meta.downsample
// etc.
```

---

## File Map

```
qupath-yolo-al/
├── build.gradle                           Gradle build
├── settings.gradle
├── setup.sh / setup.bat                   One-shot setup scripts
├── python/
│   ├── yolo_server.py                     FastAPI + Ultralytics server
│   └── requirements.txt
└── src/main/groovy/qupath/ext/yoloal/
    ├── YoloALExtension.groovy             Extension entry point
    ├── YoloALCommand.groovy               Opens panel window
    ├── PatchExtractor.groovy              Patch + label generation
    ├── ServerClient.groovy                HTTP client to Python server
    └── ui/
        └── YoloALPanel.groovy             JavaFX UI (3 tabs)
```
