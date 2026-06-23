# QuPath YOLO Active Learning

> **Annotate in QuPath · Train YOLO · Infer on whole slides · Repeat.**

A [QuPath](https://qupath.github.io) 0.7 extension that embeds a complete YOLO-based object detection and active learning workflow directly inside the pathology viewer. A local FastAPI + Ultralytics Python server handles all ML work; the extension communicates with it over HTTP on `localhost:5005`.

---

## Features

| | |
|---|---|
| **Patch extraction** | Crop YOLO-format image patches from QuPath annotations with configurable size, jitter, and augmentation count |
| **Slide role management** | Assign `train / val / test / skip` per slide in an interactive table; roles write back to QuPath project metadata automatically |
| **Incremental datasets** | Add new slides to an existing dataset without re-extracting; mark slides `redo` to force re-extraction after updating annotations |
| **Memorable dataset names** | Datasets named `swift_eagle_2026-06-01` style — easy to recognise and sort chronologically |
| **One-click training** | YOLOv8 / YOLO11 training from inside QuPath; live loss curves and mAP@50 charts update every 2 s |
| **WSI inference** | Tiles the whole slide, runs YOLO, applies cross-tile NMS, and imports detections back into QuPath |
| **Active learning loop** | Green = confident · Red = uncertain → annotate the reds, re-train |

---

## Requirements

| | Minimum |
|---|---|
| [QuPath](https://qupath.github.io) | 0.7.x |
| Java JDK | 21+ *(build only — QuPath bundles its own JRE)* |
| Gradle | 8+ |
| Python | 3.10 – 3.12 |
| GPU | CUDA 13.0 recommended; CPU works but training is slow |

---

## Setup

### 1 — Python environment

The Python server handles all training and inference (Ultralytics + FastAPI).

**Windows — one command:**
```bat
setup.bat
```

**macOS / Linux — one command:**
```bash
bash setup.sh
```

Both scripts:
1. Create the `yolo-al/` directory inside your QuPath user data folder
2. Copy `yolo_server.py` there
3. Create a Python venv `yolo-venv` in that directory
4. Install PyTorch (CUDA 13.0 wheels), Ultralytics, FastAPI, uvicorn

**Manual / custom CUDA version:**
```bash
python -m venv yolo-venv
# Windows:
yolo-venv\Scripts\activate
# macOS/Linux:
source yolo-venv/bin/activate

# PyTorch — adjust the cu1XX suffix for your CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install ultralytics fastapi "uvicorn[standard]" Pillow numpy
```
Then copy `python/yolo_server.py` to your QuPath `yolo-al/` directory:
- **Windows:** `%APPDATA%\QuPath\yolo-al\`
- **macOS/Linux:** `~/.qupath/yolo-al/`

---

### 2 — Build and install the QuPath extension JAR

**Prerequisites:** Java 21 JDK. If Gradle cannot find it automatically, copy `gradle.properties.example` to `gradle.properties` and set the path.

```bash
git clone https://github.com/sandman002/qupath-extension-yoloAL.git
cd qupath-extension-yoloAL

# Windows:
gradlew.bat jar
# macOS / Linux:
./gradlew jar
```

**Install — Windows:**
```bat
copy build\libs\qupath-yolo-al-0.1.0.jar "%USERPROFILE%\QuPath\v0.7\extensions\"
```

**Install — macOS / Linux:**
```bash
cp build/libs/qupath-yolo-al-*.jar ~/.qupath/extensions/
```

**Or use the all-in-one Gradle task:**
```bash
# Windows:
gradlew.bat installExtension
# macOS / Linux:
./gradlew installExtension
# Override destination:
gradlew.bat installExtension -PqupathExtDir="D:/QuPath/extensions"
```

> **Windows note:** The correct extensions directory is `%USERPROFILE%\QuPath\v0.7\extensions\`  
> (`%APPDATA%\QuPath\extensions\` is **not** used by QuPath 0.7.)

---

### 3 — First launch

1. **Start QuPath 0.7** — the extension loads automatically.
2. Open a WSI project (*File → Project → Create Project* if you don't have one).
3. **Extensions → YOLO Active Learning → Open Panel…**
4. Click **▶ Start Server** in the panel's top bar to launch the Python backend.

---

## Workflow

```
 1. Annotate slides  →  draw bboxes in QuPath, assign a PathClass to each
       ↓
 2. Dataset tab  →  assign slide roles (train / val / test), click Extract Patches
       ↓
 3. Start Server  (top bar, once per session)
       ↓
 4. Train tab  →  pick dataset + base model, click ▶ Start Training
       ↓
 5. Infer tab  →  click ▶ Run Inference  (test slides, or the open slide)
       ↓
 6. Review detections
        Green = confident ≥ conf threshold  →  accept or delete
        Red   = uncertain < conf threshold  →  annotate these
       ↓
 7. Add new annotations, set slide status to 'redo' in the table if needed,
    go back to step 2 and add to existing dataset
```

---

## Documentation

Full documentation is in the **[GitHub Wiki](../../wiki)**:

- [Annotation guidelines](../../wiki/Annotation-Guidelines)
- [Dataset tab — slide roles & incremental extraction](../../wiki/Dataset-Tab)
- [Training parameters guide](../../wiki/Training)
- [Inference & active learning loop](../../wiki/Inference)
- [Troubleshooting](../../wiki/Troubleshooting)

A quick-start guide is also available in [GUIDE.md](GUIDE.md).

---

## Project structure

```
qupath-extension-yoloAL/
├── build.gradle                 Gradle build (plain JAR, no fat JAR)
├── settings.gradle
├── setup.sh / setup.bat         One-shot setup scripts
├── python/
│   ├── yolo_server.py           FastAPI + Ultralytics server
│   └── requirements.txt
└── src/main/groovy/qupath/ext/yoloal/
    ├── YoloALExtension.groovy   Extension SPI entry point
    ├── YoloALCommand.groovy     Opens the floating panel window
    ├── PatchExtractor.groovy    Patch cropping + YOLO label generation
    ├── ServerClient.groovy      HTTP client (Java 11 HttpClient + Gson)
    └── ui/
        └── YoloALPanel.groovy   JavaFX UI — Dataset / Train / Infer tabs
```

---

## Author

**Sandeep Manandhar, PhD**  
manandhar.sandeep@gmail.com

---
## Citation

If you use this work, please cite:

```bibtex
@article{Sandman2026MyoSAM,
  title={Differential Cardiac Remodeling: Patterns of Cardiomyocyte Morphometry in Multiple Patient Conditions},
  author={Sandeep Manandhar and Yuxin Wu et al.},
  journal={Cardiovascular Research},
  year={2026},
}
```
---

## License

[MIT](LICENSE)
