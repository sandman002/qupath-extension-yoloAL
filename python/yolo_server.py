"""
yolo_server.py — Local FastAPI server for YOLO Active Learning QuPath extension.

Endpoints
---------
GET  /ping          health check
POST /dataset       set dataset yaml path
POST /train         start training (non-blocking, runs in background thread)
GET  /status        training progress (epoch, losses, mAP)
POST /stop          interrupt training
GET  /metrics       final evaluation metrics after training
POST /infer         run inference on a list of image paths

Usage:
    python yolo_server.py [--port 5005]
"""

import argparse
import base64
import io
import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

import numpy as np
import torch
from PIL import Image, ImageDraw
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO

try:
    import tiffslide
    HAS_TIFFSLIDE = True
except ImportError:
    HAS_TIFFSLIDE = False

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

# TIFF-native formats tiffslide handles reliably; everything else uses openslide.
_TIFFSLIDE_EXTS = {".svs", ".tif", ".tiff", ".btf", ".tf8"}

def open_slide(path: str):
    ext = Path(path).suffix.lower()
    if HAS_TIFFSLIDE and ext in _TIFFSLIDE_EXTS:
        return tiffslide.TiffSlide(path)
    if HAS_OPENSLIDE:
        return openslide.OpenSlide(path)
    if HAS_TIFFSLIDE:
        return tiffslide.TiffSlide(path)
    raise RuntimeError("Neither openslide-python nor tiffslide is installed.")

# ── App & state ──────────────────────────────────────────────────────────────

app = FastAPI(title="YOLO AL Server")

state: Dict = {
    "status":     "idle",      # idle | training | done | stopped | error
    "epoch":      0,
    "epochs":     0,
    "train_loss": None,
    "val_loss":   None,
    "map50":      None,
    "map50_95":   None,
    "error":      None,
    "infer_progress": {"active": False, "done": 0, "total": 0, "dets": 0, "slide": ""},
}

# Thread-local slide handles for parallel tile reads
_tl = threading.local()

def _make_tile_reader(slide_path: str, tile_size: int):
    """Return a tile-read function that keeps a thread-local slide handle.
    coord = (tx, ty, ds) where ds is the downsample factor.
    Reads a (tile_size*ds) × (tile_size*ds) region at level 0 and resizes to tile_size.
    """
    def read_tile(coord):
        tx, ty, ds = coord
        physical = int(tile_size * ds)
        if getattr(_tl, "slide_path", None) != slide_path:
            _tl.slide      = open_slide(slide_path)
            _tl.slide_path = slide_path
        try:
            raw = _tl.slide.read_region((tx, ty), 0, (physical, physical))
            # Composite RGBA against white (pathology background) before dropping alpha
            if raw.mode == "RGBA":
                bg = Image.new("RGB", raw.size, (255, 255, 255))
                bg.paste(raw, mask=raw.split()[3])
                img = bg
            else:
                img = raw.convert("RGB")
            if img.size != (tile_size, tile_size):
                img = img.resize((tile_size, tile_size), Image.BILINEAR)
            # ultralytics model.predict() treats numpy arrays as BGR (OpenCV convention)
            arr = np.array(img, dtype=np.uint8)[:, :, ::-1]
            return tx, ty, ds, arr
        except Exception:
            return tx, ty, ds, None
    return read_tile

dataset_yaml: Optional[str] = None
model: Optional[YOLO]       = None
best_weights: Optional[str] = None
train_thread: Optional[threading.Thread] = None
stop_event = threading.Event()


# ── Pydantic models ───────────────────────────────────────────────────────────

class DatasetBody(BaseModel):
    yaml_path: str

class TrainBody(BaseModel):
    model:   str   = "yolov8s.pt"
    epochs:  int   = 50
    imgsz:   int   = 640
    batch:   int   = 8
    lr0:     float = 0.01
    project: Optional[str] = None   # output parent dir (cycle models dir)
    name:    Optional[str] = None   # run sub-dir name
    # augmentation
    degrees:   float = 10.0
    translate: float = 0.2
    scale:     float = 0.2
    flipud:    float = 0.5
    fliplr:    float = 0.5
    mosaic:    float = 0.1
    hsv_h:     float = 0.05
    hsv_s:     float = 0.3
    hsv_v:     float = 0.25

class InferBody(BaseModel):
    image_paths: List[str]
    conf:        float = 0.25
    iou:         float = 0.45
    model_path:  Optional[str] = None  # explicit path to best.pt; uses last-trained model if omitted

class TissueBody(BaseModel):
    image_base64: str   # base64-encoded PNG thumbnail
    thumb_w:      int   # thumbnail width in pixels
    thumb_h:      int   # thumbnail height in pixels
    full_w:       int   # full-resolution slide width
    full_h:       int   # full-resolution slide height

class InferWSIBody(BaseModel):
    slide_path:   str
    model_path:   str
    conf:         float = 0.25
    iou:          float = 0.45
    tile_size:    int   = 640
    overlap:      int   = 64
    tissue_frac:  float = 0.15
    output_dir:   str
    # Each polygon is a list of [x, y] pairs in full-resolution slide coordinates.
    # Supports Rectangle, Polygon, Ellipse — whatever QuPath ROI.getAllPoints() returns.
    roi_polygons: List[List[List[float]]] = []
    debug_tiles:  bool = False   # save every tile to output_dir/debug_tiles/ for inspection
    # Downsamples to scan at — taken from extract_log.json so inference matches training scales.
    # 1.0 = full resolution; 2.0 = reads 2× larger region and downsamples to tile_size; etc.
    downsamples:  List[float] = [1.0]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/tissue")
def detect_tissue(body: TissueBody):
    """
    Detect tissue bounding box(es) from a base64-encoded thumbnail.
    Returns regions in FULL-RESOLUTION slide coordinates.
    Strategy: threshold non-white pixels (H&E background ≈ white).
    """
    try:
        img_bytes = base64.b64decode(body.image_base64)
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr       = np.array(img, dtype=np.uint8)

        # Tissue = non-white pixels
        mask = (arr[:, :, 0] < 220) | (arr[:, :, 1] < 220) | (arr[:, :, 2] < 220)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            # No tissue found — return full slide
            return {"bounds": [[0, 0, body.full_w, body.full_h]]}

        rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])

        # Add small margin in thumbnail coords
        margin = 8
        rmin = max(0, rmin - margin);  rmax = min(body.thumb_h - 1, rmax + margin)
        cmin = max(0, cmin - margin);  cmax = min(body.thumb_w - 1, cmax + margin)

        # Scale back to full-resolution coords
        sx = body.full_w / body.thumb_w
        sy = body.full_h / body.thumb_h
        return {"bounds": [[
            int(cmin * sx), int(rmin * sy),
            int(cmax * sx), int(rmax * sy)
        ]]}
    except Exception as e:
        return {"error": str(e), "bounds": [[0, 0, body.full_w, body.full_h]]}


@app.get("/device")
def get_device():
    if torch.cuda.is_available():
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem  = round(torch.cuda.get_device_properties(idx).total_memory / 1024**3, 1)
        return {"device": f"GPU — {name} ({mem} GB)"}
    return {"device": "CPU"}


@app.post("/dataset")
def set_dataset(body: DatasetBody):
    global dataset_yaml
    p = Path(body.yaml_path)
    if not p.exists():
        return {"ok": False, "error": f"File not found: {p}"}
    dataset_yaml = str(p)
    return {"ok": True, "yaml": dataset_yaml}


@app.post("/train")
def start_train(body: TrainBody):
    global train_thread, model, best_weights, stop_event

    if state["status"] == "training":
        return {"ok": False, "error": "Already training"}
    if not dataset_yaml:
        return {"ok": False, "error": "dataset_yaml not set"}

    stop_event = threading.Event()
    state.update({"status": "training", "epoch": 0, "epochs": body.epochs,
                  "train_loss": None, "val_loss": None,
                  "map50": None, "map50_95": None, "error": None})

    def _train():
        global model, best_weights
        try:
            model = YOLO(body.model)

            # Ultralytics callback to stream per-epoch metrics
            def on_train_epoch_end(trainer):
                if stop_event.is_set():
                    trainer.stop = True
                    return
                metrics = trainer.metrics or {}
                state["epoch"]      = trainer.epoch + 1
                state["train_loss"] = _safe(trainer.tloss)
                state["val_loss"]   = _safe(metrics.get("val/box_loss"))
                state["map50"]      = _safe(metrics.get("metrics/mAP50(B)"))
                state["map50_95"]   = _safe(metrics.get("metrics/mAP50-95(B)"))

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            train_kwargs = dict(
                data=dataset_yaml,
                epochs=body.epochs,
                imgsz=body.imgsz,
                batch=body.batch,
                lr0=body.lr0,
                degrees=body.degrees,
                translate=body.translate,
                scale=body.scale,
                flipud=body.flipud,
                fliplr=body.fliplr,
                mosaic=body.mosaic,
                hsv_h=body.hsv_h,
                hsv_s=body.hsv_s,
                hsv_v=body.hsv_v,
                workers=0,
                verbose=False,
            )
            if body.project: train_kwargs["project"] = body.project
            if body.name:    train_kwargs["name"]    = body.name
            results = model.train(**train_kwargs)
            best_weights = str(Path(model.trainer.best))
            state["status"] = "done" if not stop_event.is_set() else "stopped"

        except Exception as e:
            state["status"] = "error"
            state["error"]  = traceback.format_exc()
            print("[YOLO Server] Training error:", e)

    train_thread = threading.Thread(target=_train, daemon=True)
    train_thread.start()
    return {"ok": True}


@app.post("/stop")
def stop_train():
    stop_event.set()
    state["status"] = "stopped"
    return {"ok": True}


@app.post("/shutdown")
def shutdown_server():
    import threading, os
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return {"ok": True}


@app.get("/status")
def get_status():
    return {
        "state":      state["status"],
        "epoch":      state["epoch"],
        "epochs":     state["epochs"],
        "train_loss": state["train_loss"],
        "val_loss":   state["val_loss"],
        "map50":      state["map50"],
        "map50_95":   state["map50_95"],
        "error":      state.get("error"),
    }


@app.get("/infer_progress")
def get_infer_progress():
    return state["infer_progress"]


class MetricsBody(BaseModel):
    save_dir: Optional[str] = None   # write val run into this directory


@app.post("/metrics")
def get_metrics(body: MetricsBody = MetricsBody()):
    """Return final validation metrics after training."""
    if not model or state["status"] not in ("done", "stopped"):
        return {"error": "No completed training run"}

    try:
        val_kwargs = dict(data=dataset_yaml, verbose=False)
        if body.save_dir:
            val_kwargs["project"] = body.save_dir
            val_kwargs["name"]    = "val"
        val_results = model.val(**val_kwargs)
        box = val_results.box
        # per-class breakdown
        per_class = {}
        if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
            names = model.names
            for i, cls_idx in enumerate(box.ap_class_index):
                name = names.get(int(cls_idx), str(cls_idx))
                per_class[name] = {
                    "precision": _safe(box.p[i]),
                    "recall":    _safe(box.r[i]),
                    "map50":     _safe(box.ap50[i]),
                }

        # confusion matrix as 2D list
        cm = None
        if hasattr(val_results, "confusion_matrix") and val_results.confusion_matrix is not None:
            cm = val_results.confusion_matrix.matrix.tolist()

        return {
            "precision": _safe(box.mp),
            "recall":    _safe(box.mr),
            "map50":     _safe(box.map50),
            "map50_95":  _safe(box.map),
            "per_class": per_class,
            "confusion_matrix": cm,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/infer")
def run_infer(body: InferBody):
    """Run inference on a list of image file paths."""
    if not model and not best_weights:
        return {"error": "No trained model available"}

    if body.model_path:
        infer_model = YOLO(body.model_path)
    elif model is not None:
        infer_model = model
    elif best_weights:
        infer_model = YOLO(best_weights)
    else:
        return {"error": "No model available — train first or supply model_path"}

    results_out = []
    try:
        results = infer_model.predict(
            source=body.image_paths,
            conf=body.conf,
            iou=body.iou,
            workers=0,     # Windows: avoid shared-memory page-file exhaustion
            verbose=False,
        )
        for res, img_path in zip(results, body.image_paths):
            dets = []
            if res.boxes is not None:
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append({
                        "cls":  infer_model.names[int(box.cls)],
                        "conf": round(float(box.conf), 4),
                        "x1": round(x1, 1), "y1": round(y1, 1),
                        "x2": round(x2, 1), "y2": round(y2, 1),
                    })
            results_out.append({"path": img_path, "detections": dets})
    except Exception as e:
        return {"error": str(e)}

    return {"results": results_out}


# ── WSI inference ────────────────────────────────────────────────────────────

@app.post("/infer_wsi")
def infer_wsi(body: InferWSIBody):
    try:
        return _infer_wsi_impl(body)
    except Exception as exc:
        state["infer_progress"]["active"] = False
        tb = traceback.format_exc()
        print(f"[infer_wsi] UNHANDLED: {exc}\n{tb}", flush=True)
        return {"error": f"Unhandled server error: {exc}"}


def _infer_wsi_impl(body: InferWSIBody):
    """
    Full WSI inference pipeline in Python.
    Opens the slide (tiffslide for TIFF formats, openslide for everything else),
    detects tissue (or uses supplied ROI bounds), tiles the region, runs YOLO on
    kept tiles, cross-tile NMS, writes GeoJSON.
    """
    if not HAS_TIFFSLIDE and not HAS_OPENSLIDE:
        return {"error": "No slide reader installed — run: pip install openslide-python openslide-bin tiffslide"}

    try:
        infer_model = YOLO(body.model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    try:
        slide = open_slide(body.slide_path)
    except Exception as e:
        return {"error": f"Failed to open slide: {e}"}

    full_w, full_h = slide.dimensions
    tile_size  = body.tile_size
    overlap    = body.overlap
    step       = tile_size - overlap

    out_dir = Path(body.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build tissue mask at 1/32 resolution ─────────────────────────────────
    thumb_factor = 32
    thumb_hint_w = max(1, full_w // thumb_factor)
    thumb_hint_h = max(1, full_h // thumb_factor)
    thumb = slide.get_thumbnail((thumb_hint_w, thumb_hint_h))
    arr_thumb = np.array(thumb.convert("RGB"), dtype=np.uint8)
    # Use actual returned thumbnail dimensions (may differ from requested hint)
    actual_thumb_h, actual_thumb_w = arr_thumb.shape[:2]   # numpy: rows=H, cols=W
    thumb_w, thumb_h = actual_thumb_w, actual_thumb_h
    # Scale factors: full-res pixels per thumbnail pixel in each axis
    sx = thumb_w / full_w   # horizontal scale
    sy = thumb_h / full_h   # vertical scale
    tissue_mask = ((arr_thumb[:, :, 0] < 220) |
                   (arr_thumb[:, :, 1] < 220) |
                   (arr_thumb[:, :, 2] < 220))

    slide_name = Path(body.slide_path).name
    print(f"[infer_wsi] {slide_name}  slide={full_w}x{full_h}  thumb={thumb_w}x{thumb_h}", flush=True)

    # ── ROI polygons: rasterize into mask + compute bounding box ─────────────
    # When the user drew ROI annotations, restrict tiling to the ROI bounding
    # box (huge speedup for small ROIs) and also apply the polygon mask so only
    # tiles that overlap the actual shape are kept.
    if body.roi_polygons:
        valid_polys = [p for p in body.roi_polygons if len(p) >= 3]
        if valid_polys:
            all_pts = [(x, y) for poly in valid_polys for x, y in poly]
            # Full-res bounding box with one tile of padding
            scan_x0 = max(0,       int(min(p[0] for p in all_pts)) - tile_size)
            scan_y0 = max(0,       int(min(p[1] for p in all_pts)) - tile_size)
            scan_x1 = min(full_w,  int(max(p[0] for p in all_pts)) + tile_size)
            scan_y1 = min(full_h,  int(max(p[1] for p in all_pts)) + tile_size)
            # Rasterize polygons into thumbnail mask
            roi_mask = np.zeros((thumb_h, thumb_w), dtype=bool)
            for poly in valid_polys:
                scaled = [(max(0, min(int(x * sx), thumb_w - 1)),
                           max(0, min(int(y * sy), thumb_h - 1)))
                          for x, y in poly]
                img_roi = Image.new("L", (thumb_w, thumb_h), 0)
                ImageDraw.Draw(img_roi).polygon(scaled, fill=1)
                roi_mask |= np.array(img_roi, dtype=bool)
            tissue_mask = tissue_mask & roi_mask
            print(f"[infer_wsi] {len(valid_polys)} ROI polygon(s) — scan bbox "
                  f"({scan_x0},{scan_y0})→({scan_x1},{scan_y1})", flush=True)
        else:
            scan_x0, scan_y0, scan_x1, scan_y1 = 0, 0, full_w, full_h
    else:
        scan_x0, scan_y0, scan_x1, scan_y1 = 0, 0, full_w, full_h

    # ── Generate tile coordinates for each downsample scale ─────────────────
    # Each coord is (tx, ty, ds) — tx/ty in full-res pixels, ds is the downsample.
    # A ds=2 tile reads a 2×tile_size region and downsamples it → same 640px output,
    # but covering twice as much tissue → large structures fit in one tile.
    downsamples = sorted(set(body.downsamples)) or [1.0]
    tile_coords = []
    for ds in downsamples:
        physical   = int(tile_size * ds)        # full-res region size for this scale
        step_phys  = int(step * ds)             # stride in full-res pixels
        mask_tw_ds = max(1, int(physical * sx))
        mask_th_ds = max(1, int(physical * sy))
        min_px_ds  = int(mask_tw_ds * mask_th_ds * body.tissue_frac)
        x = scan_x0
        count_ds = 0
        while x < scan_x1:
            y = scan_y0
            while y < scan_y1:
                cx = min(x, full_w - physical)
                cy = min(y, full_h - physical)
                mx = max(0, min(int(cx * sx), thumb_w - mask_tw_ds))
                my = max(0, min(int(cy * sy), thumb_h - mask_th_ds))
                if tissue_mask[my : my + mask_th_ds, mx : mx + mask_tw_ds].sum() >= min_px_ds:
                    tile_coords.append((max(0, cx), max(0, cy), ds))
                    count_ds += 1
                y += step_phys
            x += step_phys
        print(f"[infer_wsi] ds={ds:.1f}×: {count_ds} candidate tiles", flush=True)

    tiles_total = len(tile_coords)
    print(f"[infer_wsi] {tiles_total} total tiles across {len(downsamples)} scale(s)", flush=True)

    if tiles_total == 0:
        slide.close()
        state["infer_progress"]["active"] = False
        return {"detections": 0, "tiles_total": 0, "tiles_used": 0,
                "geojson_path": None,
                "warning": "No tissue tiles — lower tissue_frac or check the slide/ROI."}

    slide.close()  # main handle no longer needed; workers open their own

    state["infer_progress"] = {
        "active": True, "done": 0, "total": tiles_total,
        "dets": 0, "slide": slide_name,
    }
    print(f"[infer_wsi] starting tile loop: {tiles_total} tiles  batch=64  workers=4", flush=True)

    # ── Parallel tile reads + YOLO (fp16 on GPU) ─────────────────────────────
    use_half   = torch.cuda.is_available()
    all_dets   = []
    tiles_used = 0
    batch_imgs   = []
    batch_coords = []  # (tx, ty, ds)

    def _run_batch():
        nonlocal all_dets
        if not batch_imgs:
            return
        try:
            results = infer_model.predict(
                source=batch_imgs, conf=body.conf, iou=body.iou,
                workers=0, verbose=False, half=use_half)
        except Exception as e:
            raise RuntimeError(f"YOLO inference failed: {e}")
        for res, (bx0, by0, bds) in zip(results, batch_coords):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for box in res.boxes:
                # box coords are in tile-image pixels; scale back to full-res
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_dets.append({
                    "cls":  infer_model.names[int(box.cls)],
                    "conf": round(float(box.conf), 4),
                    "x1":   round(bx0 + x1 * bds, 1), "y1": round(by0 + y1 * bds, 1),
                    "x2":   round(bx0 + x2 * bds, 1), "y2": round(by0 + y2 * bds, 1),
                })
        batch_imgs.clear()
        batch_coords.clear()

    debug_dir = None
    if body.debug_tiles:
        debug_dir = out_dir / "debug_tiles"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[infer_wsi] debug_tiles ON — saving to {debug_dir}", flush=True)

    try:
        reader_fn  = _make_tile_reader(body.slide_path, tile_size)
        chunk_size = 64 * 4   # prefetch 4 YOLO batches ahead
        with ThreadPoolExecutor(max_workers=4) as pool:
            for ci in range(0, tiles_total, chunk_size):
                chunk = tile_coords[ci:ci + chunk_size]
                for tx, ty, ds, arr in pool.map(reader_fn, chunk):
                    if arr is None:
                        continue
                    if debug_dir is not None:
                        # arr is BGR for YOLO; flip back to RGB for human-readable debug images
                        Image.fromarray(arr[:, :, ::-1]).save(
                            debug_dir / f"tile_x{tx}_y{ty}_ds{ds:.1f}.jpg", quality=90)
                    batch_imgs.append(arr)
                    batch_coords.append((tx, ty, ds))
                    tiles_used += 1
                    if len(batch_imgs) == 64:
                        _run_batch()
                        state["infer_progress"]["done"] = tiles_used
                        state["infer_progress"]["dets"] = len(all_dets)
                        print(f"[infer_wsi] {tiles_used}/{tiles_total} tiles  "
                              f"{len(all_dets)} raw dets", flush=True)
            _run_batch()
    except RuntimeError as e:
        state["infer_progress"]["active"] = False
        return {"error": str(e)}

    state["infer_progress"]["active"] = False

    # ── Cross-tile NMS ────────────────────────────────────────────────────────
    merged = _wsi_nms(all_dets, body.iou)
    print(f"[infer_wsi] raw dets: {len(all_dets)}  after NMS: {len(merged)}", flush=True)

    # ── Confidence distribution + per-class counts ────────────────────────────
    conf_hist = {"0.25-0.40": 0, "0.40-0.60": 0, "0.60-0.80": 0, "0.80-1.00": 0}
    class_counts: dict = {}
    for d in merged:
        c = d["conf"]
        if   c < 0.40: conf_hist["0.25-0.40"] += 1
        elif c < 0.60: conf_hist["0.40-0.60"] += 1
        elif c < 0.80: conf_hist["0.60-0.80"] += 1
        else:          conf_hist["0.80-1.00"] += 1
        class_counts[d["cls"]] = class_counts.get(d["cls"], 0) + 1
    print(f"[infer_wsi] conf distribution: {conf_hist}", flush=True)
    print(f"[infer_wsi] per-class counts:  {class_counts}", flush=True)

    # ── Write QuPath-compatible GeoJSON ───────────────────────────────────────
    slide_stem   = Path(body.slide_path).stem
    geojson_path = out_dir / f"{slide_stem}.geojson"
    _write_qupath_geojson(geojson_path, merged, body.conf)
    print(f"[infer_wsi] GeoJSON written: {geojson_path}", flush=True)

    return {
        "detections":   len(merged),
        "tiles_total":  tiles_total,
        "tiles_used":   tiles_used,
        "geojson_path": str(geojson_path),
        "conf_hist":    conf_hist,
        "class_counts": class_counts,
    }


def _wsi_nms(dets: list, iou_thresh: float) -> list:
    if not dets:
        return []
    dets_sorted = sorted(dets, key=lambda d: -d["conf"])
    suppress = [False] * len(dets_sorted)
    result = []
    for i, d in enumerate(dets_sorted):
        if suppress[i]:
            continue
        result.append(d)
        for j in range(i + 1, len(dets_sorted)):
            if suppress[j] or dets_sorted[j]["cls"] != d["cls"]:
                continue
            if _wsi_iou(d, dets_sorted[j]) > iou_thresh:
                suppress[j] = True
    return result


def _wsi_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"]);  iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]);  iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _write_qupath_geojson(path: Path, dets: list, conf_threshold: float):
    """Write detections as QuPath-compatible GeoJSON (drag-drop or PathIO.readObjects)."""
    RED   = -65536    # java.awt.Color.RED.getRGB()
    GREEN = -16711936 # java.awt.Color.GREEN.getRGB()
    features = []
    for d in dets:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        color = RED if d["conf"] < conf_threshold else GREEN
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]
            },
            "properties": {
                "objectType": "detection",
                "classification": {"name": d["cls"], "colorRGB": color},
                "name": f"{d['conf']:.2f}"
            }
        })
    with open(str(path), "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe(val):
    """Convert tensor/numpy scalar to Python float, or None."""
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    if torch.cuda.is_available():
        dev = f"GPU — {torch.cuda.get_device_name(0)}"
    else:
        dev = "CPU (no CUDA — reinstall PyTorch with CUDA for GPU training)"
    print(f"[YOLO Server] Device  : {dev}")
    print(f"[YOLO Server] Starting: {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
