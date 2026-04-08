"""FastAPI backend for PlotDigitizer web GUI."""

import argparse
import csv
import io
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from ..axis_reader import pixel_to_data, read_axes
from ..curve_extractor import extract_curves
from ..plot_detector import detect_plot_area

logger = logging.getLogger(__name__)

app = FastAPI(title="PlotDigitizer")

# In-memory image store: image_id → RGB uint8 ndarray
_images: dict[str, np.ndarray] = {}

_HTML = (Path(__file__).parent / "index.html").read_text()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img)
    image_id = str(uuid.uuid4())
    _images[image_id] = arr
    h, w = arr.shape[:2]
    return {"image_id": image_id, "width": w, "height": h}


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    if image_id not in _images:
        raise HTTPException(404, "Image not found")
    buf = io.BytesIO()
    Image.fromarray(_images[image_id]).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


class DetectRequest(BaseModel):
    image_id: str
    x_range: Optional[list[float]] = None
    y_range: Optional[list[float]] = None
    n_curves: Optional[int] = None
    method: str = "naive"
    has_grid: bool = False
    smoothing: float = 1.0
    n_samples: int = 500
    sat_threshold: float = 0.20
    min_col_coverage: float = 0.20
    hue_bins: int = 18
    span_frac: float = 0.55
    max_thickness: Optional[int] = None
    notch_factor: float = 3.0


@app.post("/api/detect")
def detect(req: DetectRequest):
    if req.image_id not in _images:
        raise HTTPException(404, "Image not found")

    arr = _images[req.image_id]
    plot_area = detect_plot_area(arr)

    x_range = tuple(req.x_range) if req.x_range else None
    y_range = tuple(req.y_range) if req.y_range else None
    x_calib, y_calib = read_axes(arr, plot_area, x_range=x_range, y_range=y_range)

    raw = extract_curves(
        arr, plot_area,
        method=req.method,
        smoothing=req.smoothing,
        n_curves=req.n_curves,
        has_grid=req.has_grid,
        n_samples=req.n_samples,
        sat_threshold=req.sat_threshold,
        min_col_coverage=req.min_col_coverage,
        hue_bins=req.hue_bins,
        span_frac=req.span_frac,
        max_thickness=req.max_thickness,
        notch_factor=req.notch_factor,
    )

    curves = [
        {
            "label": label,
            "color": _color_from_label(label),
            "points": pts.tolist(),
        }
        for label, pts in raw.items()
    ]

    return {
        "curves": curves,
        "calibration": {"x": list(x_calib), "y": list(y_calib)},
        "plot_area": list(plot_area),
    }


class ExportCurve(BaseModel):
    label: str
    points: list[list[float]]


class ExportRequest(BaseModel):
    curves: list[ExportCurve]
    calibration: dict
    curve_label: str = "curve"
    x_label: str = "x"
    y_label: str = "y"


@app.post("/api/export")
def export_csv(req: ExportRequest):
    x_calib = tuple(req.calibration["x"])
    y_calib = tuple(req.calibration["y"])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([req.curve_label, req.x_label, req.y_label])
    for c in req.curves:
        pts = sorted(c.points, key=lambda p: p[0])
        for px, py in pts:
            dx = pixel_to_data(px, x_calib)
            dy = pixel_to_data(py, y_calib)
            writer.writerow([c.label, f"{dx:.6f}", f"{dy:.6f}"])

    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="digitized.csv"'},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color_from_label(name: str) -> list[int]:
    m = re.search(r"_rgb(\d{3})(\d{3})(\d{3})$", name)
    if m:
        return [int(m.group(1)), int(m.group(2)), int(m.group(3))]
    m = re.search(r"_gray_v(\d{3})$", name)
    if m:
        v = int(m.group(1))
        return [v, v, v]
    return [120, 160, 220]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PlotDigitizer web GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    import uvicorn
    print(f"PlotDigitizer Web GUI → http://{args.host}:{args.port}")
    uvicorn.run(
        "plot_digitizer.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
