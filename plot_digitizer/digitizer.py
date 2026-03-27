"""Main digitisation pipeline."""

import csv
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .axis_reader import pixel_to_data, read_axes
from .curve_extractor import extract_curves
from .plot_detector import detect_plot_area

logger = logging.getLogger(__name__)


def digitize_plot(
    image_path: Path,
    output_path: Path,
    method: str = "naive",
    x_range: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
    smoothing: float = 1.0,
    n_curves: Optional[int] = None,
    has_grid: bool = False,
    n_samples: int = 500,
    sat_threshold: float = 0.20,
    min_col_coverage: float = 0.20,
    hue_bins: int = 18,
    span_frac: float = 0.55,
    max_thickness: Optional[int] = None,
    notch_factor: float = 3.0,
) -> None:
    """
    Full pipeline: load image → detect plot area → read axes → extract curves
    → convert to data coords → write CSV.
    """
    t0 = time.perf_counter()

    # 1. Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    logger.info(f"Loaded image: {image_path.name}  size={w}x{h}")

    # 2. Detect plot canvas
    plot_area = detect_plot_area(img_array)
    x_min, y_min, x_max, y_max = plot_area
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    logger.info(f"Canvas: {canvas_w}x{canvas_h} px  ({x_min},{y_min})→({x_max},{y_max})")

    # 3. Read axis calibration
    x_calib, y_calib = read_axes(
        img_array, plot_area, x_range=x_range, y_range=y_range
    )

    # 4. Extract curves
    curves = extract_curves(
        img_array, plot_area,
        method=method, smoothing=smoothing,
        n_curves=n_curves, has_grid=has_grid,
        n_samples=n_samples, sat_threshold=sat_threshold,
        min_col_coverage=min_col_coverage, hue_bins=hue_bins,
        span_frac=span_frac, max_thickness=max_thickness,
        notch_factor=notch_factor,
    )
    n_curves = len(curves)
    logger.info(f"Curves found: {n_curves}  (method={method})")

    if n_curves == 0:
        logger.warning("No curves detected — check image or try --verbose")

    # 5. Convert pixel → data coordinates and collect rows
    rows: list[tuple[str, float, float]] = []
    for curve_name, pts in curves.items():
        for px, py in pts:
            dx = pixel_to_data(px, x_calib)
            dy = pixel_to_data(py, y_calib)
            rows.append((curve_name, dx, dy))

    # Sort for cleaner output
    rows.sort(key=lambda r: (r[0], r[1]))

    # 6. Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["curve", "x", "y"])
        for curve_name, dx, dy in rows:
            writer.writerow([curve_name, f"{dx:.6f}", f"{dy:.6f}"])

    elapsed = time.perf_counter() - t0

    # Metrics log
    logger.info("--- Metrics ---")
    logger.info(f"  Total data points : {len(rows)}")
    logger.info(f"  Curves            : {n_curves}")
    for cname, pts in curves.items():
        logger.info(f"    {cname}: {len(pts)} pts")
    logger.info(f"  Processing time   : {elapsed:.2f} s")
    logger.info(f"  Output            : {output_path}")
