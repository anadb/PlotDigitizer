"""Detect the plot canvas bounding box within an image."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def detect_plot_area(img_array: np.ndarray) -> tuple[int, int, int, int]:
    """
    Detect the axis-bounded plot canvas area.

    Looks for long continuous dark lines (spines) that form the plot border.

    Returns (x_min, y_min, x_max, y_max) pixel coords of the plot canvas.
    """
    h, w = img_array.shape[:2]
    gray = img_array[:, :, :3].mean(axis=2)

    # Try increasingly permissive darkness thresholds until spine lines are found
    for threshold in (80, 120, 160, 200, 230):
        is_dark = gray < threshold

        row_max_run = _max_run_lengths(is_dark)
        col_max_run = _max_run_lengths(is_dark.T)

        h_rows = np.where(row_max_run > w * 0.30)[0]
        v_cols = np.where(col_max_run > h * 0.30)[0]

        if len(h_rows) >= 2 and len(v_cols) >= 2:
            y_min, y_max = int(h_rows[0]), int(h_rows[-1])
            x_min, x_max = int(v_cols[0]), int(v_cols[-1])
            logger.info(
                f"Detected plot area (threshold={threshold}): "
                f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]"
            )
            return x_min, y_min, x_max, y_max

    logger.warning("Spine detection failed; using 12 %/88 % margin fallback")
    y_min, y_max = int(h * 0.12), int(h * 0.88)
    x_min, x_max = int(w * 0.12), int(w * 0.88)

    return x_min, y_min, x_max, y_max


def _max_run_lengths(mask: np.ndarray) -> np.ndarray:
    """
    For each row of `mask`, return the length of the longest contiguous
    True run.  Uses a vectorised diff approach.
    """
    n_rows, n_cols = mask.shape
    result = np.zeros(n_rows, dtype=int)

    for i in range(n_rows):
        row = mask[i]
        padded = np.empty(n_cols + 2, dtype=bool)
        padded[0] = False
        padded[1:-1] = row
        padded[-1] = False
        diff = np.diff(padded.view(np.uint8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == 255)[0]   # uint8: 0-1 wraps to 255
        if starts.size:
            result[i] = int((ends - starts).max())

    return result
