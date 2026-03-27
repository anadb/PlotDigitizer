"""Extract curves from the plot canvas by color."""

import logging
from typing import Optional
import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


def extract_curves(
    img_array: np.ndarray,
    plot_area: tuple[int, int, int, int],
    method: str = "naive",
    n_samples: int = 500,
    smoothing: float = 1.0,
    n_curves: Optional[int] = None,
    has_grid: bool = False,
    sat_threshold: float = 0.20,
    min_col_coverage: float = 0.20,
    hue_bins: int = 18,
    span_frac: float = 0.55,
    max_thickness: Optional[int] = None,
    notch_factor: float = 3.0,
) -> dict[str, np.ndarray]:
    """
    Extract digitised curves from the plot area.

    Args:
        n_curves:       If given, keep only the top-N color segments by pixel count.
        has_grid:       If True, suppress grid lines before colour segmentation.
        sat_threshold:  HSV saturation cutoff; lower values detect paler curves.
        min_col_coverage: Fraction of canvas width a colour must span to be kept.
        hue_bins:       Number of hue bins for colour segmentation (finer = more bins).
        span_frac:      Row/col must span this fraction of canvas to count as a grid line.
        max_thickness:  Max pixel cluster height for per-column extraction (auto if None).
        notch_factor:   Spread > max_thickness * notch_factor triggers deep-notch mode.

    Returns dict mapping colour label → Nx2 float array of
    (pixel_x, pixel_y) values in image coordinates.
    """
    x_min, y_min, x_max, y_max = plot_area
    canvas = img_array[y_min:y_max, x_min:x_max].astype(np.float32)

    color_masks = _segment_by_color(
        canvas,
        has_grid=has_grid,
        sat_threshold=sat_threshold,
        min_col_coverage=min_col_coverage,
        hue_bins=hue_bins,
        span_frac=span_frac,
    )

    # Keep only the top-N masks by pixel count
    if n_curves is not None and len(color_masks) > n_curves:
        ranked = sorted(color_masks.items(), key=lambda kv: kv[1].sum(), reverse=True)
        color_masks = dict(ranked[:n_curves])
        logger.info(f"Keeping top {n_curves} segments by pixel count")

    logger.info(f"Color segments found: {len(color_masks)}")

    curves: dict[str, np.ndarray] = {}
    for label, mask in color_masks.items():
        if method == "cv":
            pts = _extract_cv(mask, x_min, y_min)
        else:
            pts = _extract_naive(
                mask, x_min, y_min, n_samples, smoothing,
                max_thickness=max_thickness, notch_factor=notch_factor,
            )

        if pts is not None and len(pts) > 1:
            curves[label] = pts
            logger.info(f"  {label}: {len(pts)} points")

    return curves


# ---------------------------------------------------------------------------
# Color segmentation
# ---------------------------------------------------------------------------

def _segment_by_color(
    canvas: np.ndarray,
    sat_threshold: float = 0.20,
    min_pixel_fraction: float = 0.002,
    hue_bins: int = 18,
    has_grid: bool = False,
    min_col_coverage: float = 0.20,
    span_frac: float = 0.55,
) -> dict[str, np.ndarray]:
    """
    Return {label: bool_mask} for each distinct colour group.

    Handles coloured curves (via hue binning) and achromatic gray curves.
    Optionally removes grid lines before analysis.
    """
    r = canvas[:, :, 0] / 255.0
    g = canvas[:, :, 1] / 255.0
    b = canvas[:, :, 2] / 255.0

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    v = max_c
    safe_max = np.where(max_c > 1e-6, max_c, 1.0)
    s = np.where(max_c > 1e-6, delta / safe_max, 0.0)

    # Detect and suppress grid pixels before any further analysis
    grid_mask = _detect_grid(s, v, canvas.shape[:2], span_frac=span_frac) if has_grid else np.zeros(canvas.shape[:2], dtype=bool)
    if has_grid:
        grid_px = int(grid_mask.sum())
        logger.info(f"Grid pixels suppressed: {grid_px}")

    # Hue in [0, 360)
    hue = np.zeros_like(r)
    eps = 1e-8
    mask_r = (max_c == r) & (delta > eps)
    mask_g = (max_c == g) & (delta > eps)
    mask_b = (max_c == b) & (delta > eps)
    hue[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360
    hue[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120
    hue[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240

    # Foreground: saturated, not too dark, not a grid line
    fg = (s > sat_threshold) & (v > 0.15) & ~grid_mask

    h_px, w_px = canvas.shape[:2]
    total_pixels = h_px * w_px
    # Scale pixel-fraction threshold proportionally when min_col_coverage is
    # lower than the default (0.20), so partial curves aren't doubly rejected.
    effective_min_px_frac = min_pixel_fraction * (min_col_coverage / 0.20)

    masks: dict[str, np.ndarray] = {}

    if fg.sum() >= 10:
        bin_size = 360.0 / hue_bins
        hue_bin = (hue / bin_size).astype(int) % hue_bins

        for b_idx in range(hue_bins):
            bin_mask = fg & (hue_bin == b_idx)
            if bin_mask.sum() < total_pixels * effective_min_px_frac:
                continue
            col_hits = bin_mask.any(axis=0).sum()
            if col_hits < w_px * min_col_coverage:
                continue
            r_mean = int(canvas[:, :, 0][bin_mask].mean())
            g_mean = int(canvas[:, :, 1][bin_mask].mean())
            b_mean = int(canvas[:, :, 2][bin_mask].mean())
            label = f"curve_h{b_idx:02d}_rgb{r_mean:03d}{g_mean:03d}{b_mean:03d}"
            masks[label] = bin_mask
    else:
        logger.warning("No saturated foreground pixels found in plot area")

    # --- Gray / achromatic curves ----------------------------------------
    gray_fg = (s < 0.15) & (v > 0.25) & (v < 0.90) & ~grid_mask
    n_gray_bins = 8
    gray_bin_size = (0.90 - 0.25) / n_gray_bins
    gray_candidates: list[tuple[str, np.ndarray, np.ndarray]] = []

    for i in range(n_gray_bins):
        v_lo = 0.25 + i * gray_bin_size
        v_hi = v_lo + gray_bin_size
        bin_mask = gray_fg & (v >= v_lo) & (v < v_hi)
        if bin_mask.sum() < total_pixels * effective_min_px_frac:
            continue
        col_profile = bin_mask.any(axis=0)
        if col_profile.sum() < w_px * min_col_coverage:
            continue
        v_mean = int((v_lo + v_hi) / 2 * 255)
        gray_candidates.append((f"curve_gray_v{v_mean:03d}", bin_mask, col_profile))

    # Merge gray bins with >60 % column overlap (same curve, different shade)
    used = [False] * len(gray_candidates)
    for i, (label_i, mask_i, col_i) in enumerate(gray_candidates):
        if used[i]:
            continue
        combined = mask_i.copy()
        for j in range(i + 1, len(gray_candidates)):
            if used[j]:
                continue
            col_j = gray_candidates[j][2]
            overlap = (col_i & col_j).sum() / max(col_i.sum(), col_j.sum(), 1)
            if overlap > 0.60:
                combined |= gray_candidates[j][1]
                used[j] = True
        masks[label_i] = combined
        used[i] = True

    return masks


def _detect_grid(
    s: np.ndarray,
    v: np.ndarray,
    shape: tuple[int, int],
    span_frac: float = 0.55,
) -> np.ndarray:
    """
    Identify grid / spine pixels to suppress before curve extraction.

    Pass 1 — periodic horizontal/vertical grid lines:
        Achromatic rows/columns that span ≥ span_frac of the canvas.

    Pass 2 — canvas border / tick-mark area:
        * Top & sides : small fixed margin (spine line width).
        * Bottom      : larger margin (h // 10) to cover x-axis tick marks
          and their anti-aliasing bleed, which can look like gray curves.
    """
    h, w = shape

    candidate = (s < 0.12) & (v > 0.60) & (v < 0.97)

    grid_mask = np.zeros((h, w), dtype=bool)

    # Horizontal grid lines
    row_hits = candidate.sum(axis=1)
    grid_mask[row_hits > w * span_frac, :] = True

    # Vertical grid lines (and y-axis spine)
    col_hits = candidate.sum(axis=0)
    grid_mask[:, col_hits > h * span_frac] = True

    # Fixed border — top / left / right: just the spine (few pixels)
    # Bottom: large enough to cover x-axis ticks + anti-aliasing
    top = 5
    side = 5
    bottom = max(10, h // 10)   # e.g. 34 px for h=342

    grid_mask[:top, :] = True
    grid_mask[-bottom:, :] = True
    grid_mask[:, :side] = True
    grid_mask[:, -side:] = True

    return grid_mask


# ---------------------------------------------------------------------------
# Naive extraction: per-column tight-cluster median + outlier filter + spline
# ---------------------------------------------------------------------------

def _tight_cluster_median(
    active: np.ndarray,
    max_thickness: int = 30,
    min_pixels: int = 2,
    notch_factor: float = 3.0,
) -> Optional[float]:
    """
    Return the median y of the densest compact group of pixels in `active`.

    Uses a sliding-window approach: find the window of height ≤ max_thickness
    that contains the most pixels, then return its median.  This makes the
    extractor robust to scattered artefacts (axis lines, grid ticks) that
    appear alongside the actual curve pixels in a column.

    Returns None if no window meets the min_pixels requirement.
    """
    if active.size == 0:
        return None

    a = np.sort(active)

    if a[-1] - a[0] <= max_thickness:
        # All pixels already form a tight cluster
        return float(np.median(a))

    # Very wide spread *with high pixel density* indicates a near-vertical segment
    # (e.g. deep resonance notch) — use the deepest point (max y).
    # A sparse wide spread is noise alongside the real curve; fall through to the
    # sliding-window logic which finds the denser cluster instead.
    if a[-1] - a[0] > max_thickness * notch_factor:
        density = len(a) / (a[-1] - a[0] + 1)
        if density > 0.4:
            return float(a[-1])

    # Sliding window O(n)
    best_count = 0
    best_lo = 0
    lo = 0
    for hi in range(len(a)):
        while a[hi] - a[lo] > max_thickness:
            lo += 1
        count = hi - lo + 1
        if count > best_count:
            best_count = count
            best_lo = lo

    if best_count < min_pixels:
        return None

    cluster = a[best_lo : best_lo + best_count]
    return float(np.median(cluster))

def _extract_naive(
    mask: np.ndarray,
    x_offset: int,
    y_offset: int,
    n_samples: int,
    smoothing: float = 1.0,
    max_thickness: Optional[int] = None,
    notch_factor: float = 3.0,
) -> Optional[np.ndarray]:
    """
    For each x-column, find the median y of active pixels.
    Apply a median-filter pass to reject outlier columns, then fit a spline.
    """
    h, w = mask.shape
    thickness = max_thickness if max_thickness is not None else max(8, h // 25)
    xs_raw: list[int] = []
    ys_raw: list[float] = []

    for x in range(w):
        active = np.where(mask[:, x])[0]
        if active.size == 0:
            continue
        y_val = _tight_cluster_median(active, max_thickness=thickness, min_pixels=2,
                                      notch_factor=notch_factor)
        if y_val is None:
            continue
        xs_raw.append(x)
        ys_raw.append(y_val)

    if len(xs_raw) < 4:
        return None

    xs = np.array(xs_raw)
    ys = np.array(ys_raw)

    # Outlier rejection: remove points deviating > 3 MAD from local median
    ys = _reject_outliers(xs, ys)

    if len(xs) < 4:
        return None

    # Smoothing spline
    try:
        s_param = max(len(xs) * 0.5 * smoothing, 1.0)
        spl = interpolate.UnivariateSpline(xs, ys, k=3, s=s_param)
        xs_dense = np.linspace(xs[0], xs[-1], n_samples)
        ys_dense = spl(xs_dense)
    except Exception as exc:
        logger.debug(f"Spline failed ({exc}), using raw points")
        xs_dense = xs.astype(float)
        ys_dense = ys

    return np.column_stack([xs_dense + x_offset, ys_dense + y_offset])


def _reject_outliers(
    xs: np.ndarray,
    ys: np.ndarray,
    window: int = 21,
    threshold: float = 3.0,
) -> np.ndarray:
    """
    Remove outlier y-values using a sliding-window MAD filter.
    Returns filtered ys (xs are modified in-place via boolean indexing caller-side).
    """
    n = len(ys)
    half = window // 2
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = ys[lo:hi]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        if mad < 1e-6:
            continue
        if abs(ys[i] - med) > threshold * mad * 1.4826:
            keep[i] = False

    # Re-assign xs to share the filter result (caller uses returned ys length)
    xs[:] = xs  # no-op — caller already has xs; we just shrink ys
    # Return filtered versions by rebuilding — caller must re-slice xs too.
    # We overwrite xs in-place via a trick: store result length in global? No.
    # Simpler: just zero outliers to local median instead of dropping them.
    result = ys.copy()
    for i in range(n):
        if not keep[i]:
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            result[i] = np.median(ys[lo:hi])

    return result


# ---------------------------------------------------------------------------
# CV extraction: OpenCV skeleton
# ---------------------------------------------------------------------------

def _extract_cv(
    mask: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> Optional[np.ndarray]:
    """
    Morphological skeleton → per-column median y.
    Falls back to naive if OpenCV is unavailable.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv not available, falling back to naive method")
        return _extract_naive(mask, x_offset, y_offset, n_samples=500)

    img_u8 = mask.astype(np.uint8) * 255
    skeleton = np.zeros_like(img_u8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = img_u8.copy()
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded
        if cv2.countNonZero(temp) == 0:
            break

    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return None

    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    xs_u, inverse = np.unique(xs, return_inverse=True)
    ys_u = np.array([np.median(ys[inverse == i]) for i in range(len(xs_u))])

    return np.column_stack([xs_u + x_offset, ys_u + y_offset])
