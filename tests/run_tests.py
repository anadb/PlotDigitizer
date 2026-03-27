"""
Synthetic tests for PlotDigitizer.

Generates plots with known ground-truth curves, runs the digitizer,
and reports RMS error for each curve.

Usage:  uv run python tests/run_tests.py
"""

import sys
import logging
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure the package is importable when run from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_digitizer.digitizer import digitize_plot

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_RMS_THRESHOLD = 0.05   # ≤5 % of y-range is a pass (default)


def _make_plot(curves, x_range, y_range, grid=False, dpi=150, fmt="png", quality=95,
               linewidth=2):
    """Render curves to a temp file; return (Path, ground_truth_dict)."""
    x = np.linspace(x_range[0], x_range[1], 500)
    fig, ax = plt.subplots(figsize=(8, 5))

    gt = {}
    for label, fn, color in curves:
        y = fn(x)
        # NaN values create gaps (used for partial-curve tests)
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
        valid = ~np.isnan(y)
        gt[label] = (x[valid], y[valid])

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if grid:
        ax.grid(True, alpha=0.4)

    fig.tight_layout()

    suffix = ".jpg" if fmt == "jpeg" else ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    if fmt == "jpeg":
        fig.savefig(tmp.name, format="jpeg", dpi=dpi, pil_kwargs={"quality": quality})
    else:
        fig.savefig(tmp.name, format="png", dpi=dpi)
    plt.close(fig)
    return Path(tmp.name), gt


def _match_curves(digitised, gt, x_range, y_range):
    """
    Match each digitised curve to the closest ground-truth curve and
    return {gt_label: rms_error} (normalised by y-range).
    """
    y_span = abs(y_range[1] - y_range[0]) or 1.0
    result = {}
    used_dig = set()

    for gt_label, (gt_x, gt_y) in gt.items():
        best_label = None
        best_rms = float("inf")
        for dig_label, pts in digitised.items():
            if dig_label in used_dig:
                continue
            dig_x = pts[:, 0]
            dig_y = pts[:, 1]
            # Interpolate digitised onto gt_x grid
            interp_y = np.interp(gt_x, dig_x, dig_y,
                                  left=dig_y[0], right=dig_y[-1])
            rms = float(np.sqrt(np.mean((interp_y - gt_y) ** 2))) / y_span
            if rms < best_rms:
                best_rms = rms
                best_label = dig_label
        if best_label is not None:
            used_dig.add(best_label)
        result[gt_label] = best_rms

    return result


def _run_test(name, curves, x_range, y_range, grid=False,
              dpi=150, fmt="png", quality=95, linewidth=2,
              n_curves=None, smoothing=1.0,
              n_samples=500, sat_threshold=0.20, min_col_coverage=0.20,
              hue_bins=18, span_frac=0.55, max_thickness=None, notch_factor=3.0,
              pass_threshold=None):
    img_path, gt = _make_plot(curves, x_range, y_range,
                               grid=grid, dpi=dpi, fmt=fmt, quality=quality,
                               linewidth=linewidth)
    csv_path = img_path.with_suffix(".csv")

    try:
        from plot_digitizer.axis_reader import read_axes
        from plot_digitizer.plot_detector import detect_plot_area
        from plot_digitizer.curve_extractor import extract_curves
        import numpy as _np
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        img_array = _np.array(img)
        plot_area = detect_plot_area(img_array)
        x_calib, y_calib = read_axes(img_array, plot_area,
                                      x_range=x_range, y_range=y_range)

        from plot_digitizer.axis_reader import pixel_to_data
        raw = extract_curves(img_array, plot_area,
                              method="naive", smoothing=smoothing,
                              n_curves=n_curves or len(curves),
                              has_grid=grid,
                              n_samples=n_samples,
                              sat_threshold=sat_threshold,
                              min_col_coverage=min_col_coverage,
                              hue_bins=hue_bins,
                              span_frac=span_frac,
                              max_thickness=max_thickness,
                              notch_factor=notch_factor)

        # Convert pixels → data coords
        digitised = {}
        for label, pts in raw.items():
            xs_d = _np.array([pixel_to_data(p[0], x_calib) for p in pts])
            ys_d = _np.array([pixel_to_data(p[1], y_calib) for p in pts])
            order = _np.argsort(xs_d)
            digitised[label] = _np.column_stack([xs_d[order], ys_d[order]])

        errors = _match_curves(digitised, gt, x_range, y_range)

        threshold = pass_threshold if pass_threshold is not None else PASS_RMS_THRESHOLD
        passed = all(e <= threshold for e in errors.values())
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}  (threshold={threshold*100:.0f}%)")
        for gt_lbl, rms in errors.items():
            mark = "✓" if rms <= threshold else "✗"
            print(f"       {mark} {gt_lbl}: RMS={rms*100:.2f}% of y-range")
        if len(digitised) != len(curves):
            print(f"       ! Expected {len(curves)} curves, got {len(digitised)}")
        return passed

    except Exception as exc:
        print(f"[ERROR] {name}: {exc}")
        import traceback; traceback.print_exc()
        return False
    finally:
        img_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_single_sine():
    return _run_test(
        "Single sine curve (PNG 150dpi)",
        curves=[("sin", np.sin, "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        n_curves=1,
    )


def test_two_curves_no_grid():
    return _run_test(
        "Two curves, no grid (PNG 150dpi)",
        curves=[
            ("sin", np.sin, "C0"),
            ("cos", np.cos, "C1"),
        ],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        n_curves=2,
    )


def test_three_curves_with_grid():
    return _run_test(
        "Three curves, with grid (PNG 150dpi)",
        curves=[
            ("sin", np.sin, "C0"),
            ("cos", np.cos, "C1"),
            ("half-sin2", lambda x: 0.5 * np.sin(2 * x), "C2"),
        ],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        grid=True,
        n_curves=3,
    )


def test_linear_curves():
    return _run_test(
        "Two linear curves (PNG 150dpi)",
        curves=[
            ("line1", lambda x: 0.3 * x - 1, "C0"),   # stays in [-1, 2] over [0,10]
            ("line2", lambda x: -0.2 * x + 1, "C3"),  # stays in [-1, 1] over [0,10]
        ],
        x_range=(0, 10),
        y_range=(-2, 3),
        n_curves=2,
    )


def test_jpeg_quality():
    return _run_test(
        "Two curves, JPEG q=85",
        curves=[
            ("sin", np.sin, "C0"),
            ("cos", np.cos, "C1"),
        ],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        fmt="jpeg",
        quality=85,
        n_curves=2,
    )


def test_high_dpi():
    return _run_test(
        "Three curves, PNG 300dpi",
        curves=[
            ("sin", np.sin, "C0"),
            ("cos", np.cos, "C1"),
            ("half-sin2", lambda x: 0.5 * np.sin(2 * x), "C2"),
        ],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        dpi=300,
        n_curves=3,
    )


def test_overlapping_curves():
    return _run_test(
        "Overlapping curves (crossing sin/cos)",
        curves=[
            ("sin", np.sin, "C0"),
            ("cos", np.cos, "C1"),
        ],
        x_range=(0, 4 * np.pi),
        y_range=(-1.2, 1.2),
        n_curves=2,
    )


# ---------------------------------------------------------------------------
# Parameter-specific test cases
# ---------------------------------------------------------------------------

def test_n_samples_low():
    """--n-samples 50: coarser output should still be accurate."""
    return _run_test(
        "Low n-samples=50",
        curves=[("sin", np.sin, "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        n_curves=1,
        n_samples=50,
    )


def test_pale_curve():
    """--sat-threshold 0.08: detect a very pale (low-saturation) curve.

    #DCDCFF is pale lavender-blue with HSV saturation ≈ 0.14, below the
    default 0.20 threshold, so it would be missed without lowering it.
    """
    return _run_test(
        "Pale curve (sat=0.14) with sat-threshold=0.08",
        curves=[("pale", np.sin, "#DCDCFF")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        n_curves=1,
        sat_threshold=0.08,
    )


def test_partial_curve():
    """--min-col-coverage 0.10: detect a curve spanning only ~15 % of x-axis.

    With the default 0.20 the curve would be dropped; 0.10 keeps it.
    """
    x_full = np.linspace(0, 2 * np.pi, 500)
    cutoff = 2 * np.pi * 0.15          # rightmost 15 % of x-range

    def partial_sin(x):
        y = np.where(x <= cutoff, np.sin(x), np.nan)
        return y

    return _run_test(
        "Partial curve (15 % width) with min-col-coverage=0.10",
        curves=[("partial_sin", partial_sin, "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        n_curves=1,
        min_col_coverage=0.10,
    )


def test_thick_line():
    """--max-thickness 40: correctly extract a very thick line (linewidth=8).

    Auto thickness = max(8, h//25) ≈ 18 px at 150 dpi; an 8-pt line renders
    ~24 px wide, so the auto value can misplace the centre.  Explicit 40 px
    covers the full stroke.
    """
    return _run_test(
        "Thick line (linewidth=8) with max-thickness=40",
        curves=[("sin", np.sin, "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        n_curves=1,
        linewidth=8,
        max_thickness=40,
    )


def test_hue_bins_high():
    """--hue-bins 36: finer hue separation for three distinct colours."""
    return _run_test(
        "Three curves with hue-bins=36",
        curves=[
            ("sin",       np.sin,                          "C0"),
            ("cos",       np.cos,                          "C1"),
            ("half-sin2", lambda x: 0.5 * np.sin(2 * x),  "C2"),
        ],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        n_curves=3,
        hue_bins=36,
    )


def test_span_frac_low():
    """--span-frac 0.40: less-aggressive grid suppression still passes."""
    return _run_test(
        "Grid plot with span-frac=0.40 (less aggressive suppression)",
        curves=[
            ("sin", np.sin, "C0"),
            ("cos", np.cos, "C1"),
        ],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        grid=True,
        n_curves=2,
        span_frac=0.40,
    )


def test_notch_curve():
    """--notch-factor 2.0: capture a deep narrow notch in a sine wave.

    The notch drops 80 % of the y-range in a narrow window; notch_factor=2.0
    triggers deep-notch mode (max-y) at a lower spread threshold than the
    default 3.0, recovering the minimum more faithfully.
    """
    y_span = 2.4   # y_range = (-1.2, 1.2)
    notch_depth = y_span * 0.80

    def notch_sin(x):
        notch = -notch_depth * np.exp(-((x - np.pi) ** 2) / 0.03)
        return np.sin(x) + notch

    return _run_test(
        "Notch curve with notch-factor=2.0",
        curves=[("notch_sin", notch_sin, "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-2.2, 1.2),
        n_curves=1,
        notch_factor=2.0,
        smoothing=0.2,
    )


# ---------------------------------------------------------------------------
# Noisy-data tests  (pass_threshold raised to account for residual noise)
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(42)


def _noisy(fn, sigma):
    """Wrap a function to add fixed-seed Gaussian noise."""
    def _f(x):
        return fn(x) + _rng.normal(0, sigma, x.shape)
    return _f


def test_noisy_sine_moderate():
    """Moderate noise (σ=0.05, ~2 % of y-range); smoothing=2 recovers curve."""
    return _run_test(
        "Noisy sine (σ=0.05) with smoothing=2.0",
        curves=[("sin_noisy", _noisy(np.sin, 0.05), "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.5, 1.5),
        n_curves=1,
        smoothing=2.0,
        pass_threshold=0.08,
    )


def test_noisy_two_curves():
    """Two curves both with moderate noise; smoothing=2 separates them."""
    return _run_test(
        "Two noisy curves (σ=0.05) with smoothing=2.0",
        curves=[
            ("sin_noisy", _noisy(np.sin, 0.05), "C0"),
            ("cos_noisy", _noisy(np.cos, 0.05), "C1"),
        ],
        x_range=(0, 2 * np.pi),
        y_range=(-1.5, 1.5),
        n_curves=2,
        smoothing=2.0,
        pass_threshold=0.08,
    )


def test_very_noisy_sine():
    """High noise (σ=0.15, ~5 % of y-range); higher smoothing needed."""
    return _run_test(
        "Very noisy sine (σ=0.15) with smoothing=5.0",
        curves=[("sin_very_noisy", _noisy(np.sin, 0.15), "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.8, 1.8),
        n_curves=1,
        smoothing=5.0,
        pass_threshold=0.12,
    )


def test_noisy_with_grid():
    """Noisy curves plus a grid — tests that grid suppression + smoothing co-operate."""
    return _run_test(
        "Noisy curves (σ=0.05) with grid and smoothing=2.0",
        curves=[
            ("sin_noisy", _noisy(np.sin, 0.05), "C0"),
            ("cos_noisy", _noisy(np.cos, 0.05), "C1"),
        ],
        x_range=(0, 2 * np.pi),
        y_range=(-1.5, 1.5),
        grid=True,
        n_curves=2,
        smoothing=2.0,
        pass_threshold=0.10,
    )


def test_noisy_jpeg():
    """Noisy curve saved as JPEG — double degradation (noise + compression)."""
    return _run_test(
        "Noisy sine (σ=0.05) JPEG q=85 with smoothing=3.0",
        curves=[("sin_noisy", _noisy(np.sin, 0.05), "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.5, 1.5),
        fmt="jpeg",
        quality=85,
        n_curves=1,
        smoothing=3.0,
        pass_threshold=0.10,
    )


def test_noisy_three_curves_high_dpi():
    """Three noisy curves at 300 dpi — confirms noise handling scales with resolution."""
    return _run_test(
        "Three noisy curves (σ=0.05) PNG 300dpi smoothing=2.0",
        curves=[
            ("sin_noisy",      _noisy(np.sin,                         0.05), "C0"),
            ("cos_noisy",      _noisy(np.cos,                         0.05), "C1"),
            ("half_sin2_noisy",_noisy(lambda x: 0.5 * np.sin(2 * x), 0.05), "C2"),
        ],
        x_range=(0, 2 * np.pi),
        y_range=(-1.5, 1.5),
        dpi=300,
        n_curves=3,
        smoothing=2.0,
        pass_threshold=0.08,
    )


# ---------------------------------------------------------------------------
# "Similar issues" tests — reproduce the classes of bugs that were fixed
# ---------------------------------------------------------------------------

def test_sparse_noise_alongside_curve():
    """Reproduce fig02-style spike: a dense notch curve with extra noise.

    The deep notch has density ≈ 1.0 (should trigger notch mode correctly).
    Additive Gaussian noise σ=0.04 creates occasional sparse wide-spread
    columns — these must NOT trigger notch mode (density gate blocks them).
    The density-gated _tight_cluster_median should suppress the spikes and
    recover the notch faithfully.
    """
    y_span = 2.4
    notch_depth = y_span * 0.75

    def notch_with_noise(x):
        notch = -notch_depth * np.exp(-((x - np.pi) ** 2) / 0.04)
        noise = _rng.normal(0, 0.04, x.shape)
        return np.sin(x) + notch + noise

    return _run_test(
        "Notch curve + noise (density-gate spike suppression)",
        curves=[("notch_noisy", notch_with_noise, "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-2.2, 1.2),
        n_curves=1,
        notch_factor=2.0,
        smoothing=0.5,
        pass_threshold=0.10,
    )


def test_notch_dense_vs_noise():
    """Two curves: one with a deep resonance notch, one flat + noise.

    Verifies that notch mode fires for the resonance curve but NOT for the
    noisy flat curve, i.e. the density gate discriminates correctly when both
    are present in the same image.
    """
    y_span = 2.4

    def resonance(x):
        return -0.5 * np.exp(-((x - np.pi) ** 2) / 0.05)  # single deep notch

    def flat_noisy(x):
        return 0.5 + _rng.normal(0, 0.05, x.shape)

    return _run_test(
        "Resonance notch + flat noisy curve (density-gate discrimination)",
        curves=[
            ("resonance", resonance, "C0"),
            ("flat_noisy", flat_noisy, "C1"),
        ],
        x_range=(0, 2 * np.pi),
        y_range=(-1.2, 1.2),
        n_curves=2,
        notch_factor=2.0,
        smoothing=1.0,
        pass_threshold=0.10,
    )


def test_wide_spread_sparse_noise():
    """Column with very wide y-spread but sparse pixels should NOT snap to max-y.

    Simulates the original fig02 artifact: noise pixels ~140 px below the
    real curve create a wide but sparse spread. The sliding-window should
    pick the dense upper cluster, not the outlier noise pixel.
    Uses thin linewidth so the actual curve occupies few pixels per column.
    """
    return _run_test(
        "Thin curve (lw=1) — sparse noise must not trigger notch mode",
        curves=[("sin", np.sin, "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.5, 1.5),
        linewidth=1,
        n_curves=1,
        notch_factor=3.0,
        smoothing=1.5,
        pass_threshold=0.08,
    )


def test_noisy_low_quality_jpeg():
    """Very low quality JPEG (q=50) + noise — worst-case compression artifacts."""
    return _run_test(
        "Noisy sine (σ=0.05) JPEG q=50 smoothing=4.0",
        curves=[("sin_noisy", _noisy(np.sin, 0.05), "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.5, 1.5),
        fmt="jpeg",
        quality=50,
        n_curves=1,
        smoothing=4.0,
        pass_threshold=0.12,
    )


def test_noisy_partial_curve():
    """Partial noisy curve (20 % width) — combines noise + limited span."""
    x_full = np.linspace(0, 2 * np.pi, 500)
    cutoff = 2 * np.pi * 0.20

    def partial_noisy(x):
        y = np.where(x <= cutoff, np.sin(x) + _rng.normal(0, 0.04, x.shape), np.nan)
        return y

    return _run_test(
        "Partial noisy curve (20 % width) min-col-coverage=0.15 smoothing=2",
        curves=[("partial_noisy", partial_noisy, "C0")],
        x_range=(0, 2 * np.pi),
        y_range=(-1.5, 1.5),
        n_curves=1,
        min_col_coverage=0.15,
        smoothing=2.0,
        pass_threshold=0.10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    test_single_sine,
    test_two_curves_no_grid,
    test_three_curves_with_grid,
    test_linear_curves,
    test_jpeg_quality,
    test_high_dpi,
    test_overlapping_curves,
    # Parameter-specific tests
    test_n_samples_low,
    test_pale_curve,
    test_partial_curve,
    test_thick_line,
    test_hue_bins_high,
    test_span_frac_low,
    test_notch_curve,
    # Noisy-data tests
    test_noisy_sine_moderate,
    test_noisy_two_curves,
    test_very_noisy_sine,
    test_noisy_with_grid,
    test_noisy_jpeg,
    test_noisy_three_curves_high_dpi,
    # "Similar issues" regression tests
    test_sparse_noise_alongside_curve,
    test_notch_dense_vs_noise,
    test_wide_spread_sparse_noise,
    test_noisy_low_quality_jpeg,
    test_noisy_partial_curve,
]


def main():
    print("=" * 60)
    print("PlotDigitizer — Synthetic Test Suite")
    print(f"Pass threshold: RMS ≤ {PASS_RMS_THRESHOLD*100:.0f}% of y-range")
    print("=" * 60)

    results = [t() for t in TESTS]
    n_pass = sum(results)
    n_fail = len(results) - n_pass

    print("=" * 60)
    print(f"Results: {n_pass}/{len(results)} passed, {n_fail} failed")
    print("=" * 60)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
