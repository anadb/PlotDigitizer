# PlotDigitizer

Convert a 2-D plot image into a CSV file of numeric (x, y) data points.
Works on PNG, JPEG, and most common raster formats.
Handles multiple coloured curves, overlapping curves, grid lines, and noisy or low-quality scans.

---

## Quick start

```bash
# Install with uv (recommended)
uv pip install -e .

# Basic usage — OCR reads the axis labels automatically
uv run plot_digitizer --image path/to/plot.png

# Provide axis ranges manually (bypasses OCR, more reliable)
uv run plot_digitizer --image plot.png --x-range 0 10 --y-range -1.5 1.5

# Save a comparison figure alongside the CSV
uv run plot_digitizer --image plot.png --x-range 0 10 --y-range -1.5 1.5 --plot
```

Output is written next to the image as `plot.csv` (columns: `curve`, `x`, `y`).

---

## Pipeline overview

```
Image file
    │
    ▼
1. detect_plot_area()   — locate the axis-bounded canvas
    │
    ▼
2. read_axes()          — calibrate pixel ↔ data-value mapping
    │
    ▼
3. extract_curves()     — segment by colour, extract (x, y) pixel paths
    │
    ▼
4. pixel_to_data()      — convert pixel coords to data coords
    │
    ▼
CSV  +  optional plot
```

### Stage 1 — Canvas detection (`plot_detector.py`)

The detector looks for long continuous dark runs of pixels (axis spines) that span
at least 30 % of the image width or height.  It tries progressively more permissive
darkness thresholds (80 → 120 → 160 → 200 → 230) so that both dark and light-grey
axes are found.  If no spines are found, it falls back to a 12 %/88 % margin crop.

### Stage 2 — Axis calibration (`axis_reader.py`)

**OCR mode (default):** crops the strip below (x-axis) and to the left (y-axis) of
the canvas and runs Tesseract OCR (3× upscaled, numeric characters only) to find
tick labels.  Two anchor points define a linear calibration.

**Manual mode (`--x-range` / `--y-range`):** skips OCR entirely and uses the
provided numbers directly.  Always more reliable than OCR; use it whenever the axis
labels are non-standard, too small, or partly occluded.

**Y-axis convention:** `--y-range Y_MIN Y_MAX` where Y_MIN is the *bottom* value
and Y_MAX is the *top* value, matching the natural reading of a plot.
Internally pixel y increases downward while data y increases upward, so the
calibration automatically inverts the mapping.

### Stage 3 — Curve extraction (`curve_extractor.py`)

#### Colour segmentation

1. Convert the canvas to HSV.
2. Optionally suppress grid lines: achromatic rows/columns spanning ≥ `--span-frac`
   of the canvas are masked out, plus a fixed border margin covering tick marks.
3. Saturated foreground pixels (`saturation > --sat-threshold`, `value > 0.15`) are
   grouped into hue bins of width `360 / --hue-bins` degrees.
4. Each bin that spans at least `--min-col-coverage` fraction of canvas columns is
   kept as a candidate curve.
5. A separate pass finds achromatic (grey) curves in 8 value bins between 0.25 and
   0.90, merging bins with >60 % column overlap.
6. If `--n-curves N` is given, only the top-N segments by pixel count are kept.

#### Naive method (default, `--method naive`)

For each x-column of the colour mask:

1. Collect all active (foreground) pixel rows.
2. **Tight-cluster median** — find the window of height ≤ `--max-thickness` px
   containing the most pixels via a sliding window, then return its median y.
3. **Deep-notch mode** — if the column spread exceeds `max-thickness × notch-factor`
   *and* the pixel density in that spread is > 0.4 (i.e. a dense near-vertical
   segment, not sparse noise), snap to the bottom-most pixel to capture resonance
   minima faithfully.
4. Outlier rejection: replace column medians deviating > 3 MAD from their
   21-column neighbourhood with the local median.
5. Fit a smoothing spline (scipy `UnivariateSpline`, degree 3) with smoothing
   parameter `s = n_points × 0.5 × --smoothing`, then sample `--n-samples` evenly
   spaced output points.

#### CV method (`--method cv`)

Requires `opencv-python`.  Computes a morphological skeleton of each colour mask,
then takes the per-column median of skeleton pixels.  Falls back to the naive method
if OpenCV is not available.

---

## CLI reference

```
uv run plot_digitizer --image IMAGE [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--image PATH` | *(required)* | Input plot image |
| `--output PATH` | `<image>.csv` | Output CSV path |
| `--method naive\|cv` | `naive` | Extraction method |
| `--x-range X_MIN X_MAX` | OCR | Override x-axis range |
| `--y-range Y_MIN Y_MAX` | OCR | Override y-axis range (bottom then top) |
| `--n-curves N` | all | Keep only the top-N colour segments |
| `--grid` | off | Suppress grid lines before extraction |
| `--smoothing S` | `1.0` | Spline smoothing factor (higher = smoother) |
| `--n-samples N` | `500` | Output points per curve |
| `--sat-threshold F` | `0.20` | HSV saturation cutoff for colour detection |
| `--min-col-coverage F` | `0.20` | Min fraction of canvas width a curve must span |
| `--hue-bins N` | `18` | Hue bins for colour segmentation |
| `--span-frac F` | `0.55` | Min row/column span to classify as a grid line |
| `--max-thickness N` | auto | Max pixel cluster height per column |
| `--notch-factor F` | `3.0` | Spread/thickness ratio that triggers notch mode |
| `--plot` / `-p` | off | Save a comparison figure |
| `--show` | off | Open interactive plot window |
| `--verbose` / `-v` | off | Debug logging |

---

## Known issues and how to fix them

### Wrong axis range (most common issue)

**Symptom:** all curve values are shifted or scaled incorrectly; the CSV looks like
it spans `[0, 1]` instead of your expected range.

**Cause:** OCR failed to read the tick labels (too small, font not recognised, or
non-standard notation).

**Fix:** always supply ranges manually:
```bash
plot_digitizer --image plot.png --x-range 1500 1600 --y-range -30 0
```

---

### Y-axis calibration inverted

**Symptom:** curves that should be near 0 dB appear near −14 dB and vice versa.

**Cause:** `--y-range` arguments in wrong order.  The convention is
`--y-range Y_MIN Y_MAX` where Y_MIN is the *bottom* of the plot and Y_MAX is the
*top*.  Passing them reversed (e.g. `--y-range 0 -14`) inverts the mapping.

**Fix:**
```bash
# Wrong:  --y-range 0 -14
# Correct:
plot_digitizer --image plot.png --y-range -14 0
```

---

### Canvas boundary detected incorrectly

**Symptom:** canvas is too small (cuts off part of the plot) or uses the 12 %/88 %
fallback, causing axis calibration errors of several percent.

**Cause:** light-grey axis spines (common in scientific figures) are not found at
low darkness thresholds.  The detector now tries up to threshold=230 automatically.
The fallback also fires when the image has no clear spine lines (e.g. borderless plots).

**Fix:**
- Supply `--x-range` and `--y-range` manually so that an imprecise canvas boundary
  does not affect calibration.
- Run with `--verbose` to see which threshold was used and what canvas was detected.

---

### No curves detected

**Symptom:** CSV is empty; log says "No curves detected".

**Cause A — low saturation curves:** pastel or near-grey colours fall below the
default `sat-threshold=0.20`.

**Fix:**
```bash
plot_digitizer --image plot.png --sat-threshold 0.08
```

**Cause B — curve spans a small fraction of the x-axis:** the default 20 % column
coverage requirement drops partial curves.

**Fix:**
```bash
plot_digitizer --image plot.png --min-col-coverage 0.05
```

**Cause C — curves merged into one hue bin:** two similarly-coloured curves fall in
the same hue bucket with default `hue-bins=18` (20° per bin).

**Fix:**
```bash
plot_digitizer --image plot.png --hue-bins 36   # 10° per bin
```

---

### Extra spurious curves detected

**Symptom:** more curves in the CSV than in the original plot; extras often
correspond to axis tick marks, labels, or legends.

**Fix:** cap the output with `--n-curves`:
```bash
plot_digitizer --image plot.png --n-curves 2
```

---

### Grid lines detected as curves

**Symptom:** horizontal bands or near-flat artefacts appear in the CSV.

**Fix:** enable grid suppression:
```bash
plot_digitizer --image plot.png --grid
```

Tune aggressiveness with `--span-frac` (default 0.55 — a row or column must span
55 % of the canvas to be classified as a grid line):
```bash
# Less aggressive — keep more of the plot, risk leaving some grid remnants
plot_digitizer --image plot.png --grid --span-frac 0.40

# More aggressive — removes more rows/cols, risk clipping curve endpoints
plot_digitizer --image plot.png --grid --span-frac 0.70
```

---

### Resonance notch / transmission minimum not captured (clipped too high)

**Symptom:** a sharp deep dip in the curve is flattened or its minimum is much
shallower than in the original plot.

**Cause:** the per-column tight-cluster window (`--max-thickness`) is smaller than
the notch width in pixels, so the sliding window finds a cluster above the true
minimum instead of following the near-vertical descent.

**Fix:** lower `--notch-factor` so that deep-notch mode fires sooner:
```bash
plot_digitizer --image plot.png --notch-factor 2.0
```

How it works: when the pixel spread in a column exceeds `max_thickness × notch-factor`
**and** pixel density in that spread is > 40 % (confirming a dense near-vertical
segment rather than scattered noise), the extractor snaps to the bottom-most pixel
to capture the minimum.

---

### Spikes or jumps in extracted curve (false notch triggering)

**Symptom:** a smooth curve has sudden large spikes at certain x positions,
especially near the edges of the plot or where another curve overlaps.

**Cause A — sparse noise pixels far from the curve:** anti-aliasing, JPEG
compression, or a nearby axis line leaves a few isolated pixels far below the main
curve.  Their wide y-spread used to trigger notch mode incorrectly.

**Status:** fixed automatically.  The density gate (`density > 0.4` required before
notch mode fires) ensures sparse-noise columns fall through to the sliding-window
path, which correctly finds the dense upper cluster.

**Cause B — residual smoothing artefacts:** the spline overshoots at curve endpoints
or in low-data regions.

**Fix:** increase smoothing:
```bash
plot_digitizer --image plot.png --smoothing 3.0
```

---

### Thick-stroke curves extracted with wrong centre

**Symptom:** a thick (e.g. 8 pt) curve is offset from its true centre; the error
grows with line width.

**Cause:** the auto thickness (`max(8, canvas_height / 25)`) is smaller than the
rendered stroke, so the sliding window finds the top half of the stroke rather than
its centre.

**Fix:** set `--max-thickness` explicitly to at least the stroke width in pixels
(≈ `linewidth_pt × dpi / 72`):
```bash
plot_digitizer --image plot.png --max-thickness 40
```

---

### Noisy or measured data (jagged curve)

**Symptom:** extracted curve follows every noise wiggle; CSV is noisier than desired.

**Fix:** increase `--smoothing` (default 1.0):
```bash
# Moderate noise
plot_digitizer --image plot.png --smoothing 2.0

# Heavy noise or low-quality scan
plot_digitizer --image plot.png --smoothing 5.0
```

The spline smoothing parameter scales as `s = n_points × 0.5 × smoothing`, so
larger values produce a smoother fit at the cost of following rapid transitions less
closely.

---

## Running tests

```bash
uv run python tests/run_tests.py
```

The test suite generates synthetic plots with known ground-truth curves, runs the
full digitization pipeline, and reports RMS error normalised by the y-range.
The default pass threshold is 5 % of y-range; noisy tests use relaxed thresholds
(8–12 %) set per-test.

Test categories:

- **Core accuracy** — single/multi-curve, PNG/JPEG, 150/300 dpi, overlapping curves
- **Parameter-specific** — each CLI parameter exercised in isolation
- **Noisy data** — Gaussian noise at σ=0.05 and σ=0.15, with and without grid/JPEG
- **Regression** — reproduces the specific failure modes fixed during development:
  - Density-gated notch mode (fig02-style spike suppression)
  - Notch mode discrimination between a dense resonance dip and a flat noisy curve
  - Thin curves with sparse column pixels not snapping to wrong extremum

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `Pillow` | Image loading |
| `numpy` | Array operations |
| `scipy` | Smoothing spline (`UnivariateSpline`) |
| `matplotlib` | Optional result plot |
| `pytesseract` | OCR for axis labels (optional — manual ranges bypass it) |
| `opencv-python-headless` | CV extraction method (optional) |

Tesseract OCR engine must be installed separately:
```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```
