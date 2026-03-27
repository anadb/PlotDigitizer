"""Plot digitised CSV data and optionally save / display it."""

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _color_from_label(name: str):
    """
    Extract a matplotlib colour from the curve label produced by _segment_by_color.

    Handles two label formats:
      curve_h<bin>_rgb<RRR><GGG><BBB>  →  (R/255, G/255, B/255)
      curve_gray_v<VVV>                →  grey at value V/255
    Returns None if the label doesn't match either pattern.
    """
    import re
    m = re.search(r"_rgb(\d{3})(\d{3})(\d{3})$", name)
    if m:
        return (int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255)
    m = re.search(r"_gray_v(\d{3})$", name)
    if m:
        v = int(m.group(1)) / 255
        return (v, v, v)
    return None


def plot_csv(
    csv_path: Path,
    output_path: Path,
    show: bool = False,
    original_image: Path | None = None,
) -> None:
    """
    Read the digitised CSV produced by digitize_plot and save a comparison figure.

    Args:
        csv_path:       Path to the CSV (columns: curve, x, y).
        output_path:    Where to save the plot image.
        show:           If True, open an interactive window (requires a display).
        original_image: If given, shown as a subplot alongside the digitised data.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Read CSV
    curves: dict[str, tuple[list[float], list[float]]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["curve"]
            curves.setdefault(name, ([], []))
            curves[name][0].append(float(row["x"]))
            curves[name][1].append(float(row["y"]))

    if not curves:
        logger.warning("CSV is empty — nothing to plot")
        return

    n_panels = 2 if original_image is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Panel 1 (optional): original image
    if original_image is not None:
        from PIL import Image
        img = Image.open(original_image)
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis("off")

    # Panel: digitised curves
    ax = axes[-1]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (name, (xs, ys)) in enumerate(sorted(curves.items())):
        color = _color_from_label(name) or color_cycle[idx % len(color_cycle)]
        # Sort by x for clean line rendering
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs_s = [xs[i] for i in order]
        ys_s = [ys[i] for i in order]
        short = name.split("_rgb")[0].split("_gray")[0]
        ax.plot(xs_s, ys_s, color=color, linewidth=1.5, label=short)

    ax.set_title("Digitised curves")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info(f"Plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close(fig)
