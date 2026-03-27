"""
Generate test plots with multiple overlapping curves at various image qualities.
Run with:  uv run python tests/generate_test_plots.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).parent.parent / "test_images"


def _make_axes(n_curves: int = 3) -> tuple:
    x = np.linspace(0, 10, 400)
    curves = [
        (x, np.sin(x),                   "sin(x)",        "C0"),
        (x, np.cos(x),                   "cos(x)",        "C1"),
        (x, 0.5 * np.sin(2 * x),         "0.5·sin(2x)",   "C2"),
        (x, np.sin(x) + 0.3 * np.cos(3*x), "sin+cos",    "C3"),
        (x, np.exp(-0.3 * x) * np.sin(x), "decay·sin",   "C4"),
    ]
    return x, curves[:n_curves]


def generate_single_figure(title: str, n_curves: int = 3) -> plt.Figure:
    x, curves = _make_axes(n_curves)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _x, y, label, color in curves:
        ax.plot(_x, y, label=label, color=color, linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # PNG at different DPI
    for dpi in (72, 150, 300):
        fig = generate_single_figure(f"Test plot (PNG {dpi} dpi)", n_curves=3)
        path = OUTPUT_DIR / f"test_plot_{dpi}dpi.png"
        fig.savefig(path, dpi=dpi, format="png")
        print(f"Saved: {path}")
        plt.close(fig)

    # JPEG at different qualities
    for quality in (95, 50, 15):
        fig = generate_single_figure(f"Test plot (JPEG q={quality})", n_curves=3)
        path = OUTPUT_DIR / f"test_plot_q{quality}.jpg"
        fig.savefig(path, pil_kwargs={"quality": quality}, format="jpeg", dpi=150)
        print(f"Saved: {path}")
        plt.close(fig)

    # Plot with 5 overlapping curves
    fig = generate_single_figure("Test plot (5 overlapping curves)", n_curves=5)
    path = OUTPUT_DIR / "test_plot_5curves.png"
    fig.savefig(path, dpi=150, format="png")
    print(f"Saved: {path}")
    plt.close(fig)

    print(f"\nAll test images written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
