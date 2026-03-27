"""Command-line interface for PlotDigitizer."""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a plot image to CSV data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to plot image file")
    parser.add_argument(
        "--output", help="Output CSV path (default: same as image with .csv extension)"
    )
    parser.add_argument(
        "--method",
        choices=["naive", "cv"],
        default="naive",
        help="Curve extraction method: 'naive' uses spline interpolation, 'cv' uses OpenCV",
    )
    parser.add_argument(
        "--x-range",
        nargs=2,
        type=float,
        metavar=("X_MIN", "X_MAX"),
        help="Override x-axis range (skips OCR for x-axis)",
    )
    parser.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        metavar=("Y_MIN", "Y_MAX"),
        help="Override y-axis range: minimum (bottom) then maximum (top) data values",
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Save a plot of the digitised CSV next to the output file",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive plot window (implies --plot, requires a display)",
    )
    parser.add_argument(
        "--n-curves",
        type=int,
        default=None,
        metavar="N",
        help="Expected number of curves; keeps the top-N colour segments by pixel count",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Plot has a grid — suppress grid lines before curve extraction",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        metavar="S",
        help="Spline smoothing factor (>1 = smoother, useful for noisy measured data)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        metavar="N",
        help="Number of output data points per curve",
    )
    parser.add_argument(
        "--sat-threshold",
        type=float,
        default=0.20,
        metavar="F",
        help="HSV saturation cutoff for colour detection; lower values detect paler curves",
    )
    parser.add_argument(
        "--min-col-coverage",
        type=float,
        default=0.20,
        metavar="F",
        help="Min fraction of canvas width a colour must span to be kept as a curve",
    )
    parser.add_argument(
        "--hue-bins",
        type=int,
        default=18,
        metavar="N",
        help="Number of hue bins for colour segmentation (more bins = finer separation)",
    )
    parser.add_argument(
        "--span-frac",
        type=float,
        default=0.55,
        metavar="F",
        help="Min fraction of canvas a row/col must span to be classified as a grid line",
    )
    parser.add_argument(
        "--max-thickness",
        type=int,
        default=None,
        metavar="N",
        help="Max pixel cluster height for per-column extraction (default: auto = max(8, h//25))",
    )
    parser.add_argument(
        "--notch-factor",
        type=float,
        default=3.0,
        metavar="F",
        help="Spread > max_thickness * notch_factor triggers deep-notch mode (use max y)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else image_path.with_suffix(".csv")

    x_range = tuple(args.x_range) if args.x_range else None
    y_range = tuple(args.y_range) if args.y_range else None

    from .digitizer import digitize_plot

    digitize_plot(
        image_path=image_path,
        output_path=output_path,
        method=args.method,
        x_range=x_range,
        y_range=y_range,
        smoothing=args.smoothing,
        n_curves=args.n_curves,
        has_grid=args.grid,
        n_samples=args.n_samples,
        sat_threshold=args.sat_threshold,
        min_col_coverage=args.min_col_coverage,
        hue_bins=args.hue_bins,
        span_frac=args.span_frac,
        max_thickness=args.max_thickness,
        notch_factor=args.notch_factor,
    )
    logger.info(f"Output written to: {output_path}")

    if args.plot or args.show:
        from .plotter import plot_csv
        plot_path = output_path.with_name(output_path.stem + "_digitized.png")
        plot_csv(
            csv_path=output_path,
            output_path=plot_path,
            show=args.show,
            original_image=image_path,
        )


if __name__ == "__main__":
    main()
