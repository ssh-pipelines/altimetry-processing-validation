"""CLI entry point for the validation toolkit."""

import argparse
import sys

from validation.bulk_compare import run_bulk_comparison
from validation.bulk_report import format_bulk_report
from validation.comparators.along_track import AlongTrackComparator
from validation.comparators.indicators import IndicatorsComparator
from validation.comparators.simple_grid import SimpleGridComparator
from validation.report import format_report

COMPARATORS = {
    "along_track": AlongTrackComparator,
    "simple_grid": SimpleGridComparator,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate-altimetry",
        description="Compare two altimetry NetCDF product files.",
    )
    parser.add_argument("file_a", help="Path to first NetCDF file")
    parser.add_argument("file_b", help="Path to second NetCDF file")
    parser.add_argument(
        "-t",
        "--product-type",
        required=True,
        choices=list(COMPARATORS.keys()),
        help="Product type to compare",
    )
    parser.add_argument(
        "--ignore-attrs",
        nargs="*",
        default=None,
        help="Attribute names to ignore (e.g. date_created history)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        metavar="METERS",
        help="Absolute difference threshold in metres for the pct_within_threshold metric (default: 0.05)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    comparator_cls = COMPARATORS[args.product_type]
    comparator = comparator_cls(args.file_a, args.file_b, threshold=args.threshold)

    report = comparator.run(ignore_attrs=args.ignore_attrs)
    print(format_report(report))

    return 1 if report.has_differences else 0


def build_bulk_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bulk-validate-altimetry",
        description="Compare two directories of simple-grid NetCDF files.",
    )
    parser.add_argument("dir_a", help="Path to first directory")
    parser.add_argument("dir_b", help="Path to second directory")
    parser.add_argument(
        "--ignore-attrs",
        nargs="*",
        default=None,
        help="Attribute names to ignore (e.g. date_created history)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        metavar="METERS",
        help="Absolute difference threshold in metres for pct_within_threshold (default: 0.05)",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=95.0,
        metavar="PCT",
        help="Min SSHA agreement %% for PASS status (default: 95.0)",
    )
    parser.add_argument(
        "--plot",
        metavar="OUTPUT_FILE",
        default=None,
        help="Save a timeseries plot of difference metrics to this path (e.g. report.png)",
    )
    return parser


def bulk_main(argv: list[str] | None = None) -> int:
    args = build_bulk_parser().parse_args(argv)
    report = run_bulk_comparison(args.dir_a, args.dir_b, args.threshold, args.ignore_attrs)
    print(format_bulk_report(report, pass_threshold=args.pass_threshold))

    if args.plot is not None:
        from validation.bulk_plot import plot_bulk_timeseries
        plot_bulk_timeseries(report, output_path=args.plot, pass_threshold=args.pass_threshold)

    any_fail = any(
        r.error
        or (
            r.ssha_pct_within_threshold is not None
            and r.ssha_pct_within_threshold < args.pass_threshold
        )
        for r in report.matched_pairs
    )
    return 1 if any_fail else 0


def build_indicators_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare-indicators",
        description="Compare two indicators NetCDF files and optionally plot timeseries.",
    )
    parser.add_argument("file_a", help="Path to first indicators NetCDF file")
    parser.add_argument("file_b", help="Path to second indicators NetCDF file")
    parser.add_argument(
        "--ignore-attrs",
        nargs="*",
        default=None,
        help="Attribute names to ignore (e.g. date_created history)",
    )
    parser.add_argument(
        "--label-a",
        default=None,
        metavar="LABEL",
        help="Legend label for file A (default: filename stem)",
    )
    parser.add_argument(
        "--label-b",
        default=None,
        metavar="LABEL",
        help="Legend label for file B (default: filename stem)",
    )
    parser.add_argument(
        "--plot",
        metavar="OUTPUT_FILE",
        default=None,
        help="Save a timeseries comparison plot to this path (e.g. indicators.png)",
    )
    return parser


def indicators_main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    args = build_indicators_parser().parse_args(argv)

    label_a = args.label_a or Path(args.file_a).stem
    label_b = args.label_b or Path(args.file_b).stem

    comparator = IndicatorsComparator(args.file_a, args.file_b)
    report = comparator.run(ignore_attrs=args.ignore_attrs)
    print(format_report(report))

    if args.plot is not None:
        from validation.indicators_plot import plot_indicators_comparison
        plot_indicators_comparison(
            args.file_a,
            args.file_b,
            output_path=args.plot,
            label_a=label_a,
            label_b=label_b,
        )

    return 1 if report.has_differences else 0


if __name__ == "__main__":
    sys.exit(main())
