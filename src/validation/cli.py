"""CLI entry point for the validation toolkit."""

import argparse
import sys

from validation.comparators.along_track import AlongTrackComparator
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


if __name__ == "__main__":
    sys.exit(main())
