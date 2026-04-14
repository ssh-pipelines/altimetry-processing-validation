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


def build_xover_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare-xovers",
        description=(
            "Compare daily SSH crossover RMS across pre-OER and post-OER periods, "
            "each with a set1 and set2 directory of xovers_S6-*.nc files."
        ),
    )
    parser.add_argument("pre_dir1",  help="pre-OER  set1 directory")
    parser.add_argument("pre_dir2",  help="pre-OER  set2 directory")
    parser.add_argument("post_dir1", help="post-OER set1 directory")
    parser.add_argument("post_dir2", help="post-OER set2 directory")
    parser.add_argument("--label-pre1",  default=None, metavar="LABEL")
    parser.add_argument("--label-pre2",  default=None, metavar="LABEL")
    parser.add_argument("--label-post1", default=None, metavar="LABEL")
    parser.add_argument("--label-post2", default=None, metavar="LABEL")
    parser.add_argument(
        "--plot",
        metavar="OUTPUT_FILE",
        default=None,
        help="Save the four-way RMS comparison plot to this path (e.g. xover_rms.png)",
    )
    parser.add_argument(
        "--csv-dir",
        metavar="DIR",
        default=None,
        help="Write per-series CSV files to this directory",
    )
    return parser


def xover_main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    from validation.crossover import (
        compute_daily_rms,
        plot_crossover_four_way,
        print_summary,
        write_csv,
    )

    args = build_xover_parser().parse_args(argv)

    label_pre1  = args.label_pre1  or f"pre_oer / {Path(args.pre_dir1).name}"
    label_pre2  = args.label_pre2  or f"pre_oer / {Path(args.pre_dir2).name}"
    label_post1 = args.label_post1 or f"post_oer / {Path(args.post_dir1).name}"
    label_post2 = args.label_post2 or f"post_oer / {Path(args.post_dir2).name}"

    pre_r1  = compute_daily_rms(args.pre_dir1,  label=label_pre1)
    pre_r2  = compute_daily_rms(args.pre_dir2,  label=label_pre2)
    post_r1 = compute_daily_rms(args.post_dir1, label=label_post1)
    post_r2 = compute_daily_rms(args.post_dir2, label=label_post2)

    for results, lbl in [
        (pre_r1,  label_pre1),
        (pre_r2,  label_pre2),
        (post_r1, label_post1),
        (post_r2, label_post2),
    ]:
        print_summary(results, lbl)

    if args.csv_dir is not None:
        csv_dir = Path(args.csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        for results, lbl in [
            (pre_r1,  label_pre1),
            (pre_r2,  label_pre2),
            (post_r1, label_post1),
            (post_r2, label_post2),
        ]:
            write_csv(results, lbl, csv_dir)

    if args.plot is not None:
        plot_crossover_four_way(
            pre_r1, pre_r2, post_r1, post_r2,
            label_pre1, label_pre2, label_post1, label_post2,
            output_path=args.plot,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
