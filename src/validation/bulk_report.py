"""Plain-text report formatter for bulk directory comparisons."""

from __future__ import annotations

import statistics

from validation.bulk_compare import BulkComparisonReport, FilePairResult
from validation.comparators.along_track_hilat import summarise_lat_band
from validation.report import format_lat_bands


def _status(result: FilePairResult, pass_threshold: float) -> str:
    if result.error:
        return "ERROR"
    if result.ssha_pct_within_threshold is None:
        return "N/A"
    return "PASS" if result.ssha_pct_within_threshold >= pass_threshold else "FAIL"


def format_bulk_report(
    report: BulkComparisonReport, pass_threshold: float = 95.0
) -> str:
    """Format a BulkComparisonReport as a human-readable plain-text string."""
    if report.product_type == "along_track_hilat":
        return format_bulk_hilat_report(report, pass_threshold)

    lines = _header(report, "Bulk Simple-Grid Comparison Report")

    # Per-file table
    if report.matched_pairs:
        date_width = max(len(r.date_key) for r in report.matched_pairs)
        date_width = max(date_width, len("Date"))

        header = (
            f"{'Date':<{date_width}}  "
            f"{'SSHA Agr%':>9}  "
            f"{'MaxAbs':>7}  "
            f"{'MAE':>7}  "
            f"{'RMSD':>7}  "
            f"{'Cts MAE':>7}  "
            f"{'Status':>6}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for r in report.matched_pairs:
            status = _status(r, pass_threshold)
            agr = (
                f"{r.ssha_pct_within_threshold:.1f}%"
                if r.ssha_pct_within_threshold is not None
                else "N/A"
            )
            max_abs = f"{r.ssha_max_abs_diff:.4f}" if r.ssha_max_abs_diff is not None else "N/A"
            mae = f"{r.ssha_mean_abs_diff:.4f}" if r.ssha_mean_abs_diff is not None else "N/A"
            rmsd = f"{r.ssha_rmsd:.4f}" if r.ssha_rmsd is not None else "N/A"
            cts_mae = (
                f"{r.counts_mean_abs_diff:.2f}" if r.counts_mean_abs_diff is not None else "N/A"
            )
            row = (
                f"{r.date_key:<{date_width}}  "
                f"{agr:>9}  "
                f"{max_abs:>7}  "
                f"{mae:>7}  "
                f"{rmsd:>7}  "
                f"{cts_mae:>7}  "
                f"{status:>6}"
            )
            lines.append(row)

        lines.append("")

    # Aggregate statistics
    valid = [r for r in report.matched_pairs if r.error is None]
    lines.append("--- Aggregate Statistics ---")
    if valid:
        lines.append(_stat_line("SSHA agreement %", [r.ssha_pct_within_threshold for r in valid]))
        lines.append(_stat_line("SSHA max abs diff", [r.ssha_max_abs_diff for r in valid]))
        lines.append(_stat_line("SSHA MAE", [r.ssha_mean_abs_diff for r in valid]))
        lines.append(_stat_line("SSHA RMSD", [r.ssha_rmsd for r in valid]))
        lines.append(_stat_line("Counts MAE", [r.counts_mean_abs_diff for r in valid]))
        lines.append(_stat_line("Counts max abs diff", [r.counts_max_abs_diff for r in valid]))
    else:
        lines.append("  No valid results.")

    lines.append("")
    lines.append(_pass_summary(report, pass_threshold))

    return "\n".join(lines)


def _pass_summary(report: BulkComparisonReport, pass_threshold: float) -> str:
    n_total = len(report.matched_pairs)
    n_pass = sum(1 for r in report.matched_pairs if _status(r, pass_threshold) == "PASS")
    pct_pass = (n_pass / n_total * 100) if n_total > 0 else 0.0
    return f"Files passing (SSHA >= {pass_threshold}%): {n_pass}/{n_total} ({pct_pass:.1f}%)"


def _header(report: BulkComparisonReport, title: str) -> list[str]:
    lines = ["=" * 72, title, "=" * 72]
    lines.append(f"  Dir A:     {report.dir_a}")
    lines.append(f"  Dir B:     {report.dir_b}")
    lines.append(f"  Threshold: {report.threshold} m")
    lines.append(
        f"  Matched: {len(report.matched_pairs)}  "
        f"Only in A: {len(report.only_in_a)}  "
        f"Only in B: {len(report.only_in_b)}"
    )
    lines.append("")

    # Unmatched files
    if report.only_in_a:
        lines.append("Files only in A:")
        for f in report.only_in_a:
            lines.append(f"  {f}")
        lines.append("")

    if report.only_in_b:
        lines.append("Files only in B:")
        for f in report.only_in_b:
            lines.append(f"  {f}")
        lines.append("")
    return lines


def _stat_line(label: str, values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return f"  {label}: N/A"
    return (
        f"  {label}: "
        f"mean={statistics.mean(vals):.4f}  "
        f"median={statistics.median(vals):.4f}  "
        f"min={min(vals):.4f}  "
        f"max={max(vals):.4f}"
    )


def _fmt(v, fmt: str = ".4f") -> str:
    return format(v, fmt) if v is not None else "N/A"


def format_bulk_hilat_report(
    report: BulkComparisonReport, pass_threshold: float = 95.0
) -> str:
    """Format an along_track_hilat bulk report (ref A vs high-lat B)."""
    lines = _header(report, "Bulk Reference vs High-Latitude Along-Track Comparison Report")

    if report.matched_pairs:
        header = (
            f"{'Date':<8}  {'Bias':>8}  {'RMSD':>7}  {'r':>6}  {'B/A std':>7}  "
            f"{'Self A':>7}  {'Self B':>7}  {'Noise A':>7}  {'Noise B':>7}  "
            f"{'Cells A':>7}  {'Cells B':>7}  {'Common':>6}  {'MaxAbs':>7}  "
            f"{'@lat,lon':>13}  {'Status':>6}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for r in report.matched_pairs:
            loc = (
                f"{r.max_abs_diff_lat:.1f},{r.max_abs_diff_lon:.1f}"
                if r.max_abs_diff_lat is not None
                else "N/A"
            )
            lines.append(
                f"{r.date_key:<8}  {_fmt(r.ssha_bias):>8}  {_fmt(r.ssha_rmsd):>7}  "
                f"{_fmt(r.ssha_pearson_r, '.3f'):>6}  {_fmt(r.ssha_std_ratio, '.3f'):>7}  "
                f"{_fmt(r.self_rmsd_a):>7}  {_fmt(r.self_rmsd_b):>7}  "
                f"{_fmt(r.noise_a):>7}  {_fmt(r.noise_b):>7}  "
                f"{_fmt(r.cells_a, 'd'):>7}  {_fmt(r.cells_b, 'd'):>7}  "
                f"{_fmt(r.n_common_cells, 'd'):>6}  {_fmt(r.ssha_max_abs_diff):>7}  "
                f"{loc:>13}  {_status(r, pass_threshold):>6}"
            )
        lines.append("")

    valid = [r for r in report.matched_pairs if r.error is None]
    lines.append("--- Aggregate Statistics (per-day values) ---")
    if valid:
        for label, attr in [
            ("SSHA bias (B - A)", "ssha_bias"),
            ("SSHA RMSD", "ssha_rmsd"),
            ("Self-baseline RMSD A (asc vs desc)", "self_rmsd_a"),
            ("Self-baseline RMSD B (asc vs desc)", "self_rmsd_b"),
            ("SSHA r", "ssha_pearson_r"),
            ("SSHA std ratio B/A", "ssha_std_ratio"),
            ("Noise RMS A (ssha - ssha_smoothed)", "noise_a"),
            ("Noise RMS B (ssha - ssha_smoothed)", "noise_b"),
            ("SSHA agreement %", "ssha_pct_within_threshold"),
            ("SSHA max abs diff", "ssha_max_abs_diff"),
            ("Common cells", "n_common_cells"),
        ]:
            lines.append(_stat_line(label, [getattr(r, attr) for r in valid]))

        no_overlap = [r for r in valid if not r.n_common_cells]
        if no_overlap:
            lines.append("")
            lines.append(f"Days with no common cells ({len(no_overlap)}):")
            for r in no_overlap:
                lines.append(f"  {r.date_key}  cells A={r.cells_a}  cells B={r.cells_b}")
    else:
        lines.append("  No valid results.")
    lines.append("")

    if report.lat_bands:
        lines.append("--- Latitude Bands (pooled over all days) ---")
        lines.append(
            "  Bias/RMSD/Std over common grid cells; Noise = RMS(ssha - ssha_smoothed) "
            "over all flag-good points"
        )
        lines.extend(
            line[2:] for line in format_lat_bands(
                [summarise_lat_band(b) for b in report.lat_bands]
            )
        )
        lines.append("")

    lines.append(_pass_summary(report, pass_threshold))
    return "\n".join(lines)
