"""Plain-text report formatter for bulk directory comparisons."""

from __future__ import annotations

import statistics

from validation.bulk_compare import BulkComparisonReport, FilePairResult


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
    lines: list[str] = []

    # Header
    lines.append("=" * 72)
    lines.append("Bulk Simple-Grid Comparison Report")
    lines.append("=" * 72)
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

    # Per-file table
    if report.matched_pairs:
        fname_width = max(len(r.filename) for r in report.matched_pairs)
        fname_width = max(fname_width, len("File"))

        header = (
            f"{'File':<{fname_width}}  "
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
                f"{r.filename:<{fname_width}}  "
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

        lines.append(_stat_line("SSHA agreement %", [r.ssha_pct_within_threshold for r in valid]))
        lines.append(_stat_line("SSHA max abs diff", [r.ssha_max_abs_diff for r in valid]))
        lines.append(_stat_line("SSHA MAE", [r.ssha_mean_abs_diff for r in valid]))
        lines.append(_stat_line("SSHA RMSD", [r.ssha_rmsd for r in valid]))
        lines.append(_stat_line("Counts MAE", [r.counts_mean_abs_diff for r in valid]))
        lines.append(_stat_line("Counts max abs diff", [r.counts_max_abs_diff for r in valid]))
    else:
        lines.append("  No valid results.")

    lines.append("")

    # Summary line
    n_total = len(report.matched_pairs)
    n_pass = sum(1 for r in report.matched_pairs if _status(r, pass_threshold) == "PASS")
    pct_pass = (n_pass / n_total * 100) if n_total > 0 else 0.0
    lines.append(
        f"Files passing (SSHA >= {pass_threshold}%): {n_pass}/{n_total} ({pct_pass:.1f}%)"
    )

    return "\n".join(lines)
