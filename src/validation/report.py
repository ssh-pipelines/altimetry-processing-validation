"""Plain-text report formatting for comparison results."""

from validation.comparators.base import ComparisonReport, VariableComparison


def format_report(report: ComparisonReport) -> str:
    """Format a ComparisonReport as a human-readable plain-text string."""
    lines: list[str] = []

    # Header
    lines.append("=" * 72)
    lines.append(f"Altimetry Product Comparison Report  [{report.product_type}]")
    lines.append("=" * 72)
    lines.append(f"  File A: {report.file_a}")
    lines.append(f"  File B: {report.file_b}")
    lines.append("")

    # Dimensions
    lines.append("--- Dimensions ---")
    if report.dimension_diffs:
        for dim, size_a, size_b in report.dimension_diffs:
            lines.append(f"  {dim}: A={size_a}  B={size_b}")
    else:
        lines.append("  All dimensions match.")
    lines.append("")

    # Global attributes
    lines.append("--- Global Attributes ---")
    if report.global_attr_diffs:
        for attr, val_a, val_b in report.global_attr_diffs:
            lines.append(f"  {attr}:")
            lines.append(f"    A: {_truncate(val_a)}")
            lines.append(f"    B: {_truncate(val_b)}")
    else:
        lines.append("  All global attributes match.")
    lines.append("")

    # Variable inventory
    lines.append("--- Variable Inventory ---")
    only_a = [vc.name for vc in report.variable_comparisons if vc.present_a and not vc.present_b]
    only_b = [vc.name for vc in report.variable_comparisons if vc.present_b and not vc.present_a]
    both = [vc.name for vc in report.variable_comparisons if vc.present_a and vc.present_b]
    lines.append(f"  Common: {len(both)}  |  Only in A: {len(only_a)}  |  Only in B: {len(only_b)}")
    if only_a:
        lines.append(f"  Only in A: {', '.join(only_a)}")
    if only_b:
        lines.append(f"  Only in B: {', '.join(only_b)}")
    lines.append("")

    # Per-variable statistics
    lines.append("--- Per-Variable Statistics ---")
    for vc in report.variable_comparisons:
        lines.append(_format_variable(vc))
    lines.append("")

    # Quality summary
    lines.append("--- Quality Summary ---")
    if report.quality_summary:
        for key, value in report.quality_summary.items():
            lines.append(f"  {key}:")
            if key == "ssha_agreement" and isinstance(value, dict):
                t = value["threshold_m"]
                pct = value["pct_within_threshold"]
                pct_str = f"{pct}%" if pct is not None else "N/A"
                lines.append(f"    threshold: {t} m  |  pct_within: {pct_str}")
            elif isinstance(value, dict):
                for side, data in value.items():
                    lines.append(f"    {side}: {data}")
            else:
                lines.append(f"    {value}")
    else:
        lines.append("  No quality data.")
    lines.append("")

    # Summary
    if report.has_differences:
        lines.append("RESULT: DIFFERENCES FOUND")
    else:
        lines.append("RESULT: FILES MATCH")

    return "\n".join(lines)


def _format_variable(vc: VariableComparison) -> str:
    """Format a single variable comparison block."""
    parts = [f"\n  {vc.name}:"]
    if not vc.present_a:
        parts.append("    MISSING in file A")
        return "\n".join(parts)
    if not vc.present_b:
        parts.append("    MISSING in file B")
        return "\n".join(parts)

    for label, stats in [("A", vc.stats_a), ("B", vc.stats_b)]:
        if stats:
            parts.append(
                f"    {label}: shape={stats['shape']}  dtype={stats['dtype']}  "
                f"valid={stats['valid_count']}  nan={stats['nan_count']}"
            )
            if stats["mean"] is not None:
                parts.append(
                    f"       min={stats['min']:.6g}  max={stats['max']:.6g}  "
                    f"mean={stats['mean']:.6g}  std={stats['std']:.6g}"
                )

    if vc.diff:
        d = vc.diff
        if d.get("max_abs_diff") is not None:
            r_str = f"  r={d['pearson_r']:.4f}" if d.get("pearson_r") is not None else ""
            parts.append(
                f"    Diff: max_abs={d['max_abs_diff']:.6g}  "
                f"mean_abs={d['mean_abs_diff']:.6g}  rmsd={d['rmsd']:.6g}  "
                f"bias={d['bias']:.6g}{r_str}"
            )
        else:
            parts.append("    Diff: no overlapping valid data")

    if vc.attr_diffs:
        parts.append("    Attribute diffs:")
        for attr, val_a, val_b in vc.attr_diffs:
            parts.append(f"      {attr}: A={_truncate(val_a)} | B={_truncate(val_b)}")

    return "\n".join(parts)


def _truncate(value, max_len: int = 80) -> str:
    """Truncate a value's string representation for display."""
    s = str(value) if value is not None else "<missing>"
    return s[:max_len] + "..." if len(s) > max_len else s
