"""Plain-text report formatting for comparison results."""

from validation.comparators.along_track_hilat import summarise_lat_band
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
            elif key == "ssha_gridded" and isinstance(value, dict):
                lines.extend(_format_gridded(value))
            elif key == "lat_bands" and isinstance(value, list):
                lines.extend(format_lat_bands([summarise_lat_band(b) for b in value]))
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


def _format_gridded(g: dict) -> list[str]:
    """Format the gridded SSHA comparison block (along_track_hilat)."""
    lines = [
        f"    bin: {g['bin_size_deg']} deg  |  cells A={g['cells_a']}  "
        f"B={g['cells_b']}  common={g['n_common_cells']}"
    ]
    if g.get("max_abs_diff") is None:
        lines.append("    Diff: no commonly observed cells")
        return lines
    r_str = f"  r={g['pearson_r']:.4f}" if g.get("pearson_r") is not None else ""
    lines.append(
        f"    Diff: max_abs={g['max_abs_diff']:.6g}  mean_abs={g['mean_abs_diff']:.6g}  "
        f"rmsd={g['rmsd']:.6g}  bias={g['bias']:.6g}{r_str}"
    )
    lines.append(
        f"    Largest diff at lat={g['max_abs_diff_lat']:.1f}  lon={g['max_abs_diff_lon']:.1f}"
    )
    lines.append(
        f"    threshold: {g['threshold_m']} m  |  pct_within: {g['pct_within_threshold']}%"
    )
    if g.get("std_ratio") is not None:
        lines.append(
            f"    Variability over common cells: std A={g['std_a']:.6g}  "
            f"std B={g['std_b']:.6g}  ratio B/A={g['std_ratio']:.4f}"
        )
    return lines


def format_lat_bands(bands: list[dict]) -> list[str]:
    """Format summarised latitude-band stats as a table (bands with any data)."""

    def _f(v, fmt=".4f"):
        return format(v, fmt) if v is not None else "N/A"

    header = (
        f"    {'Band':>11}  {'Cells':>7}  {'Bias':>8}  {'RMSD':>7}  "
        f"{'Std A':>7}  {'Std B':>7}  {'B/A':>6}  {'Noise A':>7}  {'Noise B':>7}"
    )
    lines = [header, "    " + "-" * (len(header) - 4)]
    for b in bands:
        if not b["n_common_cells"] and b["noise_a"] is None and b["noise_b"] is None:
            continue
        band = f"{b['lat_min']:g}..{b['lat_max']:g}"
        lines.append(
            f"    {band:>11}  {b['n_common_cells']:>7}  {_f(b['bias']):>8}  "
            f"{_f(b['rmsd']):>7}  {_f(b['std_a']):>7}  {_f(b['std_b']):>7}  "
            f"{_f(b['std_ratio'], '.3f'):>6}  {_f(b['noise_a']):>7}  {_f(b['noise_b']):>7}"
        )
    return lines


def _truncate(value, max_len: int = 80) -> str:
    """Truncate a value's string representation for display."""
    s = str(value) if value is not None else "<missing>"
    return s[:max_len] + "..." if len(s) > max_len else s
