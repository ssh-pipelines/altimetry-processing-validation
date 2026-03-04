"""Bulk directory comparison for simple-grid NetCDF files."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from validation.comparators.simple_grid import SimpleGridComparator

log = logging.getLogger(__name__)


@dataclass
class FilePairResult:
    """Comparison result for a single matched file pair."""

    date_key: str  # 8-digit YYYYMMDD extracted from both filenames
    file_a: Path
    file_b: Path
    ssha_pct_within_threshold: float | None = None
    ssha_max_abs_diff: float | None = None
    ssha_mean_abs_diff: float | None = None
    ssha_rmsd: float | None = None
    ssha_bias: float | None = None
    counts_max_abs_diff: float | None = None
    counts_mean_abs_diff: float | None = None
    counts_rmsd: float | None = None
    error: str | None = None


@dataclass
class BulkComparisonReport:
    """Aggregate report for a bulk directory comparison."""

    dir_a: Path
    dir_b: Path
    threshold: float
    matched_pairs: list[FilePairResult] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)


def _extract_date(filename: str) -> str | None:
    """Extract an 8-digit YYYYMMDD date from a NetCDF filename.

    Matches the last 8-digit sequence immediately before the .nc extension,
    e.g. 'GSFC_6.1_alt_ref_simple_grid_v1_19921102.nc' → '19921102'.
    Returns None if no such date is found.
    """
    m = re.search(r"(\d{8})\.nc$", filename)
    return m.group(1) if m else None


def _match_files(
    dir_a: Path, dir_b: Path
) -> tuple[list[tuple[Path, Path, str]], list[str], list[str]]:
    """Match NetCDF files by 8-digit date extracted from their filenames.

    Files with no parseable date, or whose date has no counterpart in the
    other directory, appear in only_a / only_b by their original filename.

    Returns (pairs, only_a, only_b) where pairs are (file_a, file_b, date_key)
    sorted by date_key.
    """

    def _keyed(directory: Path) -> tuple[dict[str, Path], list[str]]:
        dated: dict[str, Path] = {}
        no_date: list[str] = []
        for f in directory.glob("*.nc"):
            date = _extract_date(f.name)
            if date is not None:
                dated[date] = f
            else:
                log.warning("No 8-digit date found in filename, skipping: %s", f.name)
                no_date.append(f.name)
        return dated, no_date

    dated_a, no_date_a = _keyed(dir_a)
    dated_b, no_date_b = _keyed(dir_b)

    common_dates = sorted(dated_a.keys() & dated_b.keys())
    only_a = sorted(no_date_a) + sorted(dated_a[d].name for d in dated_a if d not in dated_b)
    only_b = sorted(no_date_b) + sorted(dated_b[d].name for d in dated_b if d not in dated_a)

    log.info(
        "File matching: %d paired, %d only in A, %d only in B",
        len(common_dates), len(only_a), len(only_b),
    )
    for name in only_a:
        log.debug("Unmatched (only in A): %s", name)
    for name in only_b:
        log.debug("Unmatched (only in B): %s", name)

    pairs = [(dated_a[d], dated_b[d], d) for d in common_dates]
    return pairs, only_a, only_b


def _compare_pair(
    file_a: Path,
    file_b: Path,
    date_key: str,
    threshold: float,
    ignore_attrs: list[str] | None,
) -> FilePairResult:
    """Compare a single file pair using SimpleGridComparator."""
    print(date_key)
    log.debug("Comparing %s vs %s (date=%s)", file_a.name, file_b.name, date_key)
    result = FilePairResult(date_key=date_key, file_a=file_a, file_b=file_b)

    try:
        comparator = SimpleGridComparator(str(file_a), str(file_b), threshold=threshold)
        report = comparator.run(ignore_attrs=ignore_attrs)

        ssha_agreement = report.quality_summary.get("ssha_agreement", {})
        result.ssha_pct_within_threshold = ssha_agreement.get("pct_within_threshold")

        for vc in report.variable_comparisons:
            if vc.diff is None:
                continue
            if vc.name == "ssha":
                result.ssha_max_abs_diff = vc.diff.get("max_abs_diff")
                result.ssha_mean_abs_diff = vc.diff.get("mean_abs_diff")
                result.ssha_rmsd = vc.diff.get("rmsd")
                result.ssha_bias = vc.diff.get("bias")
            elif vc.name == "counts":
                result.counts_max_abs_diff = vc.diff.get("max_abs_diff")
                result.counts_mean_abs_diff = vc.diff.get("mean_abs_diff")
                result.counts_rmsd = vc.diff.get("rmsd")

        log.debug(
            "  %s: SSHA agreement=%.1f%%  max_abs=%.4g  counts_mae=%.4g",
            date_key,
            result.ssha_pct_within_threshold or 0.0,
            result.ssha_max_abs_diff or 0.0,
            result.counts_mean_abs_diff or 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Error comparing %s (date=%s): %s", file_a.name, date_key, exc)
        result.error = str(exc)

    return result


def run_bulk_comparison(
    dir_a: str | Path,
    dir_b: str | Path,
    threshold: float = 0.05,
    ignore_attrs: list[str] | None = None,
) -> BulkComparisonReport:
    """Compare all matching NetCDF files between two directories.

    Parameters
    ----------
    dir_a, dir_b:
        Paths to directories containing simple-grid NetCDF files.
    threshold:
        Absolute difference threshold in metres for the SSHA agreement metric.
    ignore_attrs:
        Global/variable attribute names to ignore during comparison.

    Returns
    -------
    BulkComparisonReport with per-file results and unmatched file lists.
    """
    dir_a = Path(dir_a)
    dir_b = Path(dir_b)

    log.info("Starting bulk comparison: %s vs %s (threshold=%.3f m)", dir_a, dir_b, threshold)

    pairs, only_a, only_b = _match_files(dir_a, dir_b)
    matched_pairs = [_compare_pair(fa, fb, dk, threshold, ignore_attrs) for fa, fb, dk in pairs]

    n_errors = sum(1 for r in matched_pairs if r.error)
    log.info(
        "Bulk comparison complete: %d files compared, %d errors",
        len(matched_pairs), n_errors,
    )

    return BulkComparisonReport(
        dir_a=dir_a,
        dir_b=dir_b,
        threshold=threshold,
        matched_pairs=matched_pairs,
        only_in_a=only_a,
        only_in_b=only_b,
    )
