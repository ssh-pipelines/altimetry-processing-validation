"""Bulk directory comparison for simple-grid and along-track NetCDF files."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from validation.comparators.along_track_hilat import (
    HILAT_DEFAULT_CRITERIA,
    AlongTrackHilatComparator,
    pool_lat_bands,
)
from validation.comparators.simple_grid import SimpleGridComparator

log = logging.getLogger(__name__)

BULK_COMPARATORS = {
    "simple_grid": SimpleGridComparator,
    "along_track_hilat": AlongTrackHilatComparator,
}


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
    ssha_pearson_r: float | None = None
    counts_max_abs_diff: float | None = None
    counts_mean_abs_diff: float | None = None
    counts_rmsd: float | None = None
    # along_track_hilat only
    n_common_cells: int | None = None
    cells_a: int | None = None
    cells_b: int | None = None
    max_abs_diff_lat: float | None = None
    max_abs_diff_lon: float | None = None
    ssha_std_a: float | None = None
    ssha_std_b: float | None = None
    ssha_std_ratio: float | None = None
    noise_a: float | None = None
    noise_b: float | None = None
    self_rmsd_a: float | None = None
    self_rmsd_b: float | None = None
    lat_bands: list[dict] | None = None  # raw per-band sums; see pool_lat_bands
    n_outlier_cells: int | None = None
    ssha_median_abs_diff: float | None = None
    ssha_robust_std_diff: float | None = None
    ssha_rmsd_all: float | None = None  # including outlier cells
    baseline_rmsd: float | None = None
    matched_rmsd: float | None = None  # A vs B on asc/desc half-grids
    rmsd_ratio: float | None = None  # matched_rmsd / baseline_rmsd
    alignment_status: str | None = None  # PASS / FAIL / N/A
    alignment_reasons: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BulkComparisonReport:
    """Aggregate report for a bulk directory comparison."""

    dir_a: Path
    dir_b: Path
    threshold: float
    product_type: str = "simple_grid"
    matched_pairs: list[FilePairResult] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    # along_track_hilat only: per-band sums pooled across all matched pairs,
    # and the alignment criteria applied to every pair
    lat_bands: list[dict] | None = None
    criteria: dict | None = None


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
        for f in sorted(directory.glob("*.nc")):
            date = _extract_date(f.name)
            if date is None:
                log.warning("No 8-digit date found in filename, skipping: %s", f.name)
                no_date.append(f.name)
            elif date in dated:
                log.warning(
                    "Multiple files for date %s in %s; using %s, ignoring %s",
                    date, directory, dated[date].name, f.name,
                )
                no_date.append(f.name)
            else:
                dated[date] = f
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
    product_type: str = "simple_grid",
    criteria: dict | None = None,
) -> FilePairResult:
    """Compare a single file pair using the comparator for *product_type*."""
    print(date_key)
    log.debug("Comparing %s vs %s (date=%s)", file_a.name, file_b.name, date_key)
    result = FilePairResult(date_key=date_key, file_a=file_a, file_b=file_b)

    try:
        comparator_cls = BULK_COMPARATORS[product_type]
        comparator = comparator_cls(
            str(file_a), str(file_b), threshold=threshold, **(criteria or {})
        )
        report = comparator.run(ignore_attrs=ignore_attrs)

        if product_type == "along_track_hilat":
            gridded = report.quality_summary["ssha_gridded"]
            result.ssha_pct_within_threshold = gridded["pct_within_threshold"]
            result.ssha_max_abs_diff = gridded["max_abs_diff"]
            result.ssha_mean_abs_diff = gridded["mean_abs_diff"]
            result.ssha_rmsd = gridded["rmsd"]
            result.ssha_bias = gridded["bias"]
            result.ssha_pearson_r = gridded["pearson_r"]
            result.n_common_cells = gridded["n_common_cells"]
            result.cells_a = gridded["cells_a"]
            result.cells_b = gridded["cells_b"]
            result.max_abs_diff_lat = gridded["max_abs_diff_lat"]
            result.max_abs_diff_lon = gridded["max_abs_diff_lon"]
            result.ssha_std_a = gridded["std_a"]
            result.ssha_std_b = gridded["std_b"]
            result.ssha_std_ratio = gridded["std_ratio"]
            q = report.quality_summary
            result.noise_a = (q["noise"]["a"] or {}).get("rms")
            result.noise_b = (q["noise"]["b"] or {}).get("rms")
            result.self_rmsd_a = (q["self_baseline"]["a"] or {}).get("rmsd")
            result.self_rmsd_b = (q["self_baseline"]["b"] or {}).get("rmsd")
            result.lat_bands = q["lat_bands"]
            result.n_outlier_cells = gridded["n_outlier_cells"]
            result.ssha_median_abs_diff = gridded["median_abs_diff"]
            result.ssha_robust_std_diff = gridded["robust_std_diff"]
            result.ssha_rmsd_all = gridded["rmsd_all"]
            alignment = q["alignment"]
            result.baseline_rmsd = alignment["baseline_rmsd"]
            result.matched_rmsd = alignment["matched_rmsd"]
            result.rmsd_ratio = alignment["rmsd_ratio"]
            result.alignment_status = alignment["status"]
            result.alignment_reasons = alignment["reasons"]
            log.debug(
                "  %s: alignment=%s  rmsd=%.4g  common cells=%d",
                date_key,
                result.alignment_status,
                result.ssha_rmsd or 0.0,
                result.n_common_cells,
            )
            return result

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
                result.ssha_pearson_r = vc.diff.get("pearson_r")
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
    product_type: str = "simple_grid",
    criteria: dict | None = None,
) -> BulkComparisonReport:
    """Compare all matching NetCDF files between two directories.

    Parameters
    ----------
    dir_a, dir_b:
        Paths to directories of NetCDF files, matched by filename date. For
        ``along_track_hilat``, dir_a holds reference-mission along-track files
        and dir_b holds high-latitude along-track files.
    threshold:
        Absolute difference threshold in metres for the SSHA agreement metric.
    ignore_attrs:
        Global/variable attribute names to ignore during comparison.
    product_type:
        Key into BULK_COMPARATORS selecting the comparator to use.
    criteria:
        along_track_hilat only: overrides for HILAT_DEFAULT_CRITERIA
        (outlier_cutoff, max_bias, rmsd_factor, min_common_cells,
        max_outlier_cells).

    Returns
    -------
    BulkComparisonReport with per-file results and unmatched file lists.
    """
    dir_a = Path(dir_a)
    dir_b = Path(dir_b)

    log.info(
        "Starting %s bulk comparison: %s vs %s (threshold=%.3f m)",
        product_type, dir_a, dir_b, threshold,
    )

    pairs, only_a, only_b = _match_files(dir_a, dir_b)
    matched_pairs = [
        _compare_pair(fa, fb, dk, threshold, ignore_attrs, product_type, criteria)
        for fa, fb, dk in pairs
    ]

    n_errors = sum(1 for r in matched_pairs if r.error)
    log.info(
        "Bulk comparison complete: %d files compared, %d errors",
        len(matched_pairs), n_errors,
    )

    lat_bands = None
    applied_criteria = None
    if product_type == "along_track_hilat":
        lat_bands = pool_lat_bands([r.lat_bands for r in matched_pairs if r.lat_bands])
        applied_criteria = {**HILAT_DEFAULT_CRITERIA, **(criteria or {})}

    return BulkComparisonReport(
        lat_bands=lat_bands,
        criteria=applied_criteria,
        dir_a=dir_a,
        dir_b=dir_b,
        threshold=threshold,
        product_type=product_type,
        matched_pairs=matched_pairs,
        only_in_a=only_a,
        only_in_b=only_b,
    )
