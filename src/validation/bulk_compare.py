"""Bulk directory comparison for simple-grid NetCDF files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from validation.comparators.simple_grid import SimpleGridComparator


@dataclass
class FilePairResult:
    """Comparison result for a single matched file pair."""

    filename: str
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


def _match_files(
    dir_a: Path, dir_b: Path
) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    """Match NetCDF files by name across two directories.

    Returns (pairs, only_a, only_b) where pairs are sorted by filename.
    """
    files_a = {f.name: f for f in dir_a.glob("*.nc")}
    files_b = {f.name: f for f in dir_b.glob("*.nc")}

    common = sorted(files_a.keys() & files_b.keys())
    only_a = sorted(files_a.keys() - files_b.keys())
    only_b = sorted(files_b.keys() - files_a.keys())

    pairs = [(files_a[name], files_b[name]) for name in common]
    return pairs, only_a, only_b


def _compare_pair(
    file_a: Path,
    file_b: Path,
    threshold: float,
    ignore_attrs: list[str] | None,
) -> FilePairResult:
    """Compare a single file pair using SimpleGridComparator."""
    result = FilePairResult(filename=file_a.name, file_a=file_a, file_b=file_b)

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
    except Exception as exc:  # noqa: BLE001
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

    pairs, only_a, only_b = _match_files(dir_a, dir_b)
    matched_pairs = [_compare_pair(fa, fb, threshold, ignore_attrs) for fa, fb in pairs]

    return BulkComparisonReport(
        dir_a=dir_a,
        dir_b=dir_b,
        threshold=threshold,
        matched_pairs=matched_pairs,
        only_in_a=only_a,
        only_in_b=only_b,
    )
