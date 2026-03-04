"""Tests for bulk directory comparison."""

from __future__ import annotations

import pytest

from validation.bulk_compare import _extract_date, run_bulk_comparison

# Filename templates that mirror the real pipeline naming conventions.
_DATES = ["19921101", "19921102", "19921103"]
_GSFC_NAME = "GSFC_6.1_alt_ref_simple_grid_v1_{date}.nc"
_NASA_NAME = "NASA-SSH_alt_ref_simple_grid_v1_{date}.nc"


@pytest.fixture
def two_dirs_three_pairs(simple_grid_ds, tmp_path):
    """Two directories with 3 matching files using different prefixes but the same dates."""
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()

    for date in _DATES:
        simple_grid_ds.to_netcdf(dir_a / _GSFC_NAME.format(date=date))
        simple_grid_ds.to_netcdf(dir_b / _NASA_NAME.format(date=date))

    return dir_a, dir_b, simple_grid_ds


# --- date extraction ---

def test_extract_date_standard():
    assert _extract_date("GSFC_6.1_alt_ref_simple_grid_v1_19921102.nc") == "19921102"


def test_extract_date_nasa_ssh():
    assert _extract_date("NASA-SSH_alt_ref_simple_grid_v1_19921102.nc") == "19921102"


def test_extract_date_no_date():
    assert _extract_date("simple_grid_extra.nc") is None


# --- file matching ---

def test_matched_files(two_dirs_three_pairs):
    dir_a, dir_b, _ = two_dirs_three_pairs
    report = run_bulk_comparison(dir_a, dir_b)
    assert len(report.matched_pairs) == 3
    assert report.only_in_a == []
    assert report.only_in_b == []


def test_cross_prefix_matching(two_dirs_three_pairs):
    """Files with different prefixes but the same date are matched."""
    dir_a, dir_b, _ = two_dirs_three_pairs
    report = run_bulk_comparison(dir_a, dir_b)
    assert [r.date_key for r in report.matched_pairs] == sorted(_DATES)
    for r, date in zip(report.matched_pairs, sorted(_DATES)):
        assert r.file_a.name == _GSFC_NAME.format(date=date)
        assert r.file_b.name == _NASA_NAME.format(date=date)


def test_only_in_a(two_dirs_three_pairs, simple_grid_ds):
    dir_a, dir_b, _ = two_dirs_three_pairs
    extra = _GSFC_NAME.format(date="20240101")
    simple_grid_ds.to_netcdf(dir_a / extra)
    report = run_bulk_comparison(dir_a, dir_b)
    assert extra in report.only_in_a
    assert extra not in report.only_in_b


def test_only_in_b(two_dirs_three_pairs, simple_grid_ds):
    dir_a, dir_b, _ = two_dirs_three_pairs
    extra = _NASA_NAME.format(date="20240101")
    simple_grid_ds.to_netcdf(dir_b / extra)
    report = run_bulk_comparison(dir_a, dir_b)
    assert extra in report.only_in_b
    assert extra not in report.only_in_a


def test_no_date_filename_goes_to_only(two_dirs_three_pairs, simple_grid_ds):
    """A file with no parseable date is listed as unmatched, not silently dropped."""
    dir_a, dir_b, _ = two_dirs_three_pairs
    no_date = "simple_grid_extra.nc"
    simple_grid_ds.to_netcdf(dir_a / no_date)
    report = run_bulk_comparison(dir_a, dir_b)
    assert no_date in report.only_in_a
    assert len(report.matched_pairs) == 3  # existing 3 pairs unaffected


# --- comparison metrics ---

def test_ssha_identical_gives_100pct(two_dirs_three_pairs):
    dir_a, dir_b, _ = two_dirs_three_pairs
    report = run_bulk_comparison(dir_a, dir_b, threshold=0.05)
    for pair in report.matched_pairs:
        assert pair.error is None
        assert pair.ssha_pct_within_threshold == 100.0


def test_ssha_difference_detected(simple_grid_ds, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    date = "19921102"
    simple_grid_ds.to_netcdf(dir_a / _GSFC_NAME.format(date=date))
    modified = simple_grid_ds.copy(deep=True)
    modified["ssha"] = modified["ssha"] + 1.0  # all cells exceed 0.05 m threshold
    modified.to_netcdf(dir_b / _NASA_NAME.format(date=date))

    report = run_bulk_comparison(dir_a, dir_b, threshold=0.05)
    assert len(report.matched_pairs) == 1
    pair = report.matched_pairs[0]
    assert pair.error is None
    assert pair.ssha_pct_within_threshold < 100.0


def test_counts_diff_detected(simple_grid_ds, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    date = "19921102"
    simple_grid_ds.to_netcdf(dir_a / _GSFC_NAME.format(date=date))
    modified = simple_grid_ds.copy(deep=True)
    modified["counts"] = modified["counts"] + 5
    modified.to_netcdf(dir_b / _NASA_NAME.format(date=date))

    report = run_bulk_comparison(dir_a, dir_b)
    pair = report.matched_pairs[0]
    assert pair.error is None
    assert pair.counts_mean_abs_diff is not None
    assert pair.counts_mean_abs_diff > 0.0


def test_error_handling(simple_grid_ds, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    date = "19921102"
    simple_grid_ds.to_netcdf(dir_a / _GSFC_NAME.format(date=date))
    (dir_b / _NASA_NAME.format(date=date)).write_bytes(b"not a netcdf file")

    report = run_bulk_comparison(dir_a, dir_b)
    assert len(report.matched_pairs) == 1
    pair = report.matched_pairs[0]
    assert pair.error is not None
