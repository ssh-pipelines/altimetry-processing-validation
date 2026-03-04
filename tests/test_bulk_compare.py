"""Tests for bulk directory comparison."""

from __future__ import annotations

import pytest

from validation.bulk_compare import run_bulk_comparison


@pytest.fixture
def two_dirs_three_pairs(simple_grid_ds, tmp_path):
    """Two directories each with 3 matching synthetic simple-grid files."""
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()

    for i in range(3):
        name = f"simple_grid_2024-{i + 1:03d}.nc"
        simple_grid_ds.to_netcdf(dir_a / name)
        simple_grid_ds.to_netcdf(dir_b / name)

    return dir_a, dir_b, simple_grid_ds


def test_matched_files(two_dirs_three_pairs):
    dir_a, dir_b, _ = two_dirs_three_pairs
    report = run_bulk_comparison(dir_a, dir_b)
    assert len(report.matched_pairs) == 3
    assert report.only_in_a == []
    assert report.only_in_b == []


def test_only_in_a(two_dirs_three_pairs, simple_grid_ds):
    dir_a, dir_b, _ = two_dirs_three_pairs
    extra = "simple_grid_extra.nc"
    simple_grid_ds.to_netcdf(dir_a / extra)
    report = run_bulk_comparison(dir_a, dir_b)
    assert extra in report.only_in_a
    assert extra not in report.only_in_b


def test_only_in_b(two_dirs_three_pairs, simple_grid_ds):
    dir_a, dir_b, _ = two_dirs_three_pairs
    extra = "simple_grid_extra.nc"
    simple_grid_ds.to_netcdf(dir_b / extra)
    report = run_bulk_comparison(dir_a, dir_b)
    assert extra in report.only_in_b
    assert extra not in report.only_in_a


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

    name = "simple_grid_2024-001.nc"
    simple_grid_ds.to_netcdf(dir_a / name)

    # Offset ssha by 1 m so all cells exceed the 0.05 m threshold
    modified = simple_grid_ds.copy(deep=True)
    modified["ssha"] = modified["ssha"] + 1.0
    modified.to_netcdf(dir_b / name)

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

    name = "simple_grid_2024-001.nc"
    simple_grid_ds.to_netcdf(dir_a / name)

    modified = simple_grid_ds.copy(deep=True)
    modified["counts"] = modified["counts"] + 5
    modified.to_netcdf(dir_b / name)

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

    name = "simple_grid_2024-001.nc"
    simple_grid_ds.to_netcdf(dir_a / name)
    # Write invalid content to dir_b
    (dir_b / name).write_bytes(b"not a netcdf file")

    report = run_bulk_comparison(dir_a, dir_b)
    assert len(report.matched_pairs) == 1
    pair = report.matched_pairs[0]
    assert pair.error is not None
