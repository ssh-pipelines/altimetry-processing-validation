"""Tests for the crossover RMS analysis module."""

from datetime import date

import netCDF4 as nc
import numpy as np
import pytest

from validation.crossover import (
    CrossoverDayResult,
    _inner_diff,
    compute_daily_rms,
    print_summary,
    write_csv,
)


def _write_xover_file(path, ssh1: np.ndarray, ssh2: np.ndarray) -> None:
    """Write a minimal xovers_S6-*.nc file with ssh1 and ssh2 variables."""
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("obs", len(ssh1))
        v1 = ds.createVariable("ssh1", "f8", ("obs",))
        v2 = ds.createVariable("ssh2", "f8", ("obs",))
        v1[:] = ssh1
        v2[:] = ssh2


@pytest.fixture
def xover_dir(tmp_path):
    """Three days of synthetic crossover files with known RMS values."""
    days = [
        ("2024-01-01", np.zeros(10), np.full(10, 0.01)),   # diff = 0.01, RMS = 0.01
        ("2024-01-02", np.zeros(10), np.full(10, 0.02)),   # diff = 0.02, RMS = 0.02
        ("2024-01-03", np.zeros(10), np.full(10, 0.03)),   # diff = 0.03, RMS = 0.03
    ]
    for date_str, ssh1, ssh2 in days:
        fname = tmp_path / f"xovers_S6-{date_str}.nc"
        _write_xover_file(fname, ssh1, ssh2)
    return tmp_path


class TestComputeDailyRms:
    def test_returns_correct_count(self, xover_dir):
        results = compute_daily_rms(xover_dir)
        assert len(results) == 3

    def test_sorted_by_date(self, xover_dir):
        results = compute_daily_rms(xover_dir)
        dates = [r.date for r in results]
        assert dates == sorted(dates)

    def test_rms_values(self, xover_dir):
        results = compute_daily_rms(xover_dir)
        expected_rms = [0.01, 0.02, 0.03]
        for r, expected in zip(results, expected_rms):
            assert abs(r.rms_m - expected) < 1e-10

    def test_n_crossovers(self, xover_dir):
        results = compute_daily_rms(xover_dir)
        assert all(r.n_crossovers == 10 for r in results)

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_daily_rms(tmp_path)

    def test_label_used_in_output(self, xover_dir, capsys):
        compute_daily_rms(xover_dir, label="my_label")
        captured = capsys.readouterr()
        assert "my_label" in captured.out

    def test_skips_unmatched_filename(self, xover_dir, tmp_path):
        """Files without a YYYY-MM-DD date in the name are skipped."""
        bad = tmp_path / "xovers_S6-badname.nc"
        _write_xover_file(bad, np.zeros(5), np.ones(5))
        results = compute_daily_rms(xover_dir)
        # Only the three valid files should be counted
        assert len(results) == 3


class TestInnerDiff:
    def test_common_dates_only(self):
        a = [
            CrossoverDayResult(date=date(2024, 1, 1), rms_m=0.10, n_crossovers=5),
            CrossoverDayResult(date=date(2024, 1, 2), rms_m=0.20, n_crossovers=5),
        ]
        b = [
            CrossoverDayResult(date=date(2024, 1, 2), rms_m=0.25, n_crossovers=5),
            CrossoverDayResult(date=date(2024, 1, 3), rms_m=0.30, n_crossovers=5),
        ]
        pairs = _inner_diff(a, b)
        assert len(pairs) == 1
        d, diff_cm = pairs[0]
        assert d == date(2024, 1, 2)
        assert abs(diff_cm - 5.0) < 1e-9  # (0.25 - 0.20) * 100 = 5.0

    def test_empty_when_no_overlap(self):
        a = [CrossoverDayResult(date=date(2024, 1, 1), rms_m=0.1, n_crossovers=1)]
        b = [CrossoverDayResult(date=date(2024, 1, 2), rms_m=0.2, n_crossovers=1)]
        assert _inner_diff(a, b) == []

    def test_sorted_by_date(self):
        a = [
            CrossoverDayResult(date=date(2024, 1, d), rms_m=0.01 * d, n_crossovers=1)
            for d in [3, 1, 2]
        ]
        b = [
            CrossoverDayResult(date=date(2024, 1, d), rms_m=0.02 * d, n_crossovers=1)
            for d in [2, 3, 1]
        ]
        pairs = _inner_diff(a, b)
        dates = [d for d, _ in pairs]
        assert dates == sorted(dates)


class TestWriteCsv:
    def test_creates_file(self, tmp_path):
        results = [
            CrossoverDayResult(date=date(2024, 1, 1), rms_m=0.01, n_crossovers=10),
        ]
        out = write_csv(results, "test_label", tmp_path)
        assert out.exists()

    def test_csv_content(self, tmp_path):
        results = [
            CrossoverDayResult(date=date(2024, 1, 1), rms_m=0.01, n_crossovers=10),
            CrossoverDayResult(date=date(2024, 1, 2), rms_m=0.02, n_crossovers=20),
        ]
        out = write_csv(results, "series", tmp_path)
        lines = out.read_text().splitlines()
        assert lines[0] == "date,rms_m,n_crossovers"
        assert lines[1].startswith("2024-01-01,")
        assert lines[2].startswith("2024-01-02,")


class TestXoverCLI:
    def test_compare_xovers_exit_0(self, tmp_path):
        """CLI runs end-to-end and returns 0 on valid inputs."""
        for period in ("pre_set1", "pre_set2", "post_set1", "post_set2"):
            d = tmp_path / period
            d.mkdir()
            _write_xover_file(
                d / "xovers_S6-2024-01-01.nc",
                np.zeros(5),
                np.full(5, 0.01),
            )

        from validation.cli import xover_main

        rc = xover_main([
            str(tmp_path / "pre_set1"),
            str(tmp_path / "pre_set2"),
            str(tmp_path / "post_set1"),
            str(tmp_path / "post_set2"),
        ])
        assert rc == 0
