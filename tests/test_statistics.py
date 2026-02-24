"""Tests for analysis.statistics module."""

import numpy as np
import pytest
import xarray as xr

from validation.analysis.statistics import compute_variable_diff, compute_variable_stats


class TestComputeVariableStats:
    def test_basic_float(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        var = xr.DataArray(data)
        stats = compute_variable_stats(var)
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["valid_count"] == 5
        assert stats["nan_count"] == 0
        assert stats["shape"] == (5,)

    def test_with_nans(self):
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        var = xr.DataArray(data)
        stats = compute_variable_stats(var)
        assert stats["valid_count"] == 3
        assert stats["nan_count"] == 2
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_all_nan(self):
        data = np.array([np.nan, np.nan, np.nan])
        var = xr.DataArray(data)
        stats = compute_variable_stats(var)
        assert stats["valid_count"] == 0
        assert stats["nan_count"] == 3
        assert stats["min"] is None
        assert stats["mean"] is None

    def test_int8_fill_value_masked(self):
        fill = np.iinfo(np.int8).max  # 127
        data = np.array([0, 1, fill, 0, fill], dtype=np.int8)
        var = xr.DataArray(data)
        stats = compute_variable_stats(var)
        assert stats["valid_count"] == 3
        assert stats["nan_count"] == 2
        assert stats["max"] == 1.0

    def test_int32_fill_value_masked(self):
        fill = np.iinfo(np.int32).max
        data = np.array([10, 20, fill, 30], dtype=np.int32)
        var = xr.DataArray(data)
        stats = compute_variable_stats(var)
        assert stats["valid_count"] == 3
        assert stats["nan_count"] == 1
        assert stats["max"] == 30.0

    def test_empty_array(self):
        data = np.array([], dtype=np.float64)
        var = xr.DataArray(data)
        stats = compute_variable_stats(var)
        assert stats["valid_count"] == 0
        assert stats["nan_count"] == 0

    def test_inf_masked(self):
        data = np.array([1.0, np.inf, -np.inf, 2.0])
        var = xr.DataArray(data)
        stats = compute_variable_stats(var)
        assert stats["valid_count"] == 2
        assert stats["nan_count"] == 2


class TestComputeVariableDiff:
    def test_identical(self):
        data = np.array([1.0, 2.0, 3.0])
        var_a = xr.DataArray(data)
        var_b = xr.DataArray(data.copy())
        diff = compute_variable_diff(var_a, var_b)
        assert diff["max_abs_diff"] == 0.0
        assert diff["mean_abs_diff"] == 0.0
        assert diff["rmsd"] == 0.0

    def test_known_diff(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.0, 3.5])
        diff = compute_variable_diff(xr.DataArray(a), xr.DataArray(b))
        assert diff["max_abs_diff"] == pytest.approx(0.5)
        assert diff["mean_abs_diff"] == pytest.approx(1.0 / 3.0)

    def test_shape_mismatch(self):
        a = xr.DataArray(np.array([1.0, 2.0]))
        b = xr.DataArray(np.array([1.0, 2.0, 3.0]))
        assert compute_variable_diff(a, b) is None

    def test_non_numeric(self):
        a = xr.DataArray(np.array(["a", "b"]))
        b = xr.DataArray(np.array(["a", "c"]))
        assert compute_variable_diff(a, b) is None

    def test_all_nan_overlap(self):
        a = xr.DataArray(np.array([np.nan, np.nan]))
        b = xr.DataArray(np.array([1.0, 2.0]))
        diff = compute_variable_diff(a, b)
        assert diff["max_abs_diff"] is None

    def test_bias_positive(self):
        a = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0]))
        b = xr.DataArray(np.array([2.0, 3.0, 4.0, 5.0]))  # B = A + 1
        diff = compute_variable_diff(a, b)
        assert diff["bias"] == pytest.approx(1.0)

    def test_bias_negative(self):
        a = xr.DataArray(np.array([2.0, 3.0, 4.0]))
        b = xr.DataArray(np.array([1.0, 2.0, 3.0]))  # B = A - 1
        diff = compute_variable_diff(a, b)
        assert diff["bias"] == pytest.approx(-1.0)

    def test_pearson_r_perfect(self):
        a = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        b = xr.DataArray(np.array([2.0, 4.0, 6.0, 8.0, 10.0]))  # perfect linear
        diff = compute_variable_diff(a, b)
        assert diff["pearson_r"] == pytest.approx(1.0)

    def test_pearson_r_none_for_single_point(self):
        a = xr.DataArray(np.array([1.0]))
        b = xr.DataArray(np.array([2.0]))
        diff = compute_variable_diff(a, b)
        assert diff["pearson_r"] is None
