"""Tests for analysis.dimensions module."""

import numpy as np
import xarray as xr

from validation.analysis.dimensions import compare_dimensions


class TestCompareDimensions:
    def test_identical(self):
        ds_a = xr.Dataset({"x": (["time"], np.zeros(10))})
        ds_b = xr.Dataset({"x": (["time"], np.zeros(10))})
        assert compare_dimensions(ds_a, ds_b) == []

    def test_size_mismatch(self):
        ds_a = xr.Dataset({"x": (["time"], np.zeros(10))})
        ds_b = xr.Dataset({"x": (["time"], np.zeros(20))})
        diffs = compare_dimensions(ds_a, ds_b)
        assert len(diffs) == 1
        assert diffs[0] == ("time", 10, 20)

    def test_missing_dim(self):
        ds_a = xr.Dataset(
            {"x": (["time"], np.zeros(10)), "y": (["lat"], np.zeros(5))}
        )
        ds_b = xr.Dataset({"x": (["time"], np.zeros(10))})
        diffs = compare_dimensions(ds_a, ds_b)
        assert len(diffs) == 1
        assert diffs[0] == ("lat", 5, None)

    def test_extra_dim_in_b(self):
        ds_a = xr.Dataset({"x": (["time"], np.zeros(10))})
        ds_b = xr.Dataset(
            {"x": (["time"], np.zeros(10)), "y": (["lat"], np.zeros(5))}
        )
        diffs = compare_dimensions(ds_a, ds_b)
        assert len(diffs) == 1
        assert diffs[0] == ("lat", None, 5)

    def test_multiple_diffs(self):
        ds_a = xr.Dataset(
            {
                "x": (["time"], np.zeros(10)),
                "y": (["lat", "lon"], np.zeros((5, 8))),
            }
        )
        ds_b = xr.Dataset(
            {
                "x": (["time"], np.zeros(20)),
                "y": (["lat", "lon"], np.zeros((5, 10))),
            }
        )
        diffs = compare_dimensions(ds_a, ds_b)
        assert len(diffs) == 2
        dim_names = [d[0] for d in diffs]
        assert "time" in dim_names
        assert "lon" in dim_names
