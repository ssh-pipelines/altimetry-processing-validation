"""Tests for the SimpleGridComparator."""

import numpy as np
import xarray as xr

from validation.comparators.simple_grid import SimpleGridComparator


class TestSimpleGridIdentical:
    def test_no_differences(self, simple_grid_pair):
        path_a, path_b = simple_grid_pair
        comp = SimpleGridComparator(path_a, path_b)
        report = comp.run()
        assert not report.has_differences
        assert report.dimension_diffs == []
        assert report.global_attr_diffs == []


class TestSimpleGridDifferences:
    def test_value_difference(self, simple_grid_ds, tmp_path):
        ds_b = simple_grid_ds.copy(deep=True)
        ds_b["ssha"].values[0, 0] += 10.0
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        simple_grid_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = SimpleGridComparator(str(path_a), str(path_b))
        report = comp.run()
        assert report.has_differences

        ssha_comp = next(
            vc for vc in report.variable_comparisons if vc.name == "ssha"
        )
        assert ssha_comp.diff["max_abs_diff"] > 0

    def test_missing_variable(self, simple_grid_ds, tmp_path):
        ds_b = simple_grid_ds.drop_vars("counts")
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        simple_grid_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = SimpleGridComparator(str(path_a), str(path_b))
        report = comp.run()
        assert report.has_differences

    def test_changed_attr(self, simple_grid_ds, tmp_path):
        ds_b = simple_grid_ds.copy()
        ds_b.attrs["source"] = "modified"
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        simple_grid_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = SimpleGridComparator(str(path_a), str(path_b))
        report = comp.run()
        assert report.has_differences


class TestSimpleGridQuality:
    def test_coverage_and_counts(self, simple_grid_ds, tmp_path):
        ds_b = simple_grid_ds.copy(deep=True)
        # Introduce NaNs in half the SSHA grid
        ds_b["ssha"].values[:90, :] = np.nan
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        simple_grid_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = SimpleGridComparator(str(path_a), str(path_b))
        report = comp.run()

        coverage_a = report.quality_summary["ssha_coverage"]["a"]
        coverage_b = report.quality_summary["ssha_coverage"]["b"]
        assert coverage_a["coverage_pct"] == 100.0
        assert coverage_b["coverage_pct"] < coverage_a["coverage_pct"]

    def test_pct_within_threshold_default(self, simple_grid_ds, tmp_path):
        ds_b = simple_grid_ds.copy(deep=True)
        # Shift all values by 0.01 m — well within default 5 cm threshold
        ds_b["ssha"].values[:] = simple_grid_ds["ssha"].values + 0.01
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        simple_grid_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = SimpleGridComparator(str(path_a), str(path_b))
        report = comp.run()
        agreement = report.quality_summary["ssha_agreement"]
        assert agreement["threshold_m"] == 0.05
        assert agreement["pct_within_threshold"] == 100.0

    def test_pct_within_threshold_custom(self, simple_grid_ds, tmp_path):
        ds_b = simple_grid_ds.copy(deep=True)
        # Shift all values by 0.03 m — within 5 cm but not within 2 cm
        ds_b["ssha"].values[:] = simple_grid_ds["ssha"].values + 0.03
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        simple_grid_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = SimpleGridComparator(str(path_a), str(path_b), threshold=0.02)
        report = comp.run()
        agreement = report.quality_summary["ssha_agreement"]
        assert agreement["threshold_m"] == 0.02
        assert agreement["pct_within_threshold"] == 0.0

    def test_pct_within_threshold_partial(self, simple_grid_ds, tmp_path):
        ds_b = simple_grid_ds.copy(deep=True)
        nlat = ds_b["ssha"].shape[0]
        # Shift the first half of rows beyond 5 cm
        ds_b["ssha"].values[: nlat // 2, :] += 0.10
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        simple_grid_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = SimpleGridComparator(str(path_a), str(path_b))
        report = comp.run()
        pct = report.quality_summary["ssha_agreement"]["pct_within_threshold"]
        assert 40.0 < pct < 60.0

    def test_fill_value_only_counts(self, tmp_path):
        """Test with counts array full of int32 fill values."""
        fill = np.iinfo(np.int32).max
        lat = np.arange(0, 10, dtype=np.float32)
        lon = np.arange(0, 10, dtype=np.float32)
        ds = xr.Dataset(
            {
                "ssha": (["latitude", "longitude"], np.full((10, 10), np.nan)),
                "counts": (
                    ["latitude", "longitude"],
                    np.full((10, 10), fill, dtype=np.int32),
                ),
            },
            coords={"latitude": lat, "longitude": lon},
        )
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        ds.to_netcdf(path_a)
        ds.to_netcdf(path_b)

        comp = SimpleGridComparator(str(path_a), str(path_b))
        report = comp.run()
        # With all fill values, counts should report None for stats
        counts_q = report.quality_summary["counts"]["a"]
        assert counts_q["min"] is None
