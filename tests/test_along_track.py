"""Tests for the AlongTrackComparator."""

import numpy as np
import pytest
import xarray as xr

from validation.comparators.along_track import AlongTrackComparator


class TestAlongTrackIdentical:
    def test_no_differences(self, along_track_pair):
        path_a, path_b = along_track_pair
        comp = AlongTrackComparator(path_a, path_b)
        report = comp.run()
        assert not report.has_differences
        assert report.dimension_diffs == []
        assert report.global_attr_diffs == []

    def test_ignore_attrs(self, along_track_ds, tmp_path):
        ds_b = along_track_ds.copy()
        ds_b.attrs["date_created"] = "2025-06-15"
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        along_track_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = AlongTrackComparator(str(path_a), str(path_b))
        report = comp.run(ignore_attrs=["date_created"])
        assert not report.has_differences


class TestAlongTrackDifferences:
    def test_value_difference(self, along_track_ds, tmp_path):
        ds_b = along_track_ds.copy(deep=True)
        ds_b["ssha"].values[0] += 1.0
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        along_track_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = AlongTrackComparator(str(path_a), str(path_b))
        report = comp.run()
        assert report.has_differences

        ssha_comp = next(
            vc for vc in report.variable_comparisons if vc.name == "ssha"
        )
        assert ssha_comp.diff["max_abs_diff"] > 0

    def test_missing_variable(self, along_track_ds, tmp_path):
        ds_b = along_track_ds.drop_vars("oer")
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        along_track_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = AlongTrackComparator(str(path_a), str(path_b))
        report = comp.run()
        assert report.has_differences

        oer_comp = next(
            vc for vc in report.variable_comparisons if vc.name == "oer"
        )
        assert oer_comp.present_a is True
        assert oer_comp.present_b is False

    def test_dimension_mismatch(self, along_track_ds, tmp_path):
        ds_b = along_track_ds.isel(time=slice(0, 50))
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        along_track_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = AlongTrackComparator(str(path_a), str(path_b))
        report = comp.run()
        assert report.has_differences
        assert any(d[0] == "time" for d in report.dimension_diffs)

    def test_global_attr_diff(self, along_track_ds, tmp_path):
        ds_b = along_track_ds.copy()
        ds_b.attrs["title"] = "Changed title"
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        along_track_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = AlongTrackComparator(str(path_a), str(path_b))
        report = comp.run()
        assert report.has_differences
        assert any(a[0] == "title" for a in report.global_attr_diffs)


class TestAlongTrackPercentiles:
    def test_ssha_percentiles_present(self, along_track_pair):
        path_a, path_b = along_track_pair
        comp = AlongTrackComparator(path_a, path_b)
        report = comp.run()
        pcts = report.quality_summary["ssha_percentiles"]
        for side in ("a", "b"):
            assert pcts[side] is not None
            assert pcts[side]["p5"] <= pcts[side]["p25"] <= pcts[side]["p50"]
            assert pcts[side]["p50"] <= pcts[side]["p75"] <= pcts[side]["p95"]

    def test_ssha_percentiles_shifted(self, along_track_ds, tmp_path):
        ds_b = along_track_ds.copy(deep=True)
        ds_b["ssha"].values[:] += 0.5  # uniform shift
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        along_track_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = AlongTrackComparator(str(path_a), str(path_b))
        report = comp.run()
        pcts = report.quality_summary["ssha_percentiles"]
        assert pcts["b"]["p50"] == pytest.approx(pcts["a"]["p50"] + 0.5, abs=1e-5)


class TestAlongTrackQuality:
    def test_quality_counts(self, along_track_ds, tmp_path):
        ds_b = along_track_ds.copy(deep=True)
        ds_b["nasa_flag"].values[:10] = 1  # 10 bad flags
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        along_track_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp = AlongTrackComparator(str(path_a), str(path_b))
        report = comp.run()
        quality = report.quality_summary
        assert quality["nasa_flag"]["a"]["good"] == 100
        assert quality["nasa_flag"]["a"]["bad"] == 0
        assert quality["nasa_flag"]["b"]["good"] == 90
        assert quality["nasa_flag"]["b"]["bad"] == 10
