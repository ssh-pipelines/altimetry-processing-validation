"""Tests for the CLI entry point."""

from validation.cli import main


class TestCLI:
    def test_identical_files_exit_0(self, along_track_pair):
        path_a, path_b = along_track_pair
        rc = main([path_a, path_b, "-t", "along_track"])
        assert rc == 0

    def test_different_files_exit_1(self, along_track_ds, tmp_path):
        ds_b = along_track_ds.copy(deep=True)
        ds_b["ssha"].values[0] += 999.0
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        along_track_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        rc = main([str(path_a), str(path_b), "-t", "along_track"])
        assert rc == 1

    def test_ignore_attrs_flag(self, along_track_ds, tmp_path):
        ds_b = along_track_ds.copy()
        ds_b.attrs["date_created"] = "2099-12-31"
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        along_track_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        rc = main(
            [str(path_a), str(path_b), "-t", "along_track",
             "--ignore-attrs", "date_created"]
        )
        assert rc == 0

    def test_simple_grid_type(self, simple_grid_pair):
        path_a, path_b = simple_grid_pair
        rc = main([path_a, path_b, "-t", "simple_grid"])
        assert rc == 0

    def test_threshold_flag(self, simple_grid_ds, tmp_path):
        ds_b = simple_grid_ds.copy(deep=True)
        ds_b["ssha"].values[:] = simple_grid_ds["ssha"].values + 0.03
        path_a = tmp_path / "a.nc"
        path_b = tmp_path / "b.nc"
        simple_grid_ds.to_netcdf(path_a)
        ds_b.to_netcdf(path_b)

        comp_default = __import__("validation.cli", fromlist=["main"])
        from validation.comparators.simple_grid import SimpleGridComparator

        # With default 0.05 m threshold: all cells within threshold
        c1 = SimpleGridComparator(str(path_a), str(path_b), threshold=0.05)
        r1 = c1.run()
        assert r1.quality_summary["ssha_agreement"]["pct_within_threshold"] == 100.0

        # With tighter 0.02 m threshold: no cells within threshold
        c2 = SimpleGridComparator(str(path_a), str(path_b), threshold=0.02)
        r2 = c2.run()
        assert r2.quality_summary["ssha_agreement"]["pct_within_threshold"] == 0.0
