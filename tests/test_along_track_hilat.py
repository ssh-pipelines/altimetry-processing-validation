"""Tests for reference vs high-latitude along-track comparison."""

import numpy as np
import pytest
import xarray as xr

from validation.analysis.binning import bin_to_grid
from validation.bulk_compare import run_bulk_comparison
from validation.bulk_plot import plot_bulk_timeseries
from validation.bulk_report import format_bulk_report
from validation.comparators.along_track_hilat import (
    LAT_BAND_SUM_KEYS,
    AlongTrackHilatComparator,
    pool_lat_bands,
    summarise_lat_band,
)

_REF_NAME = "NASA-SSH_alt_ref_at_v1_1_{date}.nc"
_HILAT_NAME = "NASA-SSH_S3B_alt_hilat_at_v1_1_{date}.nc"


def _at_ds(lat, lon, ssha, n_src_flags=1):
    """Minimal along-track dataset on an arbitrary ground track."""
    n = len(lat)
    return xr.Dataset(
        {
            "ssha": (["time"], np.asarray(ssha, dtype=np.float64)),
            "cycle": (["time"], np.full(n, 1, dtype=np.int32)),
            "pass": (["time"], np.arange(n, dtype=np.int32)),
            "nasa_flag": (["time"], np.zeros(n, dtype=np.int8)),
            "source_flag": (
                ["time", "src_flag_dim"],
                np.zeros((n, n_src_flags), dtype=np.int8),
            ),
            "median_filter_flag": (["time"], np.zeros(n, dtype=np.int8)),
        },
        coords={
            "time": np.arange(n, dtype=np.float64),
            "latitude": (["time"], np.asarray(lat, dtype=np.float32)),
            "longitude": (["time"], np.asarray(lon, dtype=np.float32)),
        },
    )


def _field(lat, lon):
    """Smooth synthetic SSHA field so both tracks sample the same signal."""
    return 0.1 * np.sin(np.radians(lat) * 3) * np.cos(np.radians(lon))


@pytest.fixture
def ref_hilat_dss():
    """Ref track within ±66° and a differently sampled hi-lat track to ±80°."""
    rng = np.random.default_rng(0)
    lat_a = rng.uniform(-66, 66, 20000)
    lon_a = rng.uniform(0, 360, 20000)
    lat_b = rng.uniform(-80, 80, 15000)
    lon_b = rng.uniform(0, 360, 15000)
    ds_a = _at_ds(lat_a, lon_a, _field(lat_a, lon_a), n_src_flags=15)
    ds_b = _at_ds(lat_b, lon_b, _field(lat_b, lon_b))
    return ds_a, ds_b


def _write_pair(tmp_path, ds_a, ds_b):
    path_a = tmp_path / _REF_NAME.format(date="20200101")
    path_b = tmp_path / _HILAT_NAME.format(date="20200101")
    ds_a.to_netcdf(path_a)
    ds_b.to_netcdf(path_b)
    return str(path_a), str(path_b)


class TestBinToGrid:
    def test_cell_means_and_counts(self):
        lat = np.array([0.2, 0.7, 10.5, np.nan])
        lon = np.array([5.1, 5.9, -1.5, 3.0])  # -1.5 wraps to 358.5
        vals = np.array([1.0, 3.0, 5.0, 7.0])
        mean, counts = bin_to_grid(lat, lon, vals, bin_size=1.0)
        assert mean.shape == (180, 360)
        assert counts[90, 5] == 2
        assert mean[90, 5] == pytest.approx(2.0)
        assert mean[100, 358] == pytest.approx(5.0)
        assert counts.sum() == 3  # NaN latitude dropped

    def test_ignores_nan_values(self):
        mean, counts = bin_to_grid(
            np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([np.nan, 2.0])
        )
        assert counts[90, 0] == 1
        assert mean[90, 0] == pytest.approx(2.0)


class TestAlongTrackHilatComparator:
    def test_same_field_agrees(self, ref_hilat_dss, tmp_path):
        path_a, path_b = _write_pair(tmp_path, *ref_hilat_dss)
        report = AlongTrackHilatComparator(path_a, path_b, bin_size=2.0).run()
        g = report.quality_summary["ssha_gridded"]
        assert g["n_common_cells"] > 0
        assert g["cells_b"] > g["n_common_cells"]  # hi-lat extends beyond ref
        assert abs(g["bias"]) < 0.01
        assert g["pearson_r"] > 0.9
        assert g["pct_within_threshold"] > 95.0

    def test_no_pointwise_diffs(self, ref_hilat_dss, tmp_path):
        path_a, path_b = _write_pair(tmp_path, *ref_hilat_dss)
        report = AlongTrackHilatComparator(path_a, path_b).run()
        assert all(vc.diff is None for vc in report.variable_comparisons)

    def test_offset_detected(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_b = ds_b.copy(deep=True)
        ds_b["ssha"] = ds_b["ssha"] + 0.2
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b)
        report = AlongTrackHilatComparator(path_a, path_b, bin_size=2.0).run()
        g = report.quality_summary["ssha_gridded"]
        assert g["bias"] == pytest.approx(0.2, abs=0.01)
        assert g["pct_within_threshold"] == 0.0

    def test_flagged_points_excluded(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_b = ds_b.copy(deep=True)
        ds_b["nasa_flag"].values[:] = 1
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b)
        g = AlongTrackHilatComparator(path_a, path_b).run().quality_summary["ssha_gridded"]
        assert g["cells_b"] == 0
        assert g["n_common_cells"] == 0
        assert g["rmsd"] is None

    def test_latitude_range(self, ref_hilat_dss, tmp_path):
        path_a, path_b = _write_pair(tmp_path, *ref_hilat_dss)
        lr = AlongTrackHilatComparator(path_a, path_b).run().quality_summary["latitude_range"]
        assert lr["a"]["max"] <= 66.0
        assert lr["b"]["max"] > 66.0

    def test_empty_hilat_file(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b.isel(time=slice(0, 0)))
        q = AlongTrackHilatComparator(path_a, path_b).run().quality_summary
        assert q["latitude_range"]["b"] is None
        assert q["ssha_gridded"]["n_common_cells"] == 0


class TestBulkHilat:
    def test_matches_by_date_across_patterns(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        dir_a = tmp_path / "ref"
        dir_b = tmp_path / "hilat"
        dir_a.mkdir()
        dir_b.mkdir()
        for date in ["20201023", "20201024"]:
            ds_a.to_netcdf(dir_a / _REF_NAME.format(date=date))
            ds_b.to_netcdf(dir_b / _HILAT_NAME.format(date=date))
        ds_b.to_netcdf(dir_b / _HILAT_NAME.format(date="20201025"))

        report = run_bulk_comparison(dir_a, dir_b, product_type="along_track_hilat")
        assert report.product_type == "along_track_hilat"
        assert [r.date_key for r in report.matched_pairs] == ["20201023", "20201024"]
        assert report.only_in_b == [_HILAT_NAME.format(date="20201025")]
        for r in report.matched_pairs:
            assert r.error is None
            assert r.n_common_cells > 0
            assert r.ssha_rmsd is not None
            assert r.ssha_pct_within_threshold is not None


def _with_passes(ds, n_passes=40):
    """Give a dataset alternating ascending/descending passes along its points."""
    n = ds.sizes["time"]
    per = n // n_passes
    lat = np.empty(n)
    passes = np.empty(n, dtype=np.int32)
    for p in range(n_passes):
        s, e = p * per, (p + 1) * per if p < n_passes - 1 else n
        seg = np.linspace(-60, 60, e - s)
        lat[s:e] = seg if p % 2 == 0 else seg[::-1]
        passes[s:e] = p
    lon = np.linspace(0, 360 * 7, n) % 360
    ds = ds.assign_coords(latitude=("time", lat.astype(np.float32)),
                          longitude=("time", lon.astype(np.float32)))
    ds["pass"] = ("time", passes)
    ds["ssha"] = ("time", _field(lat, lon))
    return ds


class TestHilatVariabilityAndNoise:
    def test_std_ratio_detects_scaling(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_b = ds_b.copy(deep=True)
        ds_b["ssha"] = ds_b["ssha"] * 2.0
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b)
        g = AlongTrackHilatComparator(path_a, path_b, bin_size=2.0).run().quality_summary[
            "ssha_gridded"
        ]
        assert g["std_ratio"] == pytest.approx(2.0, rel=0.05)

    def test_noise_rms(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_a = ds_a.assign(ssha_smoothed=ds_a["ssha"] - 0.02)
        ds_b = ds_b.assign(ssha_smoothed=ds_b["ssha"] - 0.05)
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b)
        noise = AlongTrackHilatComparator(path_a, path_b).run().quality_summary["noise"]
        assert noise["a"]["rms"] == pytest.approx(0.02, abs=1e-6)
        assert noise["b"]["rms"] == pytest.approx(0.05, abs=1e-6)

    def test_noise_missing_smoothed(self, ref_hilat_dss, tmp_path):
        path_a, path_b = _write_pair(tmp_path, *ref_hilat_dss)
        noise = AlongTrackHilatComparator(path_a, path_b).run().quality_summary["noise"]
        assert noise == {"a": None, "b": None}

    def test_self_baseline_asc_vs_desc(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_a = _with_passes(ds_a)
        ds_b = _with_passes(ds_b)
        # Offset descending passes of B by 3 cm; A stays self-consistent.
        desc = (ds_b["pass"].values % 2) == 1
        ds_b["ssha"].values[desc] += 0.03
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b)
        base = AlongTrackHilatComparator(path_a, path_b, bin_size=2.0).run().quality_summary[
            "self_baseline"
        ]
        assert base["a"]["n_common_cells"] > 0
        assert base["a"]["rmsd"] < 0.01
        assert base["b"]["bias"] == pytest.approx(0.03, abs=0.005)


class TestLatBands:
    def test_bands_cover_offset(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_b = ds_b.copy(deep=True)
        north = ds_b["latitude"].values > 40
        ds_b["ssha"].values[north] += 0.1
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b)
        bands = AlongTrackHilatComparator(path_a, path_b, bin_size=2.0).run().quality_summary[
            "lat_bands"
        ]
        by_band = {(b["lat_min"], b["lat_max"]): summarise_lat_band(b) for b in bands}
        assert by_band[(50, 60)]["bias"] == pytest.approx(0.1, abs=0.01)
        assert abs(by_band[(0, 10)]["bias"]) < 0.01
        assert by_band[(70, 80)]["n_common_cells"] == 0  # beyond reference coverage

    def test_pooling_sums_across_files(self):
        def band(n, sum_diff):
            b = {"lat_min": 0, "lat_max": 10, **dict.fromkeys(LAT_BAND_SUM_KEYS, 0)}
            b.update(n_common_cells=n, sum_diff=sum_diff, sum_diff_sq=sum_diff**2 / n)
            return b

        pooled = pool_lat_bands([[band(10, 1.0)], [band(30, -1.0)]])
        assert len(pooled) == 1
        summary = summarise_lat_band(pooled[0])
        assert summary["n_common_cells"] == 40
        assert summary["bias"] == pytest.approx(0.0)


class TestBulkHilatReporting:
    @pytest.fixture
    def bulk_report(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_a = ds_a.assign(ssha_smoothed=ds_a["ssha"])
        ds_b = ds_b.assign(ssha_smoothed=ds_b["ssha"])
        dir_a = tmp_path / "ref"
        dir_b = tmp_path / "hilat"
        dir_a.mkdir()
        dir_b.mkdir()
        for date in ["20201023", "20201024"]:
            ds_a.to_netcdf(dir_a / _REF_NAME.format(date=date))
            ds_b.to_netcdf(dir_b / _HILAT_NAME.format(date=date))
        # A day where the hi-lat file is fully flagged: no common cells.
        flagged = ds_b.copy(deep=True)
        flagged["nasa_flag"].values[:] = 1
        ds_a.to_netcdf(dir_a / _REF_NAME.format(date="20201025"))
        flagged.to_netcdf(dir_b / _HILAT_NAME.format(date="20201025"))
        return run_bulk_comparison(dir_a, dir_b, product_type="along_track_hilat")

    def test_pair_metrics_populated(self, bulk_report):
        r = bulk_report.matched_pairs[0]
        assert r.cells_a > 0 and r.cells_b > 0
        assert r.ssha_std_ratio is not None
        assert r.noise_a == 0.0
        assert r.lat_bands

    def test_pooled_bands_double_single_day(self, bulk_report):
        day = {(b["lat_min"], b["lat_max"]): b for b in bulk_report.matched_pairs[0].lat_bands}
        for b in bulk_report.lat_bands:
            key = (b["lat_min"], b["lat_max"])
            assert b["n_common_cells"] == 2 * day[key]["n_common_cells"]

    def test_report_text(self, bulk_report):
        text = format_bulk_report(bulk_report)
        assert "Latitude Bands (pooled over all days)" in text
        assert "Self-baseline RMSD A" in text
        assert "Days with no common cells (1):" in text
        assert "20201025  cells A=" in text

    def test_plot(self, bulk_report, tmp_path):
        out = tmp_path / "hilat.png"
        plot_bulk_timeseries(bulk_report, output_path=out)
        assert out.stat().st_size > 0


def _spike_common_cell(ds_a, ds_b):
    """Set B to 4 m in one 2-degree cell (lat 4..6) that A also observes."""
    ds_b = ds_b.copy(deep=True)
    lat_a, lon_a = ds_a["latitude"].values, ds_a["longitude"].values
    lon0 = 2 * int(lon_a[(lat_a >= 4) & (lat_a < 6)][0] // 2)
    lat, lon = ds_b["latitude"].values, ds_b["longitude"].values
    spike = (lat >= 4) & (lat < 6) & (lon >= lon0) & (lon < lon0 + 2)
    assert spike.any()
    ds_b["ssha"].values[spike] = 4.0
    return ds_b


class TestOutlierExclusion:
    def test_outlier_cell_excluded_from_rmsd(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_b = _spike_common_cell(ds_a, ds_b)
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b)
        g = AlongTrackHilatComparator(path_a, path_b, bin_size=2.0).run().quality_summary[
            "ssha_gridded"
        ]
        assert g["n_outlier_cells"] == 1
        assert g["max_abs_diff"] > 3.5
        assert g["max_abs_diff_lat"] == pytest.approx(5.0)
        assert g["rmsd_all"] > 0.05
        assert g["rmsd"] < 0.02  # clean RMSD unaffected by the spike
        assert g["median_abs_diff"] < 0.02
        assert g["robust_std_diff"] < 0.02

    def test_outliers_excluded_from_lat_bands(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_b = _spike_common_cell(ds_a, ds_b)
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b)
        bands = AlongTrackHilatComparator(path_a, path_b, bin_size=2.0).run().quality_summary[
            "lat_bands"
        ]
        band = summarise_lat_band(next(b for b in bands if b["lat_min"] == 0))
        assert band["n_outlier_cells"] == 1
        assert band["rmsd"] < 0.02
        assert band["std_ratio"] == pytest.approx(1.0, abs=0.1)


class TestAlignmentCheck:
    def _alignment(self, tmp_path, ds_a, ds_b, **kwargs):
        path_a, path_b = _write_pair(tmp_path, _with_passes(ds_a), _with_passes(ds_b))
        return AlongTrackHilatComparator(path_a, path_b, bin_size=2.0, **kwargs).run()

    def test_pass(self, ref_hilat_dss, tmp_path):
        report = self._alignment(tmp_path, *ref_hilat_dss)
        a = report.quality_summary["alignment"]
        assert a["status"] == "PASS"
        assert a["reasons"] == []

    def test_bias_fails(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_b = _with_passes(ds_b)
        ds_b["ssha"] = ds_b["ssha"] + 0.05
        path_a, path_b = _write_pair(tmp_path, _with_passes(ds_a), ds_b)
        a = AlongTrackHilatComparator(path_a, path_b, bin_size=2.0).run().quality_summary[
            "alignment"
        ]
        assert a["status"] == "FAIL"
        assert any("bias" in r for r in a["reasons"])

    def test_bias_within_custom_limit_passes(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_b = _with_passes(ds_b)
        ds_b["ssha"] = ds_b["ssha"] + 0.05
        path_a, path_b = _write_pair(tmp_path, _with_passes(ds_a), ds_b)
        a = AlongTrackHilatComparator(
            path_a, path_b, bin_size=2.0, max_bias=0.1, rmsd_factor=100
        ).run().quality_summary["alignment"]
        assert a["status"] == "PASS"

    def test_rmsd_vs_baseline_fails(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_a = _with_passes(ds_a)
        ds_b = _with_passes(ds_b)
        # Zero-mean spatial pattern in B only. It is the same on ascending and
        # descending passes, so it cannot appear in either file's baseline,
        # but it raises A-vs-B RMSD. (Random per-point noise would also raise
        # B's own baseline and is caught by the noise/std metrics instead.)
        lon_b = ds_b["longitude"].values.astype(np.float64)
        ds_b["ssha"] = ds_b["ssha"] + 0.1 * np.sin(np.radians(lon_b) * 6)
        rng = np.random.default_rng(1)
        for ds in (ds_a, ds_b):
            ds["ssha"] = ds["ssha"] + rng.normal(0, 0.01, ds.sizes["time"])
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b)
        a = AlongTrackHilatComparator(
            path_a, path_b, bin_size=2.0, outlier_cutoff=10.0
        ).run().quality_summary["alignment"]
        assert a["baseline_rmsd"] is not None
        assert a["matched_rmsd"] > a["rmsd_limit"]
        assert any("baseline" in r for r in a["reasons"])

    def test_outlier_cells_fail(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        ds_b = _with_passes(ds_b)
        ds_b["ssha"].values[:5] = 4.0
        path_a, path_b = _write_pair(tmp_path, _with_passes(ds_a), ds_b)
        a = AlongTrackHilatComparator(path_a, path_b, bin_size=2.0).run().quality_summary[
            "alignment"
        ]
        assert a["status"] == "FAIL"
        assert any("cell(s)" in r for r in a["reasons"])

    def test_too_few_cells_is_na(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = ref_hilat_dss
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b.isel(time=slice(0, 5)))
        a = AlongTrackHilatComparator(path_a, path_b).run().quality_summary["alignment"]
        assert a["status"] == "N/A"

    def test_cli_exit_codes(self, ref_hilat_dss, tmp_path):
        from validation.cli import main

        ds_a, ds_b = (_with_passes(d) for d in ref_hilat_dss)
        path_a, path_b = _write_pair(tmp_path, ds_a, ds_b)
        assert main([path_a, path_b, "-t", "along_track_hilat"]) == 0
        ds_b["ssha"] = ds_b["ssha"] + 0.05
        ds_b.to_netcdf(path_b)
        assert main([path_a, path_b, "-t", "along_track_hilat"]) == 1
        assert main([path_a, path_b, "-t", "along_track_hilat", "--max-bias", "0.1",
                     "--rmsd-factor", "100"]) == 0

    def test_bulk_status_and_report(self, ref_hilat_dss, tmp_path):
        ds_a, ds_b = (_with_passes(d) for d in ref_hilat_dss)
        dir_a = tmp_path / "ref"
        dir_b = tmp_path / "hilat"
        dir_a.mkdir()
        dir_b.mkdir()
        ds_a.to_netcdf(dir_a / _REF_NAME.format(date="20201023"))
        ds_b.to_netcdf(dir_b / _HILAT_NAME.format(date="20201023"))
        ds_a.to_netcdf(dir_a / _REF_NAME.format(date="20201024"))
        (ds_b + 0).assign(ssha=ds_b["ssha"] + 0.05).to_netcdf(
            dir_b / _HILAT_NAME.format(date="20201024")
        )

        report = run_bulk_comparison(dir_a, dir_b, product_type="along_track_hilat")
        assert [r.alignment_status for r in report.matched_pairs] == ["PASS", "FAIL"]
        assert report.criteria["max_bias"] == 0.02
        text = format_bulk_report(report)
        assert "Days failing the alignment check (1):" in text
        assert "Days passing alignment check: 1/2 judged" in text
