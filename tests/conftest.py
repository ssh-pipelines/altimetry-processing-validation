"""Pytest fixtures: synthetic NetCDF datasets for testing."""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def along_track_ds():
    """Create a synthetic along-track dataset."""
    n = 100
    time = np.arange(n, dtype=np.float64)
    return xr.Dataset(
        {
            "ssha": (["time"], np.random.randn(n).astype(np.float64)),
            "ssha_smoothed": (["time"], np.random.randn(n).astype(np.float64)),
            "dac": (["time"], np.random.randn(n).astype(np.float64)),
            "cycle": (["time"], np.full(n, 42, dtype=np.int32)),
            "pass": (["time"], np.arange(n, dtype=np.int32)),
            "nasa_flag": (["time"], np.zeros(n, dtype=np.int8)),
            "source_flag": (
                ["time", "src_flag_dim"],
                np.zeros((n, 3), dtype=np.int8),
            ),
            "median_filter_flag": (["time"], np.zeros(n, dtype=np.int8)),
            "oer": (["time"], np.random.randn(n).astype(np.float64)),
            "basin_flag": (
                ["time", "basins"],
                np.zeros((n, 5), dtype=np.int32),
            ),
        },
        coords={"time": time},
        attrs={
            "title": "Along-track test",
            "source": "synthetic",
            "date_created": "2025-01-01",
        },
    )


@pytest.fixture
def simple_grid_ds():
    """Create a synthetic simple-grid dataset."""
    lat = np.arange(-90, 90, 1.0, dtype=np.float32)
    lon = np.arange(0, 360, 1.0, dtype=np.float32)
    nlat, nlon = len(lat), len(lon)
    return xr.Dataset(
        {
            "ssha": (
                ["latitude", "longitude"],
                np.random.randn(nlat, nlon).astype(np.float64),
            ),
            "counts": (
                ["latitude", "longitude"],
                np.random.randint(0, 50, size=(nlat, nlon)).astype(np.int32),
            ),
            "basin_flag": (
                ["latitude", "longitude", "basins"],
                np.zeros((nlat, nlon, 5), dtype=np.int32),
            ),
        },
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "title": "Simple grid test",
            "source": "synthetic",
            "date_created": "2025-01-01",
        },
    )


@pytest.fixture
def along_track_pair(along_track_ds, tmp_path):
    """Write two identical along-track files and return their paths."""
    path_a = tmp_path / "at_a.nc"
    path_b = tmp_path / "at_b.nc"
    along_track_ds.to_netcdf(path_a)
    along_track_ds.to_netcdf(path_b)
    return str(path_a), str(path_b)


@pytest.fixture
def simple_grid_pair(simple_grid_ds, tmp_path):
    """Write two identical simple-grid files and return their paths."""
    path_a = tmp_path / "sg_a.nc"
    path_b = tmp_path / "sg_b.nc"
    simple_grid_ds.to_netcdf(path_a)
    simple_grid_ds.to_netcdf(path_b)
    return str(path_a), str(path_b)
