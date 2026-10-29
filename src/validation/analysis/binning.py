"""Spatial binning of along-track points onto a regular lat/lon grid."""

import numpy as np


def bin_to_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    bin_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Average point values into regular lat/lon cells.

    Points with a non-finite lat, lon, or value are ignored. Longitudes are
    wrapped to [0, 360).

    Returns
    -------
    mean : np.ndarray
        2-D (nlat, nlon) array of per-cell mean values, NaN where empty.
    counts : np.ndarray
        2-D (nlat, nlon) array of the number of points in each cell.
    """
    nlat = int(np.ceil(180.0 / bin_size))
    nlon = int(np.ceil(360.0 / bin_size))

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(values)
    lat_idx = np.clip(((lat[ok] + 90.0) // bin_size).astype(int), 0, nlat - 1)
    lon_idx = np.clip(((lon[ok] % 360.0) // bin_size).astype(int), 0, nlon - 1)
    flat_idx = lat_idx * nlon + lon_idx

    sums = np.bincount(flat_idx, weights=values[ok], minlength=nlat * nlon)
    counts = np.bincount(flat_idx, minlength=nlat * nlon)

    mean = np.full(nlat * nlon, np.nan)
    filled = counts > 0
    mean[filled] = sums[filled] / counts[filled]

    return mean.reshape(nlat, nlon), counts.reshape(nlat, nlon)
