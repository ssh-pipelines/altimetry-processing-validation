"""Comparator for reference-mission vs high-latitude along-track files.

Reference and high-latitude files share the along-track layout but are
sampled along different orbits, so their time indices do not correspond.
Instead of element-wise diffs, flag-good SSHA from each file is averaged
onto a regular lat/lon grid and the cells observed by both files are
compared.

Alongside the cross-file diffs, the comparator reports:

- variability of each file's gridded SSHA over the common cells,
- small-scale noise (``ssha - ssha_smoothed``) for each file,
- a self-consistency baseline: ascending vs descending passes of the *same*
  file binned onto the same grid. This approximates the diff level expected
  purely from gridding different tracks at different times, against which
  the cross-file RMSD can be judged,
- per-latitude-band sums that can be pooled across many files.
"""

import numpy as np
import xarray as xr

from validation.analysis.binning import bin_to_grid
from validation.comparators.along_track import AlongTrackComparator

DEFAULT_LAT_BAND_EDGES = tuple(range(-90, 91, 10))

# Raw per-band accumulators. Summed across files to pool latitude-band stats.
LAT_BAND_SUM_KEYS = (
    "n_common_cells",
    "sum_diff",
    "sum_diff_sq",
    "sum_a",
    "sum_a_sq",
    "sum_b",
    "sum_b_sq",
    "noise_n_a",
    "noise_sum_sq_a",
    "noise_n_b",
    "noise_sum_sq_b",
)


def _good_ssha(ds: xr.Dataset, var: str = "ssha") -> np.ndarray:
    """Return *var* as float64 with nasa_flag != 0 points set to NaN."""
    data = ds[var].values.astype(np.float64)
    if "nasa_flag" in ds.variables:
        data = np.where(ds["nasa_flag"].values == 0, data, np.nan)
    return data


def _pass_direction(ds: xr.Dataset) -> np.ndarray | None:
    """Return +1 (ascending) / -1 (descending) / 0 per point, or None.

    Points are grouped into contiguous runs of constant (cycle, pass) and each
    run's direction is the sign of its end-minus-start latitude.
    """
    if not {"pass", "latitude"} <= set(ds.variables):
        return None
    lat = ds["latitude"].values.astype(np.float64)
    if lat.size == 0:
        return np.zeros(0)
    key = ds["pass"].values.astype(np.float64)
    if "cycle" in ds.variables:
        key = key + ds["cycle"].values.astype(np.float64) * 1e4
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], lat.size]
    direction = np.zeros(lat.size)
    for s, e in zip(starts, ends):
        seg = lat[s:e][np.isfinite(lat[s:e])]
        if seg.size > 1:
            direction[s:e] = np.sign(seg[-1] - seg[0])
    return direction


def _diff_stats(av: np.ndarray, bv: np.ndarray, threshold: float) -> dict:
    """Diff metrics between paired 1-D arrays (B - A)."""
    diff = bv - av
    abs_diff = np.abs(diff)
    pearson_r = None
    if av.size > 1:
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(av, bv)[0, 1]
        pearson_r = float(corr) if np.isfinite(corr) else None
    return {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "rmsd": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "pearson_r": pearson_r,
        "pct_within_threshold": round(float(np.mean(abs_diff <= threshold) * 100), 2),
    }


def _std_or_none(sum_x: float, sum_x_sq: float, n: float) -> float | None:
    if n < 2:
        return None
    var = sum_x_sq / n - (sum_x / n) ** 2
    return float(np.sqrt(max(var, 0.0)))


def summarise_lat_band(band: dict) -> dict:
    """Derive bias/RMSD/variability/noise stats from a band's raw sums."""
    n = band["n_common_cells"]
    std_a = _std_or_none(band["sum_a"], band["sum_a_sq"], n)
    std_b = _std_or_none(band["sum_b"], band["sum_b_sq"], n)
    return {
        "lat_min": band["lat_min"],
        "lat_max": band["lat_max"],
        "n_common_cells": int(n),
        "bias": band["sum_diff"] / n if n else None,
        "rmsd": float(np.sqrt(band["sum_diff_sq"] / n)) if n else None,
        "std_a": std_a,
        "std_b": std_b,
        "std_ratio": std_b / std_a if std_a and std_b is not None else None,
        "noise_a": (
            float(np.sqrt(band["noise_sum_sq_a"] / band["noise_n_a"]))
            if band["noise_n_a"]
            else None
        ),
        "noise_b": (
            float(np.sqrt(band["noise_sum_sq_b"] / band["noise_n_b"]))
            if band["noise_n_b"]
            else None
        ),
    }


def pool_lat_bands(per_file_bands: list[list[dict]]) -> list[dict]:
    """Sum raw latitude-band accumulators across files (same band edges)."""
    pooled: dict[tuple, dict] = {}
    for bands in per_file_bands:
        for band in bands:
            key = (band["lat_min"], band["lat_max"])
            acc = pooled.setdefault(
                key, {"lat_min": key[0], "lat_max": key[1], **dict.fromkeys(LAT_BAND_SUM_KEYS, 0)}
            )
            for k in LAT_BAND_SUM_KEYS:
                acc[k] += band[k]
    return [pooled[k] for k in sorted(pooled)]


class AlongTrackHilatComparator(AlongTrackComparator):
    """Comparator for a reference along-track file vs a high-latitude one."""

    POINTWISE_DIFFS = False

    def __init__(
        self,
        file_a: str,
        file_b: str,
        threshold: float = 0.05,
        bin_size: float = 1.0,
        lat_band_edges: tuple[float, ...] = DEFAULT_LAT_BAND_EDGES,
    ):
        super().__init__(file_a, file_b, threshold=threshold)
        self.bin_size = bin_size
        self.lat_band_edges = tuple(lat_band_edges)

    @property
    def product_type(self) -> str:
        return "along_track_hilat"

    def compare_quality(self, ds_a: xr.Dataset, ds_b: xr.Dataset) -> dict:
        """Flag/percentile summary plus gridded diffs, variability, and noise."""
        summary = super().compare_quality(ds_a, ds_b)

        for label, ds in [("a", ds_a), ("b", ds_b)]:
            lat = ds["latitude"].values if "latitude" in ds.variables else np.array([])
            lat = lat[np.isfinite(lat)]
            summary.setdefault("latitude_range", {})[label] = (
                {"min": round(float(lat.min()), 4), "max": round(float(lat.max()), 4)}
                if lat.size > 0
                else None
            )

        grid_a = self._grid(ds_a)
        grid_b = self._grid(ds_b)
        residual_a = self._residual(ds_a)
        residual_b = self._residual(ds_b)

        summary["ssha_gridded"] = self._compare_grids(grid_a, grid_b)
        summary["noise"] = {
            label: self._noise_summary(res)
            for label, res in [("a", residual_a), ("b", residual_b)]
        }
        summary["self_baseline"] = {
            label: self._self_baseline(ds) for label, ds in [("a", ds_a), ("b", ds_b)]
        }
        summary["lat_bands"] = self._lat_band_sums(
            grid_a, grid_b, ds_a, ds_b, residual_a, residual_b
        )
        return summary

    # --- helpers ---

    def _has_track(self, ds: xr.Dataset) -> bool:
        return {"ssha", "latitude", "longitude"} <= set(ds.variables)

    def _grid(self, ds: xr.Dataset, mask: np.ndarray | None = None) -> np.ndarray | None:
        """Flag-good SSHA binned to the comparison grid, or None if unavailable."""
        if not self._has_track(ds):
            return None
        ssha = _good_ssha(ds)
        if mask is not None:
            ssha = np.where(mask, ssha, np.nan)
        mean, _ = bin_to_grid(
            ds["latitude"].values, ds["longitude"].values, ssha, self.bin_size
        )
        return mean

    def _residual(self, ds: xr.Dataset) -> np.ndarray | None:
        """Small-scale residual ssha - ssha_smoothed at flag-good points."""
        if "ssha" not in ds.variables or "ssha_smoothed" not in ds.variables:
            return None
        return _good_ssha(ds) - ds["ssha_smoothed"].values.astype(np.float64)

    @staticmethod
    def _noise_summary(residual: np.ndarray | None) -> dict | None:
        if residual is None:
            return None
        valid = residual[np.isfinite(residual)]
        return {
            "n_points": int(valid.size),
            "rms": round(float(np.sqrt(np.mean(valid**2))), 6) if valid.size else None,
        }

    def _compare_grids(self, grid_a: np.ndarray | None, grid_b: np.ndarray | None) -> dict:
        """Diff metrics and variability over the cells observed by both grids."""
        result = {
            "bin_size_deg": self.bin_size,
            "threshold_m": self.threshold,
            "cells_a": None,
            "cells_b": None,
            "n_common_cells": 0,
            "max_abs_diff": None,
            "max_abs_diff_lat": None,
            "max_abs_diff_lon": None,
            "mean_abs_diff": None,
            "rmsd": None,
            "bias": None,
            "pearson_r": None,
            "pct_within_threshold": None,
            "std_a": None,
            "std_b": None,
            "std_ratio": None,
        }
        if grid_a is None or grid_b is None:
            return result

        both = np.isfinite(grid_a) & np.isfinite(grid_b)
        result["cells_a"] = int(np.isfinite(grid_a).sum())
        result["cells_b"] = int(np.isfinite(grid_b).sum())
        result["n_common_cells"] = int(both.sum())
        if not np.any(both):
            return result

        av = grid_a[both]
        bv = grid_b[both]
        result.update(_diff_stats(av, bv, self.threshold))

        abs_diff = np.where(both, np.abs(grid_b - grid_a), -np.inf)
        i, j = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        result["max_abs_diff_lat"] = -90.0 + (i + 0.5) * self.bin_size
        result["max_abs_diff_lon"] = (j + 0.5) * self.bin_size

        if av.size > 1:
            result["std_a"] = float(np.std(av))
            result["std_b"] = float(np.std(bv))
            if result["std_a"] > 0:
                result["std_ratio"] = result["std_b"] / result["std_a"]
        return result

    def _self_baseline(self, ds: xr.Dataset) -> dict | None:
        """Ascending vs descending passes of one file on the comparison grid."""
        direction = _pass_direction(ds) if self._has_track(ds) else None
        if direction is None:
            return None
        grid_asc = self._grid(ds, mask=direction > 0)
        grid_desc = self._grid(ds, mask=direction < 0)
        both = np.isfinite(grid_asc) & np.isfinite(grid_desc)
        result = {"n_common_cells": int(both.sum()), "rmsd": None, "bias": None}
        if np.any(both):
            stats = _diff_stats(grid_asc[both], grid_desc[both], self.threshold)
            result["rmsd"] = round(stats["rmsd"], 6)
            result["bias"] = round(stats["bias"], 6)
        return result

    def _lat_band_sums(
        self,
        grid_a: np.ndarray | None,
        grid_b: np.ndarray | None,
        ds_a: xr.Dataset,
        ds_b: xr.Dataset,
        residual_a: np.ndarray | None,
        residual_b: np.ndarray | None,
    ) -> list[dict]:
        """Raw per-latitude-band accumulators for pooling across files."""
        edges = self.lat_band_edges
        bands = [
            {"lat_min": lo, "lat_max": hi, **dict.fromkeys(LAT_BAND_SUM_KEYS, 0)}
            for lo, hi in zip(edges[:-1], edges[1:])
        ]

        if grid_a is not None and grid_b is not None:
            row_lat = -90.0 + (np.arange(grid_a.shape[0]) + 0.5) * self.bin_size
            cell_lat = np.broadcast_to(row_lat[:, None], grid_a.shape)
            both = np.isfinite(grid_a) & np.isfinite(grid_b)
            lat_c, a_c, b_c = cell_lat[both], grid_a[both], grid_b[both]
            d_c = b_c - a_c
            for band in bands:
                sel = (lat_c >= band["lat_min"]) & (lat_c < band["lat_max"])
                band["n_common_cells"] = int(sel.sum())
                band["sum_diff"] = float(d_c[sel].sum())
                band["sum_diff_sq"] = float((d_c[sel] ** 2).sum())
                band["sum_a"] = float(a_c[sel].sum())
                band["sum_a_sq"] = float((a_c[sel] ** 2).sum())
                band["sum_b"] = float(b_c[sel].sum())
                band["sum_b_sq"] = float((b_c[sel] ** 2).sum())

        for label, ds, residual in [("a", ds_a, residual_a), ("b", ds_b, residual_b)]:
            if residual is None or "latitude" not in ds.variables:
                continue
            lat = ds["latitude"].values.astype(np.float64)
            ok = np.isfinite(residual) & np.isfinite(lat)
            for band in bands:
                sel = ok & (lat >= band["lat_min"]) & (lat < band["lat_max"])
                band[f"noise_n_{label}"] = int(sel.sum())
                band[f"noise_sum_sq_{label}"] = float((residual[sel] ** 2).sum())

        return bands
