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
- per-latitude-band sums that can be pooled across many files,
- an alignment check (PASS/FAIL/N/A) judging bias and RMSD against the
  self-consistency baseline rather than a fixed agreement threshold. Because
  each baseline cell mean uses only one pass direction (about half the
  points), the RMSD judged against it is the *matched* cross RMSD: A's
  ascending/descending half-grids vs B's, all four pairings pooled, so
  measurement noise enters both statistics equally.

Common cells whose |B - A| exceeds ``outlier_cutoff`` are counted and excluded
from bias/RMSD/std (and from the baseline and latitude-band sums), so a single
bad cell cannot dominate them. Max abs diff and its location are still taken
over all cells so outliers can be traced.
"""

import numpy as np
import xarray as xr

from validation.analysis.binning import bin_to_grid
from validation.comparators.along_track import AlongTrackComparator

DEFAULT_LAT_BAND_EDGES = tuple(range(-90, 91, 10))

# Defaults for outlier exclusion and the per-pair alignment check.
HILAT_DEFAULT_CRITERIA = {
    "outlier_cutoff": 1.0,  # m; cells with |B - A| above this are outliers
    "max_bias": 0.02,  # m; FAIL if |bias| exceeds this
    "rmsd_factor": 1.2,  # FAIL if RMSD > factor x baseline RMSD
    "min_common_cells": 50,  # N/A below this many common cells
    "max_outlier_cells": 0,  # FAIL if more outlier cells than this
}

# 1.4826 * MAD estimates the standard deviation for normally distributed data.
_MAD_TO_STD = 1.4826

# Raw per-band accumulators. Summed across files to pool latitude-band stats.
LAT_BAND_SUM_KEYS = (
    "n_common_cells",
    "n_outlier_cells",
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
        "n_outlier_cells": int(band["n_outlier_cells"]),
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
        outlier_cutoff: float = HILAT_DEFAULT_CRITERIA["outlier_cutoff"],
        max_bias: float = HILAT_DEFAULT_CRITERIA["max_bias"],
        rmsd_factor: float = HILAT_DEFAULT_CRITERIA["rmsd_factor"],
        min_common_cells: int = HILAT_DEFAULT_CRITERIA["min_common_cells"],
        max_outlier_cells: int = HILAT_DEFAULT_CRITERIA["max_outlier_cells"],
    ):
        super().__init__(file_a, file_b, threshold=threshold)
        self.bin_size = bin_size
        self.lat_band_edges = tuple(lat_band_edges)
        self.outlier_cutoff = outlier_cutoff
        self.max_bias = max_bias
        self.rmsd_factor = rmsd_factor
        self.min_common_cells = min_common_cells
        self.max_outlier_cells = max_outlier_cells

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
        summary["alignment"] = self._alignment(
            summary["ssha_gridded"],
            summary["self_baseline"],
            self._matched_cross_rmsd(self._half_grids(ds_a), self._half_grids(ds_b)),
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
            "outlier_cutoff_m": self.outlier_cutoff,
            "n_outlier_cells": 0,
            "median_abs_diff": None,
            "robust_std_diff": None,
            "rmsd_all": None,
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
        diff = bv - av
        abs_diff = np.abs(diff)

        # Outlier-insensitive diff spread, and stats over all cells for tracing.
        result["median_abs_diff"] = float(np.median(abs_diff))
        result["robust_std_diff"] = float(
            _MAD_TO_STD * np.median(np.abs(diff - np.median(diff)))
        )
        result["rmsd_all"] = float(np.sqrt(np.mean(diff**2)))
        result["max_abs_diff"] = float(np.max(abs_diff))
        result["pct_within_threshold"] = round(
            float(np.mean(abs_diff <= self.threshold) * 100), 2
        )
        abs_grid = np.where(both, np.abs(grid_b - grid_a), -np.inf)
        i, j = np.unravel_index(np.argmax(abs_grid), abs_grid.shape)
        result["max_abs_diff_lat"] = -90.0 + (i + 0.5) * self.bin_size
        result["max_abs_diff_lon"] = (j + 0.5) * self.bin_size

        # Bias/RMSD/std exclude outlier cells.
        keep = abs_diff <= self.outlier_cutoff
        result["n_outlier_cells"] = int((~keep).sum())
        if not np.any(keep):
            return result
        av, bv = av[keep], bv[keep]
        clean = _diff_stats(av, bv, self.threshold)
        for key in ("mean_abs_diff", "rmsd", "bias", "pearson_r"):
            result[key] = clean[key]

        if av.size > 1:
            result["std_a"] = float(np.std(av))
            result["std_b"] = float(np.std(bv))
            if result["std_a"] > 0:
                result["std_ratio"] = result["std_b"] / result["std_a"]
        return result

    def _half_grids(self, ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray] | None:
        """(ascending, descending) half-grids of one file, or None."""
        direction = _pass_direction(ds) if self._has_track(ds) else None
        if direction is None:
            return None
        return self._grid(ds, mask=direction > 0), self._grid(ds, mask=direction < 0)

    def _paired_diffs(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        """y - x over cells observed by both, excluding outlier cells."""
        both = np.isfinite(grid_x) & np.isfinite(grid_y)
        diff = grid_y[both] - grid_x[both]
        return diff[np.abs(diff) <= self.outlier_cutoff]

    def _self_baseline(self, ds: xr.Dataset) -> dict | None:
        """Ascending vs descending passes of one file on the comparison grid."""
        halves = self._half_grids(ds)
        if halves is None:
            return None
        diff = self._paired_diffs(*halves)
        result = {"n_common_cells": int(diff.size), "rmsd": None, "bias": None}
        if diff.size:
            result["rmsd"] = round(float(np.sqrt(np.mean(diff**2))), 6)
            result["bias"] = round(float(np.mean(diff)), 6)
        return result

    def _matched_cross_rmsd(
        self,
        halves_a: tuple[np.ndarray, np.ndarray] | None,
        halves_b: tuple[np.ndarray, np.ndarray] | None,
    ) -> float | None:
        """RMSD of A vs B half-grids, all four direction pairings pooled.

        Each half-grid cell mean has the same sampling as the self-baseline's,
        so this is the cross-file counterpart to compare against it.
        """
        if halves_a is None or halves_b is None:
            return None
        diff = np.concatenate(
            [self._paired_diffs(ga, gb) for ga in halves_a for gb in halves_b]
        )
        return float(np.sqrt(np.mean(diff**2))) if diff.size else None

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
            outlier = np.abs(d_c) > self.outlier_cutoff
            for band in bands:
                in_band = (lat_c >= band["lat_min"]) & (lat_c < band["lat_max"])
                band["n_outlier_cells"] = int((in_band & outlier).sum())
                sel = in_band & ~outlier
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

    def _alignment(
        self, gridded: dict, baseline: dict, matched_rmsd: float | None
    ) -> dict:
        """PASS/FAIL/N/A verdict on bias, RMSD vs baseline, and outlier count.

        The baseline RMSD is the RMS of the two files' ascending-vs-descending
        RMSDs: the disagreement expected from gridding alone. It is compared
        with *matched_rmsd* (A vs B on the same half-grids); RMSD is only
        judged when both are available.
        """
        base_vals = [b["rmsd"] for b in baseline.values() if b and b["rmsd"] is not None]
        baseline_rmsd = float(np.sqrt(np.mean(np.square(base_vals)))) if base_vals else None
        rmsd_limit = self.rmsd_factor * baseline_rmsd if baseline_rmsd else None
        result = {
            "status": "N/A",
            "reasons": [],
            "baseline_rmsd": baseline_rmsd,
            "matched_rmsd": matched_rmsd,
            "rmsd_limit": rmsd_limit,
            "rmsd_ratio": (
                matched_rmsd / baseline_rmsd
                if matched_rmsd is not None and baseline_rmsd
                else None
            ),
            "criteria": {
                "outlier_cutoff": self.outlier_cutoff,
                "max_bias": self.max_bias,
                "rmsd_factor": self.rmsd_factor,
                "min_common_cells": self.min_common_cells,
                "max_outlier_cells": self.max_outlier_cells,
            },
        }

        n_cells = gridded["n_common_cells"] - gridded["n_outlier_cells"]
        if n_cells < self.min_common_cells:
            result["reasons"].append(
                f"only {n_cells} usable common cells (< {self.min_common_cells})"
            )
            return result

        reasons = result["reasons"]
        if abs(gridded["bias"]) > self.max_bias:
            reasons.append(f"|bias| {abs(gridded['bias']):.4f} m > {self.max_bias} m")
        if rmsd_limit is not None and matched_rmsd is not None and matched_rmsd > rmsd_limit:
            reasons.append(
                f"matched RMSD {matched_rmsd:.4f} m > {self.rmsd_factor}x baseline "
                f"({rmsd_limit:.4f} m)"
            )
        if gridded["n_outlier_cells"] > self.max_outlier_cells:
            reasons.append(
                f"{gridded['n_outlier_cells']} cell(s) with |B - A| > {self.outlier_cutoff} m "
                f"(max {gridded['max_abs_diff']:.3f} m at lat={gridded['max_abs_diff_lat']:.1f} "
                f"lon={gridded['max_abs_diff_lon']:.1f})"
            )
        result["status"] = "FAIL" if reasons else "PASS"
        return result
