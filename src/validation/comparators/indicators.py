"""Indicators file comparator."""

import numpy as np
import xarray as xr

from validation.comparators.base import BaseComparator

# Decimal places used when rounding decimal-year values for time alignment.
# Weekly data has a step of ~0.019 yr; 6 decimal places gives ~0.03 s precision,
# which is fine-grained enough to avoid collisions while absorbing float noise.
_TIME_ROUND_DECIMALS = 6


def align_time_indices(
    time_a: np.ndarray,
    time_b: np.ndarray,
    decimals: int = _TIME_ROUND_DECIMALS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices of common time points between two decimal-year arrays.

    Both arrays are rounded to *decimals* decimal places before matching so
    that minor floating-point discrepancies are absorbed.  Only the first
    occurrence of each rounded value is used (duplicates are ignored).

    Parameters
    ----------
    time_a, time_b:
        1-D float arrays of decimal-year time coordinates.
    decimals:
        Number of decimal places to round to before matching.

    Returns
    -------
    idx_a : np.ndarray
        Integer indices into *time_a* for each common point.
    idx_b : np.ndarray
        Integer indices into *time_b* for each common point.
    common_times : np.ndarray
        Rounded time values at the intersection, sorted ascending.
    """
    rounded_a = np.round(time_a.astype(float), decimals)
    rounded_b = np.round(time_b.astype(float), decimals)

    common = np.intersect1d(rounded_a, rounded_b)

    lookup_a = {v: i for i, v in enumerate(rounded_a)}
    lookup_b = {v: i for i, v in enumerate(rounded_b)}

    idx_a = np.array([lookup_a[v] for v in common], dtype=int)
    idx_b = np.array([lookup_b[v] for v in common], dtype=int)

    return idx_a, idx_b, common


class IndicatorsComparator(BaseComparator):
    """Comparator for indicators product files.

    Indicators files contain 1-D time-indexed scalar climate/sea-level
    indicator variables (e.g. GMSL, ENSO, PDO, IOD).  The time coordinate
    is a decimal year (float64).
    """

    EXPECTED_VARS = [
        "raw_gmsl",
        "enso",
        "pdo",
        "iod",
        "gmsl",
        "smoothed_gmsl",
    ]

    @property
    def product_type(self) -> str:
        return "indicators"

    def get_expected_variables(self) -> list[str]:
        return list(self.EXPECTED_VARS)

    def get_quality_variables(self) -> list[str]:
        return list(self.EXPECTED_VARS)

    def compare_quality(self, ds_a: xr.Dataset, ds_b: xr.Dataset) -> dict:
        """Compute per-variable diff statistics at common time points.

        Time axes are aligned by rounding decimal-year values before matching,
        so files with different lengths or minor float differences are handled
        correctly.  Statistics are computed only at the intersection of time
        points present in both files.

        Returns
        -------
        dict
            Keys are variable names; values are dicts with keys
            ``n_common``, ``bias``, ``rmsd``, and ``max_abs_diff``.
        """
        summary: dict = {}

        if "time" not in ds_a.coords or "time" not in ds_b.coords:
            return summary

        idx_a, idx_b, common_times = align_time_indices(
            ds_a["time"].values, ds_b["time"].values
        )

        common_vars = sorted(set(ds_a.data_vars) & set(ds_b.data_vars))
        for var in common_vars:
            if len(common_times) == 0:
                summary[var] = {
                    "n_common": 0,
                    "bias": None,
                    "rmsd": None,
                    "max_abs_diff": None,
                }
                continue

            a = ds_a[var].values.astype(float)[idx_a]
            b = ds_b[var].values.astype(float)[idx_b]
            diff = b - a
            valid = np.isfinite(diff)

            if np.any(valid):
                summary[var] = {
                    "n_common": int(len(common_times)),
                    "bias": float(np.mean(diff[valid])),
                    "rmsd": float(np.sqrt(np.mean(diff[valid] ** 2))),
                    "max_abs_diff": float(np.max(np.abs(diff[valid]))),
                }
            else:
                summary[var] = {
                    "n_common": int(len(common_times)),
                    "bias": None,
                    "rmsd": None,
                    "max_abs_diff": None,
                }

        return summary
