"""Indicators file comparator."""

import numpy as np
import xarray as xr

from validation.comparators.base import BaseComparator


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
        """Compute per-variable diff statistics between the two files.

        Difference metrics are only computed when the two files share an
        identical time coordinate.  If the time axes differ in length or
        values the metric is recorded as None.

        Returns
        -------
        dict
            Keys are variable names; values are dicts with keys
            ``bias``, ``rmsd``, ``max_abs_diff``, and
            ``time_axes_match`` (bool).
        """
        summary: dict = {}

        time_a = ds_a["time"].values if "time" in ds_a.coords else None
        time_b = ds_b["time"].values if "time" in ds_b.coords else None
        time_match = (
            time_a is not None
            and time_b is not None
            and len(time_a) == len(time_b)
            and bool(np.allclose(time_a, time_b, atol=1e-9))
        )

        common_vars = sorted(set(ds_a.data_vars) & set(ds_b.data_vars))
        for var in common_vars:
            a = ds_a[var].values.astype(float)
            b = ds_b[var].values.astype(float)

            if time_match and a.shape == b.shape:
                diff = b - a
                valid = np.isfinite(diff)
                if np.any(valid):
                    summary[var] = {
                        "time_axes_match": True,
                        "bias": float(np.mean(diff[valid])),
                        "rmsd": float(np.sqrt(np.mean(diff[valid] ** 2))),
                        "max_abs_diff": float(np.max(np.abs(diff[valid]))),
                    }
                else:
                    summary[var] = {
                        "time_axes_match": True,
                        "bias": None,
                        "rmsd": None,
                        "max_abs_diff": None,
                    }
            else:
                summary[var] = {
                    "time_axes_match": False,
                    "bias": None,
                    "rmsd": None,
                    "max_abs_diff": None,
                }

        return summary
