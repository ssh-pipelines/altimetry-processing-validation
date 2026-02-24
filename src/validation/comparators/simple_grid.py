"""Simple grid (Level 3, 2D lat/lon) comparator."""

import numpy as np
import xarray as xr

from validation.analysis.statistics import _mask_fill
from validation.comparators.base import BaseComparator


class SimpleGridComparator(BaseComparator):
    """Comparator for simple-grid (gridded) product files."""

    EXPECTED_DIMS = ["latitude", "longitude", "basins"]

    EXPECTED_VARS = [
        "ssha",
        "counts",
        "basin_flag",
    ]

    QUALITY_VARS = ["counts", "ssha"]

    @property
    def product_type(self) -> str:
        return "simple_grid"

    def get_expected_variables(self) -> list[str]:
        return list(self.EXPECTED_VARS)

    def get_quality_variables(self) -> list[str]:
        return list(self.QUALITY_VARS)

    def compare_quality(self, ds_a: xr.Dataset, ds_b: xr.Dataset) -> dict:
        """Compare counts distribution and spatial coverage."""
        summary = {}

        # Counts distribution
        for label, ds in [("a", ds_a), ("b", ds_b)]:
            if "counts" in ds.data_vars:
                data = ds["counts"].values
                masked = _mask_fill(data)
                valid = masked[np.isfinite(masked)]
                summary.setdefault("counts", {})[label] = {
                    "min": int(np.min(valid)) if valid.size > 0 else None,
                    "max": int(np.max(valid)) if valid.size > 0 else None,
                    "mean": float(np.mean(valid)) if valid.size > 0 else None,
                    "zero_count": int(np.sum(valid == 0)) if valid.size > 0 else None,
                }
            else:
                summary.setdefault("counts", {})[label] = None

        # SSHA spatial coverage
        for label, ds in [("a", ds_a), ("b", ds_b)]:
            if "ssha" in ds.data_vars:
                data = ds["ssha"].values
                masked = _mask_fill(data)
                total = masked.size
                valid_count = int(np.sum(np.isfinite(masked)))
                coverage_pct = (valid_count / total * 100) if total > 0 else 0.0
                summary.setdefault("ssha_coverage", {})[label] = {
                    "valid_cells": valid_count,
                    "total_cells": total,
                    "coverage_pct": round(coverage_pct, 2),
                }
            else:
                summary.setdefault("ssha_coverage", {})[label] = None

        # SSHA grid-cell agreement (cross-file, configurable threshold)
        if "ssha" in ds_a.data_vars and "ssha" in ds_b.data_vars:
            a = _mask_fill(ds_a["ssha"].values)
            b = _mask_fill(ds_b["ssha"].values)
            both_valid = np.isfinite(a) & np.isfinite(b)
            if np.any(both_valid):
                pct = float(np.mean(np.abs(b[both_valid] - a[both_valid]) <= self.threshold) * 100)
                summary["ssha_agreement"] = {
                    "threshold_m": self.threshold,
                    "pct_within_threshold": round(pct, 2),
                }
            else:
                summary["ssha_agreement"] = {
                    "threshold_m": self.threshold,
                    "pct_within_threshold": None,
                }

        return summary
