"""Along-track (Level 2, 1D time-indexed) comparator."""

import numpy as np
import xarray as xr

from validation.analysis.statistics import _mask_fill
from validation.comparators.base import BaseComparator


class AlongTrackComparator(BaseComparator):
    """Comparator for along-track daily files."""

    EXPECTED_DIMS = ["time", "src_flag_dim", "basins"]

    EXPECTED_VARS = [
        "ssha",
        "ssha_smoothed",
        "dac",
        "cycle",
        "pass",
        "nasa_flag",
        "source_flag",
        "median_filter_flag",
        "oer",
        "basin_flag",
    ]

    QUALITY_VARS = ["nasa_flag", "source_flag", "median_filter_flag"]

    @property
    def product_type(self) -> str:
        return "along_track"

    def get_expected_variables(self) -> list[str]:
        return list(self.EXPECTED_VARS)

    def get_quality_variables(self) -> list[str]:
        return list(self.QUALITY_VARS)

    def compare_quality(self, ds_a: xr.Dataset, ds_b: xr.Dataset) -> dict:
        """Compare flag value distributions between two along-track files."""
        summary = {}
        for flag_var in self.QUALITY_VARS:
            entry = {}
            for label, ds in [("a", ds_a), ("b", ds_b)]:
                if flag_var not in ds.data_vars:
                    entry[label] = None
                    continue
                data = ds[flag_var].values
                fill = np.iinfo(np.int8).max if data.dtype == np.int8 else None
                if fill is not None:
                    valid = data[data != fill]
                else:
                    valid = data
                good = int(np.sum(valid == 0))
                bad = int(np.sum(valid != 0))
                entry[label] = {"good": good, "bad": bad, "total": int(valid.size)}
            summary[flag_var] = entry

        # SSHA percentile distributions
        for label, ds in [("a", ds_a), ("b", ds_b)]:
            if "ssha" in ds.data_vars:
                masked = _mask_fill(ds["ssha"].values)
                valid = masked[np.isfinite(masked)]
                if valid.size > 0:
                    p = np.percentile(valid, [5, 25, 50, 75, 95])
                    summary.setdefault("ssha_percentiles", {})[label] = {
                        "p5": round(float(p[0]), 6),
                        "p25": round(float(p[1]), 6),
                        "p50": round(float(p[2]), 6),
                        "p75": round(float(p[3]), 6),
                        "p95": round(float(p[4]), 6),
                    }
                else:
                    summary.setdefault("ssha_percentiles", {})[label] = None
            else:
                summary.setdefault("ssha_percentiles", {})[label] = None

        return summary
