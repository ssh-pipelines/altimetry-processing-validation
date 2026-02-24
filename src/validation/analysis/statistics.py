"""Per-variable statistics computation and diff analysis."""

import numpy as np
import xarray as xr

# Fill values used by the pipeline encoding conventions.
# Integer dtype-max values and non-finite floats are treated as fill.
_INT_FILL_VALUES = {
    np.dtype("int8"): np.iinfo(np.int8).max,
    np.dtype("int16"): np.iinfo(np.int16).max,
    np.dtype("int32"): np.iinfo(np.int32).max,
    np.dtype("int64"): np.iinfo(np.int64).max,
}


def _mask_fill(data: np.ndarray) -> np.ndarray:
    """Return a float64 copy with fill/sentinel values replaced by NaN."""
    result = data.astype(np.float64)
    if data.dtype in _INT_FILL_VALUES:
        result[data == _INT_FILL_VALUES[data.dtype]] = np.nan
    # Mask non-finite floats (inf, -inf, nan already present)
    result[~np.isfinite(result)] = np.nan
    return result


def compute_variable_stats(var: xr.DataArray) -> dict:
    """Compute summary statistics for a single variable.

    Returns a dict with keys: min, max, mean, median, std,
    nan_count, valid_count, shape, dtype.
    """
    data = var.values
    shape = data.shape
    dtype = str(data.dtype)

    if not np.issubdtype(data.dtype, np.number):
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "nan_count": None,
            "valid_count": None,
            "shape": shape,
            "dtype": dtype,
        }

    masked = _mask_fill(data)
    valid = masked[np.isfinite(masked)]

    if valid.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "nan_count": int(data.size),
            "valid_count": 0,
            "shape": shape,
            "dtype": dtype,
        }

    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "nan_count": int(data.size - valid.size),
        "valid_count": int(valid.size),
        "shape": shape,
        "dtype": dtype,
    }


def compute_variable_diff(var_a: xr.DataArray, var_b: xr.DataArray) -> dict | None:
    """Compute difference statistics between two variables.

    Returns dict with max_abs_diff, mean_abs_diff, rmsd,
    or None if shapes don't match or data is non-numeric.
    """
    if var_a.shape != var_b.shape:
        return None

    if not np.issubdtype(var_a.dtype, np.number) or not np.issubdtype(
        var_b.dtype, np.number
    ):
        return None

    a = _mask_fill(var_a.values)
    b = _mask_fill(var_b.values)

    # Only compare where both are valid
    both_valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(both_valid):
        return {"max_abs_diff": None, "mean_abs_diff": None, "rmsd": None}

    av = a[both_valid]
    bv = b[both_valid]
    diff = np.abs(av - bv)
    bias = float(np.mean(bv - av))

    if av.size > 1:
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(av, bv)[0, 1]
        pearson_r = float(corr) if np.isfinite(corr) else None
    else:
        pearson_r = None

    return {
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "rmsd": float(np.sqrt(np.mean(diff**2))),
        "bias": bias,
        "pearson_r": pearson_r,
    }
