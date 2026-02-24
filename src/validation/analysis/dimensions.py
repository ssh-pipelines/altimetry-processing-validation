"""Dimension comparison between two datasets."""

import xarray as xr


def compare_dimensions(
    ds_a: xr.Dataset, ds_b: xr.Dataset
) -> list[tuple[str, int | None, int | None]]:
    """Compare dimensions of two datasets.

    Returns a list of (dim_name, size_a, size_b) tuples for every dimension
    that differs in size or is present in only one dataset.
    """
    all_dims = sorted(set(ds_a.sizes) | set(ds_b.sizes))
    diffs = []
    for dim in all_dims:
        size_a = ds_a.sizes.get(dim)
        size_b = ds_b.sizes.get(dim)
        if size_a != size_b:
            diffs.append((dim, size_a, size_b))
    return diffs
