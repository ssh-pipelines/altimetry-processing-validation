"""Global and variable attribute diffing."""


def compare_attributes(
    attrs_a: dict, attrs_b: dict, ignore: list[str] | None = None
) -> list[tuple[str, object, object]]:
    """Compare two attribute dictionaries.

    Returns a list of (attr_name, val_a, val_b) tuples for every attribute
    that differs between the two sets.  Attributes present in only one set
    use None for the missing side.

    Parameters
    ----------
    attrs_a, attrs_b : dict
        Attribute dictionaries (e.g. from ``ds.attrs``).
    ignore : list[str], optional
        Attribute names to skip (e.g. ``date_created``, ``history``).
    """
    ignore_set = set(ignore) if ignore else set()
    all_keys = sorted(set(attrs_a) | set(attrs_b))
    diffs = []
    for key in all_keys:
        if key in ignore_set:
            continue
        val_a = attrs_a.get(key)
        val_b = attrs_b.get(key)
        if _values_differ(val_a, val_b):
            diffs.append((key, val_a, val_b))
    return diffs


def _values_differ(a, b) -> bool:
    """Check whether two attribute values differ, handling numpy arrays."""
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    try:
        import numpy as np

        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return not np.array_equal(a, b)
    except ImportError:
        pass
    return a != b
