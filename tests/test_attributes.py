"""Tests for analysis.attributes module."""

import numpy as np

from validation.analysis.attributes import compare_attributes


class TestCompareAttributes:
    def test_identical(self):
        attrs = {"title": "test", "version": "1.0"}
        diffs = compare_attributes(attrs, dict(attrs))
        assert diffs == []

    def test_value_diff(self):
        a = {"title": "test", "version": "1.0"}
        b = {"title": "test", "version": "2.0"}
        diffs = compare_attributes(a, b)
        assert len(diffs) == 1
        assert diffs[0] == ("version", "1.0", "2.0")

    def test_missing_in_b(self):
        a = {"title": "test", "extra": "value"}
        b = {"title": "test"}
        diffs = compare_attributes(a, b)
        assert len(diffs) == 1
        assert diffs[0] == ("extra", "value", None)

    def test_missing_in_a(self):
        a = {"title": "test"}
        b = {"title": "test", "extra": "value"}
        diffs = compare_attributes(a, b)
        assert len(diffs) == 1
        assert diffs[0] == ("extra", None, "value")

    def test_ignore_list(self):
        a = {"title": "test", "date_created": "2025-01-01", "history": "old"}
        b = {"title": "test", "date_created": "2025-06-01", "history": "new"}
        diffs = compare_attributes(a, b, ignore=["date_created", "history"])
        assert diffs == []

    def test_numpy_array_values(self):
        a = {"data": np.array([1, 2, 3])}
        b = {"data": np.array([1, 2, 4])}
        diffs = compare_attributes(a, b)
        assert len(diffs) == 1
        assert diffs[0][0] == "data"

    def test_numpy_array_identical(self):
        a = {"data": np.array([1, 2, 3])}
        b = {"data": np.array([1, 2, 3])}
        diffs = compare_attributes(a, b)
        assert diffs == []

    def test_empty_dicts(self):
        assert compare_attributes({}, {}) == []
