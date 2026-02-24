"""Base comparator ABC and result dataclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import xarray as xr

from validation.analysis.attributes import compare_attributes
from validation.analysis.dimensions import compare_dimensions
from validation.analysis.statistics import compute_variable_diff, compute_variable_stats


@dataclass
class VariableComparison:
    """Comparison results for a single variable."""

    name: str
    present_a: bool
    present_b: bool
    stats_a: dict | None = None
    stats_b: dict | None = None
    diff: dict | None = None
    attr_diffs: list[tuple[str, object, object]] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Full comparison report between two files."""

    file_a: str
    file_b: str
    product_type: str
    dimension_diffs: list[tuple[str, int | None, int | None]] = field(
        default_factory=list
    )
    global_attr_diffs: list[tuple[str, object, object]] = field(default_factory=list)
    variable_comparisons: list[VariableComparison] = field(default_factory=list)
    quality_summary: dict = field(default_factory=dict)

    @property
    def has_differences(self) -> bool:
        if self.dimension_diffs:
            return True
        if self.global_attr_diffs:
            return True
        for vc in self.variable_comparisons:
            if not vc.present_a or not vc.present_b:
                return True
            if vc.attr_diffs:
                return True
            if vc.diff and vc.diff.get("max_abs_diff") not in (None, 0.0):
                return True
        return False


class BaseComparator(ABC):
    """Abstract base for product-type comparators."""

    def __init__(self, file_a: str, file_b: str, threshold: float = 0.05):
        self.file_a = file_a
        self.file_b = file_b
        self.threshold = threshold
        self.ds_a: xr.Dataset | None = None
        self.ds_b: xr.Dataset | None = None

    @property
    @abstractmethod
    def product_type(self) -> str:
        """Short label for this product type."""

    @abstractmethod
    def get_expected_variables(self) -> list[str]:
        """Return list of variable names expected in this product."""

    @abstractmethod
    def get_quality_variables(self) -> list[str]:
        """Return list of quality/flag variable names."""

    @abstractmethod
    def compare_quality(
        self, ds_a: xr.Dataset, ds_b: xr.Dataset
    ) -> dict:
        """Product-specific quality comparison. Returns a summary dict."""

    def load_datasets(self) -> tuple[xr.Dataset, xr.Dataset]:
        self.ds_a = xr.open_dataset(self.file_a)
        self.ds_b = xr.open_dataset(self.file_b)
        return self.ds_a, self.ds_b

    def run(self, ignore_attrs: list[str] | None = None) -> ComparisonReport:
        """Orchestrate a full comparison and return a structured report."""
        ds_a, ds_b = self.load_datasets()

        dim_diffs = compare_dimensions(ds_a, ds_b)
        global_attr_diffs = compare_attributes(
            dict(ds_a.attrs), dict(ds_b.attrs), ignore=ignore_attrs
        )

        all_vars = sorted(set(ds_a.data_vars) | set(ds_b.data_vars))
        var_comparisons = []
        for var_name in all_vars:
            in_a = var_name in ds_a.data_vars
            in_b = var_name in ds_b.data_vars
            vc = VariableComparison(name=var_name, present_a=in_a, present_b=in_b)

            if in_a:
                vc.stats_a = compute_variable_stats(ds_a[var_name])
            if in_b:
                vc.stats_b = compute_variable_stats(ds_b[var_name])
            if in_a and in_b:
                vc.diff = compute_variable_diff(ds_a[var_name], ds_b[var_name])
                vc.attr_diffs = compare_attributes(
                    dict(ds_a[var_name].attrs),
                    dict(ds_b[var_name].attrs),
                    ignore=ignore_attrs,
                )

            var_comparisons.append(vc)

        quality_summary = self.compare_quality(ds_a, ds_b)

        ds_a.close()
        ds_b.close()

        return ComparisonReport(
            file_a=self.file_a,
            file_b=self.file_b,
            product_type=self.product_type,
            dimension_diffs=dim_diffs,
            global_attr_diffs=global_attr_diffs,
            variable_comparisons=var_comparisons,
            quality_summary=quality_summary,
        )
