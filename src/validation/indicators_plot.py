"""Timeseries plots for indicators file comparison."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# Variables that carry physical units stored in a ``units`` attribute.
_UNITS_VARS = {"raw_gmsl", "gmsl", "smoothed_gmsl"}


def plot_indicators_comparison(
    file_a: str | Path,
    file_b: str | Path,
    output_path: str | Path | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Plot per-variable timeseries comparisons between two indicators files.

    Produces a figure with two columns and one row per variable:

    * **Left column** — both files overlaid on the same axes.
    * **Right column** — difference (B − A); only shown when time axes match.

    Parameters
    ----------
    file_a:
        Path to the first indicators NetCDF file.
    file_b:
        Path to the second indicators NetCDF file.
    output_path:
        Save the figure to this path.  If *None*, display interactively.
    label_a:
        Legend label for file A.
    label_b:
        Legend label for file B.
    """
    ds_a = xr.open_dataset(file_a)
    ds_b = xr.open_dataset(file_b)

    common_vars = sorted(set(ds_a.data_vars) & set(ds_b.data_vars))
    vars_to_plot = [
        v for v in common_vars
        if "time" in ds_a[v].dims and "time" in ds_b[v].dims
    ]

    if not vars_to_plot:
        log.warning("No common time-indexed variables found; nothing to plot.")
        ds_a.close()
        ds_b.close()
        return

    time_a = ds_a["time"].values.astype(float)
    time_b = ds_b["time"].values.astype(float)
    time_match = (
        len(time_a) == len(time_b)
        and bool(np.allclose(time_a, time_b, atol=1e-9))
    )

    n = len(vars_to_plot)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), sharex="col")

    # Ensure axes is always 2-D even for a single variable.
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Indicators comparison\n"
        f"{label_a}: {Path(file_a).name}   "
        f"{label_b}: {Path(file_b).name}",
        fontsize=10,
    )

    for i, var in enumerate(vars_to_plot):
        ax_val = axes[i, 0]
        ax_diff = axes[i, 1]

        vals_a = ds_a[var].values.astype(float)
        vals_b = ds_b[var].values.astype(float)

        units = ds_a[var].attrs.get("units", "")
        ylabel = f"({units})" if units else ""

        # --- left panel: both files ---
        ax_val.plot(
            time_a, vals_a,
            linewidth=0.8, color="steelblue", label=label_a,
        )
        ax_val.plot(
            time_b, vals_b,
            linewidth=0.8, linestyle="--", color="darkorange", label=label_b,
        )
        ax_val.set_title(var, fontsize=9)
        ax_val.set_ylabel(ylabel, fontsize=8)
        ax_val.grid(True, linewidth=0.4, alpha=0.5)
        if i == 0:
            ax_val.legend(fontsize=8, loc="upper left")

        # --- right panel: B − A ---
        ax_diff.set_title(f"{var}: {label_b} − {label_a}", fontsize=9)
        ax_diff.set_ylabel(ylabel, fontsize=8)
        ax_diff.grid(True, linewidth=0.4, alpha=0.5)

        if time_match and vals_a.shape == vals_b.shape:
            diff = vals_b - vals_a
            ax_diff.plot(time_a, diff, linewidth=0.8, color="purple")
            ax_diff.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
        else:
            ax_diff.text(
                0.5, 0.5, "Time axes differ — diff not shown",
                transform=ax_diff.transAxes,
                ha="center", va="center", fontsize=8, color="gray",
            )

    axes[-1, 0].set_xlabel("Year", fontsize=9)
    axes[-1, 1].set_xlabel("Year", fontsize=9)

    plt.tight_layout()

    ds_a.close()
    ds_b.close()

    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info("Plot saved to %s", output_path)
        plt.close(fig)
    else:
        plt.show()
