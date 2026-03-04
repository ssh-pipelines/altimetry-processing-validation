"""Timeseries plots for bulk comparison reports."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from validation.bulk_compare import BulkComparisonReport

log = logging.getLogger(__name__)


def plot_bulk_timeseries(
    report: BulkComparisonReport,
    output_path: str | Path | None = None,
    pass_threshold: float = 95.0,
) -> None:
    """Plot per-date SSHA and counts difference metrics as a timeseries.

    Produces a three-panel figure:
      1. SSHA agreement % with a reference line at pass_threshold
      2. SSHA MAE, RMSD, and max absolute difference
      3. Counts MAE

    Error pairs are marked with vertical red lines on all panels.

    Parameters
    ----------
    report:
        A BulkComparisonReport from run_bulk_comparison().
    output_path:
        Save the figure to this path. If None, display interactively.
    pass_threshold:
        Reference line drawn on the SSHA agreement panel.
    """
    pairs = report.matched_pairs
    if not pairs:
        log.warning("No matched pairs to plot.")
        return

    dates = [datetime.strptime(r.date_key, "%Y%m%d") for r in pairs]

    def _vals(attr: str) -> list[float | None]:
        return [getattr(r, attr) if r.error is None else None for r in pairs]

    ssha_agr  = _vals("ssha_pct_within_threshold")
    ssha_mae  = _vals("ssha_mean_abs_diff")
    ssha_rmsd = _vals("ssha_rmsd")
    ssha_max  = _vals("ssha_max_abs_diff")
    cts_mae   = _vals("counts_mean_abs_diff")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"Bulk comparison timeseries\n"
        f"A: {report.dir_a.name}   B: {report.dir_b.name}   "
        f"threshold={report.threshold} m",
        fontsize=10,
    )

    # Panel 1 — SSHA agreement %
    ax = axes[0]
    ax.plot(dates, ssha_agr, marker="o", markersize=3, linewidth=1,
            color="steelblue", label="SSHA agr %")
    ax.axhline(pass_threshold, color="red", linestyle="--", linewidth=0.8,
               label=f"Pass threshold ({pass_threshold}%)")
    ax.set_ylabel("Agreement %")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # Panel 2 — SSHA diff metrics
    ax = axes[1]
    ax.plot(dates, ssha_mae, marker="o", markersize=3, linewidth=1,
            color="steelblue", label="SSHA MAE")
    ax.plot(dates, ssha_rmsd, marker="s", markersize=3, linewidth=1,
            linestyle="--", color="darkorange", label="SSHA RMSD")
    ax.plot(dates, ssha_max, marker="^", markersize=3, linewidth=1,
            linestyle=":", color="gray", label="SSHA max abs")
    ax.set_ylabel("Difference (m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # Panel 3 — Counts MAE
    ax = axes[2]
    ax.plot(dates, cts_mae, marker="o", markersize=3, linewidth=1,
            color="steelblue", label="Counts MAE")
    ax.set_ylabel("Counts difference")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # Mark error dates across all panels
    error_dates = [d for d, r in zip(dates, pairs) if r.error]
    if error_dates:
        for ax in axes:
            for ed in error_dates:
                ax.axvline(ed, color="red", linewidth=0.8, alpha=0.4)
        log.debug("%d error dates marked on plot", len(error_dates))

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info("Plot saved to %s", output_path)
        plt.close(fig)
    else:
        plt.show()
