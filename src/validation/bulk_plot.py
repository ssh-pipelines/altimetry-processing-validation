"""Timeseries plots for bulk comparison reports."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from validation.bulk_compare import BulkComparisonReport
from validation.comparators.along_track_hilat import summarise_lat_band

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
    if report.product_type == "along_track_hilat":
        plot_bulk_hilat(report, output_path=output_path)
        return

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
    _save_or_show(fig, output_path)


def _save_or_show(fig, output_path: str | Path | None) -> None:
    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info("Plot saved to %s", output_path)
        plt.close(fig)
    else:
        plt.show()


def plot_bulk_hilat(
    report: BulkComparisonReport,
    output_path: str | Path | None = None,
) -> None:
    """Plot reference (A) vs high-latitude (B) along-track alignment.

    Left column, timeseries sharing a date axis:
      1. Bias (B - A) over common grid cells
      2. RMSD (A vs B) against each file's ascending-vs-descending baseline
      3. Pearson r and std ratio B/A
      4. Small-scale noise RMS (ssha - ssha_smoothed) for each file
      5. Grid cells observed by A, by B, and by both

    Right column, pooled over all days per latitude band:
      1. Bias and RMSD
      2. Std ratio B/A
      3. Noise RMS for each file

    Days with an error or no common cells are marked with vertical red lines.
    """
    pairs = report.matched_pairs
    if not pairs:
        log.warning("No matched pairs to plot.")
        return

    dates = [datetime.strptime(r.date_key, "%Y%m%d") for r in pairs]

    def _vals(attr: str) -> list[float | None]:
        return [getattr(r, attr) if r.error is None else None for r in pairs]

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(5, 2, width_ratios=[2.2, 1])
    ts_axes = [fig.add_subplot(gs[0, 0])]
    ts_axes += [fig.add_subplot(gs[i, 0], sharex=ts_axes[0]) for i in range(1, 5)]
    band_axes = [fig.add_subplot(gs[0:2, 1]), fig.add_subplot(gs[2, 1]),
                 fig.add_subplot(gs[3:5, 1])]
    fig.suptitle(
        f"Reference (A) vs high-latitude (B) along-track alignment\n"
        f"A: {report.dir_a.name}   B: {report.dir_b.name}",
        fontsize=10,
    )
    style = dict(marker="o", markersize=2, linewidth=0.8)

    ax = ts_axes[0]
    ax.plot(dates, _vals("ssha_bias"), color="steelblue", label="Bias (B − A)", **style)
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Bias (m)")

    ax = ts_axes[1]
    ax.plot(dates, _vals("ssha_rmsd"), color="steelblue", label="RMSD A vs B", **style)
    ax.plot(dates, _vals("self_rmsd_a"), color="gray", label="A asc vs desc", **style)
    ax.plot(dates, _vals("self_rmsd_b"), color="darkorange", label="B asc vs desc", **style)
    ax.set_ylabel("RMSD (m)")

    ax = ts_axes[2]
    ax.plot(dates, _vals("ssha_pearson_r"), color="steelblue", label="r", **style)
    ax.plot(dates, _vals("ssha_std_ratio"), color="darkorange", label="std B/A", **style)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Ratio")

    ax = ts_axes[3]
    ax.plot(dates, _vals("noise_a"), color="gray", label="Noise A", **style)
    ax.plot(dates, _vals("noise_b"), color="darkorange", label="Noise B", **style)
    ax.set_ylabel("Noise RMS (m)")

    ax = ts_axes[4]
    ax.plot(dates, _vals("cells_a"), color="gray", label="Cells A", **style)
    ax.plot(dates, _vals("cells_b"), color="darkorange", label="Cells B", **style)
    ax.plot(dates, _vals("n_common_cells"), color="steelblue", label="Common", **style)
    ax.set_ylabel("Grid cells")

    bad_dates = [d for d, r in zip(dates, pairs) if r.error or not r.n_common_cells]
    for ax in ts_axes:
        for bd in bad_dates:
            ax.axvline(bd, color="red", linewidth=0.8, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        ax.grid(True, linewidth=0.4, alpha=0.5)
    for ax in ts_axes[:-1]:
        ax.tick_params(labelbottom=False)
    ts_axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ts_axes[-1].get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    bands = [summarise_lat_band(b) for b in report.lat_bands or []]
    mid = [(b["lat_min"] + b["lat_max"]) / 2 for b in bands]

    def _band_vals(key: str) -> list[float]:
        return [b[key] if b[key] is not None else float("nan") for b in bands]

    ax = band_axes[0]
    ax.plot(_band_vals("bias"), mid, marker="o", color="steelblue", label="Bias (B − A)")
    ax.plot(_band_vals("rmsd"), mid, marker="s", color="darkorange", label="RMSD")
    ax.axvline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("m")
    ax.set_title("Pooled by latitude band", fontsize=9)

    ax = band_axes[1]
    ax.plot(_band_vals("std_ratio"), mid, marker="o", color="steelblue", label="std B/A")
    ax.axvline(1.0, color="black", linewidth=0.5, linestyle="--")

    ax = band_axes[2]
    ax.plot(_band_vals("noise_a"), mid, marker="o", color="gray", label="Noise A")
    ax.plot(_band_vals("noise_b"), mid, marker="s", color="darkorange", label="Noise B")
    ax.set_xlabel("Noise RMS (m)")

    for ax in band_axes:
        ax.set_ylabel("Latitude")
        ax.set_ylim(-90, 90)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    _save_or_show(fig, output_path)
