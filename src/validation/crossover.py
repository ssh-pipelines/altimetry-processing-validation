"""Daily RMS crossover analysis for SSH crossover files."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CrossoverDayResult:
    """RMS crossover statistics for a single day."""

    date: date
    rms_m: float
    n_crossovers: int


def compute_daily_rms(
    input_dir: str | Path, label: str = ""
) -> list[CrossoverDayResult]:
    """Compute daily RMS of (ssh2 - ssh1) for all xovers_S6-*.nc files in input_dir.

    Parameters
    ----------
    input_dir:
        Directory containing xovers_S6-YYYY-MM-DD.nc files.
    label:
        Human-readable label used in log/print output.

    Returns
    -------
    List of CrossoverDayResult sorted by date.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("xovers_S6-*.nc"))
    if not files:
        raise FileNotFoundError(f"No matching files found in {input_dir}")

    effective_label = label or str(input_dir)
    log.info("[%s]  %d files found", effective_label, len(files))
    print(f"\n[{effective_label}]  {len(files)} files found")

    results = []
    for fpath in files:
        match = re.search(r"(\d{4}-\d{2}-\d{2})\.nc$", fpath.name)
        if not match:
            log.warning("Skipping (no date in filename): %s", fpath.name)
            continue
        day = datetime.strptime(match.group(1), "%Y-%m-%d").date()

        with nc.Dataset(fpath, "r") as ds:
            ssh1 = ds.variables["ssh1"][:]
            ssh2 = ds.variables["ssh2"][:]

        diff = ssh2 - ssh1
        if isinstance(diff, np.ma.MaskedArray):
            diff = diff.compressed()
        else:
            diff = diff[np.isfinite(diff)]

        n = len(diff)
        rms = float(np.sqrt(np.mean(diff**2))) if n > 0 else float("nan")
        print(f"  {day}  n={n:6d}  RMS={rms * 100:.2f} cm")
        results.append(CrossoverDayResult(date=day, rms_m=rms, n_crossovers=n))

    return sorted(results, key=lambda r: r.date)


def print_summary(results: list[CrossoverDayResult], label: str) -> None:
    """Print a summary of crossover RMS statistics to stdout."""
    rms_values = [r.rms_m for r in results if not np.isnan(r.rms_m)]
    print(f"\n--- Summary: {label} ---")
    print(f"  Days processed : {len(results)}")
    if results:
        print(f"  Date range     : {results[0].date} → {results[-1].date}")
    if rms_values:
        mean_rms = sum(rms_values) / len(rms_values)
        print(f"  Overall RMS    : {mean_rms * 100:.2f} cm (mean of daily values)")
        print(
            f"  Min/Max RMS    : {min(rms_values) * 100:.2f} / {max(rms_values) * 100:.2f} cm"
        )


def write_csv(
    results: list[CrossoverDayResult], label: str, output_dir: str | Path
) -> Path:
    """Write a CSV of daily RMS results and return the output path.

    Parameters
    ----------
    results:
        Results from compute_daily_rms.
    label:
        Label used to derive the filename.
    output_dir:
        Directory to write the CSV into.
    """
    output_dir = Path(output_dir)
    safe_label = label.replace(" ", "_").replace("/", "-")
    out_path = output_dir / f"rms_{safe_label}.csv"
    with out_path.open("w") as f:
        f.write("date,rms_m,n_crossovers\n")
        for r in results:
            f.write(f"{r.date},{r.rms_m},{r.n_crossovers}\n")
    log.info("CSV written to %s", out_path)
    return out_path


def _inner_diff(
    results_a: list[CrossoverDayResult],
    results_b: list[CrossoverDayResult],
) -> list[tuple[date, float]]:
    """Return (date, diff_cm) pairs where diff_cm = (rms_b − rms_a) × 100."""
    lookup_a = {r.date: r.rms_m for r in results_a}
    lookup_b = {r.date: r.rms_m for r in results_b}
    common = sorted(set(lookup_a) & set(lookup_b))
    return [(d, (lookup_b[d] - lookup_a[d]) * 100) for d in common]


# Visual style: pre_oer uses blue family, post_oer uses orange family;
# set1 is solid circles, set2 is dashed squares.
_STYLES: dict[str, dict] = {
    "pre_oer_set1":  dict(color="C0", marker="o", ls="-",  ms=3, lw=1.5),
    "pre_oer_set2":  dict(color="C0", marker="s", ls="--", ms=3, lw=1.5),
    "post_oer_set1": dict(color="C1", marker="o", ls="-",  ms=3, lw=1.5),
    "post_oer_set2": dict(color="C1", marker="s", ls="--", ms=3, lw=1.5),
}
_DIFF_STYLES: dict[str, dict] = {
    "pre_oer":  dict(color="C0", alpha=0.6),
    "post_oer": dict(color="C1", alpha=0.6),
}


def plot_crossover_four_way(
    pre_r1: list[CrossoverDayResult],
    pre_r2: list[CrossoverDayResult],
    post_r1: list[CrossoverDayResult],
    post_r2: list[CrossoverDayResult],
    label_pre1: str,
    label_pre2: str,
    label_post1: str,
    label_post2: str,
    output_path: str | Path | None = None,
) -> None:
    """Plot four-way crossover RMS comparison with a difference panel.

    Top panel: all four RMS time series on one axes.
    Bottom panel: (set2 − set1) difference for pre-OER and post-OER periods.

    Parameters
    ----------
    pre_r1, pre_r2:
        Pre-OER set1 and set2 results from compute_daily_rms.
    post_r1, post_r2:
        Post-OER set1 and set2 results from compute_daily_rms.
    label_pre1, label_pre2, label_post1, label_post2:
        Legend labels for each series.
    output_path:
        Save the figure to this path. If None, display interactively.
    """
    pre_diff = _inner_diff(pre_r1, pre_r2)
    post_diff = _inner_diff(post_r1, post_r2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # Top panel: all four RMS series
    for results, label, key in [
        (pre_r1,  label_pre1,  "pre_oer_set1"),
        (pre_r2,  label_pre2,  "pre_oer_set2"),
        (post_r1, label_post1, "post_oer_set1"),
        (post_r2, label_post2, "post_oer_set2"),
    ]:
        dates = [r.date for r in results]
        rms_cm = [r.rms_m * 100 for r in results]
        ax1.plot(dates, rms_cm, label=label, **_STYLES[key])

    ax1.set_ylabel("RMS (cm)")
    ax1.set_title("Daily SSH Crossover RMS — pre OER vs post OER")
    ax1.legend(ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: set2 - set1 difference for each OER period
    ax2.axhline(0, color="black", lw=0.8, ls="--")

    bar_width = 0.4  # days
    for diff_pairs, period_label, key in [
        (pre_diff,  "pre OER  (set2 − set1)",  "pre_oer"),
        (post_diff, "post OER (set2 − set1)", "post_oer"),
    ]:
        if not diff_pairs:
            continue
        dates = [d for d, _ in diff_pairs]
        diffs = [v for _, v in diff_pairs]
        mean_d = sum(diffs) / len(diffs)
        style = _DIFF_STYLES[key]
        ax2.bar(
            dates, diffs, width=bar_width,
            label=f"{period_label}  (mean {mean_d:+.2f} cm)",
            **style,
        )
        ax2.axhline(mean_d, color=style["color"], lw=1.2, ls=":")

    ax2.set_ylabel("Δ RMS (cm)\n(set2 − set1)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=150)
        log.info("Plot saved to %s", output_path)
        plt.close(fig)
    else:
        plt.show()
