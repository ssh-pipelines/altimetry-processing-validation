# altimetry-processing-validation

Comparison toolkit for altimetry NetCDF product files. Supports single-file pair comparison and bulk directory comparison for simple-grid products. Produces structured reports of differences including per-variable statistics, bias, spatial correlation, and product-specific quality metrics.

## Supported Product Types

- **along_track** — Level 2, 1D time-indexed daily files
- **along_track_hilat** — reference-mission along-track file (A) vs a high-latitude along-track file (B). The files share a layout but sample different orbits, so SSHA is compared on a lat/lon grid rather than point-by-point
- **simple_grid** — Level 3, 2D lat/lon gridded products

## Installation

Requires Python 3.10+. Create and activate a virtual environment, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If the `validate-altimetry` entry point has a bad interpreter (e.g. in some devcontainer setups), run via the module directly:

```bash
python -m validation.cli file_a.nc file_b.nc -t along_track
```

## Usage

### Single-file comparison

```bash
# Compare two along-track files
validate-altimetry file_a.nc file_b.nc -t along_track

# Compare a reference along-track file with a high-latitude one
validate-altimetry NASA-SSH_alt_ref_at_v1_1_20201023.nc NASA-SSH_S3B_alt_hilat_at_v1_1_20201023.nc -t along_track_hilat

# Compare two simple grids
validate-altimetry file_a.nc file_b.nc -t simple_grid

# Ignore timestamp attributes that are expected to differ
validate-altimetry file_a.nc file_b.nc -t simple_grid --ignore-attrs date_created history

# Use a wider threshold for pre-offset comparisons (default is 0.05 m)
validate-altimetry file_a.nc file_b.nc -t simple_grid --threshold 0.10
```

Exit code 0 means files match; exit code 1 means differences were found. For `along_track_hilat`, exit code 0 means the alignment check passed; 1 means FAIL or N/A (see [Alignment check](#alignment-check-along_track_hilat)).

#### Options

| Flag | Default | Description |
|---|---|---|
| `-t`, `--product-type` | *(required)* | `along_track`, `along_track_hilat`, or `simple_grid` |
| `--ignore-attrs` | none | Global or variable attribute names to exclude from comparison |
| `--threshold` | `0.05` | Absolute difference threshold in metres for the `pct_within_threshold` metric (simple_grid and along_track_hilat) |
| `--outlier-cutoff` and other alignment-check flags | see below | along_track_hilat only; see [Alignment check](#alignment-check-along_track_hilat) |

### Bulk directory comparison

Compare all matching NetCDF files across two directories and produce an aggregate report. Defaults to simple grids. Use `-t along_track_hilat` to compare a directory of reference along-track files (A) against a directory of high-latitude along-track files (B):

```bash
bulk-validate-altimetry dir_a/ dir_b/

# Custom thresholds
bulk-validate-altimetry dir_a/ dir_b/ --threshold 0.10 --pass-threshold 90.0

# Ignore timestamp attributes
bulk-validate-altimetry dir_a/ dir_b/ --ignore-attrs date_created history

# Save a timeseries plot of difference metrics
bulk-validate-altimetry dir_a/ dir_b/ --plot report.png

# Reference along-track vs high-latitude along-track
bulk-validate-altimetry ref_at/ hilat_at/ -t along_track_hilat
```

Files are matched by the 8-digit date before `.nc`, so differing filename patterns (e.g. `NASA-SSH_alt_ref_at_v1_1_20201023.nc` and `NASA-SSH_S3B_alt_hilat_at_v1_1_20201023.nc`) pair up. If a directory holds more than one file for a date, the first one in sorted order is used and the rest are listed as unmatched. Unmatched files are listed but not compared. Exit code 0 means all matched files pass; exit code 1 means at least one file failed or errored. For `along_track_hilat`, PASS/FAIL comes from the alignment check, `--pass-threshold` is ignored, and N/A days (too few common cells) don't fail the run.

#### Options

| Flag | Default | Description |
|---|---|---|
| `-t`, `--product-type` | `simple_grid` | `simple_grid` or `along_track_hilat` |
| `--threshold` | `0.05` | Absolute difference threshold in metres for the SSHA agreement metric |
| `--pass-threshold` | `95.0` | Minimum SSHA agreement % for a file to be marked PASS (not used by along_track_hilat) |
| `--ignore-attrs` | none | Attribute names to exclude from comparison |
| `--plot` | none | Save a timeseries plot to the given path (e.g. `report.png`) |

#### Alignment check (along_track_hilat)

Each file pair gets PASS, FAIL, or N/A. Agreement % against a fixed threshold isn't meaningful here, because gridding two different orbits produces several cm of RMSD even for perfect data. So the check judges:

- **Bias** — FAIL if |bias (B − A)| exceeds `--max-bias`.
- **RMSD against the noise floor** — each file's ascending passes vs its own descending passes give a self-baseline: the disagreement expected from gridding alone. A baseline cell holds only one pass direction (about half the points), so it is compared with a *matched* RMSD: A vs B on the same ascending/descending half-grids. FAIL if matched RMSD exceeds `--rmsd-factor` × baseline. For data that agree, matched/baseline is ≈ 1.
- **Outlier cells** — FAIL if more than `--max-outlier-cells` common cells have |B − A| above `--outlier-cutoff`. The failure reason gives the size and location of the largest one.
- **N/A** if fewer than `--min-common-cells` usable common cells (e.g. a data gap in either file).

| Flag | Default | Description |
|---|---|---|
| `--outlier-cutoff` | `1.0` | Cells with \|B − A\| above this (m) are outliers, excluded from bias, RMSD, std, baselines, and latitude bands |
| `--max-bias` | `0.02` | Maximum \|bias\| in metres |
| `--rmsd-factor` | `1.2` | Maximum matched RMSD as a multiple of the baseline RMSD |
| `--min-common-cells` | `50` | Minimum usable common cells for a verdict |
| `--max-outlier-cells` | `0` | Maximum outlier cells before FAIL |

These flags work in both `validate-altimetry` and `bulk-validate-altimetry`.

#### Bulk report contents

- **Header** — directories, threshold, matched/unmatched counts
- **Unmatched files** — files present in only one directory
- **Per-file table** — SSHA agreement %, max abs diff, MAE, RMSD, counts MAE, and PASS/FAIL/ERROR/N/A status for each matched pair
- **Aggregate statistics** — mean, median, min, max across all valid pairs for each metric
- **Summary line** — count and percentage of files passing the SSHA agreement threshold

For `along_track_hilat` the report is organised around alignment questions instead:

- **Header** — also lists the alignment-check criteria in use
- **Per-file table** — bias, RMSD, matched RMSD / baseline, r, std ratio B/A, self-baseline RMSD for A and B, noise RMS for A and B, grid cells in A / B / both, outlier cells, max abs diff and the lat/lon of that cell, and alignment status
- **Aggregate statistics** — mean/median/min/max of each per-day metric, including RMSD with and without outliers, median abs diff, and MAD-based std. Followed by the days failing the alignment check with their reasons, and the days with no common cells showing which file was empty
- **Latitude bands** — common and outlier cell counts, bias, RMSD, std A/B and their ratio, and noise A/B per 10° band, pooled over every day and excluding outlier cells. Bands beyond the reference coverage show only high-lat noise
- **Summary line** — days passing out of those judged, with FAIL / N/A / ERROR counts

#### Timeseries plot (`--plot`)

A three-panel figure with a shared date x-axis:
1. **SSHA agreement %** — with a dashed reference line at `--pass-threshold`
2. **SSHA diff metrics** — MAE, RMSD, and max absolute difference
3. **Counts MAE** (or **common grid cells** for `along_track_hilat`)

Error pairs (files that could not be read) are marked with vertical red lines on all panels. Supports any format accepted by matplotlib (`.png`, `.pdf`, `.svg`, etc.).

For `along_track_hilat` the figure has two columns. Left, timeseries: bias with the ±`--max-bias` limits; RMSD (full grids and matched half-grids) alongside both self-baselines and the `--rmsd-factor` limit; r and std ratio B/A; noise RMS for A and B; grid cells in A, B, and both. Days with an error or no common cells get red lines; days failing the alignment check get dotted purple lines. Right, pooled by latitude band: bias and RMSD; std ratio B/A; noise RMS for A and B.

## Report Contents

The report has four sections:

**Dimensions** — flags any dimension size mismatches between the two files.

**Global Attributes** — lists attribute values that differ. Use `--ignore-attrs` to suppress expected differences like `date_created` or `history`.

**Per-Variable Statistics** — for each variable present in either file:
- Shape, dtype, valid cell count, NaN count
- Min, max, mean, std for numeric variables
- Diff metrics (where both files have matching shapes and numeric data):
  - `max_abs` — maximum absolute difference
  - `mean_abs` — mean absolute difference
  - `rmsd` — root mean square difference
  - `bias` — mean signed difference (B − A); a negative bias means B is systematically lower
  - `r` — Pearson correlation coefficient; values near 1.0 indicate strong spatial agreement

**Quality Summary** — product-type-specific metrics:

*along_track:*
- Flag distributions (`good`/`bad`/`total`) for `nasa_flag`, `source_flag`, `median_filter_flag`
- SSHA percentile distributions (p5/p25/p50/p75/p95) for each file

*along_track_hilat:*
- The along_track flag distributions and SSHA percentiles above
- `latitude_range` — min/max latitude per file
- `ssha_gridded` — SSHA points with `nasa_flag == 0` are averaged into 1° lat/lon cells for each file, and the cells observed by both files are compared:
  - over all common cells: max abs diff and its location, RMSD including outliers, median abs diff, MAD-based std of the diff (1.4826 × MAD), and % of cells within `--threshold`
  - excluding outlier cells (|B − A| > `--outlier-cutoff`): MAE, RMSD, bias (B − A), Pearson r, and the std of each file's gridded SSHA with their ratio B/A
- `noise` — RMS of `ssha − ssha_smoothed` at flag-good points for each file: small-scale noise, using every point rather than only the overlap
- `self_baseline` — each file's ascending passes vs its own descending passes, on the same grid. This approximates the RMSD that comes purely from gridding different tracks at different times. A vs B RMSD close to these values means the high-lat data adds little disagreement of its own
- `lat_bands` — per-10°-band common and outlier cell counts, bias, RMSD, std A/B, ratio, and noise A/B
- `alignment` — PASS/FAIL/N/A with reasons, matched RMSD, baseline RMSD, and the limit used
- Per-variable point-by-point diffs are skipped, because the two files' time indices refer to different orbits. The result line reads `ALIGNMENT PASS/FAIL/N/A` instead of FILES MATCH / DIFFERENCES FOUND

*simple_grid:*
- `counts` distribution (min, max, mean, zero-count) per file
- `ssha_coverage` — number and percentage of valid (non-NaN) cells per file
- `ssha_agreement` — percentage of co-located valid cells where |B − A| ≤ threshold

### Interpreting results

| Comparison type | Expected bias | Expected r | Suggested threshold |
|---|---|---|---|
| DEV vs PROD (same processing) | ≈ 0 | ≈ 1.0 | 0.05 m (default) |
| New product vs NASA-SSH, post-offset | Small | High | 0.05 m |
| New product vs NASA-SSH, pre-offset | Known offset | High | 0.10 m or wider |

For `along_track_hilat`, judge RMSD by matched RMSD / baseline (≈ 1 when the high-lat data adds no disagreement of its own) rather than by a fixed threshold. A std ratio B/A near 1 means similar variability; a noise B well above noise A means the high-lat data is noisier at small scales.

A high `r` with a nonzero bias typically means the spatial patterns agree but a systematic offset exists — expected when comparing products before a global offset has been applied.

## Running Tests

All tests use synthetic xarray datasets — no real data files are needed.

```bash
# Using the venv directly
.venv/bin/python -m pytest tests/ -v

# Or with the venv activated
pytest tests/ -v
```

## Project Structure

```
src/validation/
  cli.py                  # CLI entry points (validate-altimetry, bulk-validate-altimetry)
  report.py               # Plain-text report formatting (single-file)
  bulk_compare.py         # Bulk directory comparison logic and dataclasses
  bulk_report.py          # Plain-text report formatting (bulk)
  bulk_plot.py            # Timeseries plot of difference metrics
  comparators/
    base.py               # BaseComparator ABC + result dataclasses
    along_track.py        # AlongTrackComparator
    along_track_hilat.py  # AlongTrackHilatComparator (ref vs high-lat, gridded)
    simple_grid.py        # SimpleGridComparator
  analysis/
    statistics.py         # Per-variable stats and diff computation
    attributes.py         # Global & variable attribute diffing
    dimensions.py         # Dimension comparison
    binning.py            # Along-track → lat/lon grid binning
```
