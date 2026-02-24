# altimetry-processing-validation

Comparison toolkit for altimetry NetCDF product files. Takes two files of the same product type and produces a structured report of their differences, including per-variable statistics, bias, spatial correlation, and product-specific quality metrics.

## Supported Product Types

- **along_track** — Level 2, 1D time-indexed daily files
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

```bash
# Compare two along-track files
validate-altimetry file_a.nc file_b.nc -t along_track

# Compare two simple grids
validate-altimetry file_a.nc file_b.nc -t simple_grid

# Ignore timestamp attributes that are expected to differ
validate-altimetry file_a.nc file_b.nc -t simple_grid --ignore-attrs date_created history

# Use a wider threshold for pre-offset comparisons (default is 0.05 m)
validate-altimetry file_a.nc file_b.nc -t simple_grid --threshold 0.10
```

Exit code 0 means files match; exit code 1 means differences were found.

### Options

| Flag | Default | Description |
|---|---|---|
| `-t`, `--product-type` | *(required)* | `along_track` or `simple_grid` |
| `--ignore-attrs` | none | Global or variable attribute names to exclude from comparison |
| `--threshold` | `0.05` | Absolute difference threshold in metres for the `pct_within_threshold` metric (simple_grid only) |

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
  cli.py                  # CLI entry point and argument parsing
  report.py               # Plain-text report formatting
  comparators/
    base.py               # BaseComparator ABC + result dataclasses
    along_track.py        # AlongTrackComparator
    simple_grid.py        # SimpleGridComparator
  analysis/
    statistics.py         # Per-variable stats and diff computation
    attributes.py         # Global & variable attribute diffing
    dimensions.py         # Dimension comparison
```
