# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ripple_detection` is a Python package for detecting sharp-wave ripple events (150-250 Hz) from local field potentials (LFPs) in neuroscience research. It implements detection algorithms from Karlsson & Frank 2009, Kay et al. 2016, Yu et al. 2017, Carey et al. 2019, FMAToolbox and buzcode, along with unpublished lab variants.

## Development Commands

### Setup

```bash
# Install from source (development mode with dev dependencies)
pip install -e .[dev,examples]

# Or create conda environment with all dependencies
conda env create -f environment.yml
conda activate ripple_detection
pip install -e .[dev,examples]

# Minimal install (runtime dependencies only)
pip install -e .
```

### Testing

```bash
# Run all tests with coverage
pytest --cov=ripple_detection tests/

# Run one module
pytest tests/test_core.py          # signal processing
pytest tests/test_detectors.py     # detector behavior and conventions
pytest tests/test_simulate.py      # synthetic LFP
pytest tests/test_registry.py      # the detector registry
pytest tests/test_literature.py    # the published-parameter survey
pytest tests/test_public_api.py    # what the package exports
pytest tests/test_properties.py    # property-based (hypothesis)
pytest tests/test_snapshots.py     # regression snapshots

# Run specific test class or function
pytest tests/test_core.py::TestGetEnvelope
pytest tests/test_detectors.py::TestKayRippleDetector::test_single_channel_with_ripples

# Generate HTML coverage report
pytest --cov=ripple_detection --cov-report=html tests/
open htmlcov/index.html

# Test notebooks (as done in CI)
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute examples/detection_examples.ipynb
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute examples/test_individual_algorithm_components.ipynb
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute examples/ripple_detection_tutorial.ipynb
```

### Code Quality

```bash
# Format code with ruff
ruff format ripple_detection/ tests/

# Check formatting without modifying files
ruff format --check ripple_detection/ tests/

# Lint code with ruff
ruff check ripple_detection/ tests/

# Auto-fix ruff issues where possible
ruff check --fix ripple_detection/ tests/

# Type check with mypy
mypy ripple_detection/
```

### Building

```bash
# Build package using modern build tools (recommended)
python -m build

# Build with hatch (if installed)
hatch build
```

### Release Process

When preparing a new release:

```bash
# 1. Run all tests to ensure everything passes
pytest --cov=ripple_detection tests/

# 2. Run code quality checks
ruff format --check ripple_detection/ tests/
ruff check ripple_detection/ tests/
mypy ripple_detection/

# 3. Update CHANGELOG.md
# - Add new version section with date: ## [X.Y.Z] - YYYY-MM-DD
# - Document all changes under appropriate headers:
#   - Added (new features)
#   - Changed (changes to existing functionality)
#   - Deprecated (soon-to-be removed features)
#   - Removed (removed features)
#   - Fixed (bug fixes)
#   - Security (security fixes)
# - List closed issues: "Closes #N"
# - Update comparison links at bottom of file

# 4. Commit the changelog
git add CHANGELOG.md
git commit -m "Update CHANGELOG for vX.Y.Z release"
git push origin master

# 5. Create and push annotated git tag
git tag -a vX.Y.Z -m "Release vX.Y.Z

## New Features
- Feature description

## Improvements
- Improvement description

Closes #N"

git push origin vX.Y.Z

# The tag push triggers the automated GitHub Actions release workflow:
# - Runs tests on Python 3.10, 3.11, 3.12, 3.13
# - Builds source distribution and wheels
# - Publishes to PyPI
# - Creates GitHub release with auto-generated notes
```

**Important Notes:**
- Always update CHANGELOG.md BEFORE creating the tag
- The tag must be an annotated tag (use `-a` flag) with a meaningful message
- Version follows semantic versioning (MAJOR.MINOR.PATCH)
- The version in `ripple_detection/_version.py` is auto-generated from the git tag by hatch-vcs
- Monitor the release workflow at: https://github.com/Eden-Kramer-Lab/ripple_detection/actions

## Architecture

### Core Module Structure

The package is organized into five modules:

1. **[ripple_detection/core.py](ripple_detection/core.py)** - Low-level signal processing utilities
   - Bandpass filtering for ripple band (150-250 Hz)
   - Envelope extraction via Hilbert transform
   - Gaussian smoothing
   - Threshold detection and segment extraction
   - Movement exclusion based on speed
   - Utility functions for time series segmentation

2. **[ripple_detection/detectors.py](ripple_detection/detectors.py)** - High-level detection algorithms
   - `Kay_ripple_detector` - Multi-channel consensus approach (Kay et al. 2016)
   - `Karlsson_ripple_detector` - Per-channel detection with merging (Karlsson & Frank 2009)
   - `Roumis_ripple_detector` - Per-channel envelopes averaged (Frank-lab variant, unpublished)
   - `Shvartsman_ripple_detector` - Per-channel detection requiring a minimum number of participating channels (unpublished)
   - `Yu_ripple_detector` - Median consensus with a data-driven noise-percentile threshold (Yu et al. 2017)
   - `Zugaro_ripple_detector` - FMAToolbox `FindRipples` two-threshold rule
   - `Long_sharp_wave_ripple_detector` - Sharp wave + ripple power on two raw channels, k-means split (Long, `DetectSWR`)
   - `Carey_candidate_detector` - Joint ripple-power × multiunit score (Carey, Tanaka & van der Meer 2019)
   - `multiunit_HSE_detector` - Multiunit High Synchrony Event detector (spikes only)
   - The README's "Choosing a detector" table is the reference for how their conventions differ
   - All detectors return pandas DataFrames with event statistics

3. **[ripple_detection/simulate.py](ripple_detection/simulate.py)** - Synthetic data generation
   - Simulate LFPs with embedded ripples
   - Multiple noise types (white, pink, brown)
   - Used for testing and validation

4. **[ripple_detection/registry.py](ripple_detection/registry.py)** - `DETECTORS`, `get_detector`, `DetectorSpec`
   - Resolves a detector by name and says which signal kind it takes (`RIPPLE_BAND_LFP`, `RAW_LFP_PAIR`, `MULTIUNIT`)
   - For pipelines that store a detector's name rather than importing it

5. **[ripple_detection/literature.py](ripple_detection/literature.py)** - `load_literature_parameters`
   - The survey of detection parameters from 57 replay papers, shipped as `data/literature_detection_parameters.csv`
   - The README's "Published parameter values" table is computed from it

### Detection Pipeline Architecture

Kay, Karlsson, Roumis, Shvartsman and the HSE detector share one pipeline; `_detect_from_trace` is its tail:

1. **Preprocessing**: Validate shapes, units and time order; mark samples with NaN in any signal or in speed as missing and split the rest into contiguous blocks (`_valid_blocks`), ending a block also wherever the timestamp step exceeds 1.5 times the median step
2. **Signal Transformation**: Hilbert envelope and Gaussian smoothing within each contiguous block (`_smoothed_envelope`); combine channels (Kay: consensus trace; Karlsson and Shvartsman: per channel; Roumis: mean; HSE: population rate)
3. **Normalization**: Z-score (or median/MAD) the trace; a zero or undefined scale raises
4. **Threshold Detection**: Runs at or above the threshold for at least `minimum_sample_count` samples
5. **Extension to Mean**: Extend each run to where the trace returns to the mean
6. **Movement Exclusion**: Speed at the first and last sample at or below `speed_threshold` (Shvartsman: majority of samples)
7. **Post-processing**: Drop close events, then over-long events; compute statistics with `_get_event_stats`
8. **Output**: DataFrame indexed by `event_number` with the columns listed under "Output Format" in the README

Yu, Zugaro, Long and Carey use their own segmentation rules but the same blocks: every step of every detector runs within a block, no event spans a gap, and `_get_event_stats` flags events cut off by a block edge in `clipped_start` and `clipped_end` (Zugaro supplies its own flags, meaning a missing crossing). A block too short for a detector's transform is treated as missing with a warning (`_drop_short_blocks`): Zugaro's smoothing window, Long's sharp-wave low-pass kernel, and Carey's theta filter pad length when `theta_lfp` is given.

### Key Algorithm Differences

- **Kay detector**: Combines multiple LFP channels into single consensus trace using sum of squared envelopes
- **Karlsson detector**: Detects ripples on each channel independently, then merges overlapping events
- **Roumis detector**: Averages square-root of squared envelopes across channels
- **Yu detector**: Median of per-channel z-scored envelopes; threshold from the mirrored immobility-noise histogram
- **Zugaro detector**: Two thresholds (bounds and peak) on the z-scored squared sum; merges close events
- **Long detector**: Raw two-channel input; sharp-wave difference and ripple power clustered by k-means with local statistics
- **Carey detector**: Geometric mean of a ripple-power score and a capped multiunit score; whole event inside a low-speed interval
- **HSE detector**: Z-scored smoothed population spike rate, no LFP
- One missing-sample policy for every detector: NaN in any signal or speed, or a gap in time, ends a block; nothing crosses a gap; clipped events are flagged

### Pre-computed Filter

The package includes a pre-computed ripple bandpass filter ([ripple_detection/ripplefilter.mat](ripple_detection/ripplefilter.mat)) from the Frank lab with specific characteristics:

- 150-250 Hz bandpass
- 40 dB roll-off
- 10 Hz sidebands
- Sampling frequency: 1500 Hz

The shipped kernel is used only at 1500 Hz with the default band; for any other rate or band, `filter_ripple_band` designs an equiripple FIR with `ripple_bandpass_filter()` (`scipy.signal.remez`), scaling the tap count with the rate.

### Event Statistics

All detectors return rich event statistics via `_get_event_stats()`:

- Temporal: start_time, end_time, duration (elapsed), n_samples (the count the duration limits test)
- Z-score metrics: mean, median, max, min, max_sustained_zscore (largest z-score sustained for the minimum duration; `max_thresh` before 2.0)
- Signal metrics: area (integral), total_energy (integral of squared signal)
- Speed metrics: speed at start/end, max/min/median/mean speed during event
- Missing data: clipped_start, clipped_end (the event was cut off by a gap or the recording edge)
- Detector-specific extras: Shvartsman `participants` (sorted tuple), `n_participants`, `frac_participants`; Yu `n_suprathreshold_samples`, `detection_threshold_zscore`; Zugaro and Long `peak_time`; Long sharp-wave and ripple statistics; Carey and HSE `n_active_units`. The README's "Output Format" tables are the reference.

## Testing Strategy

**Test Coverage: 97%** (core and detector modules each 97%; registry, literature and simulate 100%)

The suite has one shared fixture module and eight test modules:

1. **[tests/conftest.py](tests/conftest.py)** - Shared pytest fixtures: 1500 Hz LFP simulations with various ripple patterns, speed data (stationary and moving), multiunit spike trains, edge cases, and the class-aware `time` and `stationary` fixtures
2. **[tests/test_core.py](tests/test_core.py)** - Core signal processing: segmentation, extension, merging, the duration and gap boundary rules, normalization (including the degenerate-scale errors), filtering at several rates and across NaN gaps, envelope, smoothing, the Yu noise-threshold estimator
3. **[tests/test_detectors.py](tests/test_detectors.py)** - One test class per detector plus shared classes for error handling, participation, duration and speed conventions, exclusion order, time ordering, and missing samples; a class states `FS` and `N_TIME` and gets `time` and `stationary` fixtures from conftest, and builds inputs with the helpers in [tests/_synthetic.py](tests/_synthetic.py)
4. **[tests/test_simulate.py](tests/test_simulate.py)** - Noise spectra, embedded ripples, per-ripple ranges, `ripple_snr`, and the inputs that used to give an all-NaN signal
5. **[tests/test_properties.py](tests/test_properties.py)** - Hypothesis-driven invariants for the signal-processing functions
6. **[tests/test_snapshots.py](tests/test_snapshots.py)** - Regression snapshots of detector output on fixed simulated data
7. **[tests/test_public_api.py](tests/test_public_api.py)** - Pins `__all__`
8. **[tests/test_registry.py](tests/test_registry.py)** - Every exported detector is registered; each spec matches its signature by kind and position
9. **[tests/test_literature.py](tests/test_literature.py)** - The shipped survey loads with the documented shape and types

**Test Execution**: about 580 tests in ~11 seconds (`pytest --collect-only -q | tail -1` for the current count)

The package also validates that example notebooks run without errors in CI.

## Build System

**Modern pyproject.toml-based build**:

- Uses `hatchling` as build backend (PEP 517/518/621 compliant)
- Dynamic versioning via `hatch-vcs` from git tags
- Version automatically determined from git tags (e.g., `v1.5.1`)
- Fallback version in `ripple_detection/_version.py`
- Pure pyproject.toml - no setup.py or setup.cfg needed

**Python version**: Requires Python >= 3.10

**Optional dependencies**:

- `dev` - Development tools (pytest, pytest-cov, ruff, mypy, hypothesis, pytest-snapshot)
- `examples` - Jupyter and visualization tools (matplotlib, jupyter, jupyterlab)

## Dependencies

**Core** (minimum versions):

- numpy >= 1.24
- scipy >= 1.10
- pandas >= 2.0

**Development** (minimum versions):

- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- ruff >= 0.16, < 0.17 (format output is stable within a minor version)
- mypy >= 1.8.0
- hypothesis >= 6.0.0 (property-based testing)
- pytest-snapshot >= 0.9.0 (snapshot testing)

**Examples** (minimum versions):

- matplotlib >= 3.5.0
- jupyter >= 1.0.0
- jupyterlab >= 3.0.0

## Standards

- Follows PEP 8 style guide
- **Type hints for all function signatures** (using modern Python 3.10+ syntax)
- Numpy Docstrings for all public functions and classes using numpy docstring best practices
- Uses f-strings for formatting
- Modular functions with single responsibility
- Comprehensive test coverage: 97% overall
- **Code quality tools**: Ruff (formatting and linting), Mypy (type checking)
- Continuous integration with GitHub Actions (tests on Python 3.10, 3.11, 3.12, 3.13)

### Type Hints

All functions in the codebase use type hints with modern Python 3.10+ syntax:
- `X | Y` instead of `Union[X, Y]`
- `X | None` instead of `Optional[X]`
- `list[X]`, `dict[K, V]`, `tuple[X, Y]` instead of `List[X]`, `Dict[K, V]`, `Tuple[X, Y]`
- `collections.abc.Generator` for generators
- `numpy.typing.ArrayLike` and `NDArray` for numpy array parameters and return types

The mypy configuration in `pyproject.toml` includes pragmatic overrides to avoid false positives with numpy's `ArrayLike` type while maintaining type safety.

### Tool Configuration

All code quality tools are configured in [pyproject.toml](pyproject.toml):

**Ruff** (`[tool.ruff]` and `[tool.ruff.lint]`) — the single formatter and linter:
- Line length: 95
- Target: Python 3.10
- Enabled checks: pycodestyle (E/W), pyflakes (F), isort (I), flake8-bugbear (B), comprehensions (C4), pyupgrade (UP), NumPy (NPY), pandas-vet (PD), Ruff-specific (RUF)
- Ignores E501 (line too long) since `ruff format` handles wrapping

**Mypy** (`[tool.mypy]`):
- Target: Python 3.10
- `ignore_missing_imports = true` (for scipy, pandas - no stubs installed)
- Module overrides disable specific error codes that cause false positives with `ArrayLike`

**Pytest** (`[tool.pytest.ini_options]`):
- Auto coverage reporting to terminal with missing lines
- Test path: `tests/`

Use the `ripple_detection` conda environment if available for testing and development to ensure consistent dependency versions.
