# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ripple_detection` is a Python package for detecting sharp-wave ripple events (150-250 Hz) from local field potentials (LFPs) in neuroscience research. It implements detection algorithms from Karlsson et al. 2009 and Kay et al. 2016, along with other variants.

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
# Run all tests with coverage (98% coverage achieved!)
pytest --cov=ripple_detection tests/

# Run specific test module
pytest tests/test_core.py          # Core signal processing tests
pytest tests/test_detectors.py     # Detector integration tests
pytest tests/test_simulate.py      # Simulation module tests

# Run specific test class or function
pytest tests/test_core.py::TestGetEnvelope
pytest tests/test_detectors.py::TestKayRippleDetector::test_single_channel_with_ripples

# Generate HTML coverage report
pytest --cov=ripple_detection --cov-report=html tests/
open htmlcov/index.html

# Test notebooks (as done in CI)
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute examples/detection_examples.ipynb
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute examples/test_individual_algorithm_components.ipynb
```

### Code Quality

```bash
# Lint code with flake8
flake8 ripple_detection/

# Format code with black
black ripple_detection/ tests/

# Check formatting without modifying files
black --check ripple_detection/ tests/
```

### Building

```bash
# Build package using modern build tools (recommended)
python -m build

# Build with hatch (if installed)
hatch build
```

## Architecture

### Core Module Structure

The package is organized into three main modules:

1. **[ripple_detection/core.py](ripple_detection/core.py)** - Low-level signal processing utilities
   - Bandpass filtering for ripple band (150-250 Hz)
   - Envelope extraction via Hilbert transform
   - Gaussian smoothing
   - Threshold detection and segment extraction
   - Movement exclusion based on speed
   - Utility functions for time series segmentation

2. **[ripple_detection/detectors.py](ripple_detection/detectors.py)** - High-level detection algorithms
   - `Kay_ripple_detector` - Multi-channel consensus approach (Kay et al. 2016)
   - `Karlsson_ripple_detector` - Per-channel detection with merging (Karlsson et al. 2009)
   - `Roumis_ripple_detector` - Variant detection method
   - `multiunit_HSE_detector` - Multiunit High Synchrony Event detector
   - All detectors return pandas DataFrames with event statistics

3. **[ripple_detection/simulate.py](ripple_detection/simulate.py)** - Synthetic data generation
   - Simulate LFPs with embedded ripples
   - Multiple noise types (white, pink, brown)
   - Used for testing and validation

### Detection Pipeline Architecture

All ripple detectors follow a common pipeline:

1. **Preprocessing**: Remove NaN values, align time/speed/LFP data
2. **Signal Transformation**:
   - Apply Hilbert transform to get envelope (instantaneous amplitude)
   - Gaussian smoothing with configurable sigma
   - Combine signals (varies by detector - Kay uses consensus trace, Karlsson uses per-channel)
3. **Normalization**: Z-score the transformed signal
4. **Threshold Detection**: Find segments above z-score threshold that persist for minimum duration
5. **Extension to Mean**: Extend threshold crossings to where signal crosses mean
6. **Movement Exclusion**: Remove events where animal speed exceeds threshold
7. **Post-processing**: Exclude events too close together, calculate statistics
8. **Output**: Return DataFrame with event times and comprehensive statistics (duration, max_thresh, mean/median/max/min z-score, area, total_energy, speed metrics)

### Key Algorithm Differences

- **Kay detector**: Combines multiple LFP channels into single consensus trace using sum of squared envelopes
- **Karlsson detector**: Detects ripples on each channel independently, then merges overlapping events
- **Roumis detector**: Averages square-root of squared envelopes across channels

### Pre-computed Filter

The package includes a pre-computed ripple bandpass filter ([ripple_detection/ripplefilter.mat](ripple_detection/ripplefilter.mat)) from the Frank lab with specific characteristics:

- 150-250 Hz bandpass
- 40 dB roll-off
- 10 Hz sidebands
- Sampling frequency: 1500 Hz

Alternative: `ripple_bandpass_filter()` can generate filters at arbitrary sampling rates using `scipy.signal.remez`.

### Event Statistics

All detectors return rich event statistics via `_get_event_stats()`:

- Temporal: start_time, end_time, duration
- Z-score metrics: mean, median, max, min, max_thresh (max threshold sustained for minimum duration)
- Signal metrics: area (integral), total_energy (integral of squared signal)
- Speed metrics: speed at start/end, max/min/median/mean speed during event

## Testing Strategy

**Test Coverage: 98%** (100% on core modules)

The test suite is organized into four modules:

1. **[tests/conftest.py](tests/conftest.py)** - Shared pytest fixtures
   - 15 fixtures providing reusable test data
   - LFP simulations with various ripple patterns
   - Speed data (stationary and movement scenarios)
   - Multiunit spike train data
   - Edge cases (no ripples, short duration, close ripples)

2. **[tests/test_core.py](tests/test_core.py)** - Core signal processing (52 tests, 100% coverage)
   - Boolean series segmentation (start/end time extraction)
   - Interval finding and extension
   - Overlapping range merging
   - Z-score thresholding and movement exclusion
   - Ripple band filtering
   - Hilbert transform envelope extraction
   - Gaussian smoothing
   - Multiunit population firing rate
   - Error handling for edge cases

3. **[tests/test_detectors.py](tests/test_detectors.py)** - Detector integration tests (25 tests, 100% coverage)
   - Kay_ripple_detector (9 tests): multi-channel consensus, parameter validation
   - Karlsson_ripple_detector (4 tests): per-channel detection with merging
   - Roumis_ripple_detector (2 tests): variant algorithm validation
   - multiunit_HSE_detector (3 tests): synchrony event detection
   - Error handling (5 tests): NaN values, empty arrays, mismatched lengths
   - Helper functions (2 tests): consensus trace generation

4. **[tests/test_simulate.py](tests/test_simulate.py)** - Simulation module (40 tests, 92% coverage)
   - Time array generation
   - Noise generation (white, pink, brown) with frequency analysis
   - LFP simulation with embedded ripples
   - Parameter validation (amplitude, duration, noise types)
   - Statistical validation and power spectrum analysis
   - Error handling for edge cases

**Test Execution**: 107 tests pass in <1 second

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

- `dev` - Development tools (pytest, black, flake8, pytest-cov)
- `examples` - Jupyter and visualization tools (matplotlib, jupyter, jupyterlab)

## Dependencies

**Core** (minimum versions):

- numpy >= 1.23.0
- scipy >= 1.9.0
- pandas >= 1.5.0

**Development** (minimum versions):

- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- flake8 >= 6.0.0
- black[jupyter] >= 23.0.0

**Examples** (minimum versions):

- matplotlib >= 3.5.0
- jupyter >= 1.0.0
- jupyterlab >= 3.0.0

## Standards

- Follows PEP 8 style guide
- Type hints for function signatures
- Numpy Docstrings for all public functions and classes using numpy docstring best practices
- Uses f-strings for formatting
- Modular functions with single responsibility
- Comprehensive test coverage: 98% overall, 100% on core modules
- Continuous integration with GitHub Actions (tests on Python 3.10, 3.11, 3.12, 3.13)

Use the `ripple_detection` conda environment if available for testing and development to ensure consistent dependency versions.
