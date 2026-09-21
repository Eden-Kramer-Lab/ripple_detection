# ripple_detection

[![PyPI version](https://badge.fury.io/py/ripple-detection.svg)](https://badge.fury.io/py/ripple-detection)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Test, Build, and Publish](https://github.com/Eden-Kramer-Lab/ripple_detection/actions/workflows/release.yml/badge.svg)](https://github.com/Eden-Kramer-Lab/ripple_detection/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/Eden-Kramer-Lab/ripple_detection/branch/master/graph/badge.svg)](https://codecov.io/gh/Eden-Kramer-Lab/ripple_detection)

A Python package for detecting [sharp-wave ripple](https://en.wikipedia.org/wiki/Sharp_waves_and_ripples) events (150-250 Hz) from local field potentials (LFPs) in neuroscience research.

## Features

- **Multiple Detection Algorithms**
  - `Kay_ripple_detector` - Multi-channel consensus approach (Kay et al. 2016)
  - `Karlsson_ripple_detector` - Per-channel detection with merging (Karlsson et al. 2009)
  - `Shvartsman_ripple_detector` - Per-channel detection requiring a minimum fraction of participating channels
  - `Roumis_ripple_detector` - Alternative detection method
  - `Yu_ripple_detector` - Median consensus with a data-driven noise-percentile threshold (Yu et al. 2017)
  - `Zugaro_ripple_detector` - The FMAToolbox/buzcode `FindRipples` two-threshold algorithm (Hirase; Zugaro)
  - `Long_sharp_wave_ripple_detector` - Two-channel detector using the sharp wave on a stratum radiatum channel (J. D. Long II, buzcode `bz_DetectSWR`); takes **raw** LFP
  - `multiunit_HSE_detector` - High Synchrony Event detection from multiunit activity

- **Comprehensive Event Statistics**
  - Temporal metrics (start time, end time, duration)
  - Z-score metrics (mean, median, max, min, sustained threshold)
  - Signal metrics (area under curve, total energy)
  - Movement metrics (speed during event)

- **Flexible Signal Processing**
  - Bandpass filtering (150-250 Hz)
  - Envelope extraction via Hilbert transform
  - Gaussian smoothing with configurable parameters
  - Movement exclusion based on speed thresholds

- **Simulation Tools**
  - Generate synthetic LFPs with embedded ripples
  - Ripple size set relative to the ripple-band background (`ripple_snr`), which is what the detectors see
  - Per-ripple frequency and duration drawn from ranges; multiple noise types (white, pink, brown)
  - Useful for testing and validation

## Installation

### From PyPI

```bash
pip install ripple_detection
```

### From Conda

```bash
conda install -c edeno ripple_detection
```

### From Source

```bash
# Clone the repository
git clone https://github.com/Eden-Kramer-Lab/ripple_detection.git
cd ripple_detection

# Install with optional dependencies
pip install -e .[dev,examples]
```

## Requirements

- Python >= 3.10
- numpy >= 1.23.0
- scipy >= 1.9.0
- pandas >= 1.5.0

## Quick Start

### Basic Usage

```python
from ripple_detection import Kay_ripple_detector, filter_ripple_band
import numpy as np

# Your data (replace the random arrays with real recordings)
sampling_frequency = 1500  # Hz; pass your true rate to filter_ripple_band
time = np.arange(0, 10, 1 / sampling_frequency)  # 10 seconds
LFPs = np.random.randn(len(time), 4)  # 4 channels of raw LFP data
speed = np.abs(np.random.randn(len(time)))  # Animal speed (cm/s)

# Filter into the ripple band (150-250 Hz) first: the detectors expect
# ripple-band-filtered LFPs, not raw signal.
filtered_lfps = filter_ripple_band(LFPs, sampling_frequency=sampling_frequency)

# Detect ripples
ripple_times = Kay_ripple_detector(
    time, filtered_lfps, speed, sampling_frequency,
    speed_threshold=4.0,        # cm/s
    minimum_duration=0.015,     # seconds
    zscore_threshold=2.0
)

print(ripple_times)
```

### Data-driven threshold (Yu et al. 2017)

```python
from ripple_detection import Yu_ripple_detector

# No z-score threshold to choose: the threshold is the 99.99th percentile of
# the noise distribution estimated from immobility, per call. Missing samples
# (NaN) are handled block-wise and never smoothed across; an event cut off by
# a gap is kept and flagged in `clipped_start` / `clipped_end`.
ripples = Yu_ripple_detector(
    time, filtered_lfps, speed, sampling_frequency,
    speed_threshold=4.0,
    minimum_duration=0.020,     # the published 20 ms
    percentile=99.99,
)
print(ripples[["start_time", "end_time", "detection_threshold_zscore"]])
```

### Advanced Usage

```python
from ripple_detection import Karlsson_ripple_detector

# Detect ripples with custom parameters (filtered_lfps from the Basic example)
ripples = Karlsson_ripple_detector(
    time, filtered_lfps, speed, sampling_frequency,
    speed_threshold=4.0,
    minimum_duration=0.015,
    zscore_threshold=3.0,
    smoothing_sigma=0.004,
    close_ripple_threshold=0.0
)

# Access detailed statistics
print(f"Detected {len(ripples)} ripple events")
print(f"Mean duration: {ripples['duration'].mean():.3f} seconds")
print(f"Mean z-score: {ripples['mean_zscore'].mean():.2f}")
```

## Output Format

All detectors return a pandas DataFrame with comprehensive event statistics:

| Column | Description |
|--------|-------------|
| `start_time` | Event start time |
| `end_time` | Event end time |
| `duration` | Event duration (seconds) |
| `max_thresh` | Maximum sustained threshold |
| `mean_zscore` | Mean z-score during event |
| `median_zscore` | Median z-score during event |
| `max_zscore` | Maximum z-score during event |
| `min_zscore` | Minimum z-score during event |
| `area` | Integral of z-score over time |
| `total_energy` | Integral of squared z-score |
| `speed_at_start` | Animal speed at event start |
| `speed_at_end` | Animal speed at event end |
| `max_speed` | Maximum speed during event |
| `min_speed` | Minimum speed during event |
| `median_speed` | Median speed during event |
| `mean_speed` | Mean speed during event |

## Examples

### Simulating realistic ripples

`simulate_LFP`'s default brown noise leaves very little power in the 150-250 Hz band (and
less the longer the record), so a ripple of any visible amplitude dominates the band. For
detector testing, set the ripple size relative to the ripple-band background with
`ripple_snr` and use pink noise, which gives a band background closer to recordings:

```python
from ripple_detection.simulate import simulate_LFP, simulate_time

time = simulate_time(15000, 1500)
lfp = simulate_LFP(
    time, [2.0, 5.0, 8.0],
    noise_type="pink",
    ripple_snr=5,                  # ripple peak = 5 x ripple-band noise SD
    ripple_frequency=(150, 250),   # drawn per ripple
    ripple_duration=(0.04, 0.12),  # drawn per ripple, seconds
    random_state=0,
)
```

`ripple_snr` is the ripple's peak after ripple-band filtering divided by the filtered
noise's SD, set per ripple so it holds at the band edges and for short bursts. The
z-score a detector reports is larger, by a factor that depends on its smoothing and
consensus rule; measure it for the detector you use rather than assuming a mapping.

See the [examples](examples/) directory for Jupyter notebooks demonstrating:

- [Detection Examples](examples/detection_examples.ipynb) - Using different detectors
- [Algorithm Components](examples/test_individual_algorithm_components.ipynb) - Testing individual components

## Troubleshooting

### Common Errors

#### "axis 1 is out of bounds" or "must be a 2D array"

Your LFP data must be 2D with shape `(n_time, n_channels)`. Even for a single channel, the array must be 2D.

```python
# Wrong - 1D array
lfps = np.random.randn(1000)  # Shape: (1000,)

# Correct - 2D array with single channel
lfps = np.random.randn(1000, 1)  # Shape: (1000, 1)
# OR reshape existing 1D array:
lfps = lfps.reshape(-1, 1)
```

#### "Array length mismatch detected"

Your `time`, `LFPs`, and `speed` arrays must have the same length. Check dimensions:

```python
print(f"time: {len(time)}, LFPs: {len(lfps)}, speed: {len(speed)}")
```

Make sure all arrays cover the same time period and sampling rate.

#### "Sampling frequency ... cannot represent the 150-250 Hz ripple band"

`filter_ripple_band(data, sampling_frequency=...)` uses the pre-computed 1500 Hz
kernel at 1500 Hz and designs a 150-250 Hz filter for any other rate, so pass the
true sampling rate. Rates at or below 550 Hz cannot hold the band and raise. To
build the filter yourself:

```python
from ripple_detection import ripple_bandpass_filter
from scipy.signal import filtfilt

filter_num, filter_denom = ripple_bandpass_filter(sampling_frequency)
filtered_lfps = filtfilt(filter_num, filter_denom, raw_lfps, axis=0)
```

#### No ripples detected (empty DataFrame)

If detection returns no events, try adjusting parameters:

```python
ripples = Kay_ripple_detector(
    time, filtered_lfps, speed, sampling_frequency,
    zscore_threshold=1.5,      # Lower from default 2.0
    minimum_duration=0.010,    # Lower from default 0.015
    speed_threshold=10.0       # Increase if too restrictive (default 4.0)
)
```

**Diagnostic steps:**
1. Check if your LFPs actually contain ripples (150-250 Hz oscillations)
2. Verify speed is in cm/s (not m/s)
3. Plot the filtered LFP to visually inspect for ripple events
4. Try different detector algorithms (Kay, Karlsson, Roumis)

### Parameter Selection Guide

| Parameter | Default | Description | When to Adjust |
|-----------|---------|-------------|----------------|
| `speed_threshold` | 4.0 cm/s | Maximum speed for ripple detection | Increase if too many events excluded during slow movement |
| `minimum_duration` | 0.015 s | Minimum ripple duration (15 ms) | Decrease to 0.010 for shorter ripples; increase to 0.020 for stricter detection |
| `zscore_threshold` | 2.0 (Kay/Roumis)<br>3.0 (Karlsson) | Detection sensitivity | Decrease for more detections; increase for fewer, higher-confidence events |
| `smoothing_sigma` | 0.004 s | Gaussian smoothing window (4 ms) | Rarely needs adjustment; increase for noisier data |
| `percentile` | 99.99 (Yu) | Percentile of the mirrored immobility-noise distribution used as the threshold | Lower for more detections; the threshold is estimated per call, so it adapts to each recording |
| `low_threshold`, `high_threshold` | 2.0, 5.0 (Zugaro) | Boundary and peak thresholds of the two-threshold rule | Lower `high_threshold` for more detections; `low_threshold` sets where events start and end |

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/Eden-Kramer-Lab/ripple_detection/issues)
- **Discussions**: For questions about usage and parameter selection
- **Email**: [edeno@bu.edu](mailto:edeno@bu.edu)

## Documentation

For detailed documentation on the detection algorithms and signal processing pipeline, see [CLAUDE.md](CLAUDE.md).

## Algorithm Comparison

| Algorithm | Approach | Best For |
|-----------|----------|----------|
| **Kay** | Multi-channel consensus (sum of squared envelopes) | High-density electrode arrays |
| **Karlsson** | Per-channel detection with merging | Independent channel analysis |
| **Roumis** | Averaged square-root of squared envelopes | Balanced multi-channel approach |

## Development

### Setup Development Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate ripple_detection

# Install in editable mode with dev dependencies
pip install -e .[dev,examples]
```

### Run Tests

The package has comprehensive test coverage (93%) across 163 tests organized in 6 modules:

```bash
# Run all tests with coverage
pytest --cov=ripple_detection --cov-report=term-missing tests/

# Run specific test modules
pytest tests/test_core.py          # Core signal processing tests (70 tests)
pytest tests/test_detectors.py     # Detector integration tests (25 tests)
pytest tests/test_simulate.py      # Simulation module tests (36 tests)
pytest tests/test_properties.py    # Property-based tests (23 tests)
pytest tests/test_snapshots.py     # Snapshot/regression tests (9 tests)

# Run specific test
pytest tests/test_core.py::TestSegmentBooleanSeries::test_single_segment

# Generate HTML coverage report
pytest --cov=ripple_detection --cov-report=html tests/
# Open htmlcov/index.html in browser
```

**Test Coverage:**
- `core.py`: 93%
- `detectors.py`: 92%
- `simulate.py`: 100%
- **Overall: 93%**

### Code Quality

```bash
# Format code with ruff
ruff format ripple_detection/ tests/

# Lint code with ruff
ruff check ripple_detection/ tests/

# Type check with mypy
mypy ripple_detection/

# Check formatting without modifying
ruff format --check ripple_detection/ tests/
```

### Release Process

Releases are automated via GitHub Actions when a version tag is pushed:

```bash
# 1. Ensure all tests pass and code quality checks pass
pytest --cov=ripple_detection tests/
ruff format --check ripple_detection/ tests/
ruff check ripple_detection/ tests/
mypy ripple_detection/

# 2. Update CHANGELOG.md with new version and changes
# - Add ## [X.Y.Z] - YYYY-MM-DD section
# - Document changes under Added/Changed/Deprecated/Removed/Fixed/Security
# - Update comparison links at bottom

# 3. Commit and push changelog
git add CHANGELOG.md
git commit -m "Update CHANGELOG for vX.Y.Z release"
git push origin master

# 4. Create and push annotated tag (triggers release workflow)
git tag -a vX.Y.Z -m "Release vX.Y.Z with feature descriptions"
git push origin vX.Y.Z
```

The automated workflow will:
- Run tests on Python 3.10, 3.11, 3.12, 3.13
- Build source distribution and wheels
- Publish to PyPI
- Create GitHub release

**Note:** Version numbers follow [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH). The package version is automatically determined from git tags via `hatch-vcs`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this package in your research, please cite the original papers:

### Karlsson Method

```bibtex
@article{karlsson2009awake,
  title={Awake replay of remote experiences in the hippocampus},
  author={Karlsson, Mattias P and Frank, Loren M},
  journal={Nature neuroscience},
  volume={12},
  number={7},
  pages={913--918},
  year={2009},
  publisher={Nature Publishing Group}
}
```

### Kay Method

```bibtex
@article{kay2016hippocampal,
  title={A hippocampal network for spatial coding during immobility and sleep},
  author={Kay, Kenneth and Sosa, Marielena and Chung, Jason E and Karlsson, Mattias P and Larkin, Margaret C and Frank, Loren M},
  journal={Nature},
  volume={531},
  number={7593},
  pages={185--190},
  year={2016},
  publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Eric Denovellis** - [edeno@bu.edu](mailto:edeno@bu.edu)

## Acknowledgments

- Frank Lab for the pre-computed ripple filter
- Original algorithm implementations by Karlsson et al. and Kay et al.

## Support

- **Issues**: [GitHub Issues](https://github.com/Eden-Kramer-Lab/ripple_detection/issues)
- **Discussions**: For questions and discussions about usage
- **Email**: [edeno@bu.edu](mailto:edeno@bu.edu)
