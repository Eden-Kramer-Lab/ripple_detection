# ripple_detection

[![PyPI version](https://badge.fury.io/py/ripple-detection.svg)](https://badge.fury.io/py/ripple-detection)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PR Test](https://github.com/Eden-Kramer-Lab/ripple_detection/actions/workflows/PR-test.yml/badge.svg)](https://github.com/Eden-Kramer-Lab/ripple_detection/actions/workflows/PR-test.yml)
[![Coverage Status](https://coveralls.io/repos/github/Eden-Kramer-Lab/ripple_detection/badge.svg?branch=master)](https://coveralls.io/github/Eden-Kramer-Lab/ripple_detection?branch=master)

A Python package for detecting [sharp-wave ripple](https://en.wikipedia.org/wiki/Sharp_waves_and_ripples) events (150-250 Hz) from local field potentials (LFPs) in neuroscience research.

## Features

- **Multiple Detection Algorithms**
  - `Kay_ripple_detector` - Multi-channel consensus approach (Kay et al. 2016)
  - `Karlsson_ripple_detector` - Per-channel detection with merging (Karlsson et al. 2009)
  - `Roumis_ripple_detector` - Alternative detection method
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
  - Multiple noise types (white, pink, brown)
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
from ripple_detection import Kay_ripple_detector
import numpy as np

# Your data
time = np.arange(0, 10, 0.001)  # 10 seconds at 1000 Hz
LFPs = np.random.randn(len(time), 4)  # 4 channels of LFP data
speed = np.abs(np.random.randn(len(time)))  # Animal speed
sampling_frequency = 1000  # Hz

# Detect ripples
ripple_times = Kay_ripple_detector(
    time, LFPs, speed, sampling_frequency,
    speed_threshold=4.0,        # cm/s
    minimum_duration=0.015,     # seconds
    zscore_threshold=2.0
)

print(ripple_times)
```

### Advanced Usage

```python
from ripple_detection import Karlsson_ripple_detector

# Detect ripples with custom parameters
ripples = Karlsson_ripple_detector(
    time, LFPs, speed, sampling_frequency,
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

See the [examples](examples/) directory for Jupyter notebooks demonstrating:

- [Detection Examples](examples/detection_examples.ipynb) - Using different detectors
- [Algorithm Components](examples/test_individual_algorithm_components.ipynb) - Testing individual components

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

The package has comprehensive test coverage (98%) across 107 tests organized in 4 modules:

```bash
# Run all tests with coverage
pytest --cov=ripple_detection --cov-report=term-missing tests/

# Run specific test modules
pytest tests/test_core.py          # Core signal processing tests (52 tests)
pytest tests/test_detectors.py     # Detector integration tests (25 tests)
pytest tests/test_simulate.py      # Simulation module tests (40 tests)

# Run specific test
pytest tests/test_core.py::TestSegmentBooleanSeries::test_single_segment

# Generate HTML coverage report
pytest --cov=ripple_detection --cov-report=html tests/
# Open htmlcov/index.html in browser
```

**Test Coverage:**
- `core.py`: 100%
- `detectors.py`: 100%
- `simulate.py`: 92%
- **Overall: 98%**

### Code Quality

```bash
# Format code with black
black ripple_detection/ tests/

# Lint code with flake8
flake8 ripple_detection/ tests/

# Check formatting without modifying
black --check ripple_detection/ tests/
```

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
