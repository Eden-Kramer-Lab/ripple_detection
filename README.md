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
  - `Karlsson_ripple_detector` - Per-channel detection with merging (Karlsson & Frank 2009)
  - `Shvartsman_ripple_detector` - Per-channel detection requiring a minimum number of participating channels (unpublished)
  - `Roumis_ripple_detector` - Per-channel envelopes averaged across channels (Frank-lab variant, unpublished)
  - `Yu_ripple_detector` - Median consensus with a data-driven noise-percentile threshold (Yu et al. 2017)
  - `Carey_candidate_detector` - Joint ripple-power x multiunit candidate events (Carey, Tanaka & van der Meer 2019); takes LFP and spikes
  - `Zugaro_ripple_detector` - The FMAToolbox/buzcode `FindRipples` two-threshold algorithm (Hirase; Zugaro)
  - `Long_sharp_wave_ripple_detector` - Two-channel detector using the sharp wave on a stratum radiatum channel (J. D. Long II, buzcode `bz_DetectSWR`); takes **raw** LFP
  - `multiunit_HSE_detector` - High Synchrony Event detection from multiunit activity (population rate, no LFP)

- **Comprehensive Event Statistics**
  - Temporal metrics (start time, end time, duration)
  - Z-score metrics (mean, median, max, min, sustained threshold)
  - Signal metrics (area under curve, total energy)
  - Movement metrics (speed during event)

- **Flexible Signal Processing**
  - Bandpass filtering, 150-250 Hz by default and any band on request
  - Envelope extraction via Hilbert transform
  - Gaussian smoothing with configurable parameters
  - Movement exclusion based on speed thresholds

- **Combining Detectors**
  - `require_overlap` - keep the events of one detector that overlap another's,
    for studies that require a ripple and a population burst together
  - `merge_close_events` / `exclude_close_events` - the two conventions for
    events separated by a short gap: join them, or keep the first and drop the rest

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

# With uv: an isolated environment with the package in editable mode, the
# development tools, and (with --extra examples) the notebook dependencies
uv sync --extra examples
uv run pytest

# With pip, into an environment of your own
pip install -e .[dev,examples]
```

## Requirements

- Python >= 3.10
- numpy >= 1.24
- scipy >= 1.10
- pandas >= 2.0

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
# (NaN) are handled block-wise, as in every detector; an event cut off by
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

### Selecting a detector by name

A pipeline that stores a detector's name rather than importing it, such as a
database-backed workflow, can resolve the name here instead of keeping its own
list that goes stale whenever this package gains a detector.

```python
from ripple_detection import DETECTORS, RIPPLE_BAND_LFP, get_detector

spec = get_detector("Kay_ripple_detector")
spec.inputs == (RIPPLE_BAND_LFP,)  # True
events = spec.detector(time, filtered_lfps, speed, sampling_frequency)

sorted(DETECTORS)                 # every detector this package has
spec.parameters                   # the tunables and their defaults, for a stored parameter dict
spec.check_parameters({"zscore_threshold": 3.0})  # raises on a name the detector does not take
```

The names in `ripple_detection.__all__` are the public API; anything else is an
implementation detail that may change without notice.

Check `inputs` before calling. The detectors do not all take the same signal,
and two of the mismatches are silent rather than loud:

| `inputs` | Detectors | What to pass |
|---|---|---|
| `("ripple_band_lfp",)` | Kay, Karlsson, Roumis, Shvartsman, Yu, Zugaro | `(n_time, n_channels)` ripple-band filtered LFP |
| `("raw_lfp_pair",)` | Long | `(n_time, 2)` **unfiltered** LFP: pyramidal-layer channel, then stratum radiatum |
| `("ripple_band_lfp", "multiunit")` | Carey | both, in that order |
| `("multiunit",)` | `multiunit_HSE_detector` | `(n_time, n_units)` spike counts or indicators |

`Long_sharp_wave_ripple_detector` takes raw LFP through a signature identical to
the ripple-band detectors', so handing it filtered data raises nothing and
returns plausible nonsense, and `multiunit_HSE_detector` has the same shape of
hazard. `spec.check_inputs(*signals)` verifies what an array can show: the number
of signals, that each is 2-D, that a raw pair has two channels, and that spike
counts are non-negative whole numbers. It cannot tell raw LFP from filtered LFP,
so `inputs` is the contract to check your pipeline against.

```python
spec = get_detector("Long_sharp_wave_ripple_detector")
try:
    spec.check_inputs(filtered_lfps)  # four channels from Basic Usage
except ValueError as error:
    print(error)
# Long_sharp_wave_ripple_detector takes raw_lfp_pair as signal 1: two channels,
# the ripple channel then the sharp-wave channel, got 4.
```

## How the detectors compare on simulated data

[`examples/simulation_study.py`](examples/simulation_study.py) runs every detector at its
defaults on `simulate_session` output: pink noise, 40 ripples in 120 s of known time, duration
and frequency, sized by `ripple_snr` (their filtered peak over the filtered background's SD),
4, 16 or 32 channels sharing half their noise, a sharp wave under each ripple for Long, and 100
Poisson units bursting with each ripple for Carey and HSE; three seeds per condition. Further
conditions put the ripple on only a quarter or half of 32 channels, sweep the thresholds of
Kay, Karlsson and Zugaro, shrink the population to 20 units, and add 20 common-mode artifacts.
An event is a hit when it overlaps the ripple's window. The
[notebook](examples/simulation_study.ipynb) plots the whole sweep; 16 channels, means over
seeds:

| Detector | Recall, `ripple_snr` 2 | Recall, `ripple_snr` 4 | False positives per minute, no ripples (4 / 16 / 32 ch) | Recall with the ripple on a quarter of 32 channels, `ripple_snr` 3 | Precision with 20 common-mode artifacts |
|---|---|---|---|---|---|
| Kay | 0.46 | 0.98 | 18 / 21 / 24 | 0.15 | 0.00 |
| Karlsson | 0.32 | 0.98 | 2 / 9 / 19 | 0.60 | 0.00 |
| Roumis | 0.46 | 0.99 | 21 / 23 / 26 | 0.10 | 0.00 |
| Shvartsman | 0.08 | 0.92 | 0 / 0.7 / 2 | 0.40 | 0.00 |
| Yu | 0.47 | 1.00 | 15 / 25 / 70 | 0.18 | 0.27 |
| Zugaro | 0.27 | 0.92 | 11 / 14 / 16 | 0.06 | 0.00 |
| Long | 0.42 | 0.81 | 8 / 8 / 8 | 0.58 | 0.94 |
| Carey | 0.92 | 0.98 | 0.7 / 0.5 / 0.2 | 0.87 | 0.79 |
| HSE | 0.99 | 0.99 | 50 / 50 / 53 | 0.98 | 0.85 |

What it shows:

- **The channel count changes the ranking.** Karlsson's false positives grow with the channel
  count, since any one channel crossing 3 SD makes an event; Yu's data-driven threshold falls the
  more correlated channels feed its median. Kay, Roumis, Zugaro, Long and Shvartsman barely move.
- **At a matched false-positive rate the ripple-band algorithms are close, and Shvartsman
  leads.** On 16 channels Kay at 3.0 SD, Karlsson at 3.5 and Zugaro at 8 each allow 1 to 2
  false positives a minute and recall 0.5 to 0.64 of ripples at `ripple_snr` 3; Shvartsman at its
  default allows 0.7 and recalls 0.67. Most of what separates the defaults is where the
  threshold sits. Kay at 2 SD is a candidate generator, about 20 events a minute on noise.
- **Which channels carry the ripple matters more than how many there are.** Pooled traces are
  diluted by silent channels (Kay by the channel count, Roumis worse), and a median cannot see a
  ripple on a minority of channels at any amplitude (Yu). Karlsson's per-channel rule is the only
  one that does not pay for silent channels, at the price of inheriting every channel's noise.
- **The spike detectors' recall is set by the population burst, not the LFP**, and their precision
  by the population size: with 20 units instead of 100, Carey's falls from 1.0 to 0.40 and HSE's
  from 0.83 to 0.32. HSE at 2 SD gives 50 events a minute on noise.
- **Long's recall plateaus near 0.9** by construction: its percentile cuts drop the weakest tenth
  of the sharp-wave cluster, and it does not evaluate candidates within 5 s of a block edge. Its
  events mark the sharp wave, so they start about 18 ms after the ripple window and end 18 ms
  before it.
- **Common-mode artifacts silence the ripple-band detectors at their defaults, except Yu**,
  whose threshold is read off the noise side of the histogram. Long cancels them in its channel
  difference, and the spike detectors barely notice them. The notebook shows what
  `normalization_method="median_mad"` recovers for Kay and Karlsson.

What it cannot show: the simulator's ripples are sinusoids under a Gaussian envelope, its
sharp wave a Gaussian, its units Poisson, its noise stationary 1/f with no theta or state
changes, and its artifacts crude; every ripple has a sharp wave and a population burst, which
builds in the advantage of the detectors that read them. The numbers rank the defaults and
expose their mechanics; they are not the recall or precision to expect on a recording.

## Output Format

All detectors return a pandas DataFrame with comprehensive event statistics:

| Column | Description |
|--------|-------------|
| `start_time` | Event start time |
| `end_time` | Event end time |
| `duration` | Elapsed time from the first to the last sample (seconds). One sample interval less than `n_samples` spans, so an event of exactly the minimum sample count has a `duration` below `minimum_duration` by half to one and a half intervals, as the rounding falls: at 15 ms and 1500 Hz, 23 samples span 14.67 ms |
| `n_samples` | Samples in the event, first to last inclusive; the quantity the duration limits test |
| `max_sustained_zscore` | The largest z-score sustained for `minimum_duration`: for Kay, Karlsson, Roumis, Yu and HSE, the highest threshold at which the detector would still find the event. Descriptive only for Shvartsman (the mean over participating channels), Zugaro and Carey (two-threshold rules) and Long (may be NaN), where it can fall below the threshold. Named `max_thresh` before 2.0 |
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
| `clipped_start` | The event begins on the first sample of its block: it was cut off by missing data or the recording edge. For Zugaro, that the run has no crossing below `low_threshold` on that side |
| `clipped_end` | The event ends on the last sample of its block |

The z-score columns describe the trace each detector thresholds: the consensus trace for Kay and Roumis, the per-sample maximum over channels for Karlsson, the mean over participating channels for Shvartsman, the immobility-normalized median for Yu, the z-scored squared power for Zugaro, the joint score for Carey, the population rate for HSE, and the globally z-scored ripple power for Long. Their scales differ, so a `mean_zscore` of 3 from one detector is not 3 from another.

The index is `event_number`. Some detectors add columns:

| Detector | Additional columns |
|---|---|
| `Shvartsman_ripple_detector` | `participants` (sorted tuple of channel indices), `n_participants`, `frac_participants` |
| `Yu_ripple_detector` | `n_suprathreshold_samples`, `detection_threshold_zscore` |
| `Zugaro_ripple_detector` | `peak_time` |
| `Long_sharp_wave_ripple_detector` | `peak_time`, `sharp_wave_zscore`, `sharp_wave_local_percentile`, `ripple_power_zscore`, `ripple_power_local_percentile`, `sharp_wave_duration`, `ripple_duration` |
| `Carey_candidate_detector` | `n_active_units` |
| `multiunit_HSE_detector` | `n_active_units` |

## Examples

### Combining two detectors

Many studies require a ripple and a population burst together. `require_overlap`
keeps the events of one inventory that overlap an event of another, so any two
detectors compose into that criterion. The events keep their own bounds.

```python
from ripple_detection import Kay_ripple_detector, multiunit_HSE_detector, require_overlap

# time, speed, filtered_lfps and sampling_frequency from the Basic Usage example
multiunit = np.random.poisson(0.01, (len(time), 20))  # (n_time, n_units) spike counts

ripples = Kay_ripple_detector(time, filtered_lfps, speed, sampling_frequency)
bursts = multiunit_HSE_detector(time, multiunit, speed, sampling_frequency)

bursts_with_a_ripple = require_overlap(bursts, ripples)
ripples_with_a_burst = require_overlap(ripples, bursts)  # the other direction
```

`require_overlap` takes a detector's DataFrame or an `(n_events, 2)` array of
start and end times, and `minimum_overlap` raises the bar above "any overlap".

### Simulating realistic ripples

The simulator's noise is pink (1/f) by default, whose ripple-band background is closest to
recordings. Set the ripple size relative to that background with `ripple_snr`, the peak of
the ripple after `filter_ripple_band` divided by the SD of the filtered noise:

```python
from ripple_detection.simulate import simulate_LFP, simulate_time

time = simulate_time(15000, 1500)
lfp = simulate_LFP(
    time, [2.0, 5.0, 8.0],
    ripple_snr=5,                  # ripple peak = 5 x ripple-band noise SD
    ripple_frequency=(150, 250),   # drawn per ripple
    ripple_duration=(0.04, 0.12),  # drawn per ripple, seconds
    random_state=0,
)
```

`simulate_session` produces every input the detectors take from one set of ripples: channels
that share the ripple in correlated noise, the raw pyramidal and stratum radiatum pair the Long
detector reads, spike trains that burst with each ripple, an immobile speed trace, and the
ground truth:

```python
from ripple_detection import filter_ripple_band, Kay_ripple_detector, simulate_session

session = simulate_session(time, [2.0, 5.0, 8.0], n_channels=4, n_units=20, ripple_snr=4, random_state=0)
filtered = filter_ripple_band(session.lfps, sampling_frequency=session.sampling_frequency)
events = Kay_ripple_detector(session.time, filtered, session.speed, session.sampling_frequency)
session.ripple_windows      # (n_ripples, 2): the interval each event should overlap
```

Brown (1/f²) noise, the default before 2.0, has almost no ripple-band power, so any ripple
dominated the band; pass `noise_type="brown"` for it. The z-score a detector reports is larger
than `ripple_snr` by a factor that depends on its smoothing and consensus rule; measure it for
the detector you use rather than assuming a mapping. The
[simulation study](examples/simulation_study.ipynb) runs every detector on these sessions.

See the [examples](examples/) directory for Jupyter notebooks demonstrating:

- [Tutorial](examples/ripple_detection_tutorial.ipynb) - A walk through detection on simulated data
- [Detection Examples](examples/detection_examples.ipynb) - Using different detectors
- [Algorithm Components](examples/test_individual_algorithm_components.ipynb) - Testing individual components
- [Simulation Study](examples/simulation_study.ipynb) - Recall, precision, timing and false positives of every detector on simulated sessions

## Troubleshooting

### Common Errors

#### "must be a 2D array"

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

#### "band upper edge ... reaches the Nyquist frequency"

`filter_ripple_band(data, sampling_frequency=...)` uses the pre-computed 1500 Hz
kernel at 1500 Hz and designs a 150-250 Hz filter for any other rate, so pass the
true sampling rate. Rates at or below 550 Hz cannot hold the band and raise. To
build the filter yourself:

```python
from ripple_detection import ripple_bandpass_filter
from scipy.signal import filtfilt

filter_num, filter_denom = ripple_bandpass_filter(sampling_frequency)
filtered_lfps = filtfilt(filter_num, filter_denom, LFPs, axis=0)  # LFPs from Basic Usage
```

#### Data sampled at 25 kHz or above

`filter_ripple_band` designs its filter for the rate it is given, and at 25 kHz
and above that design needs 2500 taps or more. There the equiripple algorithm
stops short of its specification (about 41 dB of single-pass attenuation, 82 dB
after the forward-backward pass, so the result is still clean), the kernel is
about 0.1 s long, and a run of samples shorter than that cannot be filtered.
Decimate to 3 kHz or below first (`scipy.signal.decimate`); nothing below
300 Hz is lost, and the filter then meets its specification at a tenth of the
cost. Pass the decimated rate as `sampling_frequency`.

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
4. Try a different detector (see [Choosing a detector](#choosing-a-detector))

### Parameter Selection Guide

| Parameter | Default | Description | When to Adjust |
|-----------|---------|-------------|----------------|
| `speed_threshold` | 4.0 cm/s | Maximum speed for ripple detection | Increase if too many events excluded during slow movement |
| `low_threshold`, `high_threshold` (Zugaro, Carey) | 2.0 and 5.0 on Zugaro's z-scored squared sum; 1.0 and 3.0 on Carey's joint score | An event is a run strictly above the low threshold whose peak is strictly above the high one | Raise the high threshold to keep only the largest events; the low threshold sets the bounds |
| Long durations | sharp wave 0.020–0.500 s, ripple ≥ 0.025 s | Either minimum suffices; a sharp wave longer than the maximum is dropped | Widen the sharp-wave range for slow states |
| `minimum_duration` (and every other duration limit) | 0.015 s (Kay, Karlsson, Roumis, Shvartsman, HSE)<br>0.020 s (Yu, Zugaro, Carey) | Converted to a sample count with `minimum_sample_count` (round half up from the median timestamp step); an event qualifies when its sample count is at least the minimum and at most any maximum, both inclusive (`sample_count_within`) | Decrease for shorter events; increase for stricter detection |
| `zscore_threshold` | 2.0 (Kay, Roumis, HSE)<br>3.0 (Karlsson, Shvartsman) | Detection sensitivity | Decrease for more detections; increase for fewer, higher-confidence events |
| `smoothing_sigma` | 0.004 s on Kay, Karlsson, Roumis, Shvartsman and Yu; 0.010 s on Carey (`ripple_smoothing_sigma`); 0.015 s on `multiunit_HSE_detector`. Zugaro uses a moving average (`smoothing_window`), Long its own low-pass kernels | Width of the Gaussian smoothing kernel | Rarely needs adjustment; increase for noisier data |
| `percentile` | 99.99 (Yu) | Percentile of the mirrored immobility-noise distribution used as the threshold | Lower for more detections; the threshold is estimated per call, so it adapts to each recording |
| `close_ripple_threshold` (`close_event_threshold` on the HSE detector) | 0.0 s | The later of two events closer than this is dropped | Raise (e.g. 0.05) to suppress fragments; Zugaro merges instead via `minimum_inter_ripple_interval` |
| `maximum_duration` | `None` (no limit; `Zugaro` 0.100 s). Long's ceiling is `maximum_sharp_wave_duration`, 0.500 s | Longest allowed event, applied to the event as reported rather than to the run above threshold. A sample count like the minimum, so the ceiling is one sample shorter in elapsed time than the value given | Published limits run from a few hundred milliseconds to a couple of seconds |
| `minimum_active_units` | 0 on `multiunit_HSE_detector` (no criterion), 5 on `Carey_candidate_detector` | Units with at least one spike inside the event; every event reports `n_active_units` | Published criteria are most often around five units |
| `band`, `transition_width` on `filter_ripple_band` | `None`, meaning 150-250 Hz, and `None`, meaning 25 Hz for a designed filter | Passband of the designed filter. `sampling_frequency` is always required; the shipped kernel (10 Hz transitions) is used only at 1500 Hz with the default band, whether `None` or `(150, 250)`, and no `transition_width`. Any other band, rate or width designs a filter | Published bands run from about 80-180 Hz at the lower edge to 200-300 Hz at the upper |

### Published parameter values

Where the package's defaults sit relative to the literature. Compiled from 57
papers that decode replay content (1999-2025), each value read from the paper's
Methods; counts are of papers stating a bare number for that parameter, so
entries like ">4" or "33% of the ensemble" are excluded rather than coerced.

The per-paper table ships with the package, so you can ask it your own question
rather than take the summary below:

```python
from ripple_detection import load_literature_parameters

parameters = load_literature_parameters()
parameters.groupby("Detection")["SWR Z-score Thresh. (STD)"].median()
parameters.loc[parameters["Spike sorting"] == "Clusterless", ["First Author", "Year", "DOI"]]
```

| Parameter | Papers | Published range | Median | Most common | Package default |
|---|---|---|---|---|---|
| `zscore_threshold` (ripple) | 27 | 1-8 SD | 3 SD | 3, 2, then 4 and 8 tied | 2.0 Kay/Roumis, 3.0 Karlsson/Shvartsman |
| `zscore_threshold` (multiunit) | 27 | 2-4 SD | 3 SD | 3, 2, 4 | 2.0 |
| ripple band | 30 | 80-180 Hz low, 200-300 Hz high | 150-250 Hz | 150-250 Hz (17 papers) | 150-250 Hz |
| `smoothing_sigma` (ripple) | 18 | 4-100 ms | 12.5 ms | 4, 12.5, 15 | 4 ms; 10 ms on Carey |
| `smoothing_sigma` (multiunit) | 28 | 5-30 ms | 15 ms | 15, 10, 5 | 15 ms |
| `speed_threshold` | 42 | 0.05-10 cm/s | 5 cm/s | 5, 4, 2 | 4 cm/s |
| `minimum_duration` | 41 | 15-100 ms | 50 ms | 50, 100, then 15 and 40 tied | 15 ms, 20 ms on Yu/Zugaro/Carey |
| `maximum_duration` | 25 | 400-2000 ms | 600 ms | 500, 2000, 750 | none, except Zugaro 100 ms and Long 500 ms (sharp wave) |
| merge or drop gap | 14 | 20-100 ms | 50 ms | 50, then 20, 40 and 100 tied | 0 (no exclusion); Zugaro merges within 30 ms, Long drops within 50 ms |
| `minimum_active_units` | 27 | 3-10 units | 5 units | 5, 4, 3 | 0 on the burst detector, 5 on Carey |
| channels required | 27 | 13 papers use one, 10 more than one, 4 a small number | one | one | one is enough except Shvartsman (2 by default) and Long (exactly 2); Kay, Roumis, Yu and Zugaro pool all channels |

Three cautions before treating this as a recipe:

- **The defaults are each source's where the source has one, not a consensus.** They
  reproduce the published or lab settings of the algorithm each detector is named for,
  except the speed rule on Zugaro, Long and Carey and every HSE default but its 15 ms
  smoothing (Davidson et al. 2009's), which are the package's (see [Choosing a detector](#choosing-a-detector)); so the same recording
  gives different event counts under different detectors by design.
- **Our thresholds and minimum duration sit at the permissive end.** A 2 SD threshold
  held for 15 ms admits more than the field's median of 3 SD and 50 ms. Tightening to
  the median is a defensible sensitivity check, not an extreme one.
- **Some published speed values restrict analysis rather than detection**, so that row
  overstates how many papers gate detection on speed. `minimum_duration` also means
  different things across papers: on Kay, Karlsson, Roumis, Shvartsman, Yu and the HSE
  detector it is the run above threshold, before the extension to the mean; on Zugaro,
  Carey and Long it is the event as reported, as in many papers.

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/Eden-Kramer-Lab/ripple_detection/issues)
- **Discussions**: For questions about usage and parameter selection
- **Email**: [edeno@bu.edu](mailto:edeno@bu.edu)

## Documentation

For detailed documentation on the detection algorithms and signal processing pipeline, see [CLAUDE.md](CLAUDE.md).

## Choosing a detector

All detectors take `time`, the signal, `speed`, and `sampling_frequency` as positional arguments
(`Carey_candidate_detector` takes `filtered_lfps` then `multiunit`) and return the DataFrame
described under [Output Format](#output-format). They differ in what they threshold and in the
conventions below.

| Detector | Signal input | What is thresholded | Threshold (default) | Duration (default) | Close events | Speed rule (default 4 cm/s) | Source |
|---|---|---|---|---|---|---|---|
| `Kay_ripple_detector` | ripple-band LFP `(n_time, n_channels)` | z-scored consensus √(smoothed Σ envelope²), 4 ms | `zscore_threshold` 2.0 | ≥ 0.015 s | `close_ripple_threshold` 0.0, drops the later event | speed at first and last sample ≤ threshold | Kay et al. 2016 |
| `Karlsson_ripple_detector` | same | each channel's z-scored envelope; overlapping per-channel events merged | 3.0 | ≥ 0.015 s | same | same | Karlsson & Frank 2009 |
| `Roumis_ripple_detector` | same | z-scored mean over channels of √(smoothed envelope²), 4 ms | 2.0 | ≥ 0.015 s | same | same | Frank-lab variant (D. Roumis), unpublished |
| `Shvartsman_ripple_detector` | same | per-channel z-scored envelopes; event kept when ≥ `minimum_participating_channels` (2), or `minimum_participating_fraction` of the channels, detect it | 3.0 | ≥ 0.015 s | same | at least half the event's samples ≤ threshold | lab variant (G. Shvartsman), unpublished |
| `Yu_ripple_detector` | same | median over channels of each channel's z-scored 4 ms-smoothed envelope | `percentile` 99.99 of the mirrored immobility-noise distribution, estimated per call | ≥ 0.020 s | `close_ripple_threshold` 0.0 | noise from `speed <= threshold`; event endpoints ≤ threshold | Yu et al. 2017 |
| `Zugaro_ripple_detector` | same, channels summed | z-scored smoothed squared signal, two thresholds (strictly above) | `low_threshold` 2.0 (bounds), `high_threshold` 5.0 (peak) | 0.020–0.100 s | `minimum_inter_ripple_interval` 0.030 s, merges | endpoints ≤ threshold | FMAToolbox `FindRipples` (Hirase; Zugaro) |
| `Long_sharp_wave_ripple_detector` | **raw** LFP `(n_time, 2)`: ripple channel, stratum radiatum channel | sharp-wave difference and ripple power, split by k-means with local (±5 s) statistics | `sharp_wave_thresholds`, `ripple_thresholds` (0.5, 2.5) | sharp wave 0.020–0.500 s **or** ripple ≥ 0.025 s: either minimum suffices; a sharp wave over 0.500 s is dropped | `minimum_separation` 0.050 s from the previous candidate, kept or not; drops | endpoints ≤ threshold | Long, buzcode/neurocode `DetectSWR` |
| `Carey_candidate_detector` | ripple-band LFP **and** spikes `(n_time, n_units)` | geometric mean of a ripple-envelope score and a multiunit score, two thresholds (strictly above) | `low_threshold` 1.0, `high_threshold` 3.0; ≥ `minimum_active_units` 5 | ≥ 0.020 s | none | whole event inside a low-speed interval (`speed <= threshold`) | Carey, Tanaka & van der Meer 2019 |
| `multiunit_HSE_detector` | spikes `(n_time, n_units)`, no LFP | z-scored 15 ms-smoothed population rate | `zscore_threshold` 2.0 | ≥ 0.015 s | `close_event_threshold` 0.0 | endpoints ≤ threshold | package convention; Davidson et al. 2009 lineage |

Notes:

- **Missing samples** are handled the same way by every detector. A sample is missing when any
  channel of any signal is NaN or infinite, or when the step in `time` to it exceeds 1.5 times
  the median step. The valid samples form contiguous blocks; every step runs within a block, so
  nothing is smoothed, thresholded or merged across a gap and no event spans one. An event cut off
  by a gap or by the recording edge is kept and flagged in `clipped_start` and `clipped_end`. A
  block too short for a detector's transform (Zugaro's smoothing window, Long's sharp-wave kernel,
  Carey's theta filter) or for an event of `minimum_duration` is treated as missing, with a warning
  that gives its sample ranges; a detector left with no block raises rather than return an empty
  result.
- **Unknown speed** (NaN in `speed`, as from a tracking dropout) is not a missing sample: speed
  enters no trace, so it splits no block. An event whose first or last sample has unknown speed
  fails the endpoint rule; Shvartsman's majority is taken over the samples with known speed;
  Carey treats unknown speed as not low speed. `speed_threshold=np.inf` turns the criterion off,
  unknown speed included. The speed statistics skip unknown values.
- Every detector normalizes over the whole recording unless `normalization_mask` restricts it
  (Yu defaults to immobility); a baseline period is `(time >= start) & (time <= end)`. The Long
  and Carey detectors do not take this argument.
- **Thresholds are not comparable across detectors.** Kay's 2.0 is on √(smoothed Σ envelope²),
  whose distribution depends on the channel count; Karlsson's 3.0 is on one channel's envelope;
  Zugaro's 2 and 5 are on a squared sum, which has a heavier tail; Carey's 1 and 3 are on a
  geometric mean; Yu's is estimated from the data. The false-positive rates on ripple-free
  simulations in the [simulation study](examples/simulation_study.ipynb) show how far apart the
  defaults sit. Kay, Karlsson, Roumis, Shvartsman, Yu and HSE keep a sample at or above the
  threshold; Zugaro and Carey require strictly above.
- **One channel.** Kay and Roumis return identical events on a single channel, since both reduce
  to √(smoothed envelope²). Shvartsman needs `minimum_participating_channels` channels and raises
  when it has fewer; pass `minimum_participating_channels=1`. Long needs exactly two raw channels.
- **Noisy or non-stationary data.** `normalization_method="median_mad"` (Kay, Karlsson, Roumis,
  Shvartsman, HSE) replaces the mean and SD by the median and the scaled MAD, which large events
  do not inflate; Yu, Zugaro, Long and Carey z-score only. It is unsuitable for a sparse trace
  such as a low population rate, whose MAD can be zero. `normalization_mask` restricts the
  statistics to a baseline period on every detector but Long and Carey.
- **Smoothing** differs in where it is applied: Kay after summing the squared envelopes, Roumis on
  each channel's squared envelope, Karlsson, Shvartsman and Yu on each channel's envelope, Carey on
  the channel-mean envelope with a 10 ms kernel truncated at 3 SD (the others truncate at 8). Yu
  z-scores each channel over the whole recording, takes the median, then re-normalizes that
  median to immobility (`speed <= speed_threshold`, or `normalization_mask`) before estimating
  its threshold.
- The endpoint speed rule at 4 cm/s is the package's on Zugaro and Long, whose originals have no
  speed criterion, and on Carey, whose original uses 10 pixels/s; every default of the HSE detector
  but the 15 ms smoothing, which is Davidson et al. 2009's, is the package's.
- Two conventions are the package's, not each source's: every duration limit is an inclusive
  round-half-up sample count (`sample_count_within`), and immobility is `speed <= speed_threshold`.
- "Close events" above says what each detector does by default. `merge_close_events` applies
  the other convention, joining nearby events into one, to any inventory afterwards.
  The gating rule itself (endpoints, majority, interval containment) stays as each source defines it.
- The other defaults reproduce each source's published or lab settings where one exists; they are
  not harmonized across detectors, so the same recording yields different event counts under
  different detectors by design.

## Development

### Setup Development Environment

```bash
# uv: creates .venv from the committed uv.lock with the dev tools installed
uv sync --extra examples
uv run pytest                     # any command runs in that environment
uvx pre-commit install            # optional: run the checks below on each commit

# Or conda
conda env create -f environment.yml
conda activate ripple_detection
pip install -e .[dev,examples]
```

`uv.lock` pins the development environment; `uv lock` refreshes it after a
dependency change, and CI checks that it is current.

### Run Tests

```bash
# Run all tests with coverage
pytest

# Run one module
pytest tests/test_core.py          # signal processing
pytest tests/test_detectors.py     # detector behavior and conventions
pytest tests/test_simulate.py      # synthetic LFP
pytest tests/test_registry.py      # the detector registry
pytest tests/test_literature.py    # the published-parameter survey
pytest tests/test_public_api.py    # what the package exports
pytest tests/test_properties.py    # property-based (hypothesis)
pytest tests/test_snapshots.py     # regression snapshots

# HTML coverage report (open htmlcov/index.html)
pytest --cov-report=html
```

Test modules mirror the package modules (`test_core`, `test_detectors`,
`test_simulate`, `test_registry`, `test_literature`); `test_public_api`,
`test_properties` and `test_snapshots` cut across all of them.

### Code Quality

```bash
# Format code with ruff
ruff format src/ tests/

# Lint code with ruff
ruff check src/ tests/

# Type check with mypy
mypy src/

# Check formatting without modifying
ruff format --check src/ tests/

# All of the above plus codespell and the file checks, as pre-commit runs them
uvx pre-commit run --all-files
```

### Release Process

Releases are automated via GitHub Actions when a version tag is pushed:

```bash
# 1. Ensure all tests pass and code quality checks pass
pytest tests/
ruff format --check src/ tests/
ruff check src/ tests/
mypy src/

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
- Run tests on Python 3.10 through 3.14, and at the dependency floors
- Build source distribution and wheels
- Publish to PyPI
- Create GitHub release

**Note:** Version numbers follow [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH). The package version is automatically determined from git tags via `hatch-vcs`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change. Running `uvx pre-commit install` once makes each commit run the same formatting, lint, spelling and type checks as CI.

## Citation

If you use this package, cite the source of the detector you used. Several
detectors come from lab code with no accompanying paper; those rows name the
file to cite instead.

| Detector | Source | DOI or code |
|---|---|---|
| `Kay_ripple_detector` | Kay, K., Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., & Frank, L. M. (2016). A hippocampal network for spatial coding during immobility and sleep. *Nature*, 531(7593), 185-190. | [10.1038/nature17144](https://doi.org/10.1038/nature17144) |
| `Karlsson_ripple_detector` | Karlsson, M. P., & Frank, L. M. (2009). Awake replay of remote experiences in the hippocampus. *Nature Neuroscience*, 12(7), 913-918. | [10.1038/nn.2344](https://doi.org/10.1038/nn.2344) |
| `Roumis_ripple_detector` | Unpublished Frank lab variant (D. Roumis, 2017). | no paper |
| `Shvartsman_ripple_detector` | Unpublished variant (G. Shvartsman, 2026). | no paper |
| `Yu_ripple_detector` | Yu, J. Y., Kay, K., Liu, D. F., Grossrubatscher, I., Loback, A., Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., & Frank, L. M. (2017). Distinct hippocampal-cortical memory representations for experiences associated with movement versus immobility. *eLife*, 6, e27621. | [10.7554/eLife.27621](https://doi.org/10.7554/eLife.27621) |
| `Zugaro_ripple_detector` | FMAToolbox `Analyses/FindRipples.m` (initial algorithm by H. Hirase; implemented by M. Zugaro). The summed-power rule it descends from is described in Csicsvari, J., Hirase, H., Czurkó, A., Mamiya, A., & Buzsáki, G. (1999). *Journal of Neuroscience*, 19(1), 274-287. | [FindRipples.m](https://github.com/michael-zugaro/FMAToolbox/blob/6bbb3662f7ed1ccf09c5ff4b4d233e27e17c71a6/Analyses/FindRipples.m); [10.1523/JNEUROSCI.19-01-00274.1999](https://doi.org/10.1523/JNEUROSCI.19-01-00274.1999) |
| `Long_sharp_wave_ripple_detector` | J. D. Long II, `bz_DetectSWR.m` (buzcode; converted by A. Navas-Olive; filtering after E. Stark's `detect_hfos`), carried into neurocode as `DetectSWR.m`. No accompanying paper. | [bz_DetectSWR.m](https://github.com/buzsakilab/buzcode/blob/0969ddf7f55ccaca8c71969bee4b21f310840047/analysis/SharpWaveRipples/bz_DetectSWR.m); neurocode [10.5281/zenodo.7819979](https://doi.org/10.5281/zenodo.7819979) |
| `Carey_candidate_detector` | Carey, A. A., Tanaka, Y., & van der Meer, M. A. A. (2019). Reward revaluation biases hippocampal replay content away from the preferred outcome. *Nature Neuroscience*, 22(9), 1450-1459. | [10.1038/s41593-019-0464-6](https://doi.org/10.1038/s41593-019-0464-6) |
| `multiunit_HSE_detector` | Thresholds on a smoothed population rate follow Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009). Hippocampal replay of extended experience. *Neuron*, 63(4), 497-507. The 15 ms smoothing is that paper's; the other defaults are this package's. | [10.1016/j.neuron.2009.07.027](https://doi.org/10.1016/j.neuron.2009.07.027) |
| `normalize_signal(method="median_mad")` | Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). *Journal of Experimental Social Psychology*, 49(4), 764-766. | [10.1016/j.jesp.2013.03.013](https://doi.org/10.1016/j.jesp.2013.03.013) |

Each detector's docstring carries the same reference, and the code it was
reimplemented from is named there. FMAToolbox, buzcode, neurocode, and the
van der Meer lab code are MATLAB; this package reimplements the published
algorithms rather than translating those files, which carry GPL-3 headers
(FMAToolbox, buzcode) or no license at all (neurocode).

## Other tools

Every detector here thresholds a hand-designed feature. Two other approaches are
worth knowing about, neither reimplemented here.

**Machine-learning detectors**, from Liset M. de la Prida's lab at the Cajal
Institute ([hippo-circuitlab.es](https://hippo-circuitlab.es/)):

- [rippl-AI](https://github.com/PridaLab/rippl-AI) is a toolbox of five trained
  architectures (1D-CNN, 2D-CNN, LSTM, SVM, XGBoost) with pre-trained models, so
  no threshold is chosen by hand. Navas-Olive, Rubio, Abbaspoor, Hoffman & de la
  Prida (2024), *Communications Biology* 7:211,
  [10.1038/s42003-024-05871-w](https://doi.org/10.1038/s42003-024-05871-w).
- [cnn-ripple](https://github.com/PridaLab/cnn-ripple) is the 1D convolutional
  network those build on, with a
  [MATLAB port](https://github.com/PridaLab/cnn-matlab) and an
  [Open Ephys plugin](https://github.com/PridaLab/CNNRippleDetectorOEPlugin) for
  detecting online. Navas-Olive, Amaducci, Jurado-Parras, Sebastián & de la
  Prida (2022), *eLife* 11:e77772,
  [10.7554/eLife.77772](https://doi.org/10.7554/eLife.77772).

These want a linear probe rather than tetrodes: most rippl-AI models take exactly
eight channels, "ideally centered in the SP [stratum pyramidale], with a positive
deflection on the first channels ... and a negative deflection on the last". If
your recordings span the layers that way, they detect events no amplitude
threshold will separate, and they need no threshold chosen per recording. If you
record with tetrodes, they do not apply.

**Replay scoring**, which is the step after detection and outside this package's
scope: [RnR_methods](https://github.com/DavidTingley/RnR_methods) implements
Bayesian replay scores (Radon and weighted correlation), rank-order correlation,
and reactivation strength, together with scripts comparing those methods against
each other, across bin sizes, under added noise, and for rank-order false
positives. It accompanies Tingley & Peyrache (2020), *Phil Trans R Soc B*
375:20190231, [10.1098/rstb.2019.0231](https://doi.org/10.1098/rstb.2019.0231).

Both de la Prida's `cnn-ripple` and `RnR_methods` are GPL-3, and `rippl-AI` ships
no license file, so none of them is translated into this MIT package.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Eric Denovellis** - [edeno@bu.edu](mailto:edeno@bu.edu)

## Acknowledgments

- Frank Lab for the pre-computed ripple filter
- Original algorithm implementations by Karlsson & Frank and Kay et al.

## Support

- **Issues**: [GitHub Issues](https://github.com/Eden-Kramer-Lab/ripple_detection/issues)
- **Discussions**: For questions and discussions about usage
- **Email**: [edeno@bu.edu](mailto:edeno@bu.edu)
