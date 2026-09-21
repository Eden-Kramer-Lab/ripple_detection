# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Yu_ripple_detector`: the Yu et al. 2017 (eLife) sharp-wave ripple detector. The consensus is the median across tetrodes of each tetrode's 4 ms-smoothed, z-scored ripple-band envelope (`get_Yu_ripple_consensus_trace`), and the detection threshold is estimated per call as the 99.99th percentile of the immobility noise distribution, obtained by mirroring the histogram below its mode (`estimate_noise_threshold`, a transliteration of the original MATLAB with the reflection written as intended and a warning where the original's formula would differ). Events are runs of at least `minimum_duration` (default 20 ms, counted in samples) at or above the threshold, extended to the immobility mean. Missing samples are handled block-wise instead of by row-dropping, so nothing is smoothed or joined across a gap; events truncated by a gap or the recording edge are kept and flagged (`clipped_start`, `clipped_end`). The keyword interface matches `Kay_ripple_detector` with `percentile` in place of `zscore_threshold`.
- `Zugaro_ripple_detector`: the FMAToolbox `FindRipples` algorithm (Hajime Hirase's method, implemented by Michaël Zugaro; carried into buzcode as `bz_FindRipples` by David Tingley and into AYA-lab neurocode, doi:10.5281/zenodo.7819979), reimplemented from the algorithm since the original is GPL-3. Ripple-band signal squared, summed across channels, smoothed with an 11-sample moving average at 1250 Hz (scaled with the rate), z-scored; events bounded where the trace crosses a low threshold (2 SD) and kept only if the peak exceeds a high threshold (5 SD); neighbors closer than 30 ms merged while the merged span stays under 100 ms; 20-100 ms duration limits. Defaults follow FMAToolbox and buzcode; neurocode's differing defaults are documented. Departures, documented: the package's endpoint speed rule is applied (disable with `speed_threshold=np.inf`), the reported `peak_time` is the maximum of the normalized power rather than a single channel's trough, and missing samples are handled block-wise.
- `Long_sharp_wave_ripple_detector`: John D. Long II's two-channel sharp-wave ripple detector (buzcode `bz_DetectSWR`, converted by Andrea Navas-Olive, filtering after Eran Stark's `detect_hfos`; AYA-lab neurocode `DetectSWR`), reimplemented from the algorithm. It takes **raw** LFP from a pyramidal-layer channel and a stratum radiatum channel, builds a sharp-wave feature (ripple channel minus radiatum channel in a 2-50 Hz difference-of-Gaussians band) and a ripple-power feature (smoothed rectified 80-250 Hz band of the common-average-referenced pair), pairs them per 40 ms block, separates sharp-wave ripples by two-cluster k-means with cluster-derived percentile cuts, and confirms each candidate against local +/-5 s statistics (peak >= median + 2.5 SD on both features, boundaries at median + 0.5 SD, 50 ms minimum separation, duration limits). Departures, documented: k-means is seeded (`random_state`), degenerate local windows reject rather than error, the endpoint speed rule is applied, and NaN input raises. The docstring notes that the sharp-wave cut discards roughly the weakest tenth of the SWR cluster by construction.
- `Carey_candidate_detector`: the candidate replay-event detector of Carey, Tank & van der Meer 2019 (vandermeerlab `GenCandidateEvents` with its Hilbert ripple score `OldWizard` and the multiunit score `amMUA` by Elyot Grant and A. Carey), reimplemented from the code. A ripple envelope score (10 ms Gaussian, rescaled to mean 1) and a capped, baseline-subtracted multiunit score are combined as their geometric mean, z-scored, and segmented with an edge threshold (1 SD) and a peak threshold (3 SD); candidates must exceed 20 ms, lie inside a low-speed interval and, when a theta channel is given, a low-theta interval, and contain spikes from at least five units. The docstring records that the combination is asymmetric: a ripple without a burst cannot be a candidate, a burst without a ripple can. Kernel widths are the original's 2 kHz sample counts expressed in seconds; NaN input raises.
- `simulate_LFP` gains `ripple_snr` (ripple peak after ripple-band filtering relative to the standard deviation of the filtered background noise, set per ripple from that burst's own filtered peak so it holds at the band edges and for short bursts; requires `noise_amplitude > 0`), `ripple_frequency` and `ripple_duration` that accept a `(low, high)` range drawn uniformly per ripple, and an optional `sampling_frequency`. The default call is unchanged bit-for-bit. Brown noise, the default, leaves very little power in the ripple band and less the longer the record, so pink noise is recommended for detector tests. `ripple_amplitude` is documented as peak-to-peak, which it always was.
- `random_state` parameter to `simulate_LFP()` for reproducible synthetic LFP generation (used to make the test suite deterministic).
- Regression tests for Shvartsman participation preserve the original count of distinct electrodes across each merged event, including chains of overlapping ripples and repeated ripples on the same electrode.

### Changed

- `Karlsson_ripple_detector` now computes its per-event z-score statistics (`max_thresh`, `mean_zscore`, `max_zscore`, ...) on the elementwise maximum across channels of the per-channel z-scores rather than on their mean. An event triggered by one channel at 3 SD previously reported `max_thresh` as low as 0.2 because the quiet channels were averaged in; `max_thresh` is now at least `zscore_threshold` for every event. Detected events and their boundaries are unchanged.
- Detector docstrings now state the movement rule actually applied: `Kay`, `Karlsson`, `Roumis`, and `multiunit_HSE` test the speed at an event's first and last samples only (`<=`), and `Shvartsman` keeps an event if at least half its samples are at or below the threshold. They also state the sample-count minimum-duration convention.
- **Detection-affecting.** The minimum-duration test now counts samples: a run qualifies when it holds at least `minimum_sample_count(time, minimum_duration)` = `round(minimum_duration * sampling_frequency)` (round half up, from the median timestamp step) consecutive samples, the Frank-lab `extractevents` convention. Before, the test compared end and start timestamps, which at 1500 Hz and 15 ms needed 24 samples where the convention gives 23. It could also reject runs of exactly the minimum through floating-point round-off (8-38 % of exact-15 ms runs at 1000 Hz depending on the time offset), and a run straddling a timestamp gap is now measured by the samples it holds, not the time it spans. Kay, Karlsson, Roumis, Shvartsman, Yu, and the HSE detector all use the one helper now. On 300 s pink-noise records (four channels, four seeds), the extra sample admits about 20 % more marginal noise-only events at `zscore_threshold` 2.0–2.5 for Kay and Karlsson; strong ripples are unaffected (the snapshot fixtures are unchanged). Spyglass `RippleTimesV1` populates at these settings, so re-populated tables will gain events. This is a minor-version change.
- **Detection-affecting.** One duration rule and one immobility comparison for every detector. Duration limits (minimum and maximum, including `Zugaro_ripple_detector`'s, `Long_sharp_wave_ripple_detector`'s, and `Carey_candidate_detector`'s) are inclusive round-half-up sample counts via the new `sample_count_within`; before, Zugaro and Carey compared elapsed time (inclusive and strict respectively) and Long floored to samples, so the three could differ from the rest by one sample at an exact limit. Immobility is `speed <= speed_threshold` everywhere; `Yu_ripple_detector`'s noise mask and `Carey_candidate_detector`'s low-speed intervals used `<`. Each detector's gating rule (endpoints, majority, interval containment) is unchanged.
- Development tooling: `ruff format` replaces black as the formatter, and the legacy flake8 pin is dropped. Ruff is now the single linter and formatter (`ruff format --check`, `ruff check`), and CI checks formatting with it. Black and ruff disagreed on how to wrap long `assert` messages, so the CI formatting check failed on files that ruff had formatted.
- Documentation: the README gains a "Choosing a detector" table (input, thresholded signal, defaults, close-event rule, missing-sample policy, speed rule, source) and lists the extra output columns per detector; every detector docstring states its missing-sample policy; `Roumis_ripple_detector` and `Shvartsman_ripple_detector` say they are unpublished lab variants; the package root exports the documented helpers and declares `__all__`.

### Fixed

- `multiunit_HSE_detector` now validates its inputs like the LFP detectors: a non-2D `multiunit`, mismatched lengths, time in samples, or speed in the wrong units raise, and NaN spike counts raise instead of silently blanking the smoothed rate over the full kernel width around each NaN. Its docstring now states which selection rule it implements and how it differs from Davidson et al. 2009.
- `filter_ripple_band` now designs a 150-250 Hz filter for the given `sampling_frequency` (via `ripple_bandpass_filter`) instead of applying the 1500 Hz pre-computed kernel at every rate. Previously the passband scaled with the rate — 193-340 Hz at 2000 Hz, 97-170 Hz at 1000 Hz — with only a warning. Output at 1500 Hz (or with no rate given) is unchanged. Rates whose Nyquist frequency cannot hold the band plus its 25 Hz transition (at or below 550 Hz) now raise instead of warning.
- `max_thresh` is now the largest threshold at which the event would still be detected: the maximum over every window of `minimum_sample_count` samples of that window's minimum. It used to expand greedily from the event's single highest sample, which returned a value below `zscore_threshold` (as low as 0.1) whenever that peak sat in a short excursion swept into the event by the mean-crossing extension, about 3-4 % of Karlsson events on noise. Strong single ripples are unchanged.
- `normalization_mask` is now filtered by the same NaN-row removal as the LFP/speed data in the LFP detectors (Kay, Karlsson, Roumis, Shvartsman), fixing a length-mismatch `ValueError` when the input contained NaN samples.
- A `normalization_mask` that selects no samples now raises a clear error from `normalize_signal` (matching the existing `normalization_time_range` behavior) instead of silently returning no events from a degenerate all-NaN/all-zero normalized trace.
- `normalize_signal_manually` (used by `Shvartsman_ripple_detector` for manual normalization) now treats a NaN baseline as a degenerate channel — zeroing it consistently with zero/NaN deviations rather than producing NaN — and warns naming the dropped channels. When *every* channel is degenerate (including 1-D input whose single channel is degenerate) the normalized signal would be uniformly zero, so it now raises `ValueError` instead of silently returning a signal-free array that yields no events — mirroring the empty-`normalization_mask` guard.
- `Karlsson_ripple_detector` and `multiunit_HSE_detector` now pass the caller's `minimum_duration` through to the `max_thresh` statistic instead of silently using the 15 ms default (Kay, Roumis, and Shvartsman already did). `_find_max_thresh` is also bounds-safe for events shorter than `minimum_duration`: it returns `nan` (the sustained value is undefined) instead of raising an `IndexError` that could crash `multiunit_HSE_detector` with a small `minimum_duration`. Detector output is unchanged — the detectors never produce sub-`minimum_duration` events.
- README quick start now filters the LFPs with `filter_ripple_band` before calling a detector (the detectors expect ripple-band-filtered input) and uses a sampling rate the built-in filter supports; `ripple_bandpass_filter` is now exported from the package root so the documented troubleshooting import works.

## [1.7.1] - 2026-01-22

### Fixed

- NumPy 2.x compatibility: Fixed `AttributeError` when using `np.trapz` which was removed in NumPy 2.0 (now uses `np.trapezoid` with fallback for NumPy 1.x). Closes #10.

## [1.7.0] - 2025-10-17

### Added

- **Flexible Signal Normalization**: New `normalize_signal()` function supporting both z-score (mean/std) and median/MAD normalization methods
  - Median/MAD normalization provides robust statistics less sensitive to outliers
  - Configurable normalization baseline via `normalization_mask` or `normalization_time_range` parameters
  - All detectors (`Kay_ripple_detector`, `Karlsson_ripple_detector`, `Roumis_ripple_detector`, `multiunit_HSE_detector`) now support these parameters
  - Enables advanced use cases: normalize during immobility only, use baseline period, exclude artifacts
  - Comprehensive test coverage including 23 new tests for normalization functionality

- **Tutorial Notebook**: Added `examples/ripple_detection_tutorial.ipynb` with step-by-step guide
  - Demonstrates basic ripple detection workflow
  - Shows how to use different normalization methods
  - Includes visualization examples

### Changed

- **Parameter Enhancement**: Added `normalization_method`, `normalization_mask`, and `normalization_time_range` parameters to all detector functions
  - Default behavior unchanged (z-score normalization on full signal)
  - New parameters provide fine-grained control over normalization baseline

### Deprecated

- `use_speed_threshold_for_zscore` parameter in `multiunit_HSE_detector`
  - Use `normalization_mask=speed < speed_threshold` instead for equivalent functionality
  - Deprecation warning added with migration guidance

### Fixed

- Updated README badges to reference release workflow and codecov
- Reformatted pyproject.toml for better readability

### Removed

- PR test GitHub Actions workflow (consolidated with main test workflow)

### Closed Issues

- Issue #8: Add support for using Median / MAD for ripple detection
- Issue #9: Allow users to customize where the mean and std come from

## [1.6.0] - 2025-10-16

### Added

- **Critical UX Improvements**: Comprehensive input validation to prevent common user errors
  - Speed units validation: Warns when speed appears to be in m/s instead of cm/s, preventing silent data loss
  - Time units validation: Errors when time array appears to be in samples instead of seconds
  - Sampling frequency validation: Warns when time step doesn't match expected sampling rate
  - Array shape validation: Clear error messages for wrong LFP dimensions (1D, 0D, >2D arrays)
  - Array length validation: Detailed error showing exact sample counts when arrays don't match
  - Data length validation: Checks minimum data length required for filtering (954 samples)
  - Filter sampling frequency validation: Prevents crashes with incompatible sampling rates (<1200 Hz)

- **Enhanced Documentation**
  - Added comprehensive Troubleshooting section to README with solutions for common errors
  - Added Parameter Selection Guide table with recommended adjustment strategies
  - Added complete workflow example to `Kay_ripple_detector` docstring showing filtering → detection steps
  - Clarified that `multiunit_HSE_detector` accepts both binary (0/1) and spike count formats
  - Added amplitude units documentation to `simulate_LFP` (arbitrary units with SNR guidance)
  - Bolded all unit specifications in docstrings (**seconds**, **cm/s**, **Hz**) for clarity

- **Improved Error Messages**
  - All new error messages follow WHAT/WHY/HOW pattern with actionable solutions
  - Errors include working code examples for fixing issues
  - Validation errors provide specific details (array shapes, sample counts, expected values)

### Changed

- **BREAKING**: `_preprocess_detector_inputs()` now requires `sampling_frequency` and `speed_threshold` parameters for validation (internal function, unlikely to affect users)
- Enhanced parameter descriptions across all detector functions with clearer units and adjustment guidance
- Improved `filter_ripple_band()` error handling to catch issues before scipy crashes

### Fixed

- Prevented cryptic numpy errors (`axis 1 is out of bounds`, `operands could not be broadcast`) with clear validation errors
- Fixed misleading filter warning-then-crash behavior for incompatible sampling frequencies
- Improved empty result guidance - docstrings now suggest specific parameter adjustments

## [1.5.1] - 2024-10-15

### Added
- Improved code formatting and organization
- Enhanced test coverage to 98%
- Added snapshot tests for detector outputs
- Improved type annotations with modern Python 3.10+ syntax

### Changed
- Refactored test suite for better organization
- Updated CI/CD for ruff and mypy integration
- Modernized type hints (using `|` instead of `Union`)

---

[Unreleased]: https://github.com/Eden-Kramer-Lab/ripple_detection/compare/v1.7.1...HEAD
[1.7.1]: https://github.com/Eden-Kramer-Lab/ripple_detection/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/Eden-Kramer-Lab/ripple_detection/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/Eden-Kramer-Lab/ripple_detection/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/Eden-Kramer-Lab/ripple_detection/releases/tag/v1.5.1
