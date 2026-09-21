# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Zugaro_ripple_detector`: the FMAToolbox `FindRipples` algorithm (Hajime Hirase's method, implemented by Michaël Zugaro; carried into buzcode as `bz_FindRipples` by David Tingley and into AYA-lab neurocode, doi:10.5281/zenodo.7819979), reimplemented from the algorithm since the original is GPL-3. Ripple-band signal squared, summed across channels, smoothed with an 11-sample moving average at 1250 Hz (scaled with the rate), z-scored; events bounded where the trace crosses a low threshold (2 SD) and kept only if the peak exceeds a high threshold (5 SD); neighbours closer than 30 ms merged while the merged span stays under 100 ms; 20-100 ms duration limits. Defaults follow FMAToolbox and buzcode; neurocode's differing defaults are documented. Departures, documented: the package's endpoint speed rule is applied (disable with `speed_threshold=np.inf`), the reported `peak_time` is the maximum of the normalized power rather than a single channel's trough, and missing samples are handled block-wise.
- `Yu_ripple_detector`: the Yu et al. 2017 (eLife) sharp-wave ripple detector. The consensus is the median across tetrodes of each tetrode's 4 ms-smoothed, z-scored ripple-band envelope (`get_Yu_ripple_consensus_trace`), and the detection threshold is estimated per call as the 99.99th percentile of the immobility noise distribution, obtained by mirroring the histogram below its mode (`estimate_noise_threshold`, a transliteration of the original MATLAB with the reflection written as intended and a warning where the original's formula would differ). Events are runs of at least `minimum_duration` (default 20 ms, counted in samples) at or above the threshold, extended to the immobility mean. Missing samples are handled block-wise instead of by row-dropping, so nothing is smoothed or joined across a gap; events truncated by a gap or the recording edge are kept and flagged (`clipped_start`, `clipped_end`). The keyword interface matches `Kay_ripple_detector` with `percentile` in place of `zscore_threshold`.
- `random_state` parameter to `simulate_LFP()` for reproducible synthetic LFP generation (used to make the test suite deterministic).
- Regression tests for Shvartsman participation preserve the original count of distinct electrodes across each merged event, including chains of overlapping ripples and repeated ripples on the same electrode.

### Fixed

- `max_thresh` now uses the same inclusive duration comparison as event detection, preventing spurious `NaN` values or underestimated thresholds from floating-point rounding at exact `minimum_duration` boundaries.
- `normalization_mask` is now filtered by the same NaN-row removal as the LFP/speed data in the LFP detectors (Kay, Karlsson, Roumis, Shvartsman), fixing a length-mismatch `ValueError` when the input contained NaN samples.
- A `normalization_mask` that selects no samples now raises a clear error from `normalize_signal` (matching the existing `normalization_time_range` behaviour) instead of silently returning no events from a degenerate all-NaN/all-zero normalized trace.
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
