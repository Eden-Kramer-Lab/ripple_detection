# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-09-21

Detection results change. The same recording gives different events, so upgrade
deliberately and detect again. No name was removed and no parameter changed
position, so the calls in your code still work. Nine signatures gained
parameters at the end. Pin `ripple-detection>=2,<3` and record the version
with the events that you detect. Entries marked **Breaking** change the results.

### Added

- `Yu_ripple_detector`, the detector of Yu et al. 2017. It estimates the
  threshold at each call, from the noise distribution during immobility.
- `Zugaro_ripple_detector`, the `FindRipples` algorithm of FMAToolbox. The
  bounds are at 2 SD and the peak must go above 5 SD.
- `Long_sharp_wave_ripple_detector`, the two-channel detector of J. D. Long II.
  It takes **raw** LFP from a pyramidal-layer channel and a stratum radiatum
  channel.
- `Carey_candidate_detector`, the candidate detector of Carey, Tanaka & van der
  Meer 2019. It combines a ripple score and a multiunit score.
- `Shvartsman_ripple_detector`, an unpublished laboratory variant. It keeps an
  event when at least `participation_threshold` channels detect it, and reports
  which channels took part.
- `maximum_duration` on the seven detectors that had no ceiling. It limits the
  event as the detector reports it, not the run above the threshold.
- `minimum_active_units` on `multiunit_HSE_detector`. Each event now reports
  `n_active_units`, the count of units with a spike in the event.
- `band` and `transition_width` on `filter_ripple_band` and
  `ripple_bandpass_filter`. The default band stays 150-250 Hz.
- `require_overlap`, which keeps the events of one detector that overlap an
  event of another. Use it to require a ripple and a population burst together.
- `merge_close_events`, which joins events that are close together.
  `exclude_close_events` keeps the first event and discards the others.
- `DETECTORS` and `get_detector`, which resolve a detector by name and give the
  signals that it takes. A pipeline that holds a detector by name does not need
  its own list.
- `load_literature_parameters`, the survey of detection parameters in 57 papers
  that decode replay content.
- `ripple_snr`, `random_state` and ranges for `ripple_frequency` and
  `ripple_duration` on `simulate_LFP`. The default call gives the same output as
  before.
- The package root now exports `minimum_sample_count`, `sample_count_within`,
  `nearest_sample_index`, `ripple_bandpass_filter` and `exclude_close_events`.
- README: tables for the choice of detector, for published parameter values, and
  for tools that this package does not implement.

### Changed

- **Breaking.** The minimum-duration test counts samples. It does not compare
  timestamps. At 1500 Hz a 15 ms minimum needs 23 samples, not 24. On 300 s of
  pink noise, filtered to the ripple band, this gives approximately 20 % more
  events at a `zscore_threshold` of 2.0 to 2.5. Kay gives 25 % more and
  Karlsson 20 % more.
- **Breaking.** All detectors use one duration rule: inclusive sample counts,
  rounded half up. Zugaro and Carey compared elapsed time, and Long rounded down.
- **Breaking.** Immobility is `speed <= speed_threshold` in all detectors. In
  `multiunit_HSE_detector` the deprecated `use_speed_threshold_for_zscore` path
  used `<`, so its normalization baseline, and therefore its events, change.
- **Breaking.** A duration ceiling below the minimum raises an error. Zugaro and
  Long gave an empty result.
- `Karlsson_ripple_detector` calculates its per-event z-score statistics on the
  maximum across channels, not on the mean. The events and their bounds do not
  change.
- `_detect_from_trace` and `_count_active_units` hold logic that several
  detectors stated separately. The output does not change.
- Ruff replaces black as the formatter and is now the only linter.
- Documentation. Each reference has a DOI that CrossRef resolved. Each link to
  laboratory code gives a file at a commit, and states its license. Each
  detector states its movement rule and its policy for missing samples.

### Fixed

- **Breaking.** `exclude_movement` and the per-event statistics read the speeds
  in time order, not one for each event. Nested events were kept or discarded in
  the wrong order, and a bound that was not on the sample grid discarded all
  events.
- **Breaking.** `ripple_bandpass_filter` used a fixed 101 taps, so the design
  became worse as the rate increased. At 30 kHz the stopband reached -1 dB. The
  count now follows the estimate of Kaiser.
- **Breaking.** `filter_ripple_band` designs a filter for the given rate.
  Before, it applied the 1500 Hz kernel at each rate, so the passband moved with
  the rate: 193-340 Hz at 2000 Hz. Output at 1500 Hz does not change.
- **Breaking.** `exclude_close_events` compared each event with the candidate
  before it, not with the last event that it kept. It removed more than the
  first event of a cluster.
- **Breaking.** `max_thresh` is now the largest threshold at which the detector
  would still find the event. Before, it could fall below `zscore_threshold`,
  and on noise every Karlsson event did, down to -0.16.
- **Breaking.** `Kay_ripple_detector` returns an empty result for an all-NaN
  input. Before, it raised an error. The other detectors still raise.
- These inputs now raise a clear error. Before, they gave an empty result, or
  failed with a message about something else:
  - a normalization mask that selects no samples, in any detector;
  - an all-NaN series or a negative threshold, in `segment_boolean_series` and
    `threshold_by_zscore`;
  - `multiunit` data of the wrong shape, which raised an `AxisError` before.
- `get_Kay_ripple_consensus_trace` no longer smooths across a gap when you give
  it `time`. It then operates on each continuous block. `Yu_ripple_detector`
  passes `time`. `Kay_ripple_detector` does not: it removes the rows that hold
  NaN first, and treats what remains as continuous, as its docstring states.
- `Karlsson_ripple_detector` and `multiunit_HSE_detector` give the
  `minimum_duration` of the caller to the `max_thresh` statistic. Before, they
  used the default of 15 ms.
- `Long_sharp_wave_ripple_detector` measured `max_thresh` against the ripple
  duration, but reports an event that spans the sharp wave. It gave NaN for each
  event when the two values differ. `max_thresh` is still NaN for an event that
  the ripple criterion admitted and whose sharp wave is shorter than
  `minimum_sharp_wave_duration`. The statistic has no definition there.
- The length guard of `filter_ripple_band` was one sample too permissive, so a
  signal of exactly that length failed inside scipy.
- The LFP detectors filter `normalization_mask` with the same removal of NaN
  rows as the data. A mask that selects no samples raises a clear error.
- `get_envelope` and `get_multiunit_population_firing_rate` convert their input,
  as their `array_like` annotation states.
- Warnings about units name the line of the caller, not a frame inside the
  package.
- `multiunit_HSE_detector` validates its input as the LFP detectors do.
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

[Unreleased]: https://github.com/Eden-Kramer-Lab/ripple_detection/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/Eden-Kramer-Lab/ripple_detection/compare/v1.7.1...v2.0.0
[1.7.1]: https://github.com/Eden-Kramer-Lab/ripple_detection/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/Eden-Kramer-Lab/ripple_detection/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/Eden-Kramer-Lab/ripple_detection/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/Eden-Kramer-Lab/ripple_detection/releases/tag/v1.5.1
