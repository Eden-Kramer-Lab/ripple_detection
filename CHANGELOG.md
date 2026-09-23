# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-09-22

Detection results change. The same recording gives different events, so upgrade
deliberately and detect again; events stored from 1.x follow the 1.x rules.
Pin `ripple-detection>=2,<3` and record the version with the events that you
detect. Entries marked **Breaking** change the results or the calls. Everything
below is relative to 1.7.1.

### Migrating from 1.7

Calls that stop working, and the change to make. Each of these now fails with a
message that names its replacement.

- `filter_ripple_band(lfp)` -> `filter_ripple_band(lfp, sampling_frequency)`.
  The rate is required; 1.7 assumed 1500 Hz whatever the data's rate.
- A tunable passed by position -> by name. Every detector's tunables after
  `sampling_frequency` are keyword-only:
  `Kay_ripple_detector(time, lfps, speed, 1500, speed_threshold=4.0)`.
- `normalization_time_range=(start, end)` ->
  `normalization_mask=(time >= start) & (time <= end)`, on every detector and
  on `normalize_signal`.
- `normalize_signal(data, time, method, mask)` ->
  `normalize_signal(data, method, normalization_mask)`: the `time` argument is
  gone.
- `multiunit_HSE_detector(..., use_speed_threshold_for_zscore=True)` ->
  `normalization_mask=speed <= speed_threshold`. The flag used `speed <
  speed_threshold`; the mask also counts a speed exactly at the threshold as
  immobile, as the speed rule always has.
- `pink(N, state=np.random.RandomState(seed))`, and `white` and `brown` the
  same way -> `pink(N, rng=seed)`, a seed or a `numpy.random.Generator`.
- The result column `max_thresh` -> `max_sustained_zscore`, with a different
  meaning; see below.
- `exclude_close_events` and `exclude_movement` return an array of shape
  `(n, 2)` when nothing is left, where they returned `[]`.

Apart from these, the Kay, Karlsson, Roumis and HSE signatures only add
keywords (`maximum_duration` on each, `minimum_active_units` on HSE), so a
pipeline that already passes every tunable by keyword and filters with an
explicit rate needs no code change, unless its inputs now raise (see Fixed).

### Why your events differ

The changes that move results on the same recording, in rough order of how much.
Each is described under Changed or Fixed.

- The minimum duration counts samples instead of comparing timestamps: at 1500
  Hz a 15 ms minimum needs 23 samples, not 24, which gives about 20 % more
  events (Kay 25 %, Karlsson 20 %).
- Missing samples split the recording into blocks that nothing crosses. In 1.7,
  NaN rows were dropped and the rest joined as if continuous, so an event could
  span a gap, and `filter_ripple_band` rang across the join.
- A NaN in speed no longer removes the sample; it is an unknown speed.
- The sampling rate sets the filter: at any rate but 1500 Hz a filter is
  designed for it, where 1.7 applied the 1500 Hz kernel (at 1000 Hz, 97-170
  Hz, twice the events).
- `exclude_close_events` (and `close_ripple_threshold` above 0) compares each
  event with the last one kept, not with the candidate before it.
- `exclude_movement` looks the speed up in time order, so nested events and
  bounds between samples are judged right.
- Smoothing no longer pulls the trace toward zero at the recording's ends, which
  moves the z-score statistics in the third or fourth decimal.
- Karlsson's statistics come from the per-sample maximum over channels' z-scores,
  and `max_sustained_zscore` replaces `max_thresh`.

### Added

- Five detectors:
  - `Yu_ripple_detector` (Yu et al. 2017): the threshold is estimated at each
    call from the noise distribution during immobility.
  - `Zugaro_ripple_detector`: the FMAToolbox `FindRipples` two-threshold rule,
    bounds at 2 SD and a peak above 5 SD. An event touching a gap or the
    record edge is kept and flagged, where the original drops it;
    `events[~(events.clipped_start | events.clipped_end)]` restores that.
  - `Long_sharp_wave_ripple_detector` (J. D. Long II, buzcode and neurocode
    `DetectSWR`): takes **raw**, unfiltered LFP from a pyramidal-layer channel
    and a stratum radiatum channel.
  - `Carey_candidate_detector` (Carey, Tanaka & van der Meer 2019): a joint
    ripple and multiunit score, optionally restricted to low theta.
  - `Shvartsman_ripple_detector`, an unpublished laboratory variant: an event
    is kept when at least `minimum_participating_channels` channels (2 by
    default), or `minimum_participating_fraction` of them, detect it, and it
    reports which channels took part. `normalization_method="manual"` takes
    the statistics from elsewhere, such as a whole recording day.

  The README's "Choosing a detector" table sets out how their conventions differ.
- `maximum_duration` on every detector (Zugaro and Long have their own
  ceilings); it limits the event as reported, and must not be below the
  minimum.
- `minimum_active_units` on `multiunit_HSE_detector`, and `n_active_units` in
  its result.
- `n_samples`, `clipped_start` and `clipped_end` in every result. `n_samples` is
  the fourth column, so code that reads columns by position shifts. `duration`
  is elapsed time, one sample interval less than `n_samples` spans, so an event
  of exactly the minimum count has a `duration` below `minimum_duration` by
  half to one and a half intervals (23 samples span 14.67 ms at 15 ms and 1500
  Hz). `clipped_start` and `clipped_end` flag an event cut off by missing data
  or the recording edge.
- `band` and `transition_width` on `filter_ripple_band` and
  `ripple_bandpass_filter`. The shipped kernel is used at 1500 Hz with the
  default 150-250 Hz band; any other band, rate or width designs a filter.
- Event helpers: `require_overlap` keeps the events of one detector that overlap
  another's (a ripple and a population burst together); `merge_close_events`
  joins close events where `exclude_close_events` drops all but the first;
  `exclude_movement_by_majority` keeps an event when most of its samples with
  known speed are immobile. They and the 1.7 helpers accept a detector's
  DataFrame.
- A detector registry for pipelines that store a detector by name. `DETECTORS`
  maps each name to a `DetectorSpec`, and `get_detector(name)` looks one up.
  `spec.inputs` says which signals the detector takes (`RIPPLE_BAND_LFP`,
  `RAW_LFP_PAIR`, `MULTIUNIT`); `spec.parameters` gives every tunable with its
  default; `spec.check_parameters(mapping)` and `spec.check_inputs(*signals)`
  raise on a name the detector does not take or a signal of the wrong shape;
  `spec.describe()` returns all of it as JSON-ready data, with each tunable's
  unit and meaning and the result's columns.
- Simulation. `simulate_LFP` takes `ripple_snr` (ripple size against the
  ripple-band background), `random_state`, and `(low, high)` ranges for
  `ripple_frequency` and `ripple_duration`. New simulators share one set of
  ripples: `simulate_multichannel_LFP`, `simulate_sharp_wave_ripple_pair`,
  `simulate_multiunit`, and `simulate_session`, which returns them all with the
  ground truth. Every draw uses `numpy.random.default_rng`, so a seed gives
  different noise than 1.7's `RandomState`.
- `load_literature_parameters`, the detection parameters of 57 papers that
  decode replay, and a simulation study (`examples/simulation_study.py` and its
  notebook) comparing every detector's recall, precision and false positives.
- The package root exports the helpers the detectors are built from, among
  them `get_envelope`, `gaussian_smooth`, `ripple_bandpass_filter`,
  `get_Kay_ripple_consensus_trace`, `exclude_close_events`,
  `minimum_sample_count` and `sample_count_within`; `__all__` is the public
  API.
- For people and language models writing calls: a 1.7 call fails with the 2.0
  change behind it and the call to write, a near-miss keyword gets a "did you
  mean", a duration given in milliseconds (`minimum_duration=15`) raises with
  the value to pass, every detector's docstring has a runnable example, and
  `llms.txt` at the repository root is a short map of the package.
- `py.typed`, so type checkers see the package's annotations, and
  `CITATION.cff`.

### Changed

- **Breaking.** The minimum-duration test counts samples, rounded half up from
  the median timestamp step, and includes the minimum; it no longer compares
  timestamps. At 1500 Hz a 15 ms minimum needs 23 samples, not 24. On 300 s of
  pink noise filtered to the ripple band this gives about 20 % more events at a
  `zscore_threshold` of 2.0 to 2.5: Kay 25 % more, Karlsson 20 %.
- **Breaking.** One missing-sample policy for every detector. A sample is
  missing when any channel of any signal is NaN or infinite, or when the step
  in `time` to it exceeds 1.5 times the median step. The valid samples form
  blocks, and every step of every detector runs within a block, so nothing is
  smoothed, thresholded or merged across a gap and no event spans one; an
  event cut off by a gap is kept and flagged. In 1.7 the Kay, Karlsson and
  Roumis detectors dropped the NaN rows and treated the rest as continuous, and
  the HSE detector smoothed across a NaN, which blanked its rate for a kernel
  width around each one (361 samples at 1500 Hz for one missing count). A
  block too short for an event of `minimum_duration` is treated as missing with
  a warning that gives its sample ranges, and a detector with no block left
  raises. Output on data without gaps does not change.
- **Breaking.** A NaN in `speed` is an unknown speed, not a missing sample, so a
  tracking dropout no longer removes LFP samples or cuts a ripple in two. An
  event whose first or last sample has unknown speed fails the speed rule;
  `speed_threshold=np.inf` turns the rule off, unknown speed included. The
  speed statistics skip unknown values. Speed that is NaN everywhere raises
  unless the rule is off.
- **Breaking.** `filter_ripple_band` requires `sampling_frequency`. In 1.7 the
  rate defaulted to `None`, which applied the 1500 Hz kernel to data at any
  rate without a check, so 1000 Hz data was filtered to 97-170 Hz and gave
  twice the events; a given rate other than 1500 Hz raised below 1200 Hz and
  otherwise warned and applied the same kernel. Each run of finite samples is
  filtered on its own; 1.7 joined the runs, and the step between the two sides
  of a gap rang through the filter (on noise with a 5 s gap, Kay reported a
  5 s event with a z-score near 10). A run as short as the kernel can be
  filtered (318 samples at 1500 Hz, where 1.7 needed 955 in all); a shorter one
  is returned as NaN with a warning. An infinite sample is missing, as NaN is.
- **Breaking.** `ripple_bandpass_filter` sizes the filter for the rate (Kaiser's
  estimate), where 1.7 used 101 taps at any rate: at 30 kHz its stopband
  reached -1 dB. Output at 1500 Hz with the shipped kernel does not change.
- **Breaking.** `gaussian_smooth` renormalizes its kernel where it runs past
  either end of the data instead of padding with zeros, so a trace keeps its
  level up to the recording's ends and up to a gap. Event bounds away from the
  ends are unchanged; the z-score statistics move in the third or fourth
  decimal, because the normalization no longer includes the artificially low
  samples at the two ends.
- **Breaking.** `max_thresh` is `max_sustained_zscore`: the largest z-score
  sustained for `minimum_duration`, which is the highest threshold at which the
  detector would still find the event. In 1.7 the value could fall below
  `zscore_threshold` (on noise every Karlsson event did, down to -0.16), and
  Karlsson and HSE computed it for 15 ms whatever `minimum_duration` was.
- **Breaking.** `Karlsson_ripple_detector` reports its per-event statistics on
  the per-sample maximum over the channels' z-scored envelopes; 1.7 used the
  mean of the filtered LFP. The events do not change.
- **Breaking.** `normalization_mask` is restricted to the valid samples, and a
  zero or undefined normalization scale raises and names the channel. In 1.7 a
  constant trace gave NaN, or zeros under a mask, and a channel constant over
  the mask was divided by 1.0, so its values outside the mask were reported as
  z-scores: under `median_mad` one partly disconnected channel produced a 9 s
  event with a sustained z-score of 32000. Drop a dead channel before
  detecting.
- **Breaking.** The sampling-rate check raises when `sampling_frequency` and the
  timestamps disagree by more than 10 %, and warns from 2 %, where 1.7 warned
  from 20 %. The rate sets the smoothing widths and the timestamps set the
  sample counts, so a 20 % mismatch changed Kay's event count by a fifth
  without a word.
- **Breaking.** `simulate_LFP` defaults to pink (1/f) noise. Brown noise, the 1.7
  default, has almost no ripple-band power, so a ripple of any size dominated
  the band and every detector found every ripple. Pass `noise_type="brown"` for
  the old signal.
- Faster. `filter_ripple_band` convolves by FFT: equal to `scipy`'s `filtfilt`
  to rounding (1e-15), three times faster at 1500 Hz and twelve at 30 kHz,
  where an hour of 32 channels took about 17 minutes. The thresholding of
  Karlsson and the extension of each event to the mean are no longer
  quadratic in the recording length (Karlsson: 4.2 s -> 1.1 s on ten minutes
  of 16 channels), and the per-event statistics use bisection (5.5 s -> 0.03 s
  for half an hour with 500 events).
- Warnings name the caller's line, not a line inside the package.

### Removed

- `normalization_time_range`, and the `time` argument of `normalize_signal`
  that existed only to serve it. A time range is a mask:
  `normalization_mask=(time >= start) & (time <= end)`.
- `use_speed_threshold_for_zscore` on `multiunit_HSE_detector`, deprecated in
  1.7.0. Use `normalization_mask=speed <= speed_threshold`.

### Fixed

- **Breaking.** `exclude_close_events` compared each event with the candidate
  before it, not with the last event it kept, so it removed more than the first
  event of each cluster. The detectors' `close_ripple_threshold` and
  `close_event_threshold` share the fix.
- **Breaking.** `exclude_movement` matched speeds to events with `np.isin`,
  which read them in time order rather than one per event: nested events were
  judged by the wrong speeds, and a bound between samples discarded every
  event.
- `speed_threshold=np.inf` no longer warns that speed "appears very small".
- `get_Kay_ripple_consensus_trace` raises a `ValueError` that says to reshape a
  one-dimensional input, where it raised NumPy's `AxisError`, and takes `time`
  so its envelope and smoothing split at gaps.
- `filter_ripple_band`'s length check was one sample too permissive, so a
  signal of exactly that length failed inside `scipy`.
- `simulate_LFP` adds each ripple over its own window instead of holding a
  full-length array per ripple: ten minutes at 1500 Hz with 100 ripples peaked
  at 1.5 GB.
- These inputs raise a clear error, where 1.7 returned a wrong or empty result
  or failed with a message about something else:
  - `time` that is not increasing (the event and speed lookups assume order),
    that holds NaN, or whose timestamps mostly repeat (the minimum duration
    became one sample);
  - an input whose every sample holds a NaN, naming the channels that hold
    none;
  - a normalization mask that selects no samples, is not boolean (a forgotten
    comparison made every nonzero speed count as immobile), or is 2-D;
  - an event array that is not `(n, 2)`, in the event helpers;
  - a series holding NaN, or a negative threshold, in `segment_boolean_series`
    and `threshold_by_zscore`;
  - `multiunit` of the wrong shape, or holding negative or fractional values;
  - a channel that is constant over the valid samples (a dead or disconnected
    channel);
  - a tunable that is NaN, negative, reversed or in the wrong unit: a NaN or
    negative threshold, duration, gap or speed limit, a non-positive
    `sampling_frequency`, a smoothing width of zero or of a second or more, a
    minimum duration of a second or more, a ceiling or gap longer than 10 s
    (milliseconds given as seconds), or an infinite ceiling or gap (`None` is
    the way to have no ceiling). Each disabled a criterion or emptied the
    result silently; `minimum_duration=0` still means no minimum;
  - in `simulate_LFP`, a ripple time outside `time`, a duration that is not
    positive, a frequency outside the Nyquist range, or a NaN size, each of
    which returned an all-NaN, empty or aliased signal.
- `get_envelope` and `get_multiunit_population_firing_rate` convert their input,
  as their `array_like` annotation states.

### Development

- The package lives under `src/`; imports do not change. `ripple_detection.detectors`
  is a package of modules, and every name is still importable from it and from
  the package root.
- Strict mypy, ruff as the only formatter and linter, pre-commit hooks, a
  committed `uv.lock` with a `dev` dependency group, and the license as an SPDX
  expression.
- CI tests Python 3.10 through 3.14 and the declared dependency floors, runs the
  docstring examples and every notebook, and smoke-tests the built wheel; its
  actions are pinned to commits with least-privilege permissions and kept
  current by Dependabot.
- Documentation: each reference has a DOI that CrossRef resolved, each link to
  laboratory code gives a file at a commit with its license, and each detector
  states its movement rule and its missing-sample policy.

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
