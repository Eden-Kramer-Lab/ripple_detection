# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-09-22

Results and calls change: the same recording gives different events, so detect
again rather than mixing events from 1.x and 2.0. [MIGRATING.md](https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/MIGRATING.md)
lists the calls to change, the changes without a hint, and why your events
differ. Entries marked **Breaking** change the results or the calls; everything
here is relative to 1.7.1.

### Added

- Five detectors; the README's "Choosing a detector" table sets out how their
  conventions differ:
  - `Yu_ripple_detector` (Yu et al. 2017), which estimates its threshold at each
    call from the noise distribution during immobility.
  - `Zugaro_ripple_detector`, the FMAToolbox `FindRipples` two-threshold rule. It
    keeps and flags an event touching a gap or the record edge, which the
    original drops; `events[~(events.clipped_start | events.clipped_end)]`
    restores that.
  - `Long_sharp_wave_ripple_detector` (J. D. Long II's `DetectSWR`), which takes
    **raw** LFP from a pyramidal-layer channel and, as `sharp_wave_lfp=`, a
    stratum radiatum channel.
  - `Carey_candidate_detector`, the van der Meer lab's candidate-event code
    (`GenCandidateEvents`, Hilbert option) for the data of Carey, Tanaka & van
    der Meer 2019: a joint ripple and multiunit score, optionally restricted to
    low theta. The paper's published candidates used the code's spectral score
    and a single threshold instead, as the docstring describes.
  - `Shvartsman_ripple_detector`, a laboratory variant that keeps an event when
    `minimum_participating_channels` (or `minimum_participating_fraction`) of the
    channels detect it, and can take its statistics from elsewhere
    (`normalization_method="manual"`).
- `maximum_duration` on every detector but Long, which limits the sharp wave
  instead (`maximum_sharp_wave_duration`), and `minimum_active_units` (with
  `n_active_units` in the result) on `multiunit_HSE_detector`.
- `n_samples`, `clipped_start`, `clipped_end` and `peak_time` in every result;
  `n_samples` is the fourth column. `peak_time` was a Zugaro and Long column; Long's
  is still the sharp-wave peak.
- `band` and `transition_width` on `filter_ripple_band` and
  `ripple_bandpass_filter`, to design a filter for another band; the shipped
  kernel stays in use at 1500 Hz with the default 150-250 Hz band.
- Event helpers that accept a detector's DataFrame: `require_overlap` and its
  complement `exclude_overlap` (a veto: drop the events that coincide with
  intervals marked elsewhere, such as artifacts), `merge_close_events` and
  `exclude_movement_by_majority`.
- `detect_events_from_trace`, the detectors' thresholding on a trace the caller
  builds, with `bound_threshold` (where events end), `normalization_method="none"`,
  `minimum_event_duration` (on the whole event), `speed_rule` (including
  `"restrict"`, detection on slow samples only), `close_event_rule="merge"`, a
  per-sample `threshold` array, and `bound_search_window` with fallback
  `bound_threshold` levels. `speed=None` is accepted with the speed rule off.
- `carey_spectral_ripple_score`, the van der Meer lab's spectral ripple score
  (`SWRfreak` and `amSWR`), and `ripple_score` and `threshold_method="mean"` on
  `Carey_candidate_detector`, which together reproduce the rule behind the
  candidates released with Carey, Tanaka & van der Meer 2019.
- `detect_silence_bounded_events`: events as spiking from chosen units set off by
  silence, either the groups between silences or the window after each (ending at
  its last spike, or kept whole with `window_end_rule="fixed"`), with bursts
  optionally collapsed to their first spike.
- `theta_delta_ratio` and `state_intervals`, for detection restricted to a brain
  or behavioural state, and two data-driven thresholds: `two_cluster_threshold`
  (one-dimensional k-means, warning if it stops before converging) and
  `histogram_minimum_threshold` (the first trough after a distribution's peak).
- `ripple_detection.literature_methods`: each surveyed paper's candidate-event
  rule, and its secondary ripple, HFE and MUA inventories, on measured data.
  `Recording.from_arrays` takes the selected signals, cells and curated
  intervals; `list_methods()` gives each method's DOI, role, stages, grid and
  requirements (signals, cell selections, curated intervals, external inputs,
  options needed on measured data, a fixed sampling rate); `check_method` lists
  everything a call lacks without running it; `run_method(name, recording,
  behavior_intervals=..., **options)` returns events with a common set of columns
  whose `attrs` record the method, resolved options, grid, inputs and diagnostics
  as plain JSON types, and `save_events` / `load_events` keep that provenance with
  the table. Choices a paper leaves open are explicit options, a method that needs
  an input it was not given raises and names every missing input rather than
  substituting one, and the
  [implementation guide](https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/docs/literature/implementation.md)
  sets out what remains unverified. `examples/measured_walkthrough.py` works
  through a measured recording end to end, and `examples/literature_recipes.py`
  runs the default methods on a simulated session.
- `trim_events_to_trace` and `trim_events_to_spike_windows`, which narrow events
  to where a trace stays high or to edge windows holding enough spikes.
- `require_inside` (events wholly inside intervals), `intervals_to_mask` and
  `intersect_intervals`, for restricting events or statistics to a state or to
  curated epochs. Event helpers given a DataFrame return one.
- `require_trace_peak` and `require_times_inside`, which confirm one signal's
  events with another (a ripple z-score inside a burst, a ripple peak inside it),
  and `windows_around_times`, fixed windows around peaks or crossings.
- `require_active_units` (a count, a fraction, or a spike total of chosen units,
  such as place cells) and `count_spikes_in_events`, for participation criteria
  on any event inventory. They, and `trim_events_to_spike_windows`, warn when an
  event holds missing multiunit samples, which count as no spikes.
- Close-event variants: `inclusive` and `measure="peak"` on `merge_close_events`,
  `measure_from="start"` on `exclude_close_events`, and `require_isolation`, which
  drops every event of a close pair.
- `rule` on `exclude_movement`: besides the endpoint rule, every sample (`'all'`),
  the mean (`'mean'`) or the median (`'median'`) speed at or below the threshold.
- A detector registry for pipelines that store a detector by name:
  `get_detector(name)` returns a `DetectorSpec` whose `spec.inputs` and
  `spec.keyword_inputs` name the signals (`RIPPLE_BAND_LFP`, `RAW_LFP`,
  `MULTIUNIT`), whose `spec.parameters`, `spec.check_parameters(mapping)` and
  `spec.check_inputs(*signals, **keyword_signals)` check a call, and whose
  `spec.describe()` gives it all as JSON-ready data. `check_parameters` runs the
  detector's own range, type and unit checks and names removed or misspelled
  parameters, so a stored parameter set can be checked before any data is loaded.
- Simulation: `ripple_snr`, `rng` and `(low, high)` ranges for
  `ripple_frequency` and `ripple_duration` on `simulate_LFP`, and `simulate_multichannel_LFP`,
  `simulate_sharp_wave_ripple_pair`, `simulate_multiunit` and
  `simulate_session`, which share one set of ripples and return the ground
  truth. `simulate_speed` and `simulate_theta_delta`, and `running_intervals`,
  `theta_amplitude` and `delta_amplitude` on `simulate_session`, give a session
  running bouts with theta and rest with delta.
- `load_literature_parameters`, the detection parameters of 57 replay papers,
  and a simulation study comparing every detector (`examples/simulation_study.py`).
- `load_literature_datasets`, a separate packaged catalog of public recording
  and event-data links, joined to papers by DOI, with reuse relationships,
  input availability and scoped annotation-verification statuses.
- The package root exports the helpers the detectors are built from, such as
  `minimum_sample_count` and `sample_count_within`; `__all__` is the public API.
- Help for people and language models writing calls: a 1.x call fails with the
  2.0 change and the call to write, a keyword another detector (or library)
  uses for the same role names this one's, a duration given in milliseconds raises
  with the value to pass, every detector's docstring has a runnable example,
  and `llms.txt` maps the package.
- `time=` on `filter_ripple_band` and `get_envelope`: split the transform at
  gaps in the timestamps as well as at missing samples.
- String options are `Literal` types, so a type checker catches a misspelled
  choice.
- `py.typed` and `CITATION.cff`.

### Changed

- `get_envelope` and `filter_ripple_band` work within contiguous blocks of valid
  samples: a NaN or infinity in any channel marks that sample missing in every
  channel, and the data around it stay valid rather than one missing sample
  making a channel's whole envelope NaN. A channel with no finite sample raises.
- **Breaking.** The minimum duration counts samples, rounded half up, instead of
  comparing timestamps: about 20 % more events at 1500 Hz.
- **Breaking.** One missing-sample policy: NaN or infinity in any signal, or a
  step in `time` over 1.5 median steps, ends a block that nothing crosses, and
  events cut off by one are flagged.
- **Breaking.** A NaN in `speed` is an unknown speed that fails the speed rule at
  an event's endpoints, not a missing sample; `speed_threshold=np.inf` turns
  the rule off.
- **Breaking.** `filter_ripple_band` requires `sampling_frequency` and filters
  each run of finite samples on its own, where 1.7 assumed 1500 Hz and filtered
  across gaps.
- **Breaking.** `ripple_bandpass_filter` sizes the filter for the rate, where 1.7
  used 101 taps at any rate, so its coefficients change at every rate, 1500 Hz
  included (155 taps there). `filter_ripple_band` at 1500 Hz with the default
  band still uses the shipped kernel, and its output is unchanged.
- **Breaking.** `gaussian_smooth` renormalizes its kernel at the ends of the data
  instead of padding with zeros; z-score statistics move in the third or fourth
  decimal.
- **Breaking.** `max_thresh` is `max_sustained_zscore`, computed exactly where
  1.7 approximated it, with the caller's `minimum_duration`.
- **Breaking.** Karlsson's per-event statistics come from the per-sample maximum
  of the channels' z-scores instead of the mean filtered LFP.
- **Breaking.** The normalization statistics use only valid samples, and a zero
  or undefined scale raises instead of giving NaN or dividing by 1.0.
- **Breaking.** The sampling-rate check raises beyond a 10 % mismatch and warns
  from 2 %, where 1.7 warned from 20 %.
- **Breaking.** `simulate_LFP` defaults to pink (1/f) noise instead of brown.
- **Breaking.** A simulated ripple's carrier is a cosine from the ripple's
  centre, `cos(2 pi f (t - ripple_time))`, where it was `sin(2 pi f t)` on the
  clock: the same ripple now looks the same at any time origin, and its peak is
  at its centre. Every simulated recording with ripples changes.
- Faster: `filter_ripple_band` convolves by FFT (equal to `filtfilt` to 1e-15;
  3x faster at 1500 Hz, 12x at 30 kHz, where an hour of 32 channels took 17
  min), Karlsson's thresholding is no longer quadratic in the recording (4.2 s
  -> 1.1 s on 10 min of 16 channels), and the per-event statistics use
  bisection (5.5 s -> 0.03 s for 30 min with 500 events).
- Warnings name the caller's line, not a line inside the package.

### Removed

- `normalization_time_range`, and the `time` argument of `normalize_signal`; use
  `normalization_mask=(time >= start) & (time <= end)`.
- `use_speed_threshold_for_zscore` on `multiunit_HSE_detector`, deprecated in
  1.7.0; use `normalization_mask=speed <= speed_threshold`.

### Fixed

- **Breaking.** `exclude_close_events` compared each event with the candidate
  before it, not the last event kept, so it removed more than the first of a
  cluster.
- **Breaking.** `exclude_movement` read speeds in time order rather than per
  event, misjudging nested events and discarding every event when a bound fell
  between samples.
- Inputs that 1.7 turned into a wrong or empty result now raise a clear error:
  unsorted or NaN timestamps, all-missing input, dead channels, malformed masks
  and event arrays, rates passed as spike counts, and tunables that are NaN,
  negative, reversed or given in milliseconds; [MIGRATING.md](https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/MIGRATING.md#inputs-that-now-raise)
  lists them.
- `speed_threshold=np.inf` no longer warns that speed "appears very small", and a
  session in cm/s that is mostly at rest no longer triggers it.
- Mistakes the checks can see get a message that names the fix: ripple-band
  detectors warn when their input does not look filtered; timestamps in samples,
  in milliseconds, or disagreeing with `sampling_frequency` get distinct errors
  (`filter_ripple_band(time=)` checks the rate too); a transposed signal, float32
  timestamps at large values, speed on its own clock, a non-numeric rate, 2-D
  times passed to `require_times_inside`, and a 0/1 integer array passed as
  `units` each get their own message.
- `get_Kay_ripple_consensus_trace` says to reshape a 1-D input instead of raising
  NumPy's `AxisError`, and splits at gaps in `time`.
- `filter_ripple_band`'s length check was one sample too permissive, so a signal
  of exactly that length failed inside `scipy`.
- `simulate_LFP` adds each ripple over its own window; ten minutes with 100
  ripples peaked at 1.5 GB.
- `get_envelope` and `get_multiunit_population_firing_rate` convert their input,
  as their `array_like` annotation states.

### Development

- The package moved under `src/` and `ripple_detection.detectors` became a
  package (imports unchanged), with strict mypy, ruff, pre-commit, a `uv.lock`
  and `dev` group, CI on Python 3.10-3.14 and the dependency floors, pinned
  actions, and every reference resolved to a DOI or a file at a commit.

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
