# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ripple_detection` is a Python package for detecting sharp-wave ripple events (150-250 Hz) from local field potentials (LFPs) in neuroscience research. It implements detection algorithms from Karlsson & Frank 2009, Kay et al. 2016, Yu et al. 2017, Carey et al. 2019, FMAToolbox and buzcode, along with unpublished lab variants.

## Development Commands

### Setup

```bash
# uv (recommended): .venv from uv.lock with the package editable plus the dev tools
uv sync --extra examples
uv run pytest            # prefix any command with `uv run` to use that environment
uv lock                  # after changing dependencies in pyproject.toml
uvx pre-commit install   # once; each commit then runs ruff, codespell, mypy and the file checks

# Or pip into your own environment
pip install -e .[dev,examples]

# Or conda
conda env create -f environment.yml
conda activate ripple_detection
pip install -e .[dev,examples]

# Minimal install (runtime dependencies only)
pip install -e .
```

The `dev` extra and the `dev` dependency group in `pyproject.toml` list the same
tools; keep them identical.

Releasing: the `release` skill ([.claude/skills/release/SKILL.md](.claude/skills/release/SKILL.md)). What each test module covers: [tests/CLAUDE.md](tests/CLAUDE.md).

### Testing

```bash
# Run all tests with coverage, and the docstring examples in src/ (testpaths)
pytest

# Test notebooks (as done in CI)
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute examples/detection_examples.ipynb
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute examples/test_individual_algorithm_components.ipynb
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute examples/ripple_detection_tutorial.ipynb
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute examples/simulation_study.ipynb

# Re-run the simulation study sweep the notebook reads (about two minutes)
uv run python examples/simulation_study.py
```

## Architecture

### Core Module Structure

The package lives under `src/` (the Scientific Python guide's layout, so tests import the installed package, never the checkout) and is organized into five modules, one of them a package:

1. **[src/ripple_detection/core.py](src/ripple_detection/core.py)** - Low-level signal processing utilities

2. **[src/ripple_detection/detectors/](src/ripple_detection/detectors/)** - High-level detection algorithms, a package whose `__init__` re-exports the public names so `from ripple_detection.detectors import Kay_ripple_detector` still works
   - `_validation.py` - shape, length, unit and duration-limit checks
   - `_blocks.py` - the missing-sample policy: valid samples, contiguous blocks, block-wise transforms and threshold tests
   - `_events.py` - the shared detection tail, duration ceiling, active-unit counts, and `_get_event_stats`
   - `_lfp.py` - `Kay_ripple_detector` (Kay et al. 2016), `Karlsson_ripple_detector` (Karlsson & Frank 2009), `Roumis_ripple_detector` (Frank-lab variant, unpublished), `Shvartsman_ripple_detector` (unpublished), `Yu_ripple_detector` (Yu et al. 2017), and the two consensus traces
   - `_zugaro.py` - `Zugaro_ripple_detector`, the FMAToolbox `FindRipples` two-threshold rule
   - `_long.py` - `Long_sharp_wave_ripple_detector`, sharp wave + ripple power on two raw channels, k-means split (Long, `DetectSWR`)
   - `_carey.py` - `Carey_candidate_detector`, joint ripple-envelope × multiunit score (Carey, Tanaka & van der Meer 2019)
   - `_hse.py` - `multiunit_HSE_detector`, multiunit High Synchrony Events (spikes only)
   - `_trace.py` - `detect_events_from_trace`, the shared thresholding on a trace the caller builds (bound level, raw thresholds, whole-event minimum, speed and close-event rules); not a registered detector, since its signal is whatever trace the caller passes
   - The README's "Choosing a detector" table is the reference for how their conventions differ
   - All detectors return pandas DataFrames with event statistics

3. **[src/ripple_detection/simulate.py](src/ripple_detection/simulate.py)** - Synthetic data generation
   - `simulate_LFP`: one channel of coloured noise (pink by default; brown was the 1.x default and has almost no ripple-band power) with Gaussian-windowed sine bursts, sized in signal units or by `ripple_snr`
   - `simulate_multichannel_LFP`: channels sharing one ripple (per-channel gains) in noise that is part shared, part their own; optional common-mode artifacts
   - `simulate_sharp_wave_ripple_pair`: the raw pyramidal-layer and stratum radiatum channels the Long detector takes
   - `simulate_multiunit`: Poisson units that burst with the ripples
   - `simulate_session`: all of the above from one draw of per-ripple durations and frequencies, returned with the ground truth as a `SimulatedSession` (`ripple_windows` are the intervals a detected event should overlap)
   - The basis of the integration tests and of `examples/simulation_study.py`

4. **[src/ripple_detection/registry.py](src/ripple_detection/registry.py)** - `DETECTORS`, `get_detector`, `DetectorSpec`
   - Resolves a detector by name and says which signal kinds it takes (`RIPPLE_BAND_LFP`, `RAW_LFP`, `MULTIUNIT`), positionally (`inputs`) and by name (`keyword_inputs`: Long's `sharp_wave_lfp`, Carey's `theta_lfp`)
   - `spec.describe()`: JSON-ready inputs, tunables (default, unit, meaning) and output columns, from [_descriptions.py](src/ripple_detection/_descriptions.py); `tests/test_registry.py::TestDescribe` fails when a new or renamed parameter or column has no entry there
   - For pipelines that store a detector's name rather than importing it

5. **[src/ripple_detection/literature.py](src/ripple_detection/literature.py)** - `load_literature_parameters`
   - The survey of detection parameters from 57 replay papers, shipped as `data/literature_detection_parameters.csv`
   - The README's "Published parameter values" table is computed from it
   - [docs/literature/](docs/literature/) reviews each surveyed paper's detection method, with quotes, and whether this package reproduces it; `survey_corrections.md` there holds proposed corrections to the CSV, on hold

Two private modules serve callers rather than detection: [_call_hints.py](src/ripple_detection/_call_hints.py) wraps the public functions so a call written for 1.x fails with the 2.0 change behind it (add a removed or renamed argument to `REMOVED_ARGUMENTS` there, keyed by function, and only for a name a release shipped: users upgrade from a release, so a name that changed between releases gets no hint; `SAME_ROLE` maps other detectors' and libraries' names for a parameter by what it does), and [_descriptions.py](src/ripple_detection/_descriptions.py) holds what `describe()` reports. Warnings go through `core._warn_at_caller`, which attributes them to the first frame outside the package, so no function passes a `stacklevel`.

### Detection Pipeline Architecture

Kay, Roumis and the HSE detector threshold one trace (`_threshold_trace`; Kay and Roumis through `_detect_from_trace`); Karlsson and Shvartsman run the same steps on each channel and merge the per-channel events. Those five and Yu end in `_finish_events`: every candidate criterion (speed, HSE's active units) before the proximity rule, the duration ceiling after it:

1. **Preprocessing**: Validate shapes, units and time order; mark samples with NaN in any signal as missing (NaN speed is unknown speed, handled by the movement rules, and splits nothing) and split the rest into contiguous blocks (`_valid_blocks`), ending a block also wherever the timestamp step exceeds 1.5 times the median step
2. **Signal Transformation**: Hilbert envelope and Gaussian smoothing within each contiguous block (`_smoothed_envelope`); combine channels (Kay: consensus trace; Karlsson and Shvartsman: per channel; Roumis: mean; HSE: population rate)
3. **Normalization**: Z-score (or median/MAD) the trace; a zero or undefined scale raises
4. **Threshold Detection**: Runs at or above the threshold for at least `minimum_sample_count` samples
5. **Extension to Mean**: Extend each run to where the trace returns to the mean
6. **Movement Exclusion**: Speed at the first and last sample at or below `speed_threshold` (Shvartsman: majority of samples)
7. **Post-processing**: Drop close events, then over-long events; compute statistics with `_get_event_stats`
8. **Output**: DataFrame indexed by `event_number` with the columns listed under "Output Format" in the README

Yu, Zugaro, Long and Carey use their own segmentation rules but the same blocks: every step of every detector runs within a block, no event spans a gap, and `_get_event_stats` flags events cut off by a block edge in `clipped_start` and `clipped_end` (Zugaro supplies its own flags, meaning a missing crossing). A block too short for a detector's transform, or for an event of `minimum_duration` (`_valid_blocks(..., minimum_duration=...)`), is treated as missing with a warning, and no block left raises (`_drop_short_blocks`): Zugaro's smoothing window, Long's sharp-wave low-pass kernel, and Carey's theta filter pad length when `theta_lfp` is given.

### Key Algorithm Differences

- **Kay detector**: Combines multiple LFP channels into single consensus trace using sum of squared envelopes
- **Karlsson detector**: Detects ripples on each channel independently, then merges overlapping events
- **Roumis detector**: Averages square-root of squared envelopes across channels
- **Yu detector**: Median of per-channel z-scored envelopes; threshold from the mirrored immobility-noise histogram
- **Zugaro detector**: Two thresholds (bounds and peak) on the z-scored squared sum; merges close events
- **Long detector**: Raw input, the pyramidal-layer channel and `sharp_wave_lfp` by name; sharp-wave difference and ripple power clustered by k-means with local statistics
- **Carey detector**: Geometric mean of a ripple-envelope score and a capped multiunit score; whole event inside a low-speed interval
- **HSE detector**: Z-scored smoothed population spike rate, no LFP
- One missing-sample policy for every detector: NaN in any signal, or a gap in time, ends a block; nothing crosses a gap; clipped events are flagged. NaN speed is unknown speed and splits no block: it fails the endpoint rule, is left out of Shvartsman's majority, and interrupts Carey's low-speed intervals

### Pre-computed Filter

The package includes a pre-computed ripple bandpass filter ([ripple_detection/ripplefilter.mat](src/ripple_detection/ripplefilter.mat)) from the Frank lab with specific characteristics:

- 150-250 Hz bandpass
- 40 dB roll-off
- 10 Hz sidebands
- Sampling frequency: 1500 Hz

The shipped kernel is used only at 1500 Hz with the default band; for any other rate or band, `filter_ripple_band` designs an equiripple FIR with `ripple_bandpass_filter()` (`scipy.signal.remez`), scaling the tap count with the rate.

## Standards

- **Type hints for all function signatures** (using modern Python 3.10+ syntax)
- Numpy Docstrings for all public functions and classes using numpy docstring best practices
- Tests for every behavior a change touches, including the error paths

### Type Hints

Array annotations:
- `numpy.typing.ArrayLike` for array parameters (each function casts with `np.asarray` first) and the aliases `FloatArray`, `BoolArray` and `IntArray` from `core.py` for arrays it returns or holds, so the dtype is part of the signature

mypy runs in strict mode with no per-module overrides, and `py.typed` ships so downstream type checkers see the same annotations.

### Tool Configuration

All code quality tools are configured in [pyproject.toml](pyproject.toml):

**Pytest** (`[tool.pytest.ini_options]`):
- Strict: every warning is an error (`filterwarnings = ["error"]`), `--strict-config --strict-markers`, `xfail_strict`
- `--doctest-modules` over `testpaths = ["tests", "src"]`: every public detector's docstring example runs, so what a reader copies works. Examples print column names and booleans, not event counts, which move with NumPy's random streams
- Coverage of `src/ripple_detection` reported to the terminal with missing lines

**Pre-commit** (`.pre-commit-config.yaml`):
- The standard file hooks, `ruff-check --fix` and `ruff-format` on `src/` and `tests/` (the paths CI checks), codespell (configured under `[tool.codespell]`), and mypy through `uv run`
- `uvx pre-commit install` once; `uvx pre-commit run --all-files` runs everything by hand
- CI runs the same tools directly, so the hooks are a convenience, not a second source of truth

**Task runner.** There is no `nox` or `tox` file, on purpose. `uv run <command>` against the locked environment is the task runner. This, and having no documentation site (the README and the docstrings are the documentation), are the deliberate departures from the Scientific Python development guide.

For testing and development use `uv run` (the environment `uv sync` builds from `uv.lock`) or the `ripple_detection` conda environment if available, so dependency versions are consistent.
