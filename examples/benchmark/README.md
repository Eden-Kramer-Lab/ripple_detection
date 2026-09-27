# Detector benchmark

Code and tables for a benchmark, in progress, of the package's detectors and the
packaged literature methods on simulated sessions whose events are known. None of it is
part of the installed package. It holds `simulator_targets.csv`, the measurements the
network simulator is validated against, the simulation conditions and the configurations
of the literature methods below, and the runner that simulates the conditions, runs every
method and scores it (see Running the benchmark).

## Simulation conditions

`conditions.py` defines the sessions the benchmark simulates. `REFERENCE` holds every
keyword of the three simulator calls at its reference value, the simulator's defaults
written out in full: `"session"` (600 s at 1500 Hz), `"events"` (`draw_network_events`),
`"non_events"` (`draw_non_events`) and `"render"` (`simulate_network_session`).
`conditions()` gives the 43 conditions: the reference; 15 factors varied one at a time
(28 levels); the simulator's six alternative models, one at a time; and the 8 cells of
`ripple_snr` crossed with `participation` and with `spike_leakage_rate` that are not
already in the grid. `factor_levels(factor)` lists a factor's levels in their designed
order with the reference's in place, labelled `"reference"`. A condition changes values
by dotted key, such as `"events.ripple_snr"` or `"non_events.rates.emg"`; `resolve`
gives its full parameter set and `resolved_json` the same as sorted JSON, for saving and
hashing. A reference
value changed after it was first set is recorded in `REFERENCE_REVISIONS` (its dotted
key, the previous and revised values, the reason and the evidence), never silently; the
simulator validation report lists every revision.

```python
import sys

sys.path.insert(0, "examples/benchmark")  # from the repository root
from conditions import conditions, resolved_json, simulate_condition

condition = next(c for c in conditions() if c.condition_id == "ripple_snr=high")
session = simulate_condition(condition, 0, overrides={"session.duration_s": 60.0})
print(resolved_json(condition)[:80])
```

A session starts at rest and alternates rest of 20-40 s with running bouts of 10-20 s,
dropping a bout that would leave less than 5 s of rest at the end. Replicate `k` has the
seed `session_seed(k)` in every condition. That seed gives one seed to each stage, in
the order schedule, events, non-events, rendering, so what one stage draws never moves
another's draws: replicate `k` of every condition has the same running bouts, and a
condition that changes no draw count (a size such as `ripple_snr`, or any of the six
alternative models) keeps the reference's event times, unit baseline rates and
per-unit participant draws. Conditions are therefore compared replicate by replicate.

A saved parameter set regenerates its sessions without the condition:
`parameters_from_json` reads `resolved_json`'s text, or a run's `conditions.csv`
`params`, back as `resolve` gives it (ranges as tuples again), and
`simulate_parameters(parameters, k)` simulates replicate `k` of it, as
`simulate_condition` does.

## Literature method configurations

`recipe_configs.py` configures the methods of `ripple_detection.literature_methods`
for the benchmark. It does not implement any method: a configuration names a method
from `list_methods()`, the options it runs with and the expression of the simulated
network event it is to be scored against first, and `run_recipe` calls the method through the
package's `run_method`. What each method does, which inputs it needs and why, and what
remains unverified are in the package's
[implementation guide](../../docs/literature/implementation.md), the method docstrings
and the paper notes.

### Running a configuration

`recipe_configs` is imported by name, so the directory must be on `sys.path`: paste
the example into an interpreter started in `examples/benchmark`, save it as a script
there, or run `sys.path.insert(0, "examples/benchmark")` first from the repository
root.

```python
import numpy as np
import ripple_detection as rd
from recipe_configs import RECIPES, behavior_intervals, make_recording, run_recipe

time = np.arange(90_000) / 1500.0
running = np.array([[25.0, 35.0]])
events = rd.draw_network_events(time, running_intervals=running, rng=0)
session = rd.simulate_network_session(time, events, running_intervals=running, rng=1)

config = next(config for config in RECIPES if config.config_id == "stella_2019")
recording = make_recording(session, config)
found = run_recipe(config, recording, behavior_intervals(session, config))
print(found.attrs["method"], found.attrs["role"], found.attrs["options"])
print(config.assumptions)
```

`found` is the package's result, unchanged: bounds, the method's own columns, and
`attrs` with its DOI, resolved options, grid, the inputs it ran on and diagnostics.
`check_recipe` lists what a call would lack without running it, and `method_record`
gives one flat, all-string record per configuration: its identity, the method's DOI,
role and stage, the primary expression, and the resolved options, input policy and
assumptions as JSON.

### The inputs a configuration receives

`make_recording` builds the recording with `Recording.from_arrays`, the path measured
data take, so the package's simulation fallbacks never run. The recording holds
exactly the inputs the method declares in `list_methods()`'s `requirements`, taking
into account conditions on its options (`when`). A requirement that lapses when another
input is supplied (`unless`) never lapses here, since only declared inputs are
supplied; an input a method does not ask for could change its events. The policy
(`INPUT_POLICY`) takes each input from the session:

- `lfps`, `sharp_wave_lfp`, `multiunit` and `speed`: the recorded signals, every
  pyramidal-layer channel with the ripple channel first.
- `place_cells`: the units the simulator labels `"place"`, and `pyramidal` those
  labelled `"place"` or `"pyramidal"`. These are the simulator's own labels, not a
  classification. The simulator has no place fields or trajectories, so every place
  unit stands in for a narrower selection (one template's, one directional template's,
  one probe sequence's cells), and `templates` is one template of every place unit.
- `sleep_intervals`, `baseline_intervals` and the call's `behavior_intervals`: rest,
  the recorded samples outside the running bouts, which the simulator states as it
  states unit labels. The sessions are awake, so rest stands in for a sleep state and
  for a normalization epoch; it stands in for eligible epochs because simulated events
  occur only at rest and the simulator has no position (no reward zones, track ends or
  corners).
- `reference_lfp`: zeros, so nothing is subtracted, as `examples/literature_recipes.py`
  does: the simulation has no reference electrode.
- `external_ripples` (Yang 2024, Grosmark 2016): the public `Zugaro_ripple_detector`
  on the first channel, the package's stated assumption for these papers' unspecified
  ripple detector. `example_ripples` (Carey 2019): the five largest
  `Kay_ripple_detector` events. Neither is taken from the simulation's truth.
- Options the paper does not report take the package's demonstration value only
  where the package ships one: an option it requires of measured data alone, and sets
  itself for a simulated session (Stella's wavelet, Nádasdy's RMS window and bounds,
  Kudrimoti's threshold, Wikenheiser's window anchor). Where the package requires an
  option with no default for any data, it ships no value, and the method is excluded
  (see Exclusions).

A session without unit labels gives empty selections, which `check_recipe` and
`run_recipe` report; nothing stands in for them. Each configuration's `assumptions`
list every stand-in and demonstration value it relies on, and each result's
`attrs["inputs"]` and `attrs["behavior_intervals"]` hold the values supplied.

### Stages, roles and expressions

Methods with a decoding-candidate stage run their detection stage (`configure` sets
`stage="detection"` in the options): selecting replay candidates is outside the
benchmark. Two protocol settings of the standalone demonstration are separate
configurations, `olafsdottir_2015.bayesian_candidates` and
`olafsdottir_2017.trajectory`, never pooled with their method's default.

A result's `role` comes from the package and is kept in every record: a secondary
ripple label or a candidate gate (Gupta 2010) is not a paper's candidate inventory,
even when both are scored against ripples. `primary_expression` is a benchmark
decision, made from what each implemented inventory's events require: `ripple` for
LFP ripple or SWR events, `burst` for population events alone, `network` where the
events join a ripple or SWR detection with a population-burst detection. A
participation count or spiking veto applied to ripple events keeps them `ripple`.

## Exclusions

A catalog method without a configuration, and why. `check_method` reports each
reason's missing option or rate on a recording built under the policy.

- `bush_2022_ripples`: input sampled at 4800 Hz: the method requires this rate, the
  simulated sessions are 1500 Hz, and resampling is not part of the input policy
- `olafsdottir_2017_ripples`: input sampled at 1200 Hz: the method requires this rate,
  the simulated sessions are 1500 Hz, and resampling is not part of the input policy
- `gridchyn_2020_ripples`: rms_window, bound_threshold: required options the paper
  does not report, nor do the Csicsvari et al. 1999 methods it cites; no value is
  assumed
- `xu_2019_ripples`: rms_window, bound_threshold: required options the paper does not
  report, nor do the Csicsvari et al. 1999 methods it cites; no value is assumed
- `farooq_2019_neuron_ripples`: threshold, bound_threshold, smoothing_sigma: required
  options the paper does not report; no value is assumed
- `farooq_2019_science_ripples`: power_measure, bound_threshold: required options (the
  power definition and the event bounds) the paper does not report; no value is
  assumed
- `chenani_2019_hfe`: ar_coefficients: required AR(2) coefficients fitted per channel,
  whose fit convention the paper does not specify; no value is assumed
- `liu_2019_ripples`: smoothing_sigma: a required option the paper does not report; no
  value is assumed
- `liu_2019_ripple_frames`: smoothing_sigma: a required option (of the ripple power it
  builds on) the paper does not report; no value is assumed
- `drieu_2018_ripples`: signal_measure: a required option, since the paper leaves
  amplitude or power unresolved; no value is assumed
- `diba_2007_ripples`: rms_window: a required option the paper does not report. The
  1.6 ms window of the Csicsvari et al. 1999b methods it cites is not assumed: the
  package lists RMS windows among settings its source audit did not establish

## Running the benchmark

`run.py` simulates every replicate of every condition, runs each of the package's nine
detectors at its defaults and at each point of its threshold sweep (`THRESHOLD_SWEEPS`;
Long's `peak_thresholds` sets both of its threshold pairs to `(0.5, v)`) and every
configuration in `RECIPES`, and scores each against the session's truth windows of every
expression (`ripple`, `sharp_wave`, `burst` and `network`). Events are matched one to one
(`match_events`) to the windows at 10 % of the envelope's peak, at each minimum IoU of
`MATCH_IOU_LEVELS` (0.0, the headline, then 0.2 and 0.5), and the pairs' onset and offset
errors are measured against the windows at 10, 25 and 50 %. The reference condition has
20 replicates and every other condition 10: 440 sessions, replicate `k` with the same
seed in every condition. A call's warnings change nothing but are each written to
`warnings.csv` (the session, method and setting, the warning's class and message); a
call that raises is written to `failures.csv`, with no events, results or scores, and
the run goes on. A (session, method, setting) without scores is a failure, never zero
events: a call that finds nothing still has its scores and an entry in its `results/`
sidecar.

Run it from the repository root, in this order. First validate the simulator for the
settings the run will use:

```bash
uv run python examples/benchmark/validate_simulator.py --validation-id v1 --conditions all
```

The runner checks that report before it runs any method (smoke, full run or resume): it
must be ready, with at least the predeclared 20 replicates per condition, and must have
been made from the current simulator source, target table and `validate_simulator.py`
for exactly the parameters of every selected condition, `--duration` included. Its
path, the SHA-256 of its `spec.json` and the fingerprints are saved in the run's
`run_spec.json`. Then the smoke test, one reference session in one process, which prints
each method's runtime, the simulation time, the peak resident memory, the rows and bytes
of every table, the full grid's runtime and size, the full validation's runtime (43
conditions at 20 replicates, from the per-session runtime and peak memory the report
records), the two together, and the decision rules' verdicts:

```bash
uv run python examples/benchmark/run.py --run-name smoke --smoke \
    --validation-report examples/benchmark/validation/v1/spec.json
```

The rules: if one session takes more than 5 minutes, halve `duration_s` and double the
replicates (which needs a report for the new duration first); if all conditions'
`events.csv.gz` would pass 2 GB, write sweep events only for the reference condition;
run `min(requested, free cores - 1, 0.7 x available memory / peak memory)` workers.

> **Placeholder, smoke test:** the measured numbers (per-session simulate and detect
> time, peak memory, rows and bytes per table), the extrapolation and any rule applied.

Look at single events before trusting any aggregate: `spot_check.py` simulates the
smoke session again from its seed and draws six true events of each type with their
truth windows and the events of Kay, Karlsson, the HSE detector and two recipes, one PNG
per type in the run's `spot_check/` directory:

```bash
uv run python examples/benchmark/spot_check.py --run-name smoke
```

Then the full run, by hand in a `tmux` session named `benchmark` (it is not part of CI):

```bash
uv run python examples/benchmark/run.py --run-name v1 --conditions all --workers N \
    --validation-report examples/benchmark/validation/v1/spec.json
```

> **Placeholder, full run:** the command as run, its wall time and the output's size.

Other options: `--conditions` takes `all` or condition ids separated by commas. A
crossed cell's id holds a comma itself (`ripple_snr=low,participation=low`), so the
list is read by taking the longest known id at each position: that text selects the
crossed cell, never its two one-factor conditions, and an unknown id stops the command
(`conditions.select_conditions`, which `validate_simulator.py` shares). `--replicates N`
gives every selected condition `N` replicates, `--duration S` sets the session length.

### What a run writes

Everything goes to `examples/benchmark/output/<run_name>/`, which git ignores. The
column lists of every table are in [run.py](run.py)'s module docstring:

- `manifest.json`, `run_spec.json` and `conditions.csv`, written when the run starts;
- `conditions/<condition_id>/`, one directory per condition: sessions, the truth (the
  latent event and non-event tables, restored by `load_truth`), the units active in each
  truth window, the ripple channels' gains and delays, the units, the methods run with
  their resolved options and input policies, each detected event with its active units,
  every result complete under `results/` (read back exactly by `load_results`), the
  scores, the failures and the warnings;
- `combined/`, the finished conditions' tables concatenated, which the analyses read,
  and its `manifest.json`: the run's conditions it includes and those it leaves out.

A condition is written into `conditions/<condition_id>.partial/`, its `done.json` (the
row count and SHA-256 of every other file) last, and then renamed into place, so a
condition directory is either complete or absent, and finishing one never touches
another.

### Resuming and combining

After an interruption, run the same command again with `--resume`. The runner rebuilds
the run specification (every condition's parameters, the replicates and seeds, every
method and setting with its resolved options, the package version, the git commit and
the report) and stops, naming the keys that differ, if it is not the saved one: a run is
never continued under other settings. It keeps each condition whose `done.json` matches
its files, deletes each `.partial` directory and each condition that fails that check,
printing each one it deletes and why, and runs those again.

Resume accepts only committed, clean code: the commit the run started from, with no
change under `src/` or `examples/benchmark/`. The run specification records the commit
as `<hash>-dirty` when there is one (untracked files included); a dirty run can start,
but not resume, since the flag does not identify the changes, and outside a git
checkout (commit `unknown`) no run starts or resumes. Any new commit is a new
specification, so a run cannot be resumed across one: start a new run instead.

`combined/` is rebuilt at the end of every run, and at any time by

```bash
uv run python examples/benchmark/run.py --run-name v1 --combine
```

which needs no report; `--resume` never reads it. It combines the conditions that have
finished, prints those of the run's `conditions.csv` it leaves out, and lists both in
`combined/manifest.json` (`included`, `missing`); it is built in `combined.partial/` and
renamed into place, so `combined/` is never half written.

### After changing the simulation

A change to the simulator's code or its signal helpers, to `simulator_targets.csv`, or
to a condition's parameters (including a new `--duration`) leaves the report behind, and
the runner refuses it. Make a new report for the new settings, with the same
`--conditions` and `--duration` the run will use, then start a new run with it:

```bash
uv run python examples/benchmark/validate_simulator.py --validation-id v2 --conditions all
uv run python examples/benchmark/run.py --run-name v2 --conditions all --workers N \
    --validation-report examples/benchmark/validation/v2/spec.json
```

A change confined to the detectors or the literature methods leaves the report valid,
but the git commit in `run_spec.json` changes, so it is a new run, not a resumed one.
