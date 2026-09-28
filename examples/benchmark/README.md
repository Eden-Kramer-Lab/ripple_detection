# Detector benchmark

Code and tables for a benchmark, in progress, of the package's detectors and the
packaged literature methods on simulated sessions whose events are known. None of it is
part of the installed package. It holds `simulator_targets.csv`, the measurements the
network simulator is validated against, the simulation conditions and the configurations
of the literature methods below, the runner that simulates the conditions, runs every
method and scores it (see Running the benchmark), the analyses of a run (see Analysing
a run) and the attribution of the methods' differences to the components of their rules
(see Attribution).

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
`simulate_condition` does. It refuses a set that lacks a keyword, or an entry of a
mapping such as `non_events.rates`, that the simulator would fill in itself; only
`events.type_probabilities`, which a condition replaces whole, may leave event types
out.

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
method that raises is written to `failures.csv`, with no events, results or scores, and
the run goes on. An error in the benchmark's own code (building a recipe's recording or
eligible epochs, summarizing or scoring a result) stops the run instead: it is a bug to
fix, not a method's failure. A (session, method, setting) without scores is a failure,
never zero events: a call that finds nothing still has its scores and an entry in its
`results/` sidecar.

Run it from the repository root, in this order. First validate the simulator for the
settings the run will use:

```bash
uv run python examples/benchmark/validate_simulator.py --validation-id v1 --conditions all
```

It simulates 20 validation replicates (10000-10019) of each selected condition and
measures every session, calling no detector: about 10-11 s and 1.6 GB of memory per
600 s session, so the 43 conditions' 860 sessions take about 2.5 CPU hours. `--workers
N` measures N sessions at once, each worker needing its own 1.6 GB. `--duration` must be
at least 120 s: the noise-modulation check needs two of its 60 s periods, and a check
that measures nothing fails. The report is `ready` or `not_ready`, and its `report.md`
says why; until it is ready it blocks every run. The repository keeps its
`spec.json`, `checks.csv`, `report.md` and figures but not `measurements.csv` (tens
of MB, git-ignored): the same command regenerates it, and a copy that is present
must match the hash `spec.json` records.

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
run `min(requested, free cores - 1, 0.7 x available memory / peak memory)` workers. The
smoke test reports each rule's verdict; the runner has no switch for the events rule, so
acting on it needs a change to `run.py`.

Measured on 2026-09-27 (commit 8582977, report `v1`; an 18-core, 64 GB macOS machine
shared with other work, load about 8): one reference session of 600 s simulates in 0.9 s
and runs and scores its 160 method calls (83 detector settings, 77 recipes) in 70.8 s,
74 s wall in all, with a peak resident memory of 3.56 GiB; no call failed or warned. It
writes 24,171 events (318 kB), 1,920 metrics rows (178 kB), 160 methods rows, 329 truth
rows and 2.2 MB of complete results. Extrapolated to the 440 sessions: 8.8 CPU hours and
1.15 GiB written (0.13 GiB of events); the 860-session validation adds 2.8 CPU hours
(11.7 s and at most 2.04 GiB a session). Rules: 72 s is under 300 s, so `duration_s`
stays 600; events stay under 2 GB, so every condition's events are written. Memory, not
cores, sets the workers: at 3.56 GiB a session, `0.7 x available / peak` allows 12
workers with 64 GB free and 4 with the 23 GB that were free that day. Sessions of the
`n_units=120` and `n_channels=16` conditions need more of both.

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

The full run, 2026-09-28, commit 3d1fa33, report `v1`, on the same shared machine
(load 5-15 from other work):

```bash
uv run python examples/benchmark/run.py --run-name v1 --conditions all --workers 5 \
    --validation-report examples/benchmark/validation/v1/spec.json
```

It took 2 h 23 min wall (8.9 CPU hours of user time; a session's detection 95 s on
average, 176 s at most), peaked at 4.98 GB in one process, and wrote 2.3 GB, 1.2 GB of
it `combined/` (10.7 million events, 144 MB of `events.csv.gz`). 135 of the 70,400
method calls failed, each with its reason in `failures.csv`: the 130 calls on the
single-channel condition (`n_channels=1`) of Shvartsman's detector at every setting and
of the recipes that need two or three channels, and Yu's detector at five sweep points
under brown noise, whose estimated threshold did not lie above the immobility mean. No
call warned. A first run with the same command, before commit 3d1fa33, ran only 41 of
the 43 conditions; the 41 it ran are byte-identical to this run's.

Other options: `--conditions` takes `all` or condition ids separated by commas. A
crossed cell's id holds a comma itself (`ripple_snr=low,participation=low`), so the
list is read by taking the longest known id at each position: that text selects the
crossed cell, never its two one-factor conditions, and an unknown id stops the command
(`conditions.select_conditions`, which `validate_simulator.py` shares). `--replicates N`
gives every selected condition `N` replicates, `--duration S` sets the session length.

### What a run writes

Everything goes to `examples/benchmark/output/<run_name>/`, which git ignores. The
column lists of every table are in [run.py](run.py)'s module docstring. Read a table
with `run.read_table`: it keeps text columns as text, where `pandas.read_csv` would make
a `setting` of `"3.0"` or a `level` of `"30"` a number and an empty `doi` a NaN.

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

## Analysing a run

`analyze.py` turns a finished run's `combined/` into small tables and figures:

```bash
uv run python examples/benchmark/analyze.py --run-name v1 --workers N
```

Most analyses read the reference condition's sessions and the main methods, each
detector at its defaults and every recipe (`setting` `default` or `literature`), and
match every session again, one process per session, since the runner keeps events, not
pairs. The operating curves, points and differences, the held-out thresholds and the
appendix curves also read the detectors' sweeps, in the reference condition alone.
Robustness (`robustness_*`, `robustness_crossed_*`) and model sensitivity
(`model_sensitivity`, `model_sensitivity_orders`) read every condition, the latter each
detector's sweep in the reference and the six alternative models too, and resample
replicates, not sessions, so that the conditions stay paired.

It rebuilds `examples/benchmark/results/<run_name>/`: a CSV per analysis below, a PNG
for each that has a figure (all but `failures`, `operating_differences`,
`appendix_expressions`, the `appendix_curves_*` and `model_sensitivity_orders`; none
for an empty table), `candidate_trends.csv`, and `summary.md`, which names each file with
one sentence on what it shows, lists every method that failed on some session and the
lists the analyses call for (below). What is written there by hand, `trends.md` (the
trends stated, each with what its spot check showed) and the figures in `spot_checks/`,
is carried over into the rebuilt directory, and `summary.md` links `trends.md`. No file
may pass 1 MB: the command stops before writing one and leaves the previous results in
place. `--run-directory` and `--results-directory` read and write elsewhere.

Two scoring rules. An interval method is matched one to one to the truth windows of its
primary expression at 10 % of the peak, any overlap counting (IoU 0). The two ripple
inventories whose catalog entry (`list_methods()`) gives `output` `"ripple peaks"`,
`davidson_2009_ripples` and `wu_2014_ripples`, return one time point per event, which no
interval rule can credit: a point matches a window that contains it (closed bounds, to
the timestamps' rounding), one to one, the most pairs (`match_peaks`). They get recall,
precision and false positives per minute only: in `point_inventories`, in
`appendix_expressions` against every expression, and in the rows marked
`peak_containment` of `rates_by_state`, `robustness_*` and `model_sensitivity` (and of
`summary.md`'s list of recall changes), never pooled with interval scores. Every table of
bounds, overlap, timing, agreement, consensus, splits and merges, participation, sweeps
or minimum IoU leaves them out. A method whose intervals happen to be one sample long,
such as `lee_2002`'s, keeps the interval rule: the catalog decides, not the events.

- `failures`: each method's sessions with scores and without, and its scoring rule. A
  session without a method's scores is a failure, never zero events. Every per-method
  table has a row for every method, one that never ran included (its counts 0, its
  values missing), with `n_sessions` (the sessions its numbers pool) and `n_failures`;
  a table of pairs `n_failures_a` and `n_failures_b`; a table across conditions
  `n_replicates`, `n_dropped` (replicates left out because the method failed in one of
  the conditions compared) and `n_failures`; the curves and points `n_sessions`,
  `n_dropped` (sessions left out because some setting of the sweep failed there) and
  `n_failures`. The dendrogram and consensus count the methods compared and the calls
  that failed.
- `point_inventories`: the point methods' recall, precision and false positives per
  minute.
- `detection_profile`: recall per event type against the network truth.
- `false_positive_classes`: what each method's false positives (its events matching no
  window of its primary expression) overlap longest: an event type's component
  (`swr:ripple`, `burst_only:burst`, ...), a non-event (`emg`, ...) or nothing
  (`background`).
- `pairwise_agreement` and `agreement_dendrogram`: Jaccard indices of every pair of
  methods against the network truth, the one every method is scored on (of all their
  events, of those matching a true event, of those matching none, and of the true
  events found), and the methods clustered by average linkage on `1 - jaccard`.
- `consensus`: how many methods found each true event, by type, and how many methods
  each group of overlapping false positives spans.
- `overlap_quality`: IoU, coverage and temporal precision of the matched pairs.
- `boundary_errors`: signed and absolute onset and offset errors against the truth at
  10, 25 and 50 % of the peak, each median beside its pair count and the method's
  recall, since a method that finds only easy events can time them better; a method
  whose primary expression is the network event is also timed against the ripple and
  the burst windows.
- `paired_timing_<expression>`: for two methods with the same primary expression, their
  error differences on the true events both found, which removes the difference in
  which events each found.
- `method_differences` and `error_correlations`: how every pair of methods' matched
  events differ in start and end, and whether their errors on the same true events
  are correlated: against their shared primary expression's truth for two methods that
  share one, as paired timing, else the network truth, whose correlations are beside
  every pair's.
- `splits_and_merges`: how often a method splits a true event or merges several,
  overall and on ripple doublets.
- `operating_curves`, `operating_points` and `held_out_thresholds`: recall against false
  positives per minute along each detector's sweep, pooled over the sessions on which
  every setting ran, read off at 0.5, 1, 2 and 5 per minute, and a threshold per target
  chosen on the even replicates and judged on the odd ones.
- `operating_differences`: for each pair of detectors sharing a primary expression, the
  difference in recall at each target, paired by session, with its interval and
  sign-flip test.
- `robustness_<measure>` and `robustness_crossed_<measure>` (recall, precision, onset):
  each main setting along each factor and over the cells of the two crossed pairs.
- `rates_by_state`, `participation_bias` and `boundary_effect`: event rates at rest and
  while running, which true events each method finds by their recruited cells, and what
  its bounds do to the units counted active.
- `matching_sensitivity`: the main interval methods' scores against their primary
  expression at minimum IoU 0, 0.2 and 0.5: recall, precision and F1 with intervals;
  the IoU quartiles, median absolute errors and recall by event type without them; each
  detector's recall at the target rates from `operating_points`; and a rank among the
  methods of a primary expression that is descriptive, with no interval or test.
- `model_sensitivity` and `model_sensitivity_orders`: each result, and each order of
  detectors, under each of the simulator's six alternative models.
- `appendix_expressions`: every main method against every expression (network, ripple,
  sharp wave, burst), not only its primary one: recall, precision, false positives per
  minute, median IoU and median signed and absolute errors at 10 %, with intervals.
- `appendix_curves_<expression>`: every interval method's curve against one expression,
  one file per expression.
- `candidate_trends`: patterns in the tables worth a spot check, not conclusions.

Signs: an error is detected minus truth (negative: early); a difference between two
methods is A minus B, A the method named first; a change between conditions is the
other condition minus the reference. Every interval is a 95 % paired bootstrap: over
sessions within a condition, every session drawn for all methods at once, and over
replicates across conditions, a replicate's sessions sharing its seed in every
condition. A difference between methods is summarized per session, its estimate the
mean of those values and its two-sided sign-flip p-value over them where both exist. A
change between conditions is the value pooled over the replicates minus the reference's
pooled value, over the replicates on which the method ran in both, and its p-value is
over each replicate's own change.

What each column means is in the docstring of the function that builds the table: the
function of the table's name, except `failures` (`failure_counts`),
`paired_timing_<expression>` (`paired_timing`), `robustness_<measure>` and
`robustness_crossed_<measure>` (`robustness` and `robustness_crossed`, whose columns
after the factor and levels are `paired_changes`' `CHANGE_COLUMNS`), the orders of
`model_sensitivity` (`ORDER_COLUMNS`), `operating_differences` (`DIFFERENCE_COLUMNS`),
`appendix_curves_<expression>` (`expression_curves`) and `candidate_trends`
(`TREND_COLUMNS`).

On run `v1` (the shared 18-core machine), with `--workers 6`, the command took 3.1
minutes wall and at most 3.9 GB resident: loading the reference 10 s, matching it again
at the three minimum IoUs 33 s, every condition's scores 21 s, the tables 1.8 minutes
(`robustness` 27 s, `appendix_expressions` 18 s, `model_sensitivity` 16 s,
`boundary_errors` 12 s, `robustness_crossed` 10 s, the others under 7 s each) and the
figures 11 s. It wrote 66 files (37 CSV, 28 PNG and `summary.md`), the largest 780 kB.

## Results

What each file of `results/<run_name>/` shows, and how to read it. Every number comes
from simulated sessions whose truth is known because the simulator drew it: the event
types, the non-events and their mixture are this benchmark's taxonomy
([conditions](#simulation-conditions)), not categories every recording shares, and the
events carry no sequence content, so nothing here says how well a method finds replay.
A result holds for the reference simulator unless a file compares conditions, and for
the six alternative models only one at a time.

`summary.md` lists the files, the failures, the methods whose recall moves by more than
0.1 across a factor, the rank changes with the minimum IoU and what survives each
alternative model, and links `trends.md`, where each stated trend is written by hand
with the spot check behind it.

### Detection and false positives

`detection_profile.png` is a heatmap of recall, method by event type, against the
network truth (every component of an event joined), so a ripple detector can score on a
`burst_only` event only by overlapping its burst. `false_positive_classes.png` stacks,
per method, what its unmatched events overlap longest. Neither says why a method misses
or fires: the spot checks show single events. A false positive over a non-event is a
false positive by this benchmark's taxonomy (fast gamma, EMG, leaked spikes and theta
bursts are declared non-events), not a claim about what a recording's event was.

### Agreement, consensus and overlap

`pairwise_agreement.png` shows four Jaccard indices, methods in the dendrogram's leaf
order (`agreement_dendrogram.png`, average linkage on `1 - jaccard`), all against the
network truth: two methods can agree on events that are false. `consensus.png` shows
how many methods found each true event, by type, and how many methods each group of
overlapping false positives spans. `overlap_quality.png` draws the IoU, coverage (the
fraction of the true event found) and temporal precision (the fraction of the event
that is true) of the matched pairs, 5-95 % whiskers and the interquartile box. They
cannot show agreement on events no method found.

### Boundaries and timing

`boundary_errors.png` draws the signed and absolute onset and offset errors at 10 % of
the peak as boxes and the medians at 25 and 50 % as markers, in milliseconds, detected
minus truth: a negative onset starts early, a negative offset ends early. Read each
median beside its pair count and the method's recall in the CSV: errors are measured on
the events a method found. `paired_timing_<expression>.png` shows, per pair of methods
sharing a primary expression, the mean over sessions of the median paired difference on
the true events both found, A (row) minus B (column); negative on the absolute panels
means A is closer to the truth. `method_differences.png` is the same for the methods'
own bounds against each other, `error_correlations.png` whether their errors move
together (against a shared primary expression's truth, else the network truth; the CSV
has the network truth's correlations beside every pair's). The truth windows are cut at
fractions of a latent envelope; no measured recording has such a boundary, so these
compare methods with one convention, not with an observable onset.

### Operating curves and thresholds

`operating_curves.png` plots recall against false positives per minute (log scale; a
rate of 0 drawn at the resolution, half of one false positive over the minutes counted)
along each detector's sweep, its default an open circle and each recipe a grey point on
the panel of its primary expression. False positives are counted over the minutes
outside every network window, so a rate is per minute of time without events.
`operating_points.png` gives recall at 0.5, 1, 2 and 5 per minute: a target a curve does
not reach is missing, never its nearest end, and `attained` in the CSV says in what
share of resamples it is reached. `operating_differences.csv` has, for two detectors of
one primary expression, the difference in recall at each target, paired by session,
with its interval and test: two recalls whose intervals overlap can still differ, and
two whose estimates differ may not. A sweep is pooled over the sessions on which every
setting ran, so a failed call cannot bend one point of a curve. The curves describe the
sessions they are read from: a threshold quoted from them is chosen on the even
replicates and judged on the odd ones (`held_out_thresholds.png`, held-out recall beside
the calibration recall). A recipe's point is one configured interpretation, not the
paper's own tuning. `appendix_curves_<expression>.csv` gives every curve against each
expression, not only the primary one, and `appendix_expressions.csv` every main method's
scores against each.

### Robustness across conditions

`robustness_<measure>.png` has a panel per factor, the measure against the factor's
levels with the reference level in place, one line per main setting coloured by
primary expression. `robustness_crossed_<measure>.png` shows, per crossed pair, the
change from the reference in every cell, a row per method. Changes are the level minus
the reference, over the replicates every level holds on which the method ran in every
level (`n_dropped` counts those a failure left out), with paired intervals and sign-flip
tests. One factor moves at a time (except the two crossed pairs), so the panels do not
show how factors combine.

`noise_type=brown` is confounded and supports no statement about detectors. Ripples are
sized against the ripple-band noise, which brown noise makes about 25 times smaller,
while EMG and spike-leakage artifacts keep their absolute amplitudes, so under brown
noise they are about ten times the ripples and the LFP detectors' events land on them
(`results/v1/trends.md`). Sizing those artifacts against the noise, as the gamma bursts
already are, is a simulator change for a later version with its own validation and run.

### Rates and participation

`rates_by_state.png` gives each method's events per minute at rest and while running
(an event placed by its peak time, else its midpoint), the true rates marked: network
events at rest and theta bursts, the running state's non-events. `participation_bias.png`
shows the ratio of the mean number of recruited cells (latent, some of which never fire)
of the true events a method finds to that of all true events with a burst: above 1, the
method favours events with more recruited cells. `boundary_effect.png` shows the units
active within the detected bounds minus those within the matched truth window, all units
and principal ones: zero when the bounds agree, since silent recruits, interneurons and
background spikes count on both sides. None of these says how many cells participate
in a recorded event.

### Matching sensitivity

`matching_sensitivity.png` shows recall and precision when a pair must overlap by IoU
0.2 and 0.5 instead of any overlap. The CSV keeps the IoU quartiles beside every
headline at 0, the median absolute errors, recall by event type and each method's rank
among those of its primary expression, whose changes `summary.md` lists. The ranks
and the per-type recalls have no interval: a rank change says the order moved, not by
how much or how surely.

### Model sensitivity

`model_sensitivity.png` has a panel per alternative model, each main setting's change in
recall (the alternative minus the reference, paired by replicate) and each detector's
change at 1 false positive per minute as a triangle. `model_sensitivity_orders.csv`
orders detectors of one primary expression by recall at each common target in both
conditions, with the share of resamples in which the order reverses; `summary.md` says
per alternative which supported reference orders survive, lose their support or reverse,
and how many cannot be compared because a target is out of reach there or a detector
failed (`status` `unattainable` or `failed`, never confused), beside the validation
report's target statistics that the alternative moves (or that the report could not be
read, which is not the same as nothing moving). The
alternatives are not pooled into an overall winner, and one at a time they do not test
combinations of assumptions.

### Spot checks

A trend goes into `summary.md` or this README only after its underlying events have
been looked at. `candidate_trends.csv` lists candidates from the tables, each with its
evidence and where to look; `select_events` picks the events (truth windows a method
missed or found, or its false positives, of one condition) and `spot_check` draws six of
them, the sessions simulated again from the run's parameters and seeds, with the
signals, spikes, truth windows and the methods' events (a point event as a diamond, a
method that failed on the session labelled so), into `spot_checks/`, which a rebuild
keeps. Each candidate names the condition, one or two methods each with the setting to
draw (at a false-positive target, the swept setting whose rate is nearest it) and
which events:

```python
import analyze  # with examples/benchmark on sys.path

run = "examples/benchmark/output/v1"
events = analyze.select_events(
    run, "reference", "Kay_ripple_detector", "default", "missed", event_type="weak_ripple"
)
analyze.spot_check(
    run,
    "examples/benchmark/results/v1",
    "kay_missed_weak",
    events,
    [("Kay_ripple_detector", "default"), ("Karlsson_ripple_detector", "default")],
)
```

## Attribution

Recipes differ in many components at once. `attribution.py` measures how much each
component matters: one factor at a time from a reference configuration, Sobol indices
over the space the literature spans, and Shapley decompositions of the difference between
two configurations. It reads a finished run's reference condition:

```bash
uv run python examples/benchmark/attribution.py --run-name v1 --family spikes --smoke
uv run python examples/benchmark/attribution.py --run-name v1 --family spikes \
    --analysis all --workers N
```

`--analysis` is `oat`, `sobol`, `shapley` or `all` (the default); `--smoke` times 20
configurations on one session and prints each analysis's cost, writing nothing;
`--run-directory` and `--results-directory` read and write elsewhere. Before anything
runs, the command checks the run's validation report (ready, and the one the run
recorded), simulates the reference condition's first five sessions again from the saved
parameters in `conditions.csv` (a run with a halved `duration_s` gives halved sessions)
and stops unless each has the run's seed, duration, latent events, non-events and ripple
channels.

### Templates and fixed points

A template is one flat set of factor values (`SpikeTemplate`, `LfpTemplate`), compiled
into a pipeline of the package's public primitives: a threshold core
(`detect_events_from_trace`, on `population_trace`'s 1 ms bins for spikes) and post steps
(`require_active_units`, `require_inside`, `require_overlap`, `require_times_inside`).
`TEMPLATES` maps a configuration to the template written after reading its method's body
in `literature_methods`, with the source of every value; `FIXED_POINTS` gives every
other configuration and why it has no template (a whole detector such as Karlsson's or
Kay's, silence-bounded windows, custom peak merging, finite kernels, other grids, FFT or
wavelet power, adaptive thresholds, active-fraction rules, a minimum time above
threshold). A template stands for its method only when its events equal the public
call's, bound for bound, on a session with a gap of missing samples, one at a Unix clock
origin and the five reference sessions, with some event found; `in_space.csv` lists every
configuration and the outcome. The public call stays the only source of a method's
events everywhere else: `fixed_points.csv` gives each fixed point's `Y`s (below) from
its public call, so no method silently drops out.

The families, analysed separately because their factors differ:

- `spikes`: a population rate (the units pooled, `all`, `place` or `pyramidal`, and its
  Gaussian smoothing), the normalization period, threshold, bound, whole-event minimum
  and maximum duration, merge gap, speed rule, active-unit count, state (a restriction of
  the trace to rest, containment in rest, or overlap with running) and coincidence (a
  partner event: a Long SWR, a Muessig ripple window or an external ripple peak).
- `lfp`: the mean ripple-band envelope of the first channels (band, channel count, the
  envelope or its square) and the same rules, without participation.

A factor's range is the least to the largest value among the family's templates
(continuous), its distinct values (categorical) or every whole number between them
(integer); a factor with one value is not varied. The reference configuration takes each
factor's median (integers rounded down) or mode (ties to the first in configuration
order):

| Factor | `spikes` reference | `lfp` reference |
| --- | --- | --- |
| signal | pyramidal cells, 15 ms Gaussian | 150-250 Hz, every channel, envelope |
| smoothing (LFP) | | 12.5 ms |
| normalization period | the whole session | speed below 5 cm/s |
| threshold, bound | 3 SD, the mean | 3 SD, the mean |
| minimum, maximum duration | 50 ms, none | 25 ms, 2 s |
| merge gap | none | none |
| speed rule | none | speed at most 5 cm/s at both ends |
| active units, state, coincidence | none | none |

`<family>_factor_space.csv` and `<family>_reference.csv` hold both.

Represented on run `v1`: **TBD (measured when the analyses run): the in-space count per
family and its methods**. A family with fewer than eight represented methods runs no
Sobol or Shapley analysis; the command refuses it and says so.

### Outputs and how to read them

Every configuration is scored on the five reference sessions, each `Y` the mean over them
(`Y_NAMES`): `f1` against the family's expression (burst truth for `spikes`, ripple truth
for `lfp`), `f1_network`, `events_per_minute`, `onset_error_25` (the median signed onset
error of the matched pairs against the windows at 25 % of the peak) and
`jaccard_reference` (events matched to the reference configuration's over their union).

- `<family>_oat.csv`: each factor at each of its values (five evenly spaced over a
  continuous range), every other at the reference; `change` is the value minus the
  reference's, with a 95 % paired bootstrap interval over the five sessions.
- `<family>_sobol.csv` and `.png`: first-order and total Sobol indices of each `Y`, with
  95 % bootstrap intervals over the sample rows. A first-order index is the share of the
  output's variance a factor explains alone; the total index adds every interaction it
  takes part in, so total much above first-order means the factor matters through other
  factors' settings.
- `<family>_shapley.csv` and one waterfall per pair: for two configurations `a` and `b`,
  how much of the change from `a` to `b` each differing factor carries, in `Y`, with the
  Jaccard against `b` (from `J(a, b)` to 1) and with `f1`. The values sum to the whole
  change; with more than eight differing factors they are Monte Carlo estimates, each
  beside its standard error. The pairs: each represented configuration against the
  family's reference, and the ten pairs of represented configurations that agree least.
- `output/<run_name>/attribution/<family>_<analysis>.csv.gz`: one row per configuration
  and `Y`, its factors, the mean and each session's value.

Limits. The Sobol indices assume independent factors over the sampled box, while real
recipes co-vary (a lower threshold usually comes with other changes), so an index says
what a factor does across the space, not what the literature's choices did. Fixed points
are not decomposed. The conclusions hold for the reference simulator only. A sampled
bound above a sampled threshold is not a configuration `detect_events_from_trace` runs.

Measured: **TBD: the smoke test's seconds per configuration and memory, the chosen
Sobol `N`, and each analysis's command and wall time on run `v1`**.
