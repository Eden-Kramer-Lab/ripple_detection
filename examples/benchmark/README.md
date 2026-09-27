# Detector benchmark

Code and tables for comparing the package's detectors and the packaged literature
methods on simulated sessions whose events are known. None of it is part of the
installed package. `simulator_targets.csv` holds the measurements the network
simulator is validated against.

## Literature method configurations

`recipe_configs.py` configures the methods of `ripple_detection.literature_methods`
for the benchmark. It does not implement any method: a configuration names a method
from `list_methods()`, the options it runs with and the expression of the simulated
network event it is headlined against, and `run_recipe` calls the method through the
package's `run_method`. What each method does, which inputs it needs and why, and what
remains unverified are in the package's
[implementation guide](../../docs/literature/implementation.md), the method docstrings
and the paper notes.

### Running a configuration

The benchmark's scripts import `recipe_configs` by name: run Python from this
directory, or put it on `sys.path`.

```python
import numpy as np
import ripple_detection as rd
from recipe_configs import RECIPES, behavior_intervals, make_recording, run_recipe

time = np.arange(90_000) / 1500.0
running = np.array([[25.0, 35.0]])
events = rd.draw_network_events(time, running_intervals=running, rng=0)
session = rd.simulate_network_session(time, events, running_intervals=running, rng=1)

config = next(config for config in RECIPES if config.config_id == "yang_2024")
recording = make_recording(session, config)
found = run_recipe(config, recording, behavior_intervals(session, config))
print(found.attrs["method"], found.attrs["role"], found.attrs["options"])
```

`found` is the package's result, unchanged: bounds, the method's own columns, and
`attrs` with its DOI, resolved options, grid, the inputs it ran on and diagnostics.
`check_recipe` lists what a call would lack without running it, and `method_record`
gives the configuration's row of the benchmark's `methods.csv`: the resolved options,
the input policy and the assumptions, as JSON.

### The inputs a configuration receives

`make_recording` builds the recording with `Recording.from_arrays`, the path measured
data take, so the package's simulation fallbacks never run. The recording holds
exactly the inputs the method declares in `list_methods()`'s `requirements`, taking
into account conditions on its options (`when`) and inputs that make another
unnecessary (`unless`); an input a method does not ask for could change its events,
so none is added. The policy (`INPUT_POLICY`) takes each input from the session:

- `lfps`, `sharp_wave_lfp`, `multiunit` and `speed`: the recorded signals, every
  pyramidal-layer channel with the ripple channel first.
- `place_cells`: the units the simulator labels `"place"`, and `pyramidal` those
  labelled `"place"` or `"pyramidal"`. These are the simulator's own labels, not a
  classification. The simulator has no place fields or trajectories, so every place
  unit stands in for a narrower selection (one template's, one directional template's,
  one probe sequence's cells), and `templates` is one template of every place unit.
- `sleep_intervals`, `baseline_intervals` and the call's `behavior_intervals`: rest,
  the recorded samples outside the running bouts. The sessions are awake, so rest
  stands in for a sleep state; position-defined epochs (reward zones, track ends,
  corners) have no simulated counterpart, and simulated events occur only at rest.
- `reference_lfp`: zeros; the simulated channels share no reference.
- `external_ripples` (Yang 2024, Grosmark 2016): the public `Zugaro_ripple_detector`
  on the first channel, the package's stated assumption for these papers' unspecified
  ripple detector. `example_ripples` (Carey 2019): the five largest
  `Kay_ripple_detector` events. Neither is taken from the simulation's truth.
- Options the package requires for measured data because the paper does not report
  them (Stella's wavelet, Nádasdy's RMS window and bounds, Kudrimoti's threshold,
  Wikenheiser's window anchor) take the package's demonstration values.

A session without unit labels gives empty selections, which `check_recipe` and
`run_recipe` report; nothing stands in for them. Each configuration's `assumptions`
list every stand-in and demonstration value it relies on, and each result's
`attrs["inputs"]` and `attrs["behavior_intervals"]` hold the values supplied.

### Stages, roles and expressions

Methods with a decoding-candidate stage run their detection stage
(`stage="detection"`, set explicitly): selecting replay candidates is outside the
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
