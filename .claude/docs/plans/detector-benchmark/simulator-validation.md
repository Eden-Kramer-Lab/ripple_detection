# Simulator validation and sensitivity conditions

[← back to PLAN.md](PLAN.md) · [designs](designs.md) · [phase 4](phase-4-runner.md)

The simulator supports controlled comparisons under stated assumptions. Before benchmark
detector runs, validate the **rendered measurements** against published summaries and test six
alternative assumptions. All recordings remain fully synthetic. Phase 7 checks implementation
parity and does not establish the simulator's biological validity.

## Evidence and measurement conventions

Phase 1a commits `examples/benchmark/simulator_targets.csv` before detector results are examined.
Each row records `quantity`, `population`, `state`, `conditions`, `measurement`, `source_doi`, `source_location`,
`evidence_status` (`supported` or `assumed`), `target_statistic`, `lower`, `upper`, and `rationale`.
Specify the species, recording layer and behavioral state represented by the reference. Sources
from different states or populations cannot silently supply a single reference distribution.
Resolve the existing "verify" entries; record remaining assumptions explicitly.

At minimum, establish targets for event frequency and duration, event rate during rest, baseline
unit rates, observed principal-cell participation, and the relationship of sharp-wave magnitude
to ripple magnitude/frequency. Bounds and acceptable sampling error must be recorded before the
validation simulations are examined. An unsupported quantity stays labeled assumed; it cannot
be cited as evidence that the reference reproduces physiology. Missing measurement definitions
or unresolved required source checks block the validation report.

Use the source's measurement convention. The plan's nominal duration is `3 * (rise_sigma +
decay_sigma)`: at envelope power 2, its half-maximum duration is only about 0.392 times that
value (90 ms nominal gives about 35 ms at half maximum). Report nominal, half-maximum and
10%-of-peak widths separately. Latent recruitment probability and observed probability of firing
at least one spike are also separate measurements; the latter depends on rate, gain and duration.

Two checked sources motivate sensitivity conditions, without establishing their numerical knobs:

- [Sullivan et al. (2011), abstract and Figures 1–2](https://pubmed.ncbi.nlm.nih.gov/21653864/)
  reports fast gamma at 90–140 Hz and relationships between sharp-wave magnitude and fast
  oscillation magnitude/frequency. It motivates the nearby-gamma and coupled-strength conditions.
- [Patel et al. (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3807028/) reports local ripple
  generation and variable spread. It motivates varying channel occupancy and timing.

The effect sizes below are assumed stress settings, fixed before detector runs. They are not
estimates from these papers. Any reference-parameter revision made during simulator validation
is documented with its physiological or mathematical reason and followed by a new report.

## Six sensitivity conditions

These extend the existing grid one at a time; they do not add a full factorial experiment.

| Factor | Reference | Alternative and resolved overrides |
| --- | --- | --- |
| `strength_correlation` | 0 | `coupled`: `events.strength_correlation=0.6` |
| `spatial_profile` | all channels, fixed gains, no delay | `local`: `render.spatial_profile="local"`, `render.channel_occupancy=0.5`, `render.channel_gain_range=(0.5, 1.0)`, `render.channel_delay=0.002` s |
| `noise_modulation` | stationary | `varying`: `render.noise_log_amplitude=0.35`, `render.noise_modulation_period=60.0` s |
| `fast_gamma_band` | frequency and sizing band (60, 100) Hz | `nearby`: `non_events.fast_gamma_frequency=(90, 140)`, `non_events.fast_gamma_band=(90, 140)` Hz |
| `spike_model` | `"poisson"` | `refractory`: `render.spike_model="refractory"`, `render.refractory_period=0.002` s |
| `envelope_power` | 2 | `quartic`: `events.envelope_power=4` |

### Coupled event strengths

Add `strength_correlation: float = 0.0` to `draw_network_events`, in [0, 1]. For each
candidate network event draw a shared standard normal `z`; for each applicable scalar draw an
independent standard normal `epsilon`. Set `u = scipy.special.ndtr(sqrt(rho) * z +
sqrt(1-rho) * epsilon)` and map `u` into the existing uniform range for ripple SNR, onset
frequency, sharp-wave amplitude and burst participation. Other parameters retain their current
draws. Doublet components share `z`, with independent residuals. Draw these variates even at
rho=0 so changing rho does not shift later random draws.

This preserves candidate marginal distributions while varying their dependence. `rho` is the
latent-normal correlation, not a claimed measured correlation. Measure correlations after
rendering, stratified by event type; event rejection can change realized distributions. The
reference remains independent, so report its lack of physiological coupling explicitly.

### Spatial ripple structure

Add the four renderer options in the table; `spatial_profile="global"` retains the existing
rendering. In `"local"`, each ripple component selects `max(1, ceil(channel_occupancy *
n_channels))` channels without replacement. Choose one selected anchor channel uniformly; its
event gain is 1 and delay 0. Other selected channels draw gains uniformly from
`channel_gain_range` and delays uniformly from `[-channel_delay, channel_delay]`; unselected
channels have gain 0. Multiply these gains by the existing recording-wide `channel_gains`.
Apply delays to the entire waveform, evaluating it on the native time grid.

Draw delays from the allowed interval intersected with shifts that keep that component's
±4-scale span in its original rest interval. Zero is always admissible. Store **every** channel's
gain and delay, including zero-gain channels, in `SimulatedSession.ripple_channels`; its schema
is in the shared contracts. Ripple and network truth remain anchored to the latent component's
center; channel-local bounds are those bounds plus the stored delay. Do not redefine truth
after observing detections. Radiatum leakage keeps the latent waveform and existing leak gain.
Report anchor-channel and recording-wide SNR separately; channel 0 need not contain the ripple.

### Changing background variance

Generate the stationary correlated noise first. Multiply every noise channel by
`g(t) = exp(a * sin(2*pi*(t-time[0])/T + phase))`, normalized to unit RMS over the recording;
`phase` is uniform in [0, 2*pi). Here `a=noise_log_amplitude >= 0`, `T=noise_modulation_period > 0`.
The default a=0 gives g=1. Signal amplitudes are sized against the **stationary** reference noise
SD before this modulation; do not rescale each event to its local noise. Thus local SNR changes.
Slow fields and explicit artifacts are added afterwards. Report background band power in 10 s
windows and event-local SNR, as well as the nominal SNR in the truth table.

### Nearby gamma

Add `fast_gamma_band=(60.0, 100.0)` to `draw_non_events`; validate positive ordered edges below
Nyquist and containing the frequency range. Persist its edges as `snr_band_low` and
`snr_band_high` on gamma rows. Both the background SD and burst scaling use this stored band.
This avoids boosting 90–140 Hz bursts according to their residue in the old 60–100 Hz band.
Other non-events have NaN band edges. Keep the reference gamma case and the nearby case in
separate results. Calling these bursts non-events is the benchmark's explicit taxonomy, not a
universal physiological classification by frequency alone.

### Refractory spiking

Keep the same units, recruited participants and intensity function `lambda(t)` in both models.
The alternative permits at most one spike per sample: when the elapsed time from the previous
spike is at least `refractory_period`, emit with probability `1-exp(-lambda(t)*dt)`; otherwise
emit none. Record that these are proposal intensities: dead time reduces realized rates, so do
not claim equal realized rates between models. Validate this reduction and report it alongside
count variability, active-unit counts and population silent-gap distributions. Refractory
period must be finite and non-negative. Explicit leakage spikes remain externally injected;
the hard refractory guarantee applies to endogenous spikes, tested with leakage disabled.

### Alternative envelopes and truth

Store `envelope_power` on every event and non-event row; network events accept powers 2 and 4,
while non-events retain 2. For distance u from the center on the relevant side, use
`E(u) = exp(-log(2) * (abs(u) / (sqrt(2*log(2)) * sigma))**p)`.
Power 2 recovers the current Gaussian exactly; power 4 has a flatter center and steeper edges,
with the same half-maximum width for the same side scales. Apply it to ripple, sharp-wave and
burst envelopes, including interneuron modulation. Retain nominal ±3-scale duration, ±4-scale
containment and ±8-scale rendering conventions. For p=4 the scale is not a Gaussian SD.

The analytic threshold distance is `k_p(f) * sigma`, where
`k_p(f) = sqrt(2*log(2)) * (log(1/f)/log(2))**(1/p)`. Update `truth_windows` and every consumer
to use the row's power. Test crossings against the explicit modulation envelope within one
sample. A sampled carrier's Hilbert envelope can differ from the modulation, especially for
short or sharply bounded events; measure and report that discrepancy rather than asserting
universal one-sample agreement. Truth at any fraction retains the same component row order.

## Validation report and execution order

Phase 4 adds `examples/benchmark/validate_simulator.py`, using `conditions.py` and the public
simulator only. It does not import detector/recipe APIs or execute detection. Its CLI accepts the same condition
selection and duration overrides as `run.py`; by default it validates all 43 conditions on five
replicates (10000–10004), separate from benchmark replicates. Process sessions one at a time and
retain summaries, not full arrays. Use matched noise-only renders and isolated components when
needed; fixed renderer random substreams keep the underlying noise and unit draws comparable.

Write `examples/benchmark/validation/<validation_id>/` (committed small artifacts):

- `spec.json`: resolved simulation parameters, seeds, versions, simulator source fingerprint,
  target-table hash, status and file hashes. Fingerprint the simulation code and its signal
  helpers; changes confined to detectors do not invalidate this report. Hash every other report
  artifact, excluding `spec.json` itself; the run records a separate hash of `spec.json`.
- `measurements.csv`: per condition/replicate/type measurements, including realized event rates
  and type proportions after rejection; the duration, frequency and SNR measures above; background
  PSD and windowed band power; channel occupancy, delays and coherence; per-type baseline and
  event firing rates, observed principal-cell participation, count variance/mean (fixed 10 ms
  bins), inter-spike intervals and population silent gaps; event-strength correlations.
- `checks.csv`: target, observed statistic, tolerance, evidence status and pass/fail for each
  applicable check. Source-based checks use the recorded measurement convention. Assumed stress
  cases test their specified change rather than requiring reference physiological ranges.
- `report.md` and small PNGs: source/measurement comparisons, distributions and joint plots,
  representative raw signals for each event/non-event type and model variant, limitations,
  and any parameter revisions. Plotting imports matplotlib only inside plotting functions.

The report is ready only when mathematical/rendering checks pass, required source checks are
resolved, and each source-based acceptance target passes for the conditions to which it applies.
Assumed or unsupported properties remain visible in the report; readiness does not certify
biological realism. Predeclare the independent reference as an assumed control for coupling;
the coupled condition must pass the directional source-based target. Record target applicability
by condition before simulation, and do not widen tolerances after inspecting results.

Benchmark smoke/full runs and phase-6 attribution require a ready report covering their resolved
simulation settings and matching source/target fingerprints. Record its path and hash in the
run specification. Duration or model changes require validation for the new settings; ordinary
short unit tests use explicit fixtures and do not establish scientific validation. Record the
validation runtime separately when estimating full-run cost. Simulator changes after detector
results require a documented correction and new validation plus affected benchmark runs.

Phase 5 reports each alternative alongside the reference, with paired differences and ranking
changes at common attainable false-positive rates, plus boundary and participation effects.
State which conclusions survive each alternative and which depend on the model. These six
one-factor checks do not establish robustness to their interactions or to all real recordings.
