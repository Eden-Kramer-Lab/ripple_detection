# Designs

[← back to PLAN.md](PLAN.md)

Per-component algorithms, with code where the implementation is not obvious. Types and table
schemas are in [shared-contracts.md](shared-contracts.md); this file does not repeat them.

- [Parameter sources](#parameter-sources)
- [Simulator validation and sensitivity conditions](simulator-validation.md)
- [Event types](#event-types)
- [Drawing network events](#drawing-network-events)
- [Rendering a network session](#rendering-a-network-session)
- [Non-events](#non-events)
- [Truth windows](#truth-windows)
- [Matching](#matching)
- [Agreement statistics](#agreement-statistics)
- [Recipe executor](#recipe-executor)
- [Recipe classification](#recipe-classification)
- [Conditions grid](#conditions-grid)
- [Runner](#runner)
- [Operating curves](#operating-curves)
- [Bootstrap and permutation tests](#bootstrap-and-permutation-tests)
- [Attribution](#attribution)
- [Rates and participation](#rates-and-participation)

## Parameter sources

The reference value of every simulator parameter, fixed **before any detector is run on network
sessions** (overview risk 1). Ranges are `(low, high)` drawn uniformly per event, the package's
convention (`_draw_per_ripple`, `simulate.py:221`). "Assumed" means no source. Phase 1a checked
every row the planner had marked "verify" against the source (2026-09-26), recorded citation and
location, and put the table in the `draw_network_events` docstring's Notes. Two reference values
changed with the maintainer's agreement: the event rate (0.5 to 0.3 per second, awake rest) and
the interneuron baseline (2-5 to 8-15 Hz). "Read from figure" values are approximate. The
rendered-measurement targets, source conventions,
six alternative model conditions and validation report are specified in
[simulator-validation.md](simulator-validation.md); those checks precede benchmark detector runs.

| Parameter | Reference | Source |
| --- | --- | --- |
| Sampling rate | 1500 Hz | The shipped filter's rate (`ripplefilter.mat`). |
| Session | 600 s; running bouts 10-20 s separated by 20-40 s of rest | Assumed. |
| LFP | 4 channels + radiatum, pink noise, `noise_amplitude=1.3`, `shared_noise_fraction=0.5` | `simulate_session` defaults. |
| Theta / delta | amplitude 4 each (8 Hz running, 2 Hz rest) | `examples/literature_recipes.py:70-71` (the recipes' session). |
| Units | 60: 40 place (baseline 0.1-0.5 Hz), 10 other pyramidal (0.5-1.5 Hz), 10 interneurons (8-15 Hz) | Counts and pyramidal rates `literature_recipes.py:56-58`; CA1 pyramidal rates lognormal over 0.001-10 Hz (Mizuseki & Buzsáki 2013, doi:10.1016/j.celrep.2013.07.039, Results), non-theta mean 1.4 Hz (Csicsvari et al. 1999, doi:10.1523/JNEUROSCI.19-01-00274.1999, p. 278). Interneurons: non-theta means 8.3 and 14.3 Hz for two groups (Csicsvari et al. 1999, p. 278). |
| Event rate | 0.3 per second of rest (awake immobility) | Awake immobility 0.13-0.22 ripples/s (Buzsáki 2015, doi:10.1002/hipo.22488, Fig. 3C, read from figure); 0.32-0.40 multiunit candidate events/s during stops (Davidson, Kloosterman & Wilson 2009, doi:10.1016/j.neuron.2009.07.027, Results). Sleep 0.3-0.5/s (Nguyen et al. 2009, doi:10.3389/neuro.07.011.2009, Results). Threshold-dependent. |
| Type mix | swr 0.55, weak_ripple 0.15, burst_only 0.10, ripple_doublet 0.10, sharp_wave_only 0.10 | Assumed. |
| Minimum separation | 0.05 s between events' ±3-sigma spans | Assumed. |
| Ripple span (±3 sigma) | (0.03, 0.15) s | Ripples 30-150 ms, skewed (Buzsáki 2015, "Definition of Pathological Events"), convention unstated; a nominal span, not a threshold-crossing duration. |
| Ripple skew | fraction of the span after the peak (0.5, 0.7) | Assumed (decay slower than rise). |
| Ripple frequency at onset | (160, 220) Hz | Ripples 140-220 Hz (Sullivan et al. 2011, doi:10.1523/JNEUROSCI.0294-11.2011, abstract); modal per-event spectral peaks 167/177/187 Hz in sleep/quiet waking/maze immobility (Buzsáki 2015, Fig. 4C caption). "Onset" is the model's convention. |
| Ripple chirp | decline over the span (0, 30) Hz | Deceleration supported (Nguyen et al. 2009, Results; Sullivan et al. 2011, Results), but recorded ripples fall about 15-20 Hz over some 15 ms from shortly before the peak (Nguyen, Fig. 2C, read from figure), where this linear chirp falls about 2.5 Hz over those 15 ms; about 25% rise instead (Discussion). Both are recorded limitations. |
| Ripple SNR | swr and doublet (2.5, 6.0); weak_ripple (1.2, 2.2) | Assumed; `simulate_session`'s 4.0 lies inside. |
| Sharp wave | span (0.04, 0.12) s, symmetric; amplitude (3, 8) signal units; centre lag N(0, 0.01 s) from the ripple | Amplitude after `literature_recipes.py:68` (6, above delta); span `simulate_session`'s 0.08 inside; lag assumed. |
| Burst | span = ripple span × (1.0, 1.5); centre lag N(0, 0.01 s); place-cell gain 40; participation swr/doublet (0.2, 0.6), weak_ripple (0.02, 0.1), burst_only (0.2, 0.6) with span (0.05, 0.3) s | Gain `literature_recipes.py:65`. Participation is a latent probability, assumed; observed: about 10% of CA1 pyramidal cells fire in a 50 ms window, 0-40% by event (Ylinen et al. 1995, doi:10.1523/JNEUROSCI.15-01-00030.1995, p. 35), about 30% in the largest events (Csicsvari et al. 2000, doi:10.1016/S0896-6273(00)00135-5, Fig. 3C, read from figure). Rest assumed. |
| Other pyramidal units | participate with half the place cells' probability, gain 40 | Assumed. |
| Interneurons | all take part in events with a ripple, gain 3, on the ripple's envelope | Gain: about threefold at the sharp-wave peak (Csicsvari et al. 1999, p. 279, behaving rats). All taking part is assumed: responses differ by type, O-LM cells falling silent (Klausberger et al. 2003, doi:10.1038/nature01374, p. 846, anaesthetized). |
| Doublet | 2 ripples (p 0.7) or 3 (p 0.3), centre-to-centre (0.06, 0.12) s, one burst over all | Spacing around the 8.8-11.8 ripples/s within long replay events (Davidson et al. 2009, Results); the 2/3 proportions are assumed. |
| Non-event rates (per minute) | spike_leakage 2 (rest), emg 1 (any), fast_gamma 2 (any), theta_burst 6 (running) | Assumed. |
| Spike leakage | 1-3 pyramidal units, 3-8 spikes each at ISI (3, 6) ms, waveform peak 2.0 on one channel | Intraburst ISIs peak at 2-6 ms, CA1 mode 5.04 ± 1.00 ms (Mizuseki et al. 2012, doi:10.1002/hipo.22002, Results, Figs. 2A-B; Ranck 1973 not accessed); amplitude assumed. |
| EMG | span (0.05, 0.5) s, white noise high-passed at 100 Hz, peak SD 1.5, all channels | Assumed. |
| Fast gamma | Reference 60-100 Hz; nearby condition 90-140 Hz; span (0.05, 0.15) s, SNR (1.5, 4) against the corresponding band noise | 60-100 Hz is an assumed control; Sullivan et al. 2011 reports 90-140 Hz ([abstract and Fig. 1](https://pubmed.ncbi.nlm.nih.gov/21653864/)). Non-event status is a benchmark assumption. |
| Theta burst | 5-15 place units, gain 10, span (0.1, 0.3) s, running only | Assumed. |

## Event types

Which expressions each type renders (✓), and the distributions that differ from the reference
rows above.

| Type | Ripple | Sharp wave | Burst | Differences |
| --- | --- | --- | --- | --- |
| `swr` | ✓ | ✓ | ✓ | reference |
| `weak_ripple` | ✓ weak SNR | ✓ amplitude × 0.5 | ✓ weak participation | |
| `burst_only` | | | ✓ | burst span (0.05, 0.3) s, centred on the event time |
| `ripple_doublet` | ✓ 2-3 | ✓ one per ripple | ✓ one spanning all ripples | burst ±3-sigma span = the earliest ripple start to the latest ripple end (a longer earlier ripple can end last), symmetric |
| `sharp_wave_only` | | ✓ | | |

## Drawing network events

`draw_network_events` in `simulate.py`, public, wrapped with `explain_call_errors` like every
public function there:

```python
@explain_call_errors
def draw_network_events(
    time: ArrayLike,
    *,
    event_rate: float = 0.3,
    type_probabilities: Mapping[str, float] | None = None,  # None: the reference mix
    running_intervals: ArrayLike | None = None,
    ripple_duration: tuple[float, float] = (0.03, 0.15),
    ripple_skew: tuple[float, float] = (0.5, 0.7),
    ripple_frequency: tuple[float, float] = (160.0, 220.0),
    ripple_chirp: tuple[float, float] = (0.0, 30.0),
    ripple_snr: tuple[float, float] = (2.5, 6.0),
    weak_ripple_snr: tuple[float, float] = (1.2, 2.2),
    sharp_wave_duration: tuple[float, float] = (0.04, 0.12),
    sharp_wave_amplitude: tuple[float, float] = (3.0, 8.0),
    sharp_wave_lag: float = 0.01,          # SD of the centre lag, s
    burst_duration_ratio: tuple[float, float] = (1.0, 1.5),
    burst_lag: float = 0.01,               # SD of the centre lag, s
    burst_gain: float = 40.0,
    participation: tuple[float, float] = (0.2, 0.6),
    weak_participation: tuple[float, float] = (0.02, 0.1),
    burst_only_duration: tuple[float, float] = (0.05, 0.3),
    doublet_interval: tuple[float, float] = (0.06, 0.12),
    minimum_separation: float = 0.05,
    strength_correlation: float = 0.0,
    envelope_power: int = 2,               # 2 or 4, equal half-maximum widths
    rng: int | np.random.Generator | None = None,
) -> pd.DataFrame:
```

Returns the [latent event table](shared-contracts.md#latent-event-table) with `n_participants`
0: participants are drawn when the session is rendered, which knows the units. Each row also
stores `envelope_power`. A tuple
`(x, x)` fixes a value; `ripple_chirp=(0, 0)` gives constant-frequency ripples.

Algorithm:

1. `rest` = the recording minus `running_intervals` minus 1 s at each end of the recording.
   Events occur only in `rest`.
2. Event times: a Poisson process of rate `event_rate` on the concatenated rest time, mapped back
   to recording time (draw `n ~ Poisson(rate * rest_duration)`, then `n` sorted uniform positions
   in concatenated rest time).
3. Types: `rng.choice(EVENT_TYPES, size=n, p=...)`, the probabilities normalized.
4. Per event, draw its components. The order of draws is fixed and documented in the docstring:
   types, then an `(n_events, 15)` array of standard normals and an `(n_events, 18)` array of
   uniforms, one row per event in time order. Every event draws the same block whatever its
   type (slots for up to three ripples and sharp waves), so changing one parameter or the type
   mix moves no other event's values:
   Use the marginal-preserving correlated draws in
   [coupled event strengths](simulator-validation.md#coupled-event-strengths) for SNR, onset
   frequency, sharp-wave amplitude and participation; rho=0 retains independent marginals.
   - ripple: span `s ~ U(ripple_duration)`, skew `q ~ U(ripple_skew)`,
     `rise_sigma = s (1 - q) / 3`, `decay_sigma = s q / 3`, `frequency_start ~ U(ripple_frequency)`,
     `frequency_end = frequency_start - U(ripple_chirp)`, amplitude from the type's SNR range.
     The event time is the (first) ripple's centre.
   - doublet: `m = 2` or `3`; ripple `j > 0` centred at the previous centre plus
     `U(doublet_interval)`.
   - sharp wave: one per ripple, centre = ripple centre + `N(0, sharp_wave_lag)`,
     symmetric, `rise_sigma = decay_sigma = U(sharp_wave_duration) / 6`; for `sharp_wave_only`,
     centred on the event time.
   - burst: centre = ripple centre + `N(0, burst_lag)`; span = ripple span × `U(burst_duration_ratio)`
     with the ripple's skew; for `burst_only` span `U(burst_only_duration)`, symmetric; for a
     doublet as in [Event types](#event-types); `amplitude = burst_gain`, participation from the
     type's range.
5. Rejection: walk events in time order; drop an event whose ±3-sigma union span starts less
   than `minimum_separation` after the previous kept event's span ends, or whose ±4-sigma span
   leaves the rest interval it started in. Dropping (not redrawing) keeps the draw count fixed,
   so the other events do not move when one parameter changes.
6. Renumber `event_id` in time order; sort as the contract requires.

Validation: `event_rate >= 0`; probabilities non-negative, keys in `EVENT_TYPES`, positive sum;
every range `low <= high`, durations and SNRs positive; frequencies below Nyquist;
`strength_correlation` in [0, 1], `envelope_power` in {2, 4}; `ValueError`
naming the parameter otherwise.

## Rendering a network session

```python
@explain_call_errors
def simulate_network_session(
    time: ArrayLike,
    events: pd.DataFrame,
    *,
    non_events: pd.DataFrame | None = None,
    n_channels: int = 4,
    unit_counts: Mapping[str, int] | None = None,   # None: {"place": 40, "pyramidal": 10, "interneuron": 10}
    baseline_rate: Mapping[str, tuple[float, float]] | None = None,  # None: the reference rates per type
    channel_gains: Sequence[float] | FloatArray | None = None,
    spatial_profile: str = "global",
    channel_occupancy: float = 1.0,
    channel_gain_range: tuple[float, float] = (1.0, 1.0),
    channel_delay: float = 0.0,
    shared_noise_fraction: float = 0.5,
    noise_type: NoiseType = "pink",
    noise_amplitude: float = 1.3,
    noise_log_amplitude: float = 0.0,
    noise_modulation_period: float = 60.0,
    sharp_wave_leak: float = 0.3,
    ripple_leak: float = 0.3,
    interneuron_gain: float = 3.0,
    spike_model: str = "poisson",
    refractory_period: float = 0.002,
    running_intervals: ArrayLike | None = None,
    peak_speed: float = 30.0,
    theta_amplitude: float = 4.0,
    delta_amplitude: float = 4.0,
    rng: int | np.random.Generator | None = None,
    sampling_frequency: float | None = None,
) -> SimulatedSession:
```

Allocate a fixed set of child RNGs at entry, in this documented order: noise, ripple phases,
spatial profiles, noise modulation, unit baseline rates, burst participants, non-events, spikes.
Derive their seeds from a fixed-size draw from the supplied RNG, even when an option is disabled.
Within each stream use component/table order. This permits matched noise-only renders and keeps
an alternative spike model from changing the LFP. The existing `simulate_session` draw order is
unchanged. Store the new spatial metadata as specified in the shared contracts.

Steps:

1. **Noise.** `stationary_noise = _correlated_noise(n_time, n_channels + 1, ...)`, radiatum last. Keep its stationary
   band SD for signal sizing, then apply the optional unit-RMS noise modulation from
   [changing background variance](simulator-validation.md#changing-background-variance).
2. **Ripples.** Band noise SD `sd = filter_ripple_band(stationary_noise[:, 0], sampling_frequency=rate).std()`
   (the reference channel, as `_ripple_waveform` does, `simulate.py:433-435`). Per ripple component:

   ```python
   def _render_ripple(time, center, rise_sigma, decay_sigma, f_start, f_end, phase, power=2):
       """Unit-peak asymmetric, linearly chirped burst over center -8 rise .. +8 decay."""
       first, last = np.searchsorted(time, [center - 8 * rise_sigma, center + 8 * decay_sigma])
       window = slice(int(first), max(int(last), int(first) + 1))
       t = time[window] - center
       sigma = np.where(t < 0, rise_sigma, decay_sigma)
       envelope = np.exp(-np.log(2) * (np.abs(t) / (np.sqrt(2*np.log(2)) * sigma))**power)
       # frequency linear from f_start at -3 rise to f_end at +3 decay, constant outside
       t0, t1 = -3 * rise_sigma, 3 * decay_sigma
       u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
       frequency = f_start + (f_end - f_start) * u
       step = np.diff(time[window], prepend=time[window][0])
       carrier = np.sin(phase + 2 * np.pi * np.cumsum(frequency * step))
       return window, carrier * envelope
   ```

   scaled with `_scale_to_snr(burst, snr, sd, rate, band=None)`, **extracted** from
   `_add_ripple_bursts` (`simulate.py:550-559`, the padded filter-peak scaling) so both paths
   share it. `band=None` calls `filter_ripple_band(padded, sampling_frequency=rate)` exactly as
   today (the shipped kernel at 1500 Hz); a band calls it with `band=band` (fast gamma, phase 1b).
   Apply the [spatial profile](simulator-validation.md#spatial-ripple-structure) to the scaled
   burst, multiplying by `channel_gains[c]`; add the latent waveform to the radiatum times
   `ripple_leak`. Nominal SNR precedes spatial gains and background modulation. Noise-free test
   fixtures render unit-amplitude components directly through the waveform helpers: finite SNR
   against zero noise does not define a nonzero waveform.
3. **Sharp waves.** The row's envelope of peak `-amplitude` on the radiatum and
   `+sharp_wave_leak * amplitude` on channel 0 (`_add_sharp_wave_pair`'s convention,
   `simulate.py:741-753`, generalized to side scales and powers by `_event_envelope`).
4. **Slow field and speed.** `simulate_theta_delta` (`simulate.py:998`) added to every channel
   and the radiatum; `simulate_speed` (`simulate.py:941`), exactly as `simulate_session` does
   (`simulate.py:1280-1295`).
5. **Units.** `unit_types` = `"place"` × 40, `"pyramidal"` × 10, `"interneuron"` × 10 by default
   (in that order). Baseline rates per type drawn uniformly from `baseline_rate[type]`,
   and kept per unit in `SimulatedSession.baseline_rates`.
   Modulation starts at 1 per unit and sample. Per burst component: participants are place units
   with probability `participation` and other pyramidal units with `participation / 2` (a Bernoulli
   draw per unit); `n_participants` is recorded in the returned events table (pyramidal and place
   together); participants' modulation gains `(amplitude - 1) * envelope`, using the burst row's
   `envelope_power`. Per event with a ripple: every interneuron gains
   `(interneuron_gain - 1) * envelope` on the ripple's envelope (the union for a doublet: the
   elementwise max). Spikes: `rng.poisson(rates * step * modulation)`, as `simulate_multiunit`
   (`simulate.py:918-919`) in the reference. The alternative uses
   [refractory spiking](simulator-validation.md#refractory-spiking) with the same intensity.
6. **Session.** `SimulatedSession` with `raw_lfp = lfps[:, 0].copy()`, `sharp_wave_lfp` the
   radiatum, the events table (with `n_participants`), `ripple_channels`, `unit_types`, `baseline_rates`,
   `running_intervals`, and the
   ripple arrays derived as in [SimulatedSession additions](shared-contracts.md#simulatedsession-additions).

`simulate_session` is not reimplemented on top of this; the two coexist (`simulate_session` for
the existing tests, examples and users; `simulate_network_session` for event-type truth). They
share helpers only.

## Non-events

```python
@explain_call_errors
def draw_non_events(
    time: ArrayLike,
    *,
    rates: Mapping[str, float] | None = None,  # per minute; None: the reference rates
    running_intervals: ArrayLike | None = None,
    n_channels: int = 4,
    spike_leakage_units: tuple[int, int] = (1, 3),
    spike_leakage_spikes: tuple[int, int] = (3, 8),
    spike_leakage_isi: tuple[float, float] = (0.003, 0.006),
    spike_leakage_amplitude: float = 2.0,
    emg_duration: tuple[float, float] = (0.05, 0.5),
    emg_amplitude: float = 1.5,
    fast_gamma_frequency: tuple[float, float] = (60.0, 100.0),
    fast_gamma_band: tuple[float, float] = (60.0, 100.0),
    fast_gamma_duration: tuple[float, float] = (0.05, 0.15),
    fast_gamma_snr: tuple[float, float] = (1.5, 4.0),
    theta_burst_units: tuple[int, int] = (5, 15),
    theta_burst_duration: tuple[float, float] = (0.1, 0.3),
    theta_burst_gain: float = 10.0,
    rng: int | np.random.Generator | None = None,
) -> pd.DataFrame:
```

Returns the [non-event table](shared-contracts.md#non-event-table). Where each type occurs:
`spike_leakage` rest, `emg` anywhere, `fast_gamma` anywhere, `theta_burst` running. Times are a
Poisson process per type on its allowed time (as in step 2 of drawing events), then the same
±4-sigma containment rejection. Non-events may overlap network events; that is intended (a
leakage burst during a ripple is realistic) and analyses classify by longest overlap.

For `spike_leakage` the row stores the drawn `n_spikes` and `isi`, which the renderer
reads (the spike train is regular at `isi`), and the burst's envelope as `center_time` with
`rise_sigma = decay_sigma = (n_spikes - 1) * isi / 6` for truth windows; the sigmas alone do
not determine the signal (3 spikes at 6 ms and 5 at 3 ms share them). Unit
identities are drawn at render time. `channel ~ U{0 .. n_channels - 1}`.

Rendering, in `simulate_network_session` when `non_events` is given (after the burst
participants, before the spikes, in `non_event_id` order):

| Type | Renders |
| --- | --- |
| `spike_leakage` | Picks `n_units` pyramidal or place units; adds their burst spikes to the spike counts (after the Poisson draw, so the Poisson counts do not change); adds at each spike sample a biphasic waveform `amplitude * (w)` on `channel`, `w = [-1.0, 0.45, 0.2]` over three samples (a 1500 Hz rendering of a ~1 ms spike and its after-hyperpolarization). |
| `emg` | White noise high-passed at 100 Hz (4th-order Butterworth, `sosfiltfilt`), times the asymmetric envelope, times `amplitude`, added identically to every channel and the radiatum. |
| `fast_gamma` | `_render_ripple` with `f_start = f_end = frequency`, power 2, scaled by `_scale_to_snr` using the row's stored `(snr_band_low, snr_band_high)` for both burst filtering and the stationary noise SD; added to every channel with the recording-wide channel gains. |
| `theta_burst` | Picks `n_units` place units; multiplies their modulation by `1 + (amplitude - 1) * envelope`. |

As implemented in phase 1b (2026-09-26), where the design above left a choice open:

- **Draw order.** One draw from `rng` seeds a stream per kind, in `NON_EVENT_TYPES` order.
  Each stream draws its Poisson count, the positions, then a fixed block of uniforms per
  non-event: leakage 4 (units, spikes, ISI, channel), EMG 1 (span), gamma 3 (frequency, span,
  SNR), theta 2 (units, span). One kind's rate or ranges never change another kind's rows.
  Rejected non-events are dropped, not redrawn.
- **Allowed time.** Every kind avoids the recording's first and last second, as events do.
  "Any" is `[t0 + 1, t_end - 1]`; "running" is the bouts clipped to it.
- **`rates`.** A kind left out of the mapping never occurs, as with `type_probabilities`.
- **Spans.** EMG, gamma and theta spans are symmetric, `rise = decay = span / 6`.
- **Leakage.** A burst's units fire together at the same sample, the nearest one to each
  spike time. The waveform is added once per spike sample, so `amplitude` is the leak's peak.
  At render, the side scales must equal `(n_spikes - 1) isi / 6`, so the truth windows cannot
  drift from the spikes. Leaked spikes are added after the draw under either spike model, so
  they are not subject to `refractory_period` (as [refractory
  spiking](simulator-validation.md#refractory-spiking) specifies).
- **EMG.** Noise is drawn over the envelope's window plus the filter's settling length on
  each side, which is where its impulse response keeps less than 1e-12 of its energy. It is
  filtered with `sosfiltfilt` and cropped to the window, so the kept samples are stationary
  filtered noise at any rate: near 200 Hz the settling length is hundreds of samples. It is
  scaled by the SD of forward-backward-filtered unit white noise, computed from the filter's
  frequency response (`sqrt(mean |H|^4)`), so `amplitude` is the expected SD at the peak.
- **Gamma.** Added to the pyramidal-layer channels only, not the radiatum. A band's noise SD is
  computed once per distinct stored band.
- **Theta.** The factors multiply a unit's intensity after the additive burst and interneuron
  gains. Under Poisson spiking the changed intensities also change the counts drawn for later
  units: statistically the same, not bit-identical.
- **Order.** Non-events render after the sharp waves and before the slow field and the unit
  draw (participants, then spikes). Theta factors enter the unit draw, and leaked spikes are
  added after it. With separate streams, this order changes no draw.
- **Render validation.** `envelope_power` must be 2. A leakage burst may use at most the place
  and other pyramidal units, a theta burst at most the place units. A gamma band must be
  filterable at the rate, and a gamma frequency finite. EMG needs the rate above 200 Hz.
  EMG, gamma and theta side scales, and leakage intervals, must be at least a sample.
- **One sample.** Every one-sample floor, for events and non-events, at draw and at render,
  compares with the timestamps' median step, never `1 / rate`. The draws infer the rate from
  the timestamps, while the renderer may be given one up to 0.14% away at a Unix time. So
  a table drawn on the timestamps renders at any rate given. A leak interval near one sample
  that would put two spikes on a sample still raises.
- **Leaked samples.** A spike takes its nearest sample, and a tie goes to the later one. The
  position is measured from the recording's first sample before the spike offsets are added,
  so it rounds with the centre alone. A position up to 1 µs below a tie counts as the tie;
  that is a constant, more than a Unix-time timestamp's rounding (0.12 µs). So exact ties
  take the same sample at every origin, as does every spike farther than that rounding from
  1 µs below a tie. A spike within it can move by a sample: exact invariance is impossible
  once the centre itself rounds. A row whose spikes would share a sample, or whose waveform
  would run past the recording, raises.
- **Pinning.** `tests/test_snapshots.py` pins a seeded network session with no non-events,
  checked against the renderer before non-events were added. It also pins a seeded
  non-event table and the session rendered with it.
- **Benchmark note.** The three-sample leak waveform's ripple-band share depends on the ISI
  (and a little on the spike count). For five spikes at 1500 Hz, 4 ms (250 spikes/s) gives
  14.5% of the waveform's power, 6 ms gives 10.6%, and 3 ms (333 spikes/s, above the band)
  only 2%. Leakage near the short end of the default ISI range is a weak decoy for
  ripple-band detectors.

## Truth windows

```python
@explain_call_errors
def truth_windows(table, fraction=0.1, expression=None):
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must lie in (0, 1), got {fraction}.")
    events = "event_id" in table.columns
    if not events and expression is not None:
        raise ValueError("A non-event table has no expressions; leave expression as None.")
    rows = table
    if expression not in (None, "network"):
        _check_choice("expression", expression, (*EXPRESSIONS, "network"))
        rows = table[table.expression == expression]
    k = np.sqrt(2 * np.log(2)) * (np.log(1 / fraction) / np.log(2)) ** (1 / rows.envelope_power.to_numpy())
    windows = pd.DataFrame({
        "id": rows["event_id" if events else "non_event_id"].to_numpy(),
        "type": rows["event_type" if events else "non_event_type"].to_numpy(),
        "start_time": (rows.center_time - k * rows.rise_sigma).to_numpy(),
        "end_time": (rows.center_time + k * rows.decay_sigma).to_numpy(),
        "peak_time": rows.center_time.to_numpy(),
    })
    if events and expression != "network":
        windows["expression"] = rows.expression.to_numpy()
        windows["component"] = rows.component.to_numpy()
    if expression == "network":
        priority = rows.expression.map({"ripple": 0, "burst": 1, "sharp_wave": 2})
        order = np.lexsort((rows.component.to_numpy(), priority.to_numpy(), windows.id.to_numpy()))
        first = windows.iloc[order].groupby("id", sort=True).first()
        span = windows.groupby("id", sort=True).agg(start_time=("start_time", "min"),
                                                     end_time=("end_time", "max"))
        windows = span.assign(type=first.type, peak_time=first.peak_time).reset_index()
        windows = windows[["id", "type", "start_time", "end_time", "peak_time"]]
    return windows.reset_index(drop=True)
```

Checked during planning on a toy table (an `swr`, a doublet, a `burst_only`): network spans are the
unions, the doublet's peak is its first ripple, the `burst_only` peak its burst.

`_check_choice` is the existing helper in `core.py`.

## Matching

In `src/ripple_detection/evaluate.py`:

```python
def _overlap_matrix(reference: FloatArray, detected: FloatArray) -> FloatArray:
    """Intersection lengths, shape (n_reference, n_detected); 0 where disjoint or touching."""
    start = np.maximum(reference[:, :1], detected[:, 0])
    end = np.minimum(reference[:, 1:], detected[:, 1])
    return np.clip(end - start, 0.0, None)


@explain_call_errors
def match_events(reference, detected, *, minimum_iou=0.0):
    _check_non_negative(minimum_iou=minimum_iou)
    ref, det = _checked(reference, "reference"), _checked(detected, "detected")
    ref_peaks, det_peaks = _peaks(reference, len(ref)), _peaks(detected, len(det))
    intersection = _overlap_matrix(ref, det)
    ref_len = (ref[:, 1] - ref[:, 0])[:, None]
    det_len = (det[:, 1] - det[:, 0])[None, :]
    union = ref_len + det_len - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(intersection > 0, intersection / union, 0.0)
    eligible = (intersection > 0) & (iou > minimum_iou)
    # exact one-to-one assignment, per connected component of the eligibility graph:
    # the most pairs first, then the largest summed IoU. A pair's weight is bonus + IoU with
    # bonus above any component's summed IoU, so no IoU gain outweighs one more pair; the
    # maximum pair count does not depend on which inventory is the reference, so counts
    # (and F1, Jaccard) are symmetric even when summed IoU ties
    graph = scipy.sparse.bmat([[None, scipy.sparse.csr_array(eligible)],
                               [scipy.sparse.csr_array(eligible.T), None]])
    n_components, label = scipy.sparse.csgraph.connected_components(graph, directed=False)
    ref_label, det_label = label[: len(ref)], label[len(ref):]
    rows, cols = [], []
    for component in np.unique(ref_label[eligible.any(axis=1)]):
        r = np.flatnonzero(ref_label == component)
        d = np.flatnonzero(det_label == component)
        bonus = min(len(r), len(d)) + 1.0
        weight = np.where(eligible[np.ix_(r, d)], bonus + iou[np.ix_(r, d)], 0.0)
        i, j = scipy.optimize.linear_sum_assignment(weight, maximize=True)
        keep = eligible[np.ix_(r, d)][i, j]
        rows.extend(r[i[keep]]); cols.extend(d[j[keep]])
    order = np.argsort(rows, kind="stable")
    r, d = np.asarray(rows, dtype=int)[order], np.asarray(cols, dtype=int)[order]
    pairs = pd.DataFrame({
        "reference_index": r, "detected_index": d, "iou": iou[r, d],
        "coverage": _ratio(intersection[r, d], ref_len[r, 0]),
        "temporal_precision": _ratio(intersection[r, d], det_len[0, d]),
        "onset_error": det[d, 0] - ref[r, 0], "offset_error": det[d, 1] - ref[r, 1],
        "peak_error": det_peaks[d] - ref_peaks[r],
    })
    touching = intersection > 0
    return EventMatching(ref, det, pairs, touching.sum(axis=1), touching.sum(axis=0))
```

- `_checked` wraps `core._event_bounds` plus the finite/ordered check of `_overlaps`
  (`core.py:2129-2138`); factor that check into a shared `core._check_bounds(name, bounds)` used by
  both, a behavior-preserving extraction.
- `_peaks` returns the `peak_time` column as floats, or NaNs.
- Why pairs before IoU: with IoU alone, `[[0, 1], [2, 4]]` against `[[0, 4], [2.5, 3]]` ties
  (two pairs at 0.25 or one at 0.5) and the solver returned two pairs one way and one the
  other. With the pair count first, 3000 random swapped inventories (half-integer bounds, so
  frequent ties) gave equal pair counts and summed IoU both ways. Assignments equal in both
  can still differ in which events pair; counts and summaries do not.
- Return early (no pairs, zero overlap counts) when either input is empty, before building the
  sparse graph.
- Checked during planning: this code gives the expected result for every hand case in phase 2's
  validation slice (optimal-not-greedy, IoU 8/12, touching, split, merge, empty).
- `_ratio(a, b)` is `a / b` with NaN where `b == 0`.
- Dense matrices are fine at benchmark sizes (a 10-minute session has a few hundred events per
  method; 1000 × 1000 is 8 MB). Document the O(n_reference × n_detected) memory in the docstring.
- `split_reference = flatnonzero(reference_overlaps >= 2)`,
  `merged_detected = flatnonzero(detected_overlaps >= 2)`. These count *any* overlap, not
  eligibility, so a fragment below `minimum_iou` still counts as a split.

As implemented in phase 2 (2026-09-27), where the code departs from the sketch above:

- **Tolerances from the timestamps' rounding.** With u an ulp of the largest bound: a pair is
  eligible at `minimum_iou > 0` only when `iou > minimum_iou + 8u / union` (IoU's worst-case
  rounding), so a nominal tie at the threshold is excluded at any clock origin; at 0 the rule
  stays "positive overlap". `label_by_overlap` ties overlaps within 8u (earlier row), and the
  error correlations rank errors within 3u of their group's smallest as ties (an error's
  worst-case rounding; average ranks, then Pearson): errors read off sample times are whole
  samples that round apart, and ranking that noise gave 0.33-0.40 for a true 0.29. Groups are
  measured from their smallest value, not neighbour to neighbour, so errors 1 us apart on a
  Unix clock do not chain into one constant group. `TestTimeOrigin::test_evaluation` fails
  without each tolerance.
- **Graph and solver.** No dense matrices over every pair: only overlapping pairs are built
  (candidates found by sorted starts within the longest detected event's length), components
  come from a `coo_matrix` of the eligible pairs (no `bmat`), and a component of one edge is
  taken without `linear_sum_assignment`. `label_by_overlap` uses the same pairs. Typical
  inventories at 1000 events a side: 1.6 ms and 0.2 MB, from 14 ms and 49 MB with dense
  matrices; outputs identical. Worst cases stay quadratic: a detected event spanning the
  session makes nearly every pair a candidate (about 28 MB at 1000 true events), and each
  component is still a dense block, so a chain of overlapping events costs about 16 MB at 1000
  a side and 65 MB at 2000. The `match_events` docstring states both.
- **Checks.** `minimum_iou` must be below 1; `compare_detectors` and `consensus_counts` raise
  `TypeError` for events that are not a mapping (a DataFrame iterates over its columns).
- **Indices.** `pairs` and the unmatched/split/merged arrays are row positions (for `.iloc`);
  `consensus_counts` and `label_by_overlap` return on the input DataFrame's index, so they
  assign back by label.
- **`jaccard_true`** follows the contract literally (Jaccard of the two methods' truth-matched
  detections against each other), so two methods that found different true events whose
  detections overlap agree (`n_shared_truth` 0, `jaccard_true` 1). By decision 14 it stays, as
  agreement of the detected intervals, and `jaccard_truth_ids` (`n_shared_truth` over the
  truth events either matched) answers whether they found the same events.
- Zero-length events never overlap, so never match; `coverage` NaN for a zero-length reference
  cannot occur.

## Agreement statistics

```python
@explain_call_errors
def compare_detectors(events, *, truth=None, minimum_iou=0.0):
    names = list(events)
    bounds = {name: _checked(events[name], name) for name in names}
    truth_hit = {}
    if truth is not None:
        truth_bounds = _checked(truth, "truth")
        for name in names:
            m = match_events(truth_bounds, bounds[name], minimum_iou=minimum_iou)
            truth_hit[name] = m   # m.pairs maps truth rows to this method's rows
    rows = []
    for a, b in itertools.combinations(names, 2):
        m = match_events(bounds[a], bounds[b], minimum_iou=minimum_iou)
        onset = -m.pairs.onset_error.to_numpy()   # a.start - b.start
        offset = -m.pairs.offset_error.to_numpy()
        row = {"method_a": a, "method_b": b, "n_a": len(bounds[a]), "n_b": len(bounds[b]),
               "n_matched": len(m.pairs), "jaccard": _jaccard(len(bounds[a]), len(bounds[b]), len(m.pairs)),
               "median_iou": _median(m.pairs.iou),
               **_signed_summary("onset", onset), **_signed_summary("offset", offset)}
        if truth is not None:
            row |= _truth_columns(bounds, truth_hit, a, b, minimum_iou)
        rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
```

- `_signed_summary(kind, x)` gives `median_{kind}_difference`, `{kind}_difference_iqr`
  (`q75 - q25`), `fraction_a_earlier_{kind}` (`mean(x < 0)`); NaN when `x` is empty.
- `_truth_columns`: `true_a` = `a`'s rows matched to truth, `false_a` the rest (same for `b`);
  `jaccard_true` = Jaccard of `match_events(bounds[a][true_a], bounds[b][true_b])`,
  `jaccard_false` likewise on the false rows; shared truth rows = intersection of the truth
  indices each matched; the two methods' onset (offset) errors on those rows; Spearman via
  `scipy.stats.spearmanr`, NaN below 3 rows or when either vector is constant.
- `consensus_counts`: match each method to `truth`, set a boolean column per method from
  `pairs.reference_index`, `n_methods` = row sum.
- `label_by_overlap`: `_overlap_matrix(events, windows)`; per event the argmax column's label
  where the row max is > 0, else `unlabeled`. Ties go to the earlier window row.

Hierarchical clustering of methods (phase 5) uses `scipy.cluster.hierarchy.linkage` on the
condensed `1 - jaccard` matrix with `method="average"`; no new dependency.

## Recipe executor

`examples/benchmark/recipe_configs.py` is a thin adapter to the installed API:

```python
from ripple_detection.literature_methods import (
    Recording, bounds, check_method, list_methods, run_method,
)

def run_recipe(
    config: RecipeConfig, rec: Recording, behavior_intervals: FloatArray | None = None
) -> pd.DataFrame:
    return run_method(
        config.method, rec, behavior_intervals=behavior_intervals, **dict(config.options)
    )
```

Use the [configuration contract](shared-contracts.md#recipe-config). Build recordings
from observed arrays and declared selections via `Recording.from_arrays`. Unit types,
state/baseline intervals, reference channels and templates are explicit input policies;
record which labels the simulator supplies. Required external ripple inventories come
from a named detector and recorded settings, never event truth. Unknown settings need
an explicit benchmark assumption or exclusion. Avoid using the simulation-only
fallback paths intended for demonstration.

The installed package owns filtering, normalization, native bins, event construction,
postprocessing, roles and stage semantics. The benchmark neither copies its `Recording`
class nor caches mutable recordings behind public methods. Any future cache belongs to
an explicitly bounded immutable analysis context and requires lifetime/invalidation tests.

## Recipe classification

Use exact public method names from `list_methods()`. Classify by implemented output
and role, not survey row or demonstration grouping. Every method is configured or
listed in `EXCLUSIONS` with a reason. Configure supported stages/protocols separately.

Published-method results always come from `run_recipe`. Experimental component
representations are phase 6's responsibility. Custom peak merging, adaptive feedback,
FFT windows and finite/native-grid kernels are not assumed equivalent to generic
thresholding. A method without a verified experimental representation remains a fixed
comparison point; no detector body is replaced to make attribution possible.

As implemented in phase 3 (2026-09-27), where the plan left a choice open:

- **Coverage.** 77 configurations (the 57 default inventories, the demo's two protocol
  variants `olafsdottir_2015.bayesian_candidates` and `olafsdottir_2017.trajectory`, and 18
  additional inventories) and 11 exclusions partition the 86 `list_methods()` names. Excluded:
  `bush_2022_ripples` (4800 Hz) and `olafsdottir_2017_ripples` (1200 Hz), since resampling is
  not an input policy; nine additional inventories whose required options have no published
  value (`gridchyn_2020_ripples`, `xu_2019_ripples`, `farooq_2019_neuron_ripples`,
  `farooq_2019_science_ripples`, `chenani_2019_hfe`, `liu_2019_ripples`,
  `liu_2019_ripple_frames`, `drieu_2018_ripples`, `diba_2007_ripples`). The maintainer kept
  Diba excluded rather than assume Csicsvari 1999b's 1.6 ms RMS window.
- **One input policy** (`simulated_awake_session`): a recording holds exactly the inputs the
  method declares for its options (`when` evaluated on the resolved options; `unless` never
  lapses, since only declared inputs are supplied), built with `Recording.from_arrays`, so no
  simulation-only fallback runs. Place cells are the units labelled `place` (also standing in
  for any template, probe-sequence or block-specific selection; `templates` is one template of
  them), pyramidal cells `place` or `pyramidal`. Sleep, baseline and behavior intervals are all
  rest, the samples outside the known running bouts (a session without rest raises); the
  reference LFP is zeros. External ripples (Yang 2024, Grosmark 2016) come from
  `Zugaro_ripple_detector` on channel 0 at 130-200 Hz (low 2, high 5, at most 0.2 s, no speed
  rule), Carey's example ripples are the five largest default Kay events by `max_zscore`: the
  package's own simulation proxies, run by name through the registry and recorded with every
  resolved setting. Unreported measured-only options take the package's demonstration value
  (Stella, Nádasdy, Kudrimoti, Wikenheiser). Each stand-in is written into the
  configuration's `assumptions`, which a configuration must match (`configure` derives them;
  a stale copy raises when a recording or record is built).
- **Stage.** `configure` sets `stage="detection"` for every method with a decoding stage;
  decoding-candidate stages (replay gating) are not configured.
- **Primary expressions**, pinned in a test: `network` where events join an LFP ripple or SWR
  detection with a population burst (12, including Krause 2022's SWRs trimmed to place-cell
  activity), `ripple` for LFP events even with a participation or spiking gate (30: Harvey,
  Wikenheiser, Bhattarai's ripples, Gupta's gate), `burst` for population-only events (35,
  including Igata 2021, whose implemented candidates are population-only). None is
  `sharp_wave`.
- **Records.** `method_record(config)` is one all-string row of `methods.csv` minus
  `session_id`. Method options follow the package's attrs JSON convention (non-finite as
  None, so they equal `attrs["options"]`); detector settings in `input_policy` write
  non-finite values as `"inf"`, since None is itself a setting there.
- **For the runner** (measured on a 600 s reference-size session): building the 77 recordings
  takes about 13 s and running them about 40 s. 51 configurations take spike counts, and
  `from_arrays` validates float64 counts chunkwise (136 ms each); integer counts made
  recordings identical and results exactly equal on the configurations checked, and cut
  recording time from 10.5 s to 3.7 s. Each such recording peaks at about 476 MB on top of the
  session's 432 MB of float64 counts, so a worker needs about 0.9 GB, not 0.5 GB.

## Conditions grid

`REFERENCE` is a dict of four dicts, keyed by where the values go: `"session"`
(`duration_s=600`), `"events"` (`draw_network_events` keywords), `"non_events"`
(`draw_non_events` keywords) and `"render"` (`simulate_network_session` keywords), holding the
[parameter sources](#parameter-sources) values. A condition's `params` override entries by dotted
key (`"events.ripple_snr"`); a factor that sets several entries lists them all.

**Running schedule** (`running_schedule(duration_s, rng)`): alternating rest `U(20, 40)` s and
bout `U(10, 20)` s, starting with a rest; a bout that would leave less than 5 s of rest before the
end is dropped. So a session shorter than 35 s has no bout.

**Seeding** (common random numbers): `session_seed(replicate) = 20260924 + replicate`, the same
for every condition. Replicate `k` of two conditions shares its schedule and, where the factor does
not change the draw count, its event times (the drop-not-redraw rule in
[drawing network events](#drawing-network-events)), so robustness comparisons are paired by
replicate. Draw order within a session: schedule, events, non-events, render. The six model
alternatives change no draw count: the strength variates are drawn at every correlation and
the renderer's substreams are allocated whether or not an option is active, so each
alternative's replicate `k` has the reference's event times, unit baseline rates and
recruitment, and their comparison is paired.

**Condition ids** use only `[A-Za-z0-9_.=,-]`: `"reference"`, `f"{factor}={label}"` for one
factor, `f"{factor1}={label1},{factor2}={label2}"` for a crossed cell. Labels are the column
below.

One factor at a time (reference level in bold):

| Factor | Sets | Levels (label: value) |
| --- | --- | --- |
| `ripple_snr` | `events.ripple_snr` | low: (1.5, 3.0), **(2.5, 6.0)**, high: (4.0, 10.0) |
| `participation` | `events.participation` | low: (0.05, 0.2), **(0.2, 0.6)**, high: (0.5, 0.9) |
| `n_units` | `render.unit_counts` | 30: {place 20, pyramidal 5, interneuron 5}, **60**, 120: {80, 20, 20} |
| `n_channels` | `render.n_channels`, `non_events.n_channels` | 1, **4**, 16 |
| `shared_noise_fraction` | `render.shared_noise_fraction` | 0.2, **0.5**, 0.8 |
| `noise_type` | `render.noise_type` | **pink**, brown |
| `event_rate` | `events.event_rate` | 0.15, **0.3**, 0.6 |
| `type_mix` | `events.type_probabilities` | **reference**, swr_only: {swr 1.0}, hard: {swr 0.25, weak_ripple 0.3, burst_only 0.15, ripple_doublet 0.15, sharp_wave_only 0.15} |
| `burst_lag` | `events.burst_lag` | 0.0, **0.01**, 0.03 |
| `ripple_chirp` | `events.ripple_chirp` | none: (0, 0), **(0, 30)** |
| `spike_leakage_rate` | `non_events.rates["spike_leakage"]` | 0, **2**, 6 |
| `emg_rate` | `non_events.rates["emg"]` | 0, **1**, 3 |
| `fast_gamma_rate` | `non_events.rates["fast_gamma"]` | 0, **2**, 6 |
| `theta_burst_rate` | `non_events.rates["theta_burst"]` | 0, **6**, 18 |
| `slow_amplitude` | `render.theta_amplitude`, `render.delta_amplitude` | 0, **4**, 8 |

A rate override replaces one entry of the `rates` mapping, the others keeping their reference
values. Numeric labels are the value as written (`n_units=30`, `emg_rate=0`).

Crossed pairs (3 × 3 each, the four one-factor points and the reference shared with the grid):
`ripple_snr × participation`, `ripple_snr × spike_leakage_rate`.

Add the six one-factor alternatives in
[simulator-validation.md](simulator-validation.md#six-sensitivity-conditions), using that table's
factor names, labels and exact overrides. Their reference defaults are part of `REFERENCE` and
every saved resolved specification, including options inactive in the reference.

Totals: 1 reference + 28 original one-factor levels + 6 model alternatives + 8 new crossed
cells = 43 conditions. Replicates: 20 for the reference, 10 for every other condition = 440
sessions. Validation simulations are additional and use separate replicates.

## Runner

`examples/benchmark/run.py`, a CLI:

```
uv run python examples/benchmark/run.py --run-name NAME [--conditions all|ID,ID] [--replicates N]
    [--duration S] [--workers N] [--resume] [--smoke] [--combine] [--validation-report PATH]
```

`--validation-report` is required for every run that simulates or detects (`--smoke`, a full
run, `--resume`); `--combine` alone needs none.

Before detector execution, verify the ready report and simulation settings per
[simulator validation](simulator-validation.md#validation-report-and-execution-order).
`--combine` only rebuilds saved outputs and does not run this preflight.

Per session, in a worker (`ProcessPoolExecutor`, default `os.cpu_count() - 1` workers):

1. `simulate_condition(condition, replicate)`: seed from `session_seed(replicate)`, then the draw
   order in [Conditions grid](#conditions-grid).
2. Build package recordings via `make_recording(session, config)` using declared input
   policies. Save resolved options and input provenance; release recordings after use.
3. Each detector at defaults and along its [sweep](shared-contracts.md#threshold-sweeps); each
   recipe via `run_recipe(config, recording)`. Every call inside `warnings.catch_warnings()` with
   `simplefilter("ignore")` and `try/except Exception` (broader than `simulation_study.py:108-118`'s
   `ValueError`: a recipe can raise others, e.g. a required baseline containing no valid spikes),
   recording `f"{type(error).__name__}: {error}"`.
4. Per method × setting × expression in (`ripple`, `sharp_wave`, `burst`, `network`) ×
   `minimum_iou` in `MATCH_IOU_LEVELS`:
   `match_events(truth_windows(events, 0.1, expression), detected, minimum_iou=minimum_iou)`,
   boundary errors at 0.25 and 0.5 via `boundary_errors`, signed and absolute medians, the
   metrics row.
5. Return the session's truth, units, events and metrics frames; the parent collects them per
   condition and, when a condition's sessions are all done, writes every table into
   `conditions/<condition_id>.partial/`, then `done.json` (row counts and SHA-256 of every
   other file there), then renames the directory to `conditions/<condition_id>/`. A condition
   writes nothing outside its own directory.
6. After the last condition, `--combine` builds `combined/` by concatenating the finished
   conditions' tables; it can be re-run at any time and is never an input to `--resume`.

**Run specification and resume.** A new run writes `run_spec.json` before any session: the
resolved parameters of every condition (after CLI overrides such as `--duration`), the
replicate count and seeds, every method and setting with its resolved options, and the
package version and git commit. `--resume` rebuilds the specification from its arguments and
stops with the differing keys if it is not equal to the saved one; it never reuses outputs
made under another specification. A condition counts as finished only when
`conditions/<condition_id>/done.json` exists and the directory's files match its counts and
hashes. A leftover `conditions/<condition_id>.partial/` (an interrupted write), or a
condition directory that fails that check, is deleted and the condition runs again; only
that condition's own directory is ever deleted.

`--smoke` runs the reference condition, 1 replicate, 1 worker; prints per-method runtime,
simulate time, peak resident memory (`resource.getrusage(RUSAGE_SELF).ru_maxrss`, bytes on macOS
and kilobytes on Linux), rows per table and bytes written; and extrapolates total runtime and
output size for the full grid at the requested worker count. Decision rules:

- one reference session takes more than 5 minutes on one core → halve `duration_s` and double
  the replicates;
- extrapolated events size (all conditions' `events.csv.gz`) passes 2 GB → write sweep events
  only for the reference condition,
  metrics only elsewhere;
- workers = min(requested, free cores − 1, ⌊0.7 × available memory / peak memory per session⌋).

As implemented in phase 4 (2026-09-27), where the plan left a choice open or the maintainer
decided during the phase:

- **Seeding.** `default_rng(session_seed(k))` draws four int64 seeds at once, one per stage
  (schedule, events, non-events, render; `conditions.stage_seeds`), so a factor that changes
  one stage's draw count leaves the others' streams alone (`event_rate=0.6` keeps the
  reference's schedule, rates and non-events). The coupled-strength alternative redraws each
  burst's participation by design, so "the reference's burst participants" holds for the other
  five alternatives only; coupled shares the per-unit draws, and the test checks that.
- **Reference revisions**, both chosen by the maintainer after the first validation report and
  recorded in `conditions.REFERENCE_REVISIONS` (the report lists them): the ripple span range
  from (0.03, 0.15) to (0.042, 0.21) s (the smallest 0.1-step scale, 1.4, whose calibrated swr
  RMS-duration median lies inside the 40-60 ms target by two standard errors; 33.3 ms at 1.0,
  42.7 at 1.4), then the burst gain from 40 to 34 (the whole gain whose reference
  pyramidal-rate ratio is closest to Csicsvari et al. 1999's 8.6; 8.64 at 34), since the
  longer bursts had put that ratio at the target's upper bound. Calibrations used replicates
  20000-20019, never the report's, and no detector. The package defaults are unchanged; the
  `draw_network_events` docstring says the benchmark's reference differs.
- **Validation.** 20 replicates (10000-10019) per condition, not 5 (the maintainer's decision:
  with 5, the pyramidal gain under coupling failed by sampling noise); a report with fewer
  cannot back a run. `measurements.csv` (37 MB) is git-ignored: its hash stays in spec.json, a
  present copy must match, none is required. The report also records the validator's source
  hash, per-session runtime and memory, and the revisions; rendering checks carry explicit
  applicability, and a check that measured nothing fails. Report v1: ready.
- **Runner.** Only the method call is guarded: benchmark input preparation and scoring raise.
  Warnings are recorded in `warnings.csv`. Condition ids are parsed by longest known id
  (crossed ids contain a comma) and then looked up, never re-joined: a first full run lost
  `ripple_snr=high` and `participation=low` that way and was discarded. The git commit carries
  a dirty flag; resume refuses a dirty or unknown commit. `--combine` stages `combined/` and
  records included and missing conditions. Recipe recordings take an int16 copy of the counts
  (identical results, 7 s less per session).
- **Measured** (reference session, 600 s, a machine shared with other work): 74 s and 3.56 GiB
  per session, the heaviest condition 5.0 GB; validation 11 s per session; the full run
  (43 conditions, 440 sessions, 5 workers) 2 h 23 min wall, 8.9 CPU hours, 2.3 GB
  written; 135 explicit failures (single-channel condition, Yu under brown noise).
- **Not changed, noted:** sharing each detector's threshold-independent trace across its sweep
  points would save about 20% of detection time but needs a package API.

## Operating curves

Per detector, condition, expression and `minimum_iou`, pooled over sessions: at each setting,
`recall = Σ n_matched / Σ n_reference` and `fp_rate = Σ unmatched / Σ non-event minutes`.
Curves are drawn in threshold order. For a target FP rate `r` in `(0.5, 1, 2, 5)` per minute:

```python
def at_fp_rate(curve, target, floor, columns):
    """curve: one row per setting of one detector, condition and expression, in threshold
    order, with `fp_rate`, `recall` and the metric `columns`. Settings with the same floored
    FP rate keep one row, the best recall (ties: the first in threshold order), whole; every
    column is then interpolated linearly in log FP rate between the same two bracketing rows,
    so recall and the boundary errors describe the same settings. NaN outside the range.
    floor: half of 1 / total non-event minutes, the estimate's resolution, used for 0."""
    x = np.log(np.maximum(curve.fp_rate.to_numpy(float), floor))
    ranked = curve.assign(_x=x, _order=np.arange(len(curve)))
    ranked = ranked.sort_values(["_x", "recall", "_order"], ascending=[True, False, True])
    points = ranked.drop_duplicates("_x", keep="first")
    xs, t = points._x.to_numpy(), np.log(target)
    if not (xs[0] <= t <= xs[-1]):
        return pd.Series(np.nan, index=list(columns))
    return pd.Series({c: float(np.interp(t, xs, points[c].to_numpy(float))) for c in columns})
```

Settings with equal FP rates (several at 0, floored to the same value) collapse to one setting,
the best recall, so `np.interp` gets strictly increasing `x`. The setting is chosen once, by
recall, and its whole row is kept: median onset and offset errors at the target come from the
same settings as the recall, never from a per-column maximum (which, on a curve where two
settings share an FP rate, paired the better setting's recall with the other's +20 ms onset
error instead of its own -10 ms). Recipes are points `(fp_rate, recall)` on their primary expression's axes, drawn over the
curves of the detectors with the same primary expression.

The curves describe performance on the sessions they are read from. A threshold the benchmark
recommends, or a single tuned number per method, is chosen on calibration replicates and its
recall, false-positive rate and boundary errors are reported on the held-out replicates, beside
the descriptive curve. Membership is by replicate id, the same in every condition: even ids
calibrate, odd ids are held out (`is_held_out(replicate) = replicate % 2 == 1`), so the
reference's 20 split 10/10 and another condition's 10 split 5/5, and a replicate never
calibrates in one condition while being held out in another. Replicate `k` keeps its seed in
every condition, so held-out comparisons across conditions still pair by replicate (odd ids
0-9 are shared by every condition).

## Bootstrap and permutation tests

```python
def paired_bootstrap(frame, statistic, *, key, n_resamples=2000, seed=0, level=0.95):
    """Resample the values of `key` with replacement, one draw shared by every row with that
    value: key="session_id" within one condition (every method shares the draw), and
    key="replicate" whenever the statistic compares conditions (a replicate's sessions in every
    condition share a seed, so drawing the replicate keeps the pairs together). `frame` has
    `session_id` and `replicate` columns; statistic(frame) returns a Series; returns estimate,
    low, high per Series entry (percentile interval)."""
    rng = np.random.default_rng(seed)
    values = frame[key].unique()
    groups = {v: g for v, g in frame.groupby(key)}
    estimate = statistic(frame)
    draws = []
    for _ in range(n_resamples):
        pick = rng.choice(values, size=values.size, replace=True)
        # a value drawn twice is two draws: relabel both ids so per-session and per-replicate
        # grouping keep both copies
        resampled = [
            groups[v].assign(
                session_id=groups[v].session_id.astype(str) + f"#{k}",
                replicate=groups[v].replicate.astype(str) + f"#{k}",
            )
            for k, v in enumerate(pick)
        ]
        draws.append(statistic(pd.concat(resampled, ignore_index=True)))
    draws = pd.DataFrame(draws)
    alpha = (1 - level) / 2
    return pd.DataFrame({"estimate": estimate, "low": draws.quantile(alpha),
                         "high": draws.quantile(1 - alpha)})


def sign_flip_test(differences, *, n_resamples=10_000, seed=0):
    """Two-sided paired test of mean difference 0 over sessions; exact below 17 sessions,
    else Monte Carlo with the (k + 1) / (n + 1) estimate, which is never 0. Differences must be
    finite: the caller pairs sessions where both values exist and reports how many were dropped.
    With no pairs there is no test: NaN."""
    d = np.asarray(differences, dtype=float)
    if not np.isfinite(d).all():
        raise ValueError("sign_flip_test needs finite paired differences; drop incomplete pairs first.")
    if d.size == 0:
        return float("nan")
    observed = abs(d.mean())
    if d.size <= 16:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=d.size)))
        null = np.abs((signs * d).mean(axis=1))
        return float((null >= observed - 1e-12).mean())
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(n_resamples, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float(((null >= observed - 1e-12).sum() + 1) / (n_resamples + 1))
```

Both go in `examples/benchmark/analyze.py`; phase 6 imports `paired_bootstrap` from there.
Settings: 2000 resamples, seed 0, 95% percentile intervals. Across conditions (robustness) the
resampling unit is the replicate, shared by the conditions compared (common random numbers).
Mixed models are not used (overview Open Question 1).

As implemented in phase 5 (2026-09-28), where the plan left a choice open or the maintainer
decided during the phase:

- **Point inventories.** `davidson_2009_ripples` and `wu_2014_ripples` return ripple peaks (the
  catalog's output "ripple peaks"), which interval matching can never credit. The maintainer
  chose to score them by one-to-one peak containment in the method's primary-expression windows
  (windows by end, earliest unused peak: optimal; checked against a brute force), recall,
  precision and false positives per minute only, in their own rows, excluded from every interval
  analysis. `lee_2002`'s single-sample events keep the interval rule.
- **Statistics.** Intervals within a condition come from the replicate-weighted bootstrap
  (`resample_weights`), identical draw for draw to `paired_bootstrap`, which stays public and
  now refuses a statistic indexed by the draws. Comparisons across conditions use only the
  replicates on which a method ran in every condition compared (`n_dropped`). Paired timing and
  method differences carry the interval and sign-flip test of per-session values beside the
  pooled median. Error correlations use the pair's shared primary expression, network beside.
  Every "A differs from B" statement carries a paired interval and sign-flip p; rank moves are
  labelled descriptive.
- **Failures.** A method that never ran keeps a row with its failures in every per-method table;
  model sensitivity separates "failed" from "unattainable"; finite counts are reported per
  statistic.
- **Additions.** `appendix_expressions` and `appendix_curves_<expression>` score every method
  against every expression; `operating_differences` gives paired detector differences at the
  targets; files are split to stay under 1 MB. `trends.md` and `spot_checks/` are hand-written and
  survive rebuilds.
- **Run v1** (analysis 3.1 min, 3.95 GB): the maintainer chose five stated trends (local-ripple
  order reversals; Carey's dependence on unit count and slow field; long-event recipes; the
  participation selection effect; no supported Kay-Roumis order), no results summary in the
  package README, and to keep `noise_type=brown` but flag it as confounded: ripples are sized to
  the ripple-band noise, brown noise makes that 25 times smaller, and EMG and spike-leakage
  artifacts keep absolute amplitudes, so they dominate every LFP detector there. **Follow-up for
  a later simulator version:** size EMG and spike leakage against the noise, as fast gamma is,
  then revalidate and rerun.

As implemented after an external statistical review (2026-09-28), the maintainer approving:

- **One statistic for estimate, interval and test.** A change between conditions
  (`paired_changes`: robustness, crossed cells, model sensitivity's measures; and model
  sensitivity's recall at a target) was reported as a pooled value with a bootstrap interval of
  that pooled value, but tested by a sign flip of the mean of per-replicate changes, which can
  differ in sign (Carey under coupled strengths at 0.5 per minute: change -0.0145, p 0.645).
  `swap_test` now tests the pooled change itself: each replicate's two sessions are
  exchangeable under the null, so the change is pooled again, through the same `Pool`
  (`_swap_pool`: each unit's rows twice, as they are and moved to the partner condition,
  weighted `1 - s` and `s`), under every exchange up to 16 replicates, else 10,000 random
  ones (seed 0, `(k + 1) / (n + 1)`); patterns whose statistic is undefined are left out.
  Carey's p is now 0.064. `n_paired` is the replicates pooled. `sign_flip_test` stays for
  per-session means.
- **Audit of the other p-values.** `method_differences` and `paired_timing` test the mean of
  per-session values that they report as the estimate (consistent; the pooled median beside it
  carries no test); matching sensitivity, candidate ranks, the attribution's one factor at a
  time and the orders carry no p-value; candidate trends copy their tables'.
- **Operating differences.** Their difference is read off pooled curves, but was tested by a
  sign flip of per-session read-offs, and a swap of two detectors' sessions has no pooled
  counterpart when their sweeps have different settings (15 of the 21 pairs). The maintainer
  chose the bootstrap-inverted p-value (`bootstrap_p`): `2 min(share of draws <= 0, share of
  draws >= 0)`, capped at 1, over the defined draws (`n_draws`, replacing `n_paired`) of the
  same resampled pooled differences as the interval, so estimate, interval and p-value are one
  statistic for every pair and p < 0.05 when the 95 % interval excludes 0 (to within one draw,
  where the interpolated bound can fall either side). Approximate, not an exact randomization
  test; a p of 0 means no draw on one side. On v1, 22 of 29 compared rows have p < 0.05 (23
  by sign flip; three rows changed side); Roumis minus Kay at 1 per minute went from 0.123 to
  0.29, its interval still holding 0.
- **Rebuilds keep other commands' results.** `analyze.py` rebuilt `results/<run>` keeping
  only `trends.md` and `spot_checks/`, deleting the attribution's `attribution/`; `KEPT` now
  lists all three and a rebuild copies each over.
- **Reversals.** `reversed` required only opposite point estimates. It now requires a supported
  reference order, opposite signs and an alternative interval excluding 0; opposite point
  estimates whose alternative interval holds 0 are `point_reversed`, counted in `summary.md`
  among the orders that lose support. On v1: `spatial_profile=local` 13 reversals (16 before),
  `noise_modulation=varying` 0 (1 before, interval -0.089 to +0.168).

## Attribution

Module `examples/benchmark/attribution.py`. This phase defines experimental `Step`,
`ThresholdCore` and `Pipeline` dataclasses and `run_pipeline`, using public package
primitives. `Step` names a transform/state/postprocessing operation and hashable
parameters; `ThresholdCore` specifies its signal, normalization and threshold/bound
rules; `Pipeline` combines a core and ordered postprocessing steps. Implement only
operations needed by the factor templates below. Do not copy paper-specific bodies.
A named-method call remains `run_recipe(config, rec)`; experimental results have
separate identifiers. Methods with no verified template are fixed points, reported
alongside attribution results. The installed API remains the source of their events.

**Families.** `spikes` (signal source `rate`, `counts`, `per_cell_rate`) and `lfp` (every other
source), each analyzed separately; the factor spaces differ.

**Templates.** A template is a flat frozen dataclass per family (`SpikeTemplate`,
`LfpTemplate`), one field per factor below, compiled to a `Pipeline` in one fixed order:

1. `ThresholdCore` with signal `(rate(units, smoothing_sigma),)` for `spikes`, or
   `(mean_envelope(band, channels),)` plus `square` when `trace = "squared"` for `lfp` (the
   `lfp` template's smoothing goes to `smoothing_sigma` of the core); `restrict_to = state` when
   the state level is a trace restriction; `normalization_period`, `threshold`, `bound_threshold`,
   `minimum_event_duration`, `maximum_duration`; `speed` as `speed_rule` and `speed_threshold`;
   `merge_gap` as `close_event_threshold` with `close_event_rule = "merge"`.
2. Post steps: `active_units(units, minimum_active_units)` when above 0, then the state when it is
   an overlap or containment step, then the coincidence step.

| Factor | Family | Kind |
| --- | --- | --- |
| `units` | spikes | categorical |
| `band`, `channels`, `trace` | lfp | categorical |
| `smoothing_sigma` | both | continuous |
| `normalization_period` | both | categorical |
| `threshold` | both | continuous (SD; recipes with `normalization_method="none"` contribute no value) |
| `bound_threshold` | both | categorical |
| `minimum_event_duration` | both | continuous |
| `maximum_duration` | both | categorical |
| `merge_gap` | both | continuous |
| `speed` | both | categorical (rule and threshold together) |
| `minimum_active_units` | spikes | integer |
| `state` | both | categorical (an explicit state selection and how it is applied) |
| `coincidence` | both | categorical (a whole post step with its partner) |

Rule for ranges and levels: **continuous range = [min, max] over verified method
templates in the family; categorical levels = their distinct values**, including 0
or "none" only when a represented method omits that step. An explicit mapping from
`config_id` to a template records the source of each factor and benchmark assumption;
do not infer decomposition from the function name or the survey CSV alone.

A template represents a method only after reviewing operation order, normalization,
native grids and boundary rules, and comparing its events exactly with
`run_recipe(config, rec)` on positive controls, relevant edge cases and all `K`
reference sessions. Matching empty outputs alone is insufficient. Store exclusions
and their reasons; fixed-point results come from public calls. Pin the eligible set
and factor-space ranges. Fewer than eight eligible methods in a family triggers the
existing stop/report rule before Sobol or Shapley. Factor-space membership cannot be
expanded by changing a public method to fit the template.

**Reference configuration** per family: each factor at the median (continuous, integer; for
integers rounded down) or mode (categorical, ties to the first level in configuration order) over the
family's in-space recipes.

**Outputs `Y`** per configuration, each averaged over the `K = 5` reference-condition sessions
(replicates 0-4, so session noise is not attributed to factors): F1 against the family's
expression, `burst` for `spikes` and `ripple` for `lfp`; the same F1 against `network`; events per
minute; median onset error at 0.25; Jaccard against the reference configuration's events. One
expression per family, not each recipe's primary expression: a variance decomposition needs one
output, and a spike configuration with a ripple coincidence step is still scored on the burst
(with the `network` F1 beside it).

**One factor at a time.** From the reference, each factor set to each level (continuous: 5 evenly
spaced values over the range), every other factor at reference; report `ΔY`.

**Sobol indices.** `N = 256`, `d` factors; `A`, `B` from `scipy.stats.qmc.Sobol(d=2 d, scramble=True, seed=0)`
split in two; a uniform `u` maps to a continuous factor by `low + u (high - low)`, to a categorical
or integer factor by `levels[min(int(u * n), n - 1)]`. `AB_i` = `A` with column `i` from `B`.

```python
def sobol_indices(y_a, y_b, y_ab):
    """First-order (Saltelli et al. 2010) and total (Jansen 1999) indices.
    y_a, y_b: (N,); y_ab: (d, N)."""
    variance = np.var(np.concatenate([y_a, y_b]), ddof=1)
    # centred on the mean of every finite output, A, B and AB together (as SALib
    # does), so one missing AB output leaves only its own index missing
    everything = np.concatenate([y_a, y_b, y_ab.ravel()])
    finite = everything[np.isfinite(everything)]
    centre = finite.mean() if finite.size else np.nan
    first = np.mean((y_b - centre) * (y_ab - y_a), axis=1) / variance
    total = 0.5 * np.mean((y_a - y_ab) ** 2, axis=1) / variance
    return first, total
```

The first-order product is of centred outputs. Uncentred, adding a constant `c` to every
output adds `c mean(y_ab - y_a) / variance` to each first-order index, a term zero only in
expectation, so the estimate moves with the outputs' origin (an F1 near 0.5 is far from 0) and
its interval widens with it. Each bootstrap resample is centred on its own mean; the total is a
difference of outputs and needs none. (This design first had the uncentred product; see the
note on run v1.)

Intervals: bootstrap over the `N` sample rows (1000 resamples, percentile), not over sessions:
each `Y` already averages the `K` sessions. Cost `N (d + 2)` configs × `K` sessions; phase 6
smoke-tests it.

**Shapley decomposition** between two in-space configs `a` and `b` differing in factor set `D`.
`v(S)` = `Y` of `a` with the factors in `S` taken from `b`; primary `Y` = Jaccard of that
config's events with `b`'s events, so `v(∅) = J(a, b)` and `v(D) = 1`; also run with F1.

```python
def shapley(value, factors, *, exact_up_to=8, n_permutations=128, seed=0):
    """value(frozenset) -> float, memoized by the caller. Exact over subsets for
    len(factors) <= exact_up_to (standard errors 0), else Monte Carlo over
    permutations. Returns ({factor: phi}, {factor: standard error})."""
    factors = tuple(factors)
    n = len(factors)
    if n <= exact_up_to:
        phi = dict.fromkeys(factors, 0.0)
        for i in factors:
            others = [f for f in factors if f != i]
            for k in range(n):
                weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                for subset in itertools.combinations(others, k):
                    s = frozenset(subset)
                    phi[i] += weight * (value(s | {i}) - value(s))
        return phi, dict.fromkeys(factors, 0.0)
    rng = np.random.default_rng(seed)
    contributions = {i: [] for i in factors}
    for _ in range(n_permutations):
        s = frozenset()
        for i in rng.permutation(factors):
            contributions[i].append(value(s | {i}) - value(s))
            s = s | {i}
    phi = {i: float(np.mean(c)) for i, c in contributions.items()}
    error = {i: float(np.std(c, ddof=1) / np.sqrt(n_permutations)) for i, c in contributions.items()}
    return phi, error
```

Checked numerically while planning: the exact path gives `φ_i = w_i` for an additive `v` and
satisfies efficiency; on a 10-factor toy (`v(S) = |S|^1.5 + 2·[x0, x1 ∈ S]`, values ≈ 3.2-4.2) the
Monte Carlo path's largest error was 0.16 at 128 permutations, 0.07 at 1000, 0.03 at 4000. The
standard errors are reported beside every Monte Carlo φ.

Efficiency check (a test): `sum(phi) == v(D) - v(∅)` to 1e-9 for the exact path. The Sobol
estimator above reproduced the analytic Ishigami indices to within 0.001 at `N = 2^13` when
checked during planning (the values are in phase 6's test).

**Pairs.** Each in-space recipe against its family reference, plus the 10 in-space pairs with the
lowest Jaccard at the reference condition. Not all pairs: 300 pairs × 256 subsets × 5 sessions is
past a workstation.

As implemented in phase 6 (2026-09-28), where the plan left a choice open or the maintainer
decided during the phase:

- **Coverage.** Templates reproduce a method only when their events equal its public call's
  exactly (bounds, not only counts) on two 60 s edge sessions (a 20 ms gap cutting a
  sharp-wave ripple; a Unix clock origin) and the K = 5 reference sessions of run v1, with at
  least one event found. In-space: `spikes` 15 configurations, 14 distinct templates
  (`grosmark_2016` equals `yang_2024` and counts once); `lfp` 4; 58 fixed points, each with
  its reason and public-call Ys. Spike templates use 1 ms population bins and one-sample
  minimum time above threshold (no factor for either), so other grids and run minima are
  fixed points.
- **Sensitivity of the verification** (`<family>_sensitivity.csv`): each template value is
  perturbed (continuous x0.9/x1.1, other levels, integers +-1); 17 of 116 values are not
  exercised by the verification sessions, mostly duration limits the simulated events never
  reach and `liu_2023`'s threshold (decided by its coincidence with Long's sharp-wave
  ripples). They still set factor ranges and the reference; the table says which.
- **Maintainer decisions.** The bound is the factor `bound_fraction` (bound / threshold),
  so every Sobol row and Shapley subset is valid (bound above threshold made 12.5% of Sobol
  rows and some Shapley subsets invalid). The `lfp` family, below the minimum of eight, runs
  every analysis anyway under `--below-minimum`, every output labelled "rests on 4 methods,
  below the design's 8; the maintainer chose to run it".
- **Run v1.** Sobol at N = 256 (spikes 3584 configurations, lfp 2048); Shapley against the
  family reference for every distinct template and the ten lowest-agreement pairs; wall 9 min
  52 s (spikes) and 6 min 42 s (lfp) at 8 workers after caching binned traces and pipeline
  stages and grouping configurations (0.65 to 0.077 s per spike configuration, outputs
  identical). Indices are read with their intervals.
- **Centred first-order estimates** (after an external review, 2026-09-28). The first-order
  estimator of run v1 used uncentred outputs (the code above now centres them), so its
  estimates moved with the outputs' origin and their intervals were inflated: spike F1
  `bound_fraction` 0.416 (-0.109, 0.844) became 0.137 (0.051, 0.226), the LFP F1
  threshold's interval width 5.81 became 0.29, and the spike F1 first-order estimates,
  which summed to 1.33, sum to 0.96. The spike first-order estimates summing past 1 was this
  bug, not noise at N = 256. Both families' Sobol tables were rebuilt from the same
  configurations (`--analysis sobol`); the raw rows came out identical and the totals are
  unchanged.

## Rates and participation

- **Rates by state.** Per method, events per minute in rest and in running (event assigned by its
  `peak_time`, else midpoint), against the true rates (network events at rest; theta bursts in
  running as the non-event rate).
- **Participation bias.** For truth events with a burst: the distribution of latent
  `n_participants` (recruited cells, some of which stay silent) among those a method matched,
  against all truth events (ratio of means with bootstrap interval; `scipy.stats.ks_2samp`
  statistic). Reported on its own, never subtracted from an observed count.
- **Boundary effect on counts.** For matched pairs against an expression: observed active units
  within the detected bounds minus observed active units within the matched truth window of that
  expression (fraction 0.1, from `truth_counts.csv.gz`), both from
  `count_spikes_in_events` with the same unit selection (`n_active_units`: all units;
  `n_active_principal`: place and pyramidal units). Interneurons, background spikes and silent
  recruits count on both sides, so the difference is zero when the bounds agree.
