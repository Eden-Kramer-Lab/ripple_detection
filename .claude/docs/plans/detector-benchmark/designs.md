# Designs

[← back to PLAN.md](PLAN.md)

Per-component algorithms, with code where the implementation is not obvious. Types and table
schemas are in [shared-contracts.md](shared-contracts.md); this file does not repeat them.

- [Parameter sources](#parameter-sources)
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
convention (`_draw_per_ripple`, `simulate.py:221`). "Assumed" means no source; "verify" means the
planner recalls a source but did not check it. Phase 1a replaces each "verify" with a citation
and page, or changes it to "assumed", before merging, and records the table in the
`draw_network_events` docstring's Notes.

| Parameter | Reference | Source |
| --- | --- | --- |
| Sampling rate | 1500 Hz | The shipped filter's rate (`ripplefilter.mat`). |
| Session | 600 s; running bouts 10-20 s separated by 20-40 s of rest | Assumed. |
| LFP | 4 channels + radiatum, pink noise, `noise_amplitude=1.3`, `shared_noise_fraction=0.5` | `simulate_session` defaults. |
| Theta / delta | amplitude 4 each (8 Hz running, 2 Hz rest) | `examples/literature_recipes.py:70-71` (the recipes' session). |
| Units | 60: 40 place (baseline 0.1-0.5 Hz), 10 other pyramidal (0.5-1.5 Hz), 10 interneurons (2-5 Hz) | `literature_recipes.py:56-58`. |
| Event rate | 0.5 per second of rest | Verify: awake and sleep SWR rates of roughly 0.1-1 Hz (Buzsáki 2015, Hippocampus 25:1073). |
| Type mix | swr 0.55, weak_ripple 0.15, burst_only 0.10, ripple_doublet 0.10, sharp_wave_only 0.10 | Assumed. |
| Minimum separation | 0.05 s between events' ±3-sigma spans | Assumed. |
| Ripple span (±3 sigma) | (0.03, 0.15) s | Verify: 30-100 ms typical with a tail (Buzsáki 2015). |
| Ripple skew | fraction of the span after the peak (0.5, 0.7) | Assumed (decay slower than rise). |
| Ripple frequency at onset | (160, 220) Hz | Verify: 140-220 Hz in rat CA1 (Buzsáki 2015). |
| Ripple chirp | decline over the span (0, 30) Hz | Verify: within-event frequency decline (Sullivan et al. 2011, J Neurosci 31:8605; Nguyen et al. 2009). |
| Ripple SNR | swr and doublet (2.5, 6.0); weak_ripple (1.2, 2.2) | Assumed; `simulate_session`'s 4.0 lies inside. |
| Sharp wave | span (0.04, 0.12) s, symmetric; amplitude (3, 8) signal units; centre lag N(0, 0.01 s) from the ripple | Amplitude after `literature_recipes.py:68` (6, above delta); span `simulate_session`'s 0.08 inside; lag assumed. |
| Burst | span = ripple span × (1.0, 1.5); centre lag N(0, 0.01 s); place-cell gain 40; participation swr/doublet (0.2, 0.6), weak_ripple (0.02, 0.1), burst_only (0.2, 0.6) with span (0.05, 0.3) s | Gain `literature_recipes.py:65`; participation verify: 10-30% of CA1 pyramidal cells per SWR (Csicsvari et al. 2000; Ylinen et al. 1995); rest assumed. |
| Other pyramidal units | participate with half the place cells' probability, gain 40 | Assumed. |
| Interneurons | all take part in events with a ripple, gain 3, on the ripple's envelope | Verify: interneurons strongly recruited during ripples (Klausberger et al. 2003). |
| Doublet | 2 ripples (p 0.7) or 3 (p 0.3), centre-to-centre (0.06, 0.12) s, one burst over all | Verify: ripple doublets and triplets (Davidson, Kloosterman & Wilson 2009). |
| Non-event rates (per minute) | spike_leakage 2 (rest), emg 1 (any), fast_gamma 2 (any), theta_burst 6 (running) | Assumed. |
| Spike leakage | 1-3 pyramidal units, 3-8 spikes each at ISI (3, 6) ms, waveform peak 2.0 on one channel | Verify: complex-spike bursts with 3-6 ms ISIs (Ranck 1973); amplitude assumed. |
| EMG | span (0.05, 0.5) s, white noise high-passed at 100 Hz, peak SD 1.5, all channels | Assumed. |
| Fast gamma | 60-100 Hz, span (0.05, 0.15) s, SNR (1.5, 4) against 60-100 Hz noise | Verify: fast gamma 65-140 Hz, distinct from ripples (Colgin et al. 2009; Sullivan et al. 2011). |
| Theta burst | 5-15 place units, gain 10, span (0.1, 0.3) s, running only | Assumed. |

## Event types

Which expressions each type renders (✓), and the distributions that differ from the reference
rows above.

| Type | Ripple | Sharp wave | Burst | Differences |
| --- | --- | --- | --- | --- |
| `swr` | ✓ | ✓ | ✓ | reference |
| `weak_ripple` | ✓ weak SNR | ✓ amplitude × 0.5 | ✓ weak participation | |
| `burst_only` | | | ✓ | burst span (0.05, 0.3) s, centred on the event time |
| `ripple_doublet` | ✓ 2-3 | ✓ one per ripple | ✓ one spanning all ripples | burst ±3-sigma span = first ripple's start to last ripple's end, symmetric |
| `sharp_wave_only` | | ✓ | | |

## Drawing network events

`draw_network_events` in `simulate.py`, public, wrapped with `explain_call_errors` like every
public function there:

```python
@explain_call_errors
def draw_network_events(
    time: ArrayLike,
    *,
    event_rate: float = 0.5,
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
    rng: int | np.random.Generator | None = None,
) -> pd.DataFrame:
```

Returns the [latent event table](shared-contracts.md#latent-event-table) with `n_participants`
0: participants are drawn when the session is rendered, which knows the units. A tuple
`(x, x)` fixes a value; `ripple_chirp=(0, 0)` gives constant-frequency ripples.

Algorithm:

1. `rest` = the recording minus `running_intervals` minus 1 s at each end of the recording.
   Events occur only in `rest`.
2. Event times: a Poisson process of rate `event_rate` on the concatenated rest time, mapped back
   to recording time (draw `n ~ Poisson(rate * rest_duration)`, then `n` sorted uniform positions
   in concatenated rest time).
3. Types: `rng.choice(EVENT_TYPES, size=n, p=...)`, the probabilities normalized.
4. Per event, draw its components (the order of draws is fixed and documented in the docstring:
   types, then per event in time order ripple(s), sharp wave(s), burst):
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
every range `low <= high`, durations and SNRs positive; frequencies below Nyquist; `ValueError`
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
    shared_noise_fraction: float = 0.5,
    noise_type: NoiseType = "pink",
    noise_amplitude: float = 1.3,
    sharp_wave_leak: float = 0.3,
    ripple_leak: float = 0.3,
    interneuron_gain: float = 3.0,
    running_intervals: ArrayLike | None = None,
    peak_speed: float = 30.0,
    theta_amplitude: float = 4.0,
    delta_amplitude: float = 4.0,
    rng: int | np.random.Generator | None = None,
    sampling_frequency: float | None = None,
) -> SimulatedSession:
```

Order of random draws (documented in the docstring): noise (`_correlated_noise`,
`simulate.py:565`, `n_channels + 1` channels, the radiatum last), ripple initial phases, unit
baseline rates, burst participants, non-event randomness (phase 1b), spike counts.

Steps:

1. **Noise.** `_correlated_noise(n_time, n_channels + 1, ...)`.
2. **Ripples.** Band noise SD `sd = filter_ripple_band(noise[:, 0], sampling_frequency=rate).std()`
   (the reference channel, as `_ripple_waveform` does, `simulate.py:433-435`). Per ripple component:

   ```python
   def _render_ripple(time, center, rise_sigma, decay_sigma, f_start, f_end, phase):
       """Unit-peak asymmetric, linearly chirped burst over center -8 rise .. +8 decay."""
       first, last = np.searchsorted(time, [center - 8 * rise_sigma, center + 8 * decay_sigma])
       window = slice(int(first), max(int(last), int(first) + 1))
       t = time[window] - center
       sigma = np.where(t < 0, rise_sigma, decay_sigma)
       envelope = np.exp(-(t**2) / (2 * sigma**2))
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
   today (the shipped kernel at 1500 Hz); a band calls it with `band=band` (fast gamma, phase 1b). The scaled burst is added to channel `c` times `channel_gains[c]` and to the
   radiatum times `ripple_leak`.
3. **Sharp waves.** Asymmetric Gaussian of peak `-amplitude` on the radiatum and
   `+sharp_wave_leak * amplitude` on channel 0 (`_add_sharp_wave_pair`'s convention,
   `simulate.py:741-753`, generalized to rise/decay sigmas by a helper `_half_gaussians`).
4. **Slow field and speed.** `simulate_theta_delta` (`simulate.py:998`) added to every channel
   and the radiatum; `simulate_speed` (`simulate.py:941`), exactly as `simulate_session` does
   (`simulate.py:1280-1295`).
5. **Units.** `unit_types` = `"place"` × 40, `"pyramidal"` × 10, `"interneuron"` × 10 by default
   (in that order). Baseline rates per type drawn uniformly from `baseline_rate[type]`,
   and kept per unit in `SimulatedSession.baseline_rates`.
   Modulation starts at 1 per unit and sample. Per burst component: participants are place units
   with probability `participation` and other pyramidal units with `participation / 2` (a Bernoulli
   draw per unit); `n_participants` is recorded in the returned events table (pyramidal and place
   together); participants' modulation gains `(amplitude - 1) * envelope`, `envelope` the burst's
   asymmetric Gaussian. Per event with a ripple: every interneuron gains
   `(interneuron_gain - 1) * envelope` on the ripple's envelope (the union for a doublet: the
   elementwise max). Spikes: `rng.poisson(rates * step * modulation)`, as `simulate_multiunit`
   (`simulate.py:918-919`).
6. **Session.** `SimulatedSession` with `raw_lfp = lfps[:, 0].copy()`, `sharp_wave_lfp` the
   radiatum, the events table (with `n_participants`), `unit_types`, `baseline_rates`,
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
| `fast_gamma` | `_render_ripple` with `f_start = f_end = frequency`, scaled by `_scale_to_snr(..., band=(60.0, 100.0))` against the SD of channel 0's noise filtered to 60-100 Hz (`filter_ripple_band(noise[:, 0], sampling_frequency=rate, band=(60.0, 100.0))`), added to every channel with the channel gains. |
| `theta_burst` | Picks `n_units` place units; multiplies their modulation by `1 + (amplitude - 1) * envelope`. |

## Truth windows

```python
@explain_call_errors
def truth_windows(table, fraction=0.1, expression=None):
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must lie in (0, 1), got {fraction}.")
    k = np.sqrt(-2.0 * np.log(fraction))
    events = "event_id" in table.columns
    if not events and expression is not None:
        raise ValueError("A non-event table has no expressions; leave expression as None.")
    rows = table
    if expression not in (None, "network"):
        _check_choice("expression", expression, (*EXPRESSIONS, "network"))
        rows = table[table.expression == expression]
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
    # exact one-to-one assignment, per connected component of the eligibility graph
    graph = scipy.sparse.bmat([[None, scipy.sparse.csr_array(eligible)],
                               [scipy.sparse.csr_array(eligible.T), None]])
    n_components, label = scipy.sparse.csgraph.connected_components(graph, directed=False)
    ref_label, det_label = label[: len(ref)], label[len(ref):]
    rows, cols = [], []
    for component in np.unique(ref_label[eligible.any(axis=1)]):
        r = np.flatnonzero(ref_label == component)
        d = np.flatnonzero(det_label == component)
        weight = np.where(eligible[np.ix_(r, d)], iou[np.ix_(r, d)], 0.0)
        i, j = scipy.optimize.linear_sum_assignment(weight, maximize=True)
        keep = weight[i, j] > 0
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
replicate. Draw order within a session: schedule, events, non-events, render.

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
| `event_rate` | `events.event_rate` | 0.2, **0.5**, 1.0 |
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

Totals: 1 reference + 28 one-factor levels + 8 new crossed cells = 37 conditions. Replicates: 20
for the reference, 10 for every other condition = 380 sessions.

## Runner

`examples/benchmark/run.py`, a CLI:

```
uv run python examples/benchmark/run.py --run-name NAME [--conditions all|ID,ID] [--replicates N]
    [--duration S] [--workers N] [--resume] [--smoke]
```

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
4. Per method × setting × expression in (`ripple`, `sharp_wave`, `burst`, `network`):
   `match_events(truth_windows(events, 0.1, expression), detected)`, boundary errors at 0.25 and
   0.5 via `boundary_errors`, the metrics row.
5. Return the session's truth, units, events and metrics frames; the parent appends them and writes
   each condition's files when its sessions are done (so `--resume` skips finished conditions).

`--smoke` runs the reference condition, 1 replicate, 1 worker; prints per-method runtime,
simulate time, peak resident memory (`resource.getrusage(RUSAGE_SELF).ru_maxrss`, bytes on macOS
and kilobytes on Linux), rows per table and bytes written; and extrapolates total runtime and
output size for the full grid at the requested worker count. Decision rules:

- one reference session takes more than 5 minutes on one core → halve `duration_s` and double
  the replicates;
- extrapolated `events/` size passes 2 GB → write sweep events only for the reference condition,
  metrics only elsewhere;
- workers = min(requested, free cores − 1, ⌊0.7 × available memory / peak memory per session⌋).

## Operating curves

Per detector, condition and expression, pooled over sessions: at each setting,
`recall = Σ n_matched / Σ n_reference` and `fp_rate = Σ unmatched / Σ non-event minutes`.
Curves are drawn in threshold order. For a target FP rate `r` in `(0.5, 1, 2, 5)` per minute:

```python
def recall_at(fp_rate, recall, target, floor):
    """Linear in log FP rate between the settings bracketing target; NaN outside.
    floor: half of 1 / total non-event minutes, the estimate's resolution, used for 0."""
    points = pd.DataFrame({"x": np.log(np.maximum(fp_rate, floor)), "y": recall})
    points = points.groupby("x", sort=True).y.max()   # equal rates: the best recall
    x, y = points.index.to_numpy(), points.to_numpy()
    if not (x[0] <= np.log(target) <= x[-1]):
        return np.nan
    return float(np.interp(np.log(target), x, y))
```

Settings with equal FP rates (several at 0, floored to the same value) collapse to their largest
recall, so `np.interp` gets strictly increasing `x`. The same interpolation gives median onset and offset error at the
target. Recipes are points `(fp_rate, recall)` on their primary expression's axes, drawn over the
curves of the detectors with the same primary expression.

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
    else Monte Carlo with the (k + 1) / (n + 1) estimate, which is never 0."""
    d = np.asarray(differences, dtype=float)
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
    first = np.mean(y_b * (y_ab - y_a), axis=1) / variance
    total = 0.5 * np.mean((y_a - y_ab) ** 2, axis=1) / variance
    return first, total
```

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

## Rates and participation

- **Rates by state.** Per method, events per minute in rest and in running (event assigned by its
  `peak_time`, else midpoint), against the true rates (network events at rest; theta bursts in
  running as the non-event rate).
- **Participation bias.** For truth events with a burst: the distribution of latent
  `n_participants` (recruited cells, some of which stay silent) among those a method matched,
  against all truth events (ratio of means with bootstrap interval; `scipy.stats.ks_2samp`
  statistic). Reported on its own, never subtracted from an observed count.
- **Boundary effect on counts.** For matched pairs: observed active units within the detected
  bounds minus observed active units within the matched truth window (fraction 0.1), both from
  `count_spikes_in_events` with the same unit selection (`n_active_units`: all units;
  `n_active_principal`: place and pyramidal units). Interneurons, background spikes and silent
  recruits count on both sides, so the difference is zero when the bounds agree.
