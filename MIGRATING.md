# Migrating from 1.x to 2.0

2.0 changes results as well as calls. The same recording gives different
events, so detect again rather than mixing events from the two versions:
events stored from 1.x follow the 1.x rules. Pin `ripple-detection>=2,<3`, and
record the version with the events you detect. Everything here is relative to
1.7.1; [CHANGELOG.md](CHANGELOG.md) lists every change.

A pipeline that stores parameters, such as a database table keyed by a parameter
set, gets different events from the same stored parameters. Put the package
version in the parameter set's name or key, so events detected under 1.x and 2.0
are never stored as the same result.

## Calls to change

Each of these fails in 2.0 with a message that names its replacement, so code
written for 1.x, by a person or by a language model trained on 1.x examples,
finds out at the first call.

| 1.x | 2.0 |
|---|---|
| `filter_ripple_band(lfp)` | `filter_ripple_band(lfp, sampling_frequency)`: the rate is required; 1.x assumed 1500 Hz whatever the data's rate |
| `Kay_ripple_detector(time, lfps, speed, fs, 4.0, 0.015)` | tunables after `sampling_frequency` are keyword-only: `..., fs, speed_threshold=4.0, minimum_duration=0.015` |
| `normalization_time_range=(start, end)`, on any detector or `normalize_signal` | `normalization_mask=(time >= start) & (time <= end)` |
| `multiunit_HSE_detector(..., use_speed_threshold_for_zscore=True)` | `normalization_mask=speed <= speed_threshold`. The flag used `speed < speed_threshold`; the mask also counts a speed exactly at the threshold as immobile, as the speed rule always has |
| `normalize_signal(data, time, method, mask)` | `normalize_signal(data, method, normalization_mask)`: the `time` argument is gone |
| `pink(N, state=np.random.RandomState(seed))`, and `white`, `brown` | `pink(N, rng=seed)`, a seed or a `numpy.random.Generator`. A `RandomState` passed by name or position to a noise function or simulator raises |

Apart from these, the Kay, Karlsson, Roumis and HSE signatures only add keywords
(`maximum_duration` on each, `minimum_active_units` on HSE). A pipeline that
already passes every tunable by keyword and filters with an explicit rate needs
no code change, unless its inputs now raise: 2.0 rejects inputs that 1.x turned
into a wrong or empty result (see [Inputs that now raise](#inputs-that-now-raise)).

## Changes without a hint

Look for these yourself:

- The result column `max_thresh` is `max_sustained_zscore`: the same quantity,
  computed exactly (see below). Code that reads `max_thresh` fails with pandas'
  own `KeyError` or `AttributeError`, which does not name the new column, and
  `events.get("max_thresh")` returns `None`. Selecting columns by position also
  shifts: the fourth column is now `n_samples`.
- `simulate_LFP` draws pink noise by default, where 1.x drew brown. Pass
  `noise_type="brown"` for the old signal. Brown noise has almost no ripple-band
  power, so a ripple of any size dominated the band and every detector found
  every ripple; on pink noise a ripple of `ripple_snr` 1 to 4 is a real test.
- `exclude_close_events` and `exclude_movement` return an empty array of shape
  `(0, 2)` when nothing is left, where they returned `[]`.
- Every random draw goes through `numpy.random.default_rng`, so a seed gives
  different noise than it did through 1.x's `RandomState`.

Units are as in 1.x: seconds, hertz and cm/s. A duration given in milliseconds
(`minimum_duration=15`) now raises and says what to pass.

## Inputs that now raise

2.0 rejects, with a message naming the problem, inputs that 1.x turned into a
wrong or empty result or a failure about something else:

- `time` that is not increasing (the event and speed lookups assume order),
  that holds NaN, or whose timestamps mostly repeat (the minimum duration
  became one sample, so every single-sample crossing was an event).
- An input whose every sample holds a NaN; the message names the channels that
  hold none.
- A normalization mask that selects no samples, is not boolean (a forgotten
  comparison made every nonzero speed count as immobile), or is 2-D (it pooled
  every channel's statistics into one).
- An event array that is not `(n, 2)`, in the event helpers, which reshaped an
  18-column frame into pairs.
- A series holding NaN, or a negative threshold, in `segment_boolean_series` and
  `threshold_by_zscore`.
- `multiunit` of the wrong shape (1.x raised NumPy's `AxisError`), or holding
  negative or fractional values: counts per sample, not a rate.
- A channel that is constant over the valid samples, as a dead or disconnected
  channel is.
- A tunable that is NaN, negative, reversed or in the wrong unit: a NaN or
  negative threshold, duration, gap or speed limit; a non-positive
  `sampling_frequency`; a smoothing width of zero or of a second or more; a
  minimum duration of a second or more, or a ceiling or gap longer than 10 s
  (each is milliseconds given as seconds, and the message gives the value to
  pass); or an infinite ceiling or gap (`None` is the way to have no ceiling).
  Each disabled a criterion or emptied the result silently. `minimum_duration=0`
  still means no minimum.
- In `simulate_LFP`, a ripple time outside `time`, a duration that is not
  positive, a frequency outside the Nyquist range, or a NaN size, each of which
  returned an all-NaN, empty or aliased signal.
- `get_Kay_ripple_consensus_trace` with a one-dimensional input raises a
  `ValueError` that says to reshape it, where 1.x raised NumPy's `AxisError`.

## New output columns

Every result has `n_samples`, `clipped_start` and `clipped_end`. `n_samples` is
the fourth column, so code that reads columns by position shifts.

- `n_samples` is the number of samples in the event, first to last inclusive:
  the quantity the duration limits test. `duration` stays elapsed time, one
  sample interval less than `n_samples` spans, so an event of exactly the
  minimum count has a `duration` below `minimum_duration` by half to one and a
  half intervals (23 samples span 14.67 ms at 15 ms and 1500 Hz).
- `clipped_start` and `clipped_end` flag an event cut off by missing data or
  the recording edge.

## Why your events differ

The changes that move results on the same recording, in rough order of how much.

**Checking that the difference is expected.** On a recording without gaps, most
of the difference is the minimum duration's sample count. Pass a minimum one
sample longer, `minimum_duration=(round(0.015 * sampling_frequency) + 1) /
sampling_frequency` for 1.x's 15 ms, and every other argument as before: 1.x's
events come back. On 300 s of simulated data at 1500 Hz this reproduced the 1.x
events of Kay, Karlsson and the HSE detector exactly. Remaining differences come
from gaps, NaN, speed or the filter rate, described below.

**The minimum duration counts samples.** A run qualifies when it holds at least
`round(minimum_duration * rate)` samples, rounded half up from the median
timestamp step, where 1.x compared timestamps. At 1500 Hz a 15 ms minimum needs
23 samples, not 24. On 300 s of pink noise filtered to the ripple band this
gives about 20 % more events at a `zscore_threshold` of 2.0 to 2.5: Kay 25 %
more, Karlsson 20 %. Every duration limit follows the same inclusive rule.

**Missing samples split the recording.** A sample is missing when any channel
of any signal is NaN or infinite, and a step in `time` larger than 1.5 times the
median step ends a block as a missing sample does. Every step of every detector
runs within a block, so nothing is smoothed, thresholded or merged across a gap
and no event spans one; an event cut off by a gap is kept and flagged. In 1.x
the Kay, Karlsson and Roumis detectors dropped the NaN rows and treated the rest
as continuous, so an event could span a gap, and the HSE detector smoothed
across a NaN, which blanked its rate for a kernel width around each one (361
samples at 1500 Hz for one missing count). A block too short for an event of
`minimum_duration` is treated as missing with a warning that gives its sample
ranges, and a detector with no block left raises. Data without gaps gives the
same output.

**A NaN in speed is an unknown speed**, not a missing sample, so a tracking
dropout no longer removes LFP samples or cuts a ripple in two. An event whose
first or last sample has unknown speed fails the speed rule;
`speed_threshold=np.inf` turns the rule off, unknown speed included. The speed
statistics skip unknown values. Speed that is NaN everywhere raises unless the
rule is off.

**The sampling rate sets the filter.** 1.x's `filter_ripple_band` defaulted to
`sampling_frequency=None`, which applied the 1500 Hz kernel to data at any rate
without a check: 1000 Hz data was filtered to 97-170 Hz and gave twice the
events. A given rate other than 1500 Hz raised below 1200 Hz and otherwise
warned and applied the same kernel. 2.0 designs a filter for any rate but 1500
Hz, sized by Kaiser's estimate, where 1.x's `ripple_bandpass_filter` used 101
taps at any rate (at 30 kHz its stopband reached -1 dB). At 1500 Hz the shipped
kernel is used as before.

**The filter no longer rings across gaps.** 1.x joined the runs of finite
samples and filtered them as one, so the step between the two sides of a gap
rang through the filter: on noise with a 5 s gap, Kay reported a 5 s event with
a z-score near 10. 2.0 filters each run on its own; a run as short as the kernel
can be filtered (318 samples, 212 ms, at 1500 Hz, where 1.x needed 955 in all),
and a shorter one is returned as NaN with a warning.

**`exclude_close_events` compares with the last event kept.** 1.x compared each
event with the candidate before it, so it removed more than the first event of
each cluster. The detectors' `close_ripple_threshold` and
`close_event_threshold` share the fix; at their default of 0 nothing changes.

**`exclude_movement` looks the speed up per event.** 1.x matched speeds to
events with `np.isin`, which read them in time order: nested events were judged
by the wrong speeds, and a bound between samples discarded every event.

**Normalization.** The mask is restricted to the valid samples, as the data is,
so a mask that selected NaN rows no longer changes the statistics. A zero or
undefined scale raises and names the channel, as does a channel that is
constant over the valid samples. In 1.x a constant trace gave NaN, or zeros
under a mask, and a channel constant over the mask was divided by 1.0, so its
values outside the mask were reported as z-scores: under `median_mad`, one
partly disconnected channel produced a 9 s event with a sustained z-score of
32000. Drop a dead channel before detecting.

**Smoothing keeps its level at the ends.** `gaussian_smooth` renormalizes its
kernel where it runs past either end of the data instead of padding with zeros,
so a trace no longer sags toward zero at the recording's ends or at a gap. Event
bounds away from the ends are unchanged; the z-score statistics move in the
third or fourth decimal, because the normalization no longer includes the
artificially low samples at the two ends.

**Statistics.** `max_sustained_zscore` is the largest z-score sustained for
`minimum_duration`, the highest threshold at which the detector would still find
the event. 1.x approximated it as `max_thresh` by growing a window greedily from
the peak, so its value could fall below `zscore_threshold` (on noise every
Karlsson event did, down to -0.16), and Karlsson and HSE computed it for 15 ms
whatever `minimum_duration` was. Karlsson's per-event statistics come from the
per-sample maximum over the channels' z-scored envelopes, where 1.x used the
mean of the filtered LFP; its events do not change.

**The sampling-rate check.** A `sampling_frequency` that disagrees with the
timestamps by more than 10 % raises, and one that disagrees by more than 2 %
warns, where 1.x warned from 20 %. The rate sets the smoothing widths and the
timestamps set the sample counts, so a 20 % mismatch changed Kay's event count
by a fifth without a word.

## Private names

Names beginning with an underscore were never part of the API, and several were
removed or changed in 2.0, among them `ripple_detection.core._validate_normalization_params`
and `ripple_detection.detectors._get_event_stats`. `ripple_detection.__all__` is the
public API. Code that rebuilt the HSE detector from private helpers to handle missing
samples can call `multiunit_HSE_detector` directly: it now splits the recording at NaN
samples and timestamp gaps, and `normalization_mask` restricts its statistics.
