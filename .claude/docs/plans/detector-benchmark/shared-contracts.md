# Shared Contracts

[← back to PLAN.md](PLAN.md)

Types, table schemas and conventions used by more than one phase. Each is defined once here;
phases link by anchor. "Do not weaken" marks an invariant a later phase relies on.

- [Vocabularies](#vocabularies) — event types, non-event types, expressions, unit types
- [Latent event table](#latent-event-table) — `SimulatedSession.events` (phases 1a, 2, 4, 5)
- [Non-event table](#non-event-table) — `SimulatedSession.non_events` (phases 1b, 4, 5)
- [SimulatedSession additions](#simulatedsession-additions) (phases 1a, 1b, 3, 4)
- [Truth windows](#truth-windows) — `truth_windows` (phases 1a, 2, 4, 5)
- [Event inventory input](#event-inventory-input) (phases 2, 3, 4)
- [Matching and pair metrics](#matching-and-pair-metrics) — `match_events`, `EventMatching` (phases 2, 4, 5, 6)
- [Detector comparison table](#detector-comparison-table) — `compare_detectors` (phases 2, 5)
- [Recipe config](#recipe-config) — `RecipeConfig`, public dispatch and input policies (phases 3, 4, 6)
- [Primary expression](#primary-expression) (phases 3, 4, 5, 6)
- [Threshold sweeps](#threshold-sweeps) (phases 4, 5)
- [Conditions](#conditions) (phases 4, 5, 6)
- [Benchmark outputs](#benchmark-outputs) (phases 4, 5, 6)

## Vocabularies

Module-level tuples in `src/ripple_detection/simulate.py`, exported:

```python
EVENT_TYPES = ("swr", "weak_ripple", "burst_only", "ripple_doublet", "sharp_wave_only")
NON_EVENT_TYPES = ("spike_leakage", "emg", "fast_gamma", "theta_burst")
EXPRESSIONS = ("ripple", "sharp_wave", "burst")   # "network" is derived, never stored
UNIT_TYPES = ("place", "pyramidal", "interneuron")
```

Which expressions each event type has is in [designs.md#event-types](designs.md#event-types).
Adding a type later means adding it to the tuple and to that table; no schema changes.

## Latent event table

`SimulatedSession.events`: a `pd.DataFrame`, **one row per rendered component** of a latent
network event. A `swr` has three rows (ripple, sharp wave, burst); a `ripple_doublet` with two
ripples has five (two ripples, two sharp waves, one burst).

| Column | dtype | Meaning |
| --- | --- | --- |
| `event_id` | int64 | The latent event, `0 .. n_events - 1` in order of the earliest component's centre. |
| `event_type` | str | One of `EVENT_TYPES`. |
| `expression` | str | One of `EXPRESSIONS`. |
| `component` | int64 | `0 ..` within (`event_id`, `expression`); above 0 only for the extra ripples and sharp waves of a `ripple_doublet`. |
| `center_time` | float64 | Seconds; the envelope's peak. |
| `rise_sigma`, `decay_sigma` | float64 | Seconds; SD of the half-Gaussian before and after the peak. |
| `amplitude` | float64 | Ripple: target SNR (peak after `filter_ripple_band` over the ripple-band noise SD). Sharp wave: peak deflection on the radiatum channel in signal units (rendered negative). Burst: peak rate gain of a participating unit (multiplier of baseline). |
| `frequency_start`, `frequency_end` | float64 | Hz, ripple rows only (linear chirp over the component's ±3-sigma span); NaN otherwise. |
| `participation` | float64 | Burst rows: probability a pyramidal unit takes part. NaN otherwise. |
| `n_participants` | int64 | Burst rows: place and pyramidal units that took part, drawn and filled in by `simulate_network_session` (`draw_network_events` leaves 0). 0 otherwise. |

Invariants (do not weaken):

- Sorted by (`event_id`, `expression`, `component`); index is a RangeIndex.
- Every component's ±4-sigma span lies inside the recording, so truth windows never need clipping.
- An empty table has exactly these columns and dtypes (`simulate_session` returns it so).
- `center_time` of an event's components may differ (the coupling lags); `event_id` ties them.

## Non-event table

`SimulatedSession.non_events`: one row per non-event (activity that a detector should not report).

| Column | dtype | Meaning |
| --- | --- | --- |
| `non_event_id` | int64 | `0 ..` in order of `center_time`. |
| `non_event_type` | str | One of `NON_EVENT_TYPES`. |
| `center_time`, `rise_sigma`, `decay_sigma` | float64 | As in the event table. |
| `amplitude` | float64 | `spike_leakage`: peak waveform amplitude in signal units; `emg`: peak SD of the broadband burst in signal units; `fast_gamma`: target SNR as for ripples, against the 60-100 Hz band noise; `theta_burst`: peak rate gain. |
| `frequency` | float64 | `fast_gamma` only, Hz; NaN otherwise. |
| `channel` | int64 | `spike_leakage`: the LFP channel the leaking units sit on; -1 (all channels) otherwise. |
| `n_units` | int64 | `spike_leakage` and `theta_burst`: units involved; 0 otherwise. |
| `n_spikes` | int64 | `spike_leakage`: spikes per leaking unit; 0 otherwise. |
| `isi` | float64 | `spike_leakage`: inter-spike interval in seconds; NaN otherwise. |

Same ordering, span and empty-table invariants as the event table.

## SimulatedSession additions

`src/ripple_detection/simulate.py:1077` gains five fields **after** `sampling_frequency`, each
with a default so every existing constructor call keeps working:

```python
events: pd.DataFrame = field(default_factory=_empty_events)          # latent event table
non_events: pd.DataFrame = field(default_factory=_empty_non_events)  # non-event table
unit_types: StrArray = field(default_factory=lambda: np.empty(0, dtype="<U11"))  # (n_units,)
baseline_rates: FloatArray = field(default_factory=lambda: np.empty(0))  # (n_units,) spikes/s
running_intervals: FloatArray = field(default_factory=lambda: np.empty((0, 2)))  # (n_bouts, 2)
```

- `unit_types[i]` is one of `UNIT_TYPES`; `"place"` units are pyramidal units with place fields
  (a subset in meaning, a distinct label in the array). Empty when the simulator did not assign
  types (`simulate_session`).
- `baseline_rates[i]` is unit `i`'s realized baseline rate in spikes/s, as drawn by the
  renderer; the runner writes it to `units.csv.gz`. Empty when the simulator did not record
  them (`simulate_session`).
- `simulate_session` sets `running_intervals` from its argument (empty for None) and leaves the
  other four at their defaults. Its signals are unchanged.
- `__post_init__` checks `unit_types` and `baseline_rates` are each empty or have
  `multiunit.shape[1]` entries.
- For network sessions, `ripple_times`, `ripple_durations`, `ripple_frequencies` hold one entry per
  ripple component: `ripple_times = center + 1.5 (decay_sigma - rise_sigma)`,
  `ripple_durations = 3 (rise_sigma + decay_sigma)`, `ripple_frequencies = frequency_start`, so
  the existing `ripple_windows` property gives each ripple's ±3-sigma span.

`StrArray = NDArray[np.str_]` is added to `core.py` beside `FloatArray` (`core.py:26-32`).

## Truth windows

```python
def truth_windows(
    table: pd.DataFrame,
    fraction: float = 0.1,
    expression: str | None = None,
) -> pd.DataFrame:
```

- `table` is an event table or a non-event table.
- Returns columns `id`, `type`, `start_time`, `end_time`, `peak_time`, and for the event table
  `expression` and `component`.
- A component's window is where its envelope is at or above `fraction` of its peak:
  `[center - k rise_sigma, center + k decay_sigma]`, `k = sqrt(-2 ln fraction)`.
- `expression="ripple"`, `"sharp_wave"` or `"burst"`: one row per component of that expression.
- `expression="network"`: one row per `event_id`, spanning the union of its components' windows.
  `peak_time` is the ripple's centre when there is one (the first ripple for a doublet), else the
  burst's, else the sharp wave's.
- For the non-event table `expression` must be None: one row per non-event.
- **Rows are in the same order for every `fraction`** (do not weaken: phase 2 and phase 4 compute
  boundary errors at other fractions by row position after matching at 0.1).
- `0 < fraction < 1`, else `ValueError`.

The benchmark's fractions are `TRUTH_FRACTIONS = (0.1, 0.25, 0.5)`; matching uses 0.1.

## Event inventory input

Every function in `ripple_detection.evaluate` takes events as anything `core._event_bounds`
accepts (`src/ripple_detection/core.py:797`): an `(n, 2)` array of `[start_time, end_time]` or a
DataFrame with those columns. A DataFrame with a `peak_time` column also supplies peaks; otherwise
peak errors are NaN. Rows need not be sorted; returned indices refer to the input's row positions.
Bounds must be finite with start ≤ end, else `ValueError` (the check `_overlaps` makes,
`core.py:2129-2138`).

## Matching and pair metrics

```python
def match_events(reference, detected, *, minimum_iou: float = 0.0) -> EventMatching: ...

@dataclass(frozen=True)
class EventMatching:
    reference: FloatArray        # (n_reference, 2)
    detected: FloatArray         # (n_detected, 2)
    pairs: pd.DataFrame          # one row per matched pair, columns below
    reference_overlaps: IntArray # (n_reference,): detected events overlapping each
    detected_overlaps: IntArray  # (n_detected,): reference events overlapping each

    recall: float     # n_pairs / n_reference; NaN when n_reference == 0
    precision: float  # n_pairs / n_detected;  NaN when n_detected == 0
    f1: float         # 2 n_pairs / (n_reference + n_detected); NaN when both are 0
    unmatched_reference: IntArray
    unmatched_detected: IntArray
    split_reference: IntArray    # reference events overlapped by >= 2 detected events
    merged_detected: IntArray    # detected events overlapping >= 2 reference events

    def boundary_errors(self, reference: ArrayLike) -> pd.DataFrame: ...
```

- **Overlap** is intersection of positive length; touching endpoints do not overlap (matches
  `require_overlap`'s default, `core.py:2201-2203`).
- **Matching is one-to-one** over pairs whose IoU exceeds `minimum_iou`, maximizing first the
  number of pairs and then their summed IoU, solved exactly per connected component of the
  overlap graph. The pair count is the same with the inventories swapped, so recall and
  precision exchange roles and F1 and Jaccard are unchanged (algorithm in
  [designs.md#matching](designs.md#matching)). Do not weaken to greedy: split and merge counts and
  boundary errors depend on which detected event is the match.
- `pairs` columns, per pair (`r` reference, `d` detected, `∩` intersection length):

| Column | Definition |
| --- | --- |
| `reference_index`, `detected_index` | Row positions in the inputs. |
| `iou` | `∩ / (len(r) + len(d) - ∩)` |
| `coverage` | `∩ / len(r)`, the fraction of the true event found. NaN for a zero-length `r`. |
| `temporal_precision` | `∩ / len(d)`, the fraction of the detected event that is true. NaN for a zero-length `d`. |
| `onset_error` | `d.start - r.start`, seconds. **Positive: detected late.** |
| `offset_error` | `d.end - r.end`, seconds. **Positive: detected late (too long).** |
| `peak_error` | `d.peak_time - r.peak_time` when both inputs have `peak_time`, else NaN. |

- `boundary_errors(reference)` recomputes `onset_error` and `offset_error` for the same pairs
  against another `(n_reference, 2)` array of bounds for the same reference events (same rows),
  which is how errors at fractions 0.25 and 0.5 are computed after matching at 0.1.
- Sign convention (do not weaken): every signed quantity in this project is *detected minus
  reference* or *method A minus method B*; negative means earlier.

## Detector comparison table

```python
def compare_detectors(
    events: Mapping[str, EventInventory],
    *,
    truth: EventInventory | None = None,
    minimum_iou: float = 0.0,
) -> pd.DataFrame: ...

def consensus_counts(events: Mapping[str, EventInventory], truth: EventInventory,
                     *, minimum_iou: float = 0.0) -> pd.DataFrame: ...

def label_by_overlap(events: EventInventory, windows: pd.DataFrame,
                     *, unlabeled: str = "background") -> pd.Series: ...
```

`compare_detectors` returns one row per unordered pair `(a, b)`, `a` before `b` in the mapping's
order, with `a`'s events as the reference in `match_events(a, b)`:

| Column | Definition |
| --- | --- |
| `method_a`, `method_b`, `n_a`, `n_b`, `n_matched` | |
| `jaccard` | `n_matched / (n_a + n_b - n_matched)`; NaN when both are empty. |
| `median_iou` | Over matched pairs. |
| `median_onset_difference`, `onset_difference_iqr` | Of `a.start - b.start` over matched pairs, seconds. |
| `fraction_a_earlier_onset` | Fraction of matched pairs with `a.start < b.start` (strict). |
| `median_offset_difference`, `offset_difference_iqr`, `fraction_a_earlier_offset` | Same for ends. |

Always present, NaN when `truth` is None, so the columns do not depend on the call (each method
matched to `truth` first):

| Column | Definition |
| --- | --- |
| `jaccard_true` | `jaccard` between the two methods' events that matched a truth event. |
| `jaccard_false` | `jaccard` between the two methods' events that matched none. |
| `n_shared_truth` | Truth events both methods matched. |
| `onset_error_correlation`, `offset_error_correlation` | Spearman correlation over the shared truth events of the two methods' signed errors against truth; NaN below 3 shared events. |

`consensus_counts` returns one row per truth event: one boolean column per method (matched it or
not) and `n_methods`. `label_by_overlap` returns, per event, the `label` column of the row of
`windows` (`start_time`, `end_time`, `label`) it overlaps longest, or `unlabeled` when it overlaps
none.

## Recipe config

In `examples/benchmark/recipe_configs.py`. This is a benchmark call specification;
the implementation and scientific metadata belong to `ripple_detection.literature_methods`.

```python
Params = tuple[tuple[str, Any], ...]  # serializable, hashable scalar/tuple settings

@dataclass(frozen=True)
class RecipeConfig:
    config_id: str                # unique, stable benchmark configuration identifier
    method: str                   # exact name from list_methods()
    primary_expression: str       # ripple, sharp_wave, burst or network
    options: Params = ()          # method options, including stage when supported
    input_policy: str = ""        # named, documented simulation-to-Recording policy
    assumptions: tuple[str, ...] = ()  # benchmark choices absent from the source
```

- `RECIPES: tuple[RecipeConfig, ...]` configures supported methods; `EXCLUSIONS`
  maps every remaining catalog name to a reason. Derive coverage from the integrated
  catalog, not paper-row counts: a paper can supply several inventories or only a label.
- `make_recording(session, config) -> Recording` uses the installed constructor and
  explicit input policies. Do not copy `Recording` or silently use simulation fallbacks.
- `run_recipe(config, recording, behavior_intervals=None) -> pd.DataFrame` calls
  `run_method` with the configured name/options and the session's behavior intervals
  (per call, not on `Recording`) and preserves all diagnostics and attrs. The package owns defaults;
  persist the resolved options, not just overrides.
- `bounds` is imported from the package. Event comparison accepts the public output
  contract; no conversion discards metadata before it is saved.
- DOI, paper, output role and interpretation come from the package. Demonstration
  grouping (default/additional) is distinct from role. Use detection stage by default;
  any other stage is a separate named configuration and must not be pooled with it.
- Input policies serialize supplied intervals, selections and external detector
  configurations, or stable references to per-session values stored with the run.
  External ripple inventories cannot be substituted with simulator truth.
- Phase 6 owns experimental `Step`, `ThresholdCore`, `Pipeline` and template types,
  plus their executor built from public package primitives. These are experimental
  compositions, not method configurations or the source of named-method results.
  Their definitions and equivalence rule are in [attribution](designs.md#attribution).

## Primary expression

The truth a method is headlined against. Every method is also scored against the other
expressions and `"network"`.

| Method | Primary expression |
| --- | --- |
| `Kay_ripple_detector`, `Karlsson_ripple_detector`, `Roumis_ripple_detector`, `Shvartsman_ripple_detector`, `Yu_ripple_detector`, `Zugaro_ripple_detector`, `Long_sharp_wave_ripple_detector` | `ripple` |
| `Carey_candidate_detector` | `network` |
| `multiunit_HSE_detector` | `burst` |
| Paper-method configuration | Explicit per-method assignment in `RecipeConfig`, reviewed against its implemented output. |

The detector mapping lives in `examples/benchmark/run.py` as `DETECTOR_EXPRESSION`.
Method configurations carry their primary expression explicitly: do not infer it from
free-text trigger prefixes or whether a function is a demonstration default. Preserve
roles such as secondary labels and candidate gates in reports, even when both are
scored against ripple truth. Tests cover representative ripple, burst and joint methods.

## Threshold sweeps

`examples/benchmark/run.py`:

```python
THRESHOLD_SWEEPS: dict[str, tuple[str, tuple[Any, ...]]] = {
    "Kay_ripple_detector":        ("zscore_threshold", (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)),
    "Karlsson_ripple_detector":   ("zscore_threshold", (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)),
    "Roumis_ripple_detector":     ("zscore_threshold", (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)),
    "Shvartsman_ripple_detector": ("zscore_threshold", (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)),
    "multiunit_HSE_detector":     ("zscore_threshold", (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)),
    "Zugaro_ripple_detector":     ("high_threshold",   (2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)),
    "Carey_candidate_detector":   ("high_threshold",   (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)),
    "Yu_ripple_detector":         ("percentile",       (99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999)),
    "Long_sharp_wave_ripple_detector": ("peak_thresholds", (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)),
}
```

- Zugaro keeps `low_threshold=2.0` (every swept high is above it); Carey keeps `low_threshold=1.0`.
- Long's `peak_thresholds` is a runner alias: value `v` sets `sharp_wave_thresholds=(0.5, v)`
  and `ripple_thresholds=(0.5, v)`.
- Every detector also runs at its defaults, `setting="default"`.
- Sweeps run with `speed_threshold` at each detector's default (4 cm/s); events occur only at rest,
  so the speed rule removes theta-state non-events, which is part of what is measured.

## Conditions

`examples/benchmark/conditions.py`:

```python
@dataclass(frozen=True)
class Condition:
    condition_id: str             # "reference", f"{factor}={label}", or f"{f1}={l1},{f2}={l2}"
    factor: str                   # "reference", a factor name, or "f1,f2" for a crossed pair
    level: str                    # the label(s)
    params: Params                # overrides of REFERENCE by dotted key, e.g. "events.ripple_snr"

REFERENCE: dict[str, dict[str, Any]]  # "session", "events", "non_events", "render" -> keywords
def conditions() -> tuple[Condition, ...]: ...   # reference + one-factor grid + crossed pairs
def session_seed(replicate: int) -> int: ...     # the same for every condition
def running_schedule(duration_s: float, rng: np.random.Generator) -> FloatArray: ...
def simulate_condition(condition: Condition, replicate: int) -> SimulatedSession: ...
```

Invariant (do not weaken): replicate `k` has the same seed in every condition (common random
numbers), so comparisons across conditions pair by replicate, and phase 6 finds the reference
sessions by replicate. Values, labels, the factor-to-keyword mapping, the schedule and the draw
order are in [designs.md#conditions-grid](designs.md#conditions-grid).

## Benchmark outputs

Written by phase 4 under `examples/benchmark/output/<run_name>/` (git-ignored). Read by phases 5
and 6. Every table is CSV; `.csv.gz` for the large ones.

- **Run files** (`manifest.json`, `run_spec.json`, `conditions.csv`) are written once when the
  run starts; `manifest.json` gains `finished` at the end.
- **Condition files** live in `conditions/<condition_id>/` and are immutable: a condition is
  written into `conditions/<condition_id>.partial/`, its `done.json` last, and the directory is
  then renamed into place. No condition writes to another's directory or to a shared table, so
  finishing one condition never changes another's files or invalidates its marker.
- **Combined tables** in `combined/` (the same file names as a condition directory, with
  `events.csv.gz` and `results/<condition_id>/...`) are concatenations of the finished
  conditions, built by `run.py --combine` (and at the end of a run). They are derived:
  rebuilt from the condition directories at any time, never read by `--resume`. Phases 5 and 6
  read `combined/`.

| File | One row per | Columns |
| --- | --- | --- |
| `manifest.json` | run | `run_name`, `git_commit`, `package_version`, `numpy_version`, `scipy_version`, `command`, `started`, `finished`, `n_workers` |
| `run_spec.json` | run | the resolved specification `--resume` checks: every condition's parameters after overrides, `replicates`, seeds, every method and setting with its resolved options, `package_version`, `git_commit` |
| `conditions/<condition_id>/done.json` | finished condition | row count and SHA-256 of every other file in the condition's directory; written last, before the directory is renamed into place |
| `conditions/<condition_id>/methods.csv` | session × method × setting | `session_id`, `method`, `setting`, `doi`, `role`, `inventory`, `stage`, `primary_expression`, `resolved_options` (JSON), `input_policy` (JSON or references), `assumptions` (JSON), `interpretation` |
| `conditions.csv` | condition | `condition_id`, `factor`, `level`, `params` (JSON of the full parameter set after the condition's and the command line's overrides, such as `--duration`; phase 6 regenerates sessions from it) |
| `conditions/<condition_id>/sessions.csv.gz` | session | `session_id` (`f"{condition_id}/{replicate}"`), `condition_id`, `replicate`, `seed`, `duration_s`, `rest_s`, `event_time_s` (union of network windows at 0.1), `n_events_<type>` per `EVENT_TYPES`, `n_non_events_<type>` per `NON_EVENT_TYPES`, `simulate_s`, `detect_s` |
| `conditions/<condition_id>/truth.csv.gz` | truth component | `session_id`, `table` (`"event"`/`"non_event"`), `id`, `type`, `expression`, `component`, then the remaining columns of the event or non-event table (NaN where not applicable) |
| `conditions/<condition_id>/truth_counts.csv.gz` | truth window × expression | `session_id`, `expression` (`ripple`, `sharp_wave`, `burst`, `network`), `row` (position in `truth_windows(events, 0.1, expression)`, the rows matching uses), `n_active_units`, `n_active_principal` (observed within that window) |
| `conditions/<condition_id>/units.csv.gz` | session × unit | `session_id`, `unit`, `unit_type`, `baseline_rate` (from `SimulatedSession.baseline_rates`) |
| `conditions/<condition_id>/events.csv.gz` | detected event | `session_id`, `method`, `setting`, `event_index`, `start_time`, `end_time`, `peak_time`, `n_active_units`, `n_active_principal` |
| `conditions/<condition_id>/results/<method_slug>__<setting>.csv.gz` and `.json` | detected event, complete | `session_id`, then every column the method returned, in its order (clipping flags, per-event statistics, method-specific columns); the JSON sidecar holds each column's dtype and, per `session_id`, the result's complete `attrs` (recipes: method, DOI, resolved options, grid, inputs, diagnostics such as adaptive threshold updates, `ripple_detection_version`; detectors: name, resolved parameters, version). `method_slug` is `method` with `:` replaced by `--`. |
| `conditions/<condition_id>/metrics.csv.gz` | session × method × setting × expression | `session_id`, `method`, `setting`, `expression`, `n_reference`, `n_detected`, `n_matched`, `recall`, `precision`, `f1`, `false_positives_per_minute`, `median_iou`, `median_coverage`, `median_temporal_precision`, `median_onset_error_<f>`, `median_offset_error_<f>` for `f` in 10, 25, 50, `n_split`, `n_merged` |
| `conditions/<condition_id>/failures.csv` | failed call | `session_id`, `method`, `setting`, `error` (`f"{type(error).__name__}: {error}"`, first 200 characters) |

- `method` is a registry name for a detector or `recipe:<config_id>` for a method
  configuration. `config_id` includes the stable package method name and any protocol
  or stage discriminator; it does not depend on mutable survey row numbers.
- `setting` is `"default"`, the swept value formatted with `repr(float(v))`, or `"literature"`. The latter labels a configured interpretation; `assumptions`
  discloses settings not established by the paper. These three are the only values; the main
  analyses select `setting in {"default", "literature"}` (detectors at defaults, every recipe).
- `false_positives_per_minute` = unmatched detected events / (minutes of the session outside every
  network window at fraction 0.1).
- `n_active_units` counts units with a spike in the event (`count_spikes_in_events`, all units),
  `n_active_principal` the same over place and pyramidal units, for the participation
  analysis. `truth_counts.csv.gz` holds both counts for every truth window of every expression
  at fraction 0.1, with the same unit selections, so a pair matched against any expression
  compares observed counts at its own truth window.
- `results/` keeps every result complete, so nothing a method reports is lost; `events.csv.gz`
  holds only the columns analyses read. Written and read with the conventions of
  `literature_methods.save_events`/`load_events` (`float_precision="round_trip"`, saved
  dtypes): a reloaded result equals the original exactly, `attrs` included.
- A method that raises on a session writes a `failures.csv` row (in its condition's directory)
  and neither events nor metrics
  rows; analyses count a missing (session, method, setting) as a failure, never as zero events.
  Runs never abort on one method's error.
- Phase 5 and 6 summaries go to `examples/benchmark/results/` (committed): small CSVs and PNG
  figures only, each under 1 MB.
