# Phase 1a — Latent network events with per-expression truth

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#drawing-network-events)

Add to `ripple_detection.simulate` a simulator whose truth is a latent network event expressed as a
ripple, a sharp wave and a population burst, in five event types, with truth windows computable
at any envelope fraction. Public, because users can test a detector on it.

**Inputs to read first:**

- [src/ripple_detection/simulate.py:1075-1147](../../../../src/ripple_detection/simulate.py) — `SimulatedSession`; the fields and invariants being extended.
- [src/ripple_detection/simulate.py:1150-1316](../../../../src/ripple_detection/simulate.py) — `simulate_session`: the order of draws and the slow-field/speed steps the new renderer mirrors; its output must not change.
- [src/ripple_detection/simulate.py:407-600](../../../../src/ripple_detection/simulate.py) — `_ripple_waveform`, `_gaussian_window`, `_add_ripple_bursts` (the SNR scaling to extract at 548-557), `_correlated_noise`.
- [src/ripple_detection/simulate.py:726-1072](../../../../src/ripple_detection/simulate.py) — sharp-wave helpers, `simulate_multiunit`, `simulate_speed`, `simulate_theta_delta`.
- [src/ripple_detection/_call_hints.py:143](../../../../src/ripple_detection/_call_hints.py) — `explain_call_errors`; every new public function is wrapped.
- [tests/test_simulate.py:897-1128](../../../../tests/test_simulate.py) — `TestSimulateSession`, `TestSessionStates` (including `test_the_defaults_are_unchanged` at 1092): the style new tests follow and the guards that must pass untouched.
- [src/ripple_detection/__init__.py:72-85,146-153](../../../../src/ripple_detection/__init__.py) and `tests/test_public_api.py` — exports and the pinned `__all__`.

**Contracts referenced:**

- [Vocabularies](shared-contracts.md#vocabularies) — defines `EVENT_TYPES`, `EXPRESSIONS`, `UNIT_TYPES` (`NON_EVENT_TYPES` too, used in 1b).
- [Latent event table](shared-contracts.md#latent-event-table) — produced here; do not weaken the ordering, span and empty-table invariants.
- [SimulatedSession additions](shared-contracts.md#simulatedsession-additions) — all four fields land here (`non_events` stays empty until 1b).
- [Truth windows](shared-contracts.md#truth-windows) — implemented here; same-row-order-for-every-fraction is load-bearing for phases 2 and 4.

**Designs referenced:** [parameter sources](designs.md#parameter-sources),
[event types](designs.md#event-types), [drawing network events](designs.md#drawing-network-events),
[rendering a network session](designs.md#rendering-a-network-session),
[truth windows](designs.md#truth-windows).

## Tasks

- **Settle the parameter table first.** For each "verify" row in
  [designs.md#parameter-sources](designs.md#parameter-sources), find the source by its DOI (publisher
  or PubMed Central; the maintainer's reference library if at hand) and record citation and page, or change the
  row to "assumed". Put the final table in `draw_network_events`' docstring under Notes. No detector
  is run on network sessions before this is committed (overview risk 1); running the simulator to
  check its own truth is fine.
- **Behavior-preserving extraction:** move the padded filter-peak scaling at
  `simulate.py:548-557` into `_scale_to_snr(burst, snr, band_noise_sd, rate, band=None) -> float`
  (`band=None` filters exactly as today; see [designs.md#rendering-a-network-session](designs.md#rendering-a-network-session)) and call it
  from `_add_ripple_bursts`. Run the full suite before and after; `test_snapshots.py` and
  `test_simulate.py` must pass without snapshot updates. Separate commit.
- Add the vocabularies to `simulate.py` and `StrArray = NDArray[np.str_]` to `core.py`'s alias
  block (`core.py:25-31`, beside `FloatArray`).
- Extend `SimulatedSession` (`simulate.py:1075`) with `events`, `non_events`, `unit_types`,
  `running_intervals` per the contract: `field(default_factory=...)` defaults, an
  `_empty_events()` / `_empty_non_events()` pair building the typed empty frames, the `unit_types`
  length check in `__post_init__` (`simulate.py:1120`), and docstring entries. In the same edit,
  correct the `speed` attribute's docstring ("Zeros: an immobile animal", now wrong since
  `running_intervals` was added) to "Zeros unless `running_intervals` was given".
- `simulate_session` passes `running_intervals` (empty `(0, 2)` for None) into the result. Nothing
  else in it changes.
- Implement `draw_network_events` per [designs.md#drawing-network-events](designs.md#drawing-network-events),
  including validation, the fixed draw order and the drop-not-redraw rejection.
- Implement `simulate_network_session` per [designs.md#rendering-a-network-session](designs.md#rendering-a-network-session),
  without the `non_events` parameter (1b adds it). Helpers: `_render_ripple`, `_half_gaussians`
  (asymmetric Gaussian over ±8 sigma, the sharp-wave and burst envelope), `_draw_units`. Fill
  `n_participants` in the returned events table and derive the `ripple_*` arrays.
- Implement `truth_windows` per [designs.md#truth-windows](designs.md#truth-windows), in
  `simulate.py` (it reads the simulator's tables).
- Export `draw_network_events`, `simulate_network_session`, `truth_windows`, `EVENT_TYPES`,
  `NON_EVENT_TYPES`, `EXPRESSIONS`, `UNIT_TYPES` from `ripple_detection` (`__init__.py:72-85`
  imports, `__all__` at 146-153) and update `tests/test_public_api.py`'s pinned list.
- **Docs:**
  - README: a subsection "Simulating sessions with known event types" after "Simulating realistic
    ripples" (`README.md:522`): drawing events, editing the table, rendering, `truth_windows` at
    two fractions; a runnable snippet that prints column names, not counts (the doctest convention
    in CLAUDE.md).
  - CHANGELOG `[2.0.0]` → Added (`CHANGELOG.md:16`): one entry for the three functions and the
    `SimulatedSession` fields.
  - `llms.txt`: one line under "Building blocks for published variants" (`llms.txt:51`).
  - CLAUDE.md "Core Module Structure", item 3 (simulate.py): a bullet for the network simulator.
  - `tests/CLAUDE.md` item 4: the new test classes.
  - Docstring examples on each new public function (they run under `--doctest-modules`).

## Deliberately not in this phase

- Non-events and the `non_events` parameter — phase 1b.
- Any evaluation or matching code — phase 2. Tests here check truth against the rendered signal
  directly, not through `match_events`.
- Running detectors on network sessions, even to "sanity check" — that is the benchmark, and it
  waits for the parameter table.
- Reimplementing `simulate_session` on top of the new renderer — the two coexist
  ([designs.md#rendering-a-network-session](designs.md#rendering-a-network-session), last paragraph).
- Changing `_ripple_waveform`, `_add_ripple_bursts` or any default of an existing function beyond
  the `_scale_to_snr` extraction.

## Validation slice

| Test | Asserts |
| --- | --- |
| existing `tests/test_simulate.py`, `tests/test_snapshots.py` | Pass with no edits and no snapshot updates (the extraction and the new fields change nothing). |
| CI | Green, including the dependency-floors job (overview Rollout Strategy). |
| `TestSimulatedSessionFields::test_defaults_keep_old_constructors_working` | A `SimulatedSession(...)` built with only the 11 original fields has empty `events`/`non_events` with the contract's columns and dtypes, `unit_types.shape == (0,)`, `running_intervals.shape == (0, 2)`. |
| `TestSimulatedSessionFields::test_unit_types_length_is_checked` | `unit_types` of the wrong length raises `ValueError`. |
| `TestSimulatedSessionFields::test_simulate_session_records_running_intervals` | `simulate_session(..., running_intervals=[(2, 4)])` returns them as a `(1, 2)` array, and `(0, 2)` for None. Signals are guarded by the existing `test_the_defaults_are_unchanged`, which must pass unedited. |
| `TestDrawNetworkEvents::test_schema_and_order` | Columns, dtypes, sort order, RangeIndex; `event_id` increasing with the earliest component's centre. |
| `TestDrawNetworkEvents::test_components_per_type` | `swr` has ripple, sharp wave, burst; `burst_only` only a burst; `sharp_wave_only` only a sharp wave; a doublet has 2-3 ripples, as many sharp waves, one burst whose ±3-sigma span runs from the first ripple's start to the last's end. |
| `TestDrawNetworkEvents::test_events_only_at_rest_and_inside` | No component's ±4-sigma span touches a running interval or the first/last second. |
| `TestDrawNetworkEvents::test_rate` | With nothing to drop (`minimum_separation=0`, only `swr`, `ripple_duration=(0.01, 0.01)`, `sharp_wave_duration=(0.01, 0.01)`, `burst_duration_ratio=(1, 1)`, no running), 3600 s at rate 0.5 gives a count within 4 SD of 1800 (fixed seed). |
| `TestDrawNetworkEvents::test_seeded_and_parameter_local` | Same seed → identical table; changing `sharp_wave_amplitude` leaves every event time and ripple column unchanged. |
| `TestDrawNetworkEvents::test_validation` | Negative rate, unknown type key, `low > high`, frequency above Nyquist each raise `ValueError` naming the parameter. |
| `TestSimulateNetworkSession::test_shapes_and_types` | `lfps (n_time, n_channels)`, `multiunit (n_time, 60)`, `unit_types` 40/10/10 in order, `n_participants` filled for burst rows and 0 elsewhere. |
| `TestSimulateNetworkSession::test_ripple_snr_is_met` | For isolated ripples, peak of `filter_ripple_band(lfps[:, 0])` within the window over the band-noise SD is within 20% of `amplitude` (noise adds; single seed, 20 ripples, median ratio). |
| `TestSimulateNetworkSession::test_chirp` | Instantaneous frequency (Hilbert phase derivative of the noise-free rendering) at −2 and +2 sigma matches the linear chirp within 5 Hz. |
| `TestSimulateNetworkSession::test_burst_follows_its_envelope` | Summed spike rate over 200 simulated copies of one `burst_only` event (different seeds, fixed table) peaks within 5 ms of the burst centre and exceeds baseline only inside the 0.01-fraction window. |
| `TestSimulateNetworkSession::test_sharp_wave_sign_and_leak` | Radiatum deflection negative with peak `amplitude`, channel 0 positive at `sharp_wave_leak` of it (noise-free: `noise_amplitude=0`). |
| `TestSimulateNetworkSession::test_ripple_windows_match_components` | `session.ripple_windows` equals the ripple rows' ±3-sigma spans. |
| `TestTruthWindows::test_fraction_formula` | Windows equal `center ∓ sqrt(-2 ln f) sigma` for f in (0.1, 0.25, 0.5); narrower as f rises. |
| `TestTruthWindows::test_row_order_is_the_same_for_every_fraction` | `id` (and `expression`, `component`) columns identical across fractions. |
| `TestTruthWindows::test_measured_on_the_rendered_envelope` | Noise-free rendering: the samples where the ripple's Hilbert envelope is ≥ f × peak span the analytic window within one sample (f = 0.1, 0.25, 0.5). |
| `TestTruthWindows::test_network_union_and_peak` | Network row spans the union of components; `peak_time` is the ripple's centre, else the burst's, else the sharp wave's. |
| `TestTruthWindows::test_validation` | `fraction` 0 or 1 raises; unknown expression raises. |
| doctests | Each new public function's example runs. |

## Fixtures

All synthesized in the tests: short sessions (10-60 s at 1500 Hz) with fixed seeds, and
hand-built event tables for the noise-free checks (`noise_amplitude=0`, `theta_amplitude=0`,
`delta_amplitude=0`). Add a module-level helper `_one_event_table(event_type, **overrides)` in
`tests/test_simulate.py` to build single-event tables. No real data: the simulator is the truth.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- The parameter table's "verify" rows are all resolved, and the `_scale_to_snr` extraction is its own commit with the suite green before and after.
