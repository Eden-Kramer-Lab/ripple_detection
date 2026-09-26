# Phase 1b — Non-events: activity a detector should not report

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#non-events)

Add four kinds of non-event to the network simulator: spike-waveform leakage (high-frequency
complex-spike bursts leaking into one LFP channel), broadband EMG artifacts, fast-gamma bursts,
and theta-state population bursts while running. Each has a truth table row, so a false positive
can be attributed to its cause.

**Inputs to read first:**

- Phase 1a's merged code in `src/ripple_detection/simulate.py`: `draw_network_events`,
  `simulate_network_session`, `_render_ripple`, `_scale_to_snr`, `_half_gaussians`, `truth_windows`.
- [src/ripple_detection/simulate.py:589-601](../../../../src/ripple_detection/simulate.py) — `_add_common_mode_artifacts`, the pattern EMG generalizes (asymmetric envelope, high-passed noise).
- [src/ripple_detection/core.py](../../../../src/ripple_detection/core.py) — `filter_ripple_band(..., band=(60, 100))` designs a FIR for a non-default band; used to size fast gamma.

**Contracts referenced:**

- [Vocabularies](shared-contracts.md#vocabularies) — `NON_EVENT_TYPES` (declared in 1a).
- [Non-event table](shared-contracts.md#non-event-table) — produced here; same ordering, span and empty-table invariants as the event table.
- [SimulatedSession additions](shared-contracts.md#simulatedsession-additions) — `non_events` is filled here.
- [Truth windows](shared-contracts.md#truth-windows) — the non-event branch (`expression=None`).

**Designs referenced:** [non-events](designs.md#non-events), [parameter sources](designs.md#parameter-sources) (non-event rows).

## Tasks

- Implement `draw_non_events` per [designs.md#non-events](designs.md#non-events): per-type Poisson
  times on the allowed time (rest, anywhere, running), the ±4-sigma containment rejection, the
  per-type draws, validation (unknown type key, negative rate, `low > high`, `fast_gamma_frequency`
  above Nyquist → `ValueError` naming the parameter). Wrapped with `explain_call_errors`.
- Add `non_events: pd.DataFrame | None = None` to `simulate_network_session` and render each type
  per the table in the design, in the documented position of the draw order (after burst
  participants, before spikes; leakage spikes added after the Poisson draw so the Poisson counts do
  not move). With `non_events=None` the output must be identical to phase 1a's for the same seed.
- Record the table in `SimulatedSession.non_events`.
- Fast gamma is sized with `_scale_to_snr(..., band=(60.0, 100.0))` (the `band` argument 1a added);
  without it the burst would be scaled by its ripple-band residue and come out far too large.
- `truth_windows` already handles the non-event table (1a); add tests for that branch here if 1a
  did not.
- Export `draw_non_events`; update `tests/test_public_api.py`.
- **Docs:** extend the README subsection added in 1a with a paragraph and snippet on non-events
  (draw, render, `truth_windows(session.non_events)`); CHANGELOG `[2.0.0]` Added entry extended
  (same entry as 1a, one more function); `llms.txt` line extended; the `draw_non_events` docstring
  lists each type's rendering and states the parameter sources (from the design table).

## Deliberately not in this phase

- Classifying detections by the non-event they overlap — `label_by_overlap` is phase 2.
- New event types, or changing network-event rendering — 1a is settled.
- Non-events in `simulate_session` — it stays as it is.
- Realistic spike waveforms at higher sampling rates: the three-sample biphasic waveform is a
  1500 Hz rendering; a rate-aware waveform is not needed for the benchmark (1500 Hz throughout).

## Validation slice

| Test | Asserts |
| --- | --- |
| CI | Green, including the dependency-floors job. |
| `TestDrawNonEvents::test_schema_and_order` | Columns, dtypes, sort by `center_time`, RangeIndex, empty table when all rates are 0. |
| `TestDrawNonEvents::test_where_each_type_occurs` | `spike_leakage` only at rest, `theta_burst` only inside running intervals, `emg`/`fast_gamma` in both; all spans inside the recording. |
| `TestDrawNonEvents::test_validation` | Each invalid input raises `ValueError` naming the parameter. |
| `TestSimulateNetworkSession::test_no_non_events_is_unchanged` | `non_events=None` and an empty table both give arrays equal to phase 1a's output for the same seed. |
| `TestNonEventRendering::test_spike_leakage_adds_ripple_band_power_on_one_channel` | Noise-free: `filter_ripple_band` power inside the window rises on `channel` only; the leaking units' spike counts rise by exactly the burst's spikes. |
| `TestNonEventRendering::test_emg_is_common_mode_and_high_passed` | Noise-free: identical on every channel and the radiatum; < 5% of its power below 80 Hz. |
| `TestNonEventRendering::test_fast_gamma_snr_and_band` | Filtered 60-100 Hz peak over the 60-100 Hz noise SD within 20% of `amplitude`; ripple-band (150-250 Hz) power inside the window < 10% of the 60-100 Hz power. |
| `TestNonEventRendering::test_theta_burst_modulates_only_its_units` | Summed over seeds, the chosen place units' rate peaks at the centre; other units unchanged; no LFP change. |
| `TestTruthWindows::test_non_event_table` | One row per non-event, analytic windows; passing `expression` raises. |
| doctest | `draw_non_events`' example runs. |

## Fixtures

Synthesized in the tests, as in 1a: short sessions, fixed seeds, hand-built one-row non-event
tables, noise-free renderings for the waveform checks. Reuse 1a's `_one_event_table` pattern with a
`_one_non_event_table(non_event_type, **overrides)` helper.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Rendering with `non_events=None` is bit-identical to phase 1a's.
