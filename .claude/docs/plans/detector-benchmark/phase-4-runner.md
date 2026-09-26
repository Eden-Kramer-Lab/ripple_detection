# Phase 4 — Benchmark runner: conditions, sweeps, recipes, event-level outputs

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#runner)

A command-line runner in `examples/benchmark/` that simulates sessions for every condition, runs
the nine detectors at their defaults and along their threshold sweeps plus every recipe, scores
each against every truth expression, and writes event-level outputs and per-session metrics. Then
the smoke test, the extrapolation, and the full run.

**Inputs to read first:**

- Phase 1a/1b: `draw_network_events`, `draw_non_events`, `simulate_network_session`, `truth_windows`.
- Phase 2: `ripple_detection.evaluate` (`match_events`, `EventMatching.boundary_errors`).
- Phase 3: `examples/benchmark/recipe_configs.py` (`make_recording`, `RECIPES`, `EXCLUSIONS`, `run_recipe`).
- [src/ripple_detection/registry.py:462-478](../../../../src/ripple_detection/registry.py) — `DETECTORS`; detectors are resolved by name.
- [examples/simulation_study.py:108-141](../../../../examples/simulation_study.py) — `call` and `detector_calls`: the error-recording call and the per-detector input wiring to mirror.
- [src/ripple_detection/detectors/_units.py:34](../../../../src/ripple_detection/detectors/_units.py) — `count_spikes_in_events`, for `n_active_units`.

**Contracts referenced:**

- [Threshold sweeps](shared-contracts.md#threshold-sweeps) — defined in `run.py` exactly as written there.
- [Primary expression](shared-contracts.md#primary-expression) — `DETECTOR_EXPRESSION` lives in `run.py`.
- [Conditions](shared-contracts.md#conditions) — defined here; the common-random-numbers seeding is load-bearing (robustness pairs by replicate; phase 6 reuses reference replicates 0-4).
- [Benchmark outputs](shared-contracts.md#benchmark-outputs) — written here; phases 5 and 6 read these schemas, do not weaken.
- [Truth windows](shared-contracts.md#truth-windows), [Matching and pair metrics](shared-contracts.md#matching-and-pair-metrics) — used per session.

**Designs referenced:** [conditions grid](designs.md#conditions-grid), [runner](designs.md#runner).

## Tasks

- `examples/benchmark/conditions.py`: `Condition`, `REFERENCE`, `running_schedule`,
  `conditions()`, `session_seed`, `simulate_condition`, each as the contract and
  [designs.md#conditions-grid](designs.md#conditions-grid) define them (values, labels, the
  factor-to-keyword table, the schedule, the seeding and the draw order live there).
  `simulate_condition` applies a condition's dotted-key overrides to a deep copy of `REFERENCE`.
- `examples/benchmark/run.py`: `THRESHOLD_SWEEPS`, `DETECTOR_EXPRESSION`, `method_calls(session)`
  (detector defaults, sweep points and recipes as `(method, setting, callable)`;
  build each method recording from its input policy, and expand Long's
  `peak_thresholds` alias here), `run_session(condition, replicate, methods=None)`
  returning the session's `truth`, `units`, `methods`, `events`, `metrics`, `failures` frames and timings, and
  the CLI in [designs.md#runner](designs.md#runner) with `ProcessPoolExecutor`, per-condition writes,
  `--resume`, `--smoke`, and `manifest.json`. Every method call is guarded by `except Exception`
  (the design's runner step 3), so no recipe or detector can abort a run. Metrics per the output schema, using
  `truth_windows(events, 0.1, expression)` for matching and `boundary_errors` at 0.25 and 0.5.
- Persist `methods.csv` with the resolved public-call metadata and input policy for
  each session/configuration. Reports separate output roles and stages; failed or
  excluded methods never count as successful zero-event calls.
- `.gitignore`: add `examples/benchmark/output/`.
- **Smoke test.** Check capacity first (`sysctl -n hw.ncpu` on macOS or `nproc`; `df -h .`; no other
  heavy job running). Run `uv run python examples/benchmark/run.py --run-name smoke --smoke`.
  Record per-method runtime, simulate time, peak memory, rows and bytes per table, and the
  extrapolation, and apply the smoke-test decision rules in [designs.md#runner](designs.md#runner)
  (session length, sweep events, worker count from memory). Write the measured numbers and any rule
  applied into the PR description and `examples/benchmark/README.md`.
- **Spot check before the full run** (look at individual events before trusting aggregates): from the smoke
  session, plot 6 true events of each type with the truth windows at the three fractions and the
  events of Kay, Karlsson, HSE and two recipes (`examples/benchmark/spot_check.py`, re-simulating by
  seed; PNGs to the run's output directory, not committed). Look for misalignment, unit errors,
  missing events; fix before continuing.
- **Full run**, only after the smoke test and spot check: in a `tmux` session named `benchmark`,
  `uv run python examples/benchmark/run.py --run-name v1 --conditions all --workers N`. Record the
  command, wall time and output size in `examples/benchmark/README.md`. The run is by hand; it is
  not part of CI.
- `examples/benchmark/README.md`: a "Running the benchmark" section (commands, outputs and their
  schemas by link to the column lists in `run.py`'s module docstring, the smoke-test numbers,
  resuming).
- **Docs:** CLAUDE.md Development Commands: the smoke and full-run commands beside the simulation
  study's. README "How the detectors compare on simulated data" (`README.md:245`): one sentence
  pointing to `examples/benchmark/README.md` for the full benchmark. `tests/CLAUDE.md`: an item
  for `tests/test_benchmark.py`. No CHANGELOG entry (examples only).

## Deliberately not in this phase

- Any analysis or figure beyond the spot check — phase 5.
- Attribution runs (Sobol, Shapley) — phase 6; the runner has no factor-space code.
- Removing `examples/simulation_study.py` and its notebook. It stays as the fast study CI runs and
  the README cites. Revisit when the benchmark's reference results make the study's README table
  redundant: then replace the table with benchmark summaries and delete the study, the notebook
  and their CSV in one change.
- New dependencies (parquet, joblib, dask). `csv.gz` and `concurrent.futures` only.
- Tuning simulator parameters after seeing detector results (overview risk 1). A simulator bug found
  in the spot check is fixed in `src/` with a test, and the fix is described in the PR.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_benchmark.py::test_conditions_are_unique_and_complete` | The condition count in [designs.md#conditions-grid](designs.md#conditions-grid), unique ids in the allowed characters, the reference first; each one-factor condition overrides exactly the keys its factor sets, each crossed cell those of two factors. |
| `test_common_random_numbers` | `session_seed(k)` is the same for every condition; replicate 0 of `reference` and of `ripple_snr=high` have identical running schedules and event centre times, and different ripple amplitudes. |
| `test_running_schedule` | Sorted, non-overlapping bouts inside the session, rest first and last, bout and rest lengths in range. |
| `test_run_session_schema` | A 60 s reference session (long enough for one bout: see the schedule) with two detectors, one sweep point and two recipes including Gridchyn 2020: every frame has exactly the contract's columns; `metrics` has one row per method × setting × expression. |
| `test_metrics_agree_with_match_events` | For one method, the metrics row equals `match_events` called directly on the written events and `truth_windows`. |
| `test_failures_are_recorded_not_raised` | Stub methods raising `ValueError` and `IndexError` each yield a `failures` row (`"IndexError: ..."`) and no events or metrics rows; the others still run. A Gridchyn configuration with a missing or invalid pre-rest baseline fails explicitly; absence of a running bout alone does not imply a missing baseline. |
| `test_resume_skips_finished_conditions` | With `--resume`, a condition whose files exist is not re-simulated (monkeypatched `run_session` call count). |
| manual | Smoke-test numbers and extrapolation recorded; spot-check PNGs inspected. |

The tests load `examples/benchmark` modules through a module-scoped fixture that prepends the
directory to `sys.path` and removes it afterwards (as phase 3's adapter tests do). Keep the suite in
seconds: 30-60 s sessions, method subsets via `run_session(..., methods=...)`. No test imports
`spot_check.py` or matplotlib (the dependency-floors job has no matplotlib).

## Fixtures

Simulated only: 30 s and 60 s sessions from `simulate_condition` with `session.duration_s` overrides; a
tmp_path output directory for the resume and schema tests.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- `examples/benchmark/output/` is ignored and nothing under it is committed; the smoke numbers are in the PR.
