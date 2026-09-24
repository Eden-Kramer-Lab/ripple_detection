# Overview — Decisions, scope, integration, risks

[← back to PLAN.md](PLAN.md)

## Decisions (settled with the maintainer, 2026-09-24)

Each was asked and answered; do not reopen them without asking.

1. **Purpose.** A methods benchmark of the field's detectors. It may become a paper; that is not
   decided, so nothing is organized around a paper.
2. **Ground truth.** Fully simulated only. No semi-synthetic data (real background with injected
   events), no hand-labeled real data, no unlabeled-real-data validity analysis in this plan.
3. **Simulator.** A latent-event model that extends `simulate_session`. A biophysical
   NEURON/LFPy network and MEArec (used by the Spyglass `spikesorting-v2` branch for spike-sorting
   ground truth; it simulates extracellular spikes, not ripple LFP) were considered and not chosen.
4. **What is true.** The latent *network event* is the truth, expressed as a ripple (LFP), a sharp
   wave (radiatum LFP) and a population burst (spikes), each with its own truth window and with
   coupling the simulation controls. Every method is scored against the network event and against
   each expression; see [primary expression](shared-contracts.md#primary-expression).
5. **Event types:** canonical SWR with a burst; ripple with weak participation; burst without a
   ripple; long burst spanning a ripple doublet or triplet; sharp wave without a ripple.
   **Non-events:** spike-waveform leakage into the LFP, broadband EMG/chewing artifacts, fast-gamma
   bursts (60-100 Hz), theta-state population bursts during running. The maintainer accepted this
   taxonomy "for now"; prevalences are condition parameters, not fixed.
6. **Scope, layered.** The nine package detectors, each swept over its threshold (operating
   curves); then the published recipes placed as points on those curves.
7. **Questions answered:** same or different events (detection profile per event type, classified
   false positives, pairwise agreement, consensus per event); how much matched events overlap (IoU,
   coverage, temporal precision per matched pair); signed onset and offset error against truth, at
   three truth definitions, *and* signed differences between detectors, with the correlation of
   their errors; splits and merges; operating curves at matched false-positive rates; robustness;
   parameter attribution; downstream event rates and cell participation.
8. **Deferred by the maintainer:** sequence capture (ordered replay content inside bounds) and
   replay-candidate yield. Not in any phase.
9. **Location.** Everything in this repository. Only what helps a user with their own data goes in
   `src/ripple_detection/`: the network-event simulator and the evaluation module. The recipe
   configurations, runner, analyses and figures live in `examples/benchmark/`. Large outputs are
   git-ignored; small summary tables are committed.
10. **Recipe configs live in `examples/benchmark/`**, not the public API; promote to a public module
    later only if users ask to run a published rule by name.
11. **Primary use of the package stays user detection.** Benchmark-only code must not enter the
    public API. (Maintainer's words: "The primary use of this package is for users to be able to do
    ripple detection on their data.")

Defaults chosen by the planner, open to override, recorded in [Open Questions](#open-questions):
no new dependencies (outputs as `.csv.gz`, estimators in NumPy, `concurrent.futures` for
parallelism, paired bootstrap and permutation tests in place of mixed models).

## Current codebase integration points

Branch `detector-benchmark` starts from `literature-methods` at commit 6d866b8. Line numbers are
from that commit.

- `src/ripple_detection/simulate.py:1150` — `simulate_session`: **untouched**; its output must stay
  identical (tests `tests/test_simulate.py::TestSessionStates::test_the_defaults_are_unchanged` and
  the snapshot tests guard it).
- `src/ripple_detection/simulate.py:1075-1147` — `SimulatedSession` (frozen dataclass, fields at
  1108-1118, `__post_init__` 1120-1129, `ripple_windows` property 1131-1147): gains the fields in
  [SimulatedSession additions](shared-contracts.md#simulatedsession-additions), each with a
  default, so existing constructors keep working (phase 1a).
- `src/ripple_detection/simulate.py:407` `_ripple_waveform`, `:509` `_gaussian_window`, `:523`
  `_add_ripple_bursts` — symmetric Gaussian envelope, constant frequency; the new simulator adds its
  own asymmetric, chirped renderer rather than changing these (phase 1a).
- `src/ripple_detection/simulate.py:563` `_correlated_noise`, `:587` `_add_common_mode_artifacts`,
  `:739` `_add_sharp_wave_pair`, `:844` `simulate_multiunit` (rate modulation at 910-917), `:939`
  `simulate_speed`, `:996` `simulate_theta_delta` — reused by the new simulator.
- `src/ripple_detection/core.py:663` `_event_bounds` — every evaluation function reads event tables
  through it. `:1813` `_overlaps`, `:1855` `require_overlap`, `:1681` `merge_close_events` —
  untouched.
- `src/ripple_detection/registry.py:408` `DETECTORS`, `:427` `get_detector` — the runner resolves
  detectors by name; untouched.
- `src/ripple_detection/__init__.py:72-85` — simulate exports; new public names are added here and
  to `__all__`; `tests/test_public_api.py` pins `__all__`.
- `examples/simulation_study.py:79-105` — `score`: any-overlap matching with median boundary errors.
  **Removed in phase 2**, replaced by `ripple_detection.evaluate` (and likewise the recipes'
  `score`, `examples/literature_recipes.py:1185-1194`). The study itself, its `THRESHOLD_SWEEPS`
  (`:41`) and its notebook stay as the fast study CI runs and the README cites; phase 4 names the
  trigger for retiring them.
- `examples/literature_recipes.py` — recipes as functions (`Recording` at :60, `make_recording`
  :153, `Recipe`/`recipe` :227-243, `NOT_REPRODUCED` :1171, `score` :1185, `run_all` :1197). Phase 3
  rewrites it on top of declarative configs and removes the function bodies; its public behaviour
  (the results table) must not change.
- `tests/test_literature_recipes.py` — runs every recipe; kept and pointed at the configs in phase 3.
- `.gitignore` — gains `examples/benchmark/output/` (phase 4).
- `CLAUDE.md` "Core Module Structure" lists five modules; phase 2 adds `evaluate.py` as the sixth.

## Scope and dependency policy

### Goals

- A simulator users can call to test a detector on data whose events, event types and bounds are
  known, including events a detector should *not* report.
- Evaluation functions users can call on any two event inventories, with or without a truth.
- A reproducible benchmark answering the questions in decision 7, re-runnable with one command.

### Non-Goals

- Real data of any kind (decision 2).
- Sequence capture, replay-candidate yield (decision 8).
- A public recipe API (decision 10).
- Changing any existing detector, its defaults or its output.
- Tuning the simulator until detectors look good: simulator parameters are set from the literature
  before results are seen (see [Risks](#risks-and-mitigations)).
- Paper figures styled for a journal; phase 5 produces analysis figures only.
- A notebook for the benchmark results: it would read git-ignored outputs, which CI cannot run;
  `examples/benchmark/README.md` documents the figures instead.

### Dependency policy

No new runtime or optional dependencies. Outputs are gzip CSV through pandas; the Sobol estimator,
Shapley values and bootstrap are NumPy; parallelism is `concurrent.futures.ProcessPoolExecutor`;
`scipy.optimize.linear_sum_assignment` (already a dependency through scipy) does the matching;
figures use matplotlib from the existing `examples` extra.

## Metrics

- Phase 1a/1b: truth windows computed analytically equal those measured on the rendered signal
  (envelope threshold crossings) to within one sample; defaults of existing functions unchanged
  (existing tests pass untouched).
- Phase 2: on hand-built cases, every metric equals its hand-computed value exactly. The
  examples' scorers become one-to-one, so their recall and precision can fall where one event
  overlapped two windows or two events one window; the PR shows the old and new README tables side
  by side.
- Phase 3: every declarative recipe returns *exactly* the event bounds the current function returns
  (array equality, in-process, while both exist), and a rounded snapshot guards it afterwards.
- Phase 4: the smoke test's measured runtime and output size, and the extrapolation to the full
  grid, are recorded in the PR; outputs follow the [output schema](shared-contracts.md#benchmark-outputs).
- Phase 5/6: every summary table is regenerated from outputs by one command; every reported
  interval is a 95% bootstrap interval over sessions (replicates, across conditions), except the
  Sobol indices', which resample the sample rows because each value already averages sessions.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The simulator's choices decide the winners ("tuned to the benchmark"). | Parameters are set from published ranges in [designs.md](designs.md#parameter-sources) *before* running detectors; the robustness grid (phase 4-5) varies each; conclusions are reported per condition, not only at the reference. |
| Truth bounds are a convention (where does a ripple start?). | Truth windows are analytic functions of the envelope, evaluated at 10%, 25% and 50% of peak; boundary errors are reported at all three. |
| MUA detectors judged against ripple bounds, or ripple detectors against burst bounds. | Every method is scored against every expression and the network event; headline numbers use the method's [primary expression](shared-contracts.md#primary-expression). |
| Declarative recipes drift from the audited functions. | Phase 3 regression test: exact equality of event bounds for every recipe before the functions are removed. |
| Recipes that cannot be decomposed (Kay, Karlsson, Zugaro, Long, Carey, silence-bounded) break attribution. | They are fixed points in agreement analyses and excluded from the Sobol factor space; phase 6 says so. |
| Runtime grows past a workstation. | Phase 4 smoke-tests one session, measures, and extrapolates before any full run; the grid is one-factor-at-a-time plus two crossed pairs, not full factorial. |

## Rollout Strategy

Phase order: 1a, then 1b and 2 (2 needs 1a's `truth_windows` for its integration test); 3 after
2, since both edit `examples/literature_recipes.py` (2 replaces its `score`, 3 its recipes); 4
needs 1a, 1b, 2 and 3; 5 needs 4; 6 needs 3, 4 and 5 (it imports phase 5's `paired_bootstrap`).
Every phase must pass CI's dependency-floors job (Python 3.10, NumPy 1.24, SciPy 1.10, pandas
2.0, no matplotlib, `.github/workflows/release.yml:94-111`), which runs the whole test suite,
benchmark tests included: plotting code imports matplotlib inside the plotting functions, and no
test calls them.
Each phase is one PR onto `detector-benchmark` (itself on `literature-methods`); rebase the stack
when `literature-methods` merges to `master`.

Additive only. New public functions (`draw_network_events`, `simulate_network_session`,
`draw_non_events`, `truth_windows`, and the `evaluate` module) ship with docs and CHANGELOG entries
in the phase that adds them. `simulate_session` and every detector are unchanged. The benchmark is
run by hand (`uv run python examples/benchmark/run.py ...`); nothing runs in CI except the tests
each phase adds, which use short sessions.

## Open Questions

1. **Mixed models for headline effects.** The design summary mentioned them; the plan uses paired
   bootstrap and permutation tests instead, to avoid a `statsmodels` dependency. Revisit only if the
   work becomes a paper and a reviewer asks; then add `statsmodels` to an optional extra.
2. **Output format.** `.csv.gz` (no dependency) rather than parquet (needs `pyarrow`). Revisit if
   outputs pass ~1 GB.
3. **Event-type taxonomy.** Accepted "for now" (decision 5). New types go in
   [designs.md](designs.md#event-types) and the truth table's `event_type` vocabulary; no phase
   depends on the list being final.
4. **Promoting recipe configs to the public API** (decision 10): deferred until users ask.

## Estimated Effort

Phase 1a ~600 LOC src + ~400 tests; 1b ~300 + ~200; phase 2 ~450 + ~450; phase 3 ~900 (configs and
executor) + ~100 tests, net shrink of `literature_recipes.py`; phase 4 ~400 + ~100; phase 5 ~600 +
`examples/benchmark/README.md`; phase 6 ~400 + a README section. Compute: the full grid
([designs.md#conditions-grid](designs.md#conditions-grid)) is estimated at 10-15 core-hours before
the smoke test, and about 0.5 GB of memory per worker for 600 s sessions; phase 6 at about 1 core-hour per
family.
