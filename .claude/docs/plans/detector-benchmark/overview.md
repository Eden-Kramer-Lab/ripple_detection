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
   bursts (60-100 Hz reference plus a 90-140 Hz challenge), theta-state population bursts during running. The maintainer accepted this
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
10. **Published methods live in the installed package.** The maintainer clarified on
    2026-09-25 that running published paper methods is a primary package use. Benchmark
    call configurations live in `examples/benchmark/` and dispatch to the installed API.
    This supersedes the earlier decision to keep executable recipes in examples.
11. **Primary use of the package stays user detection.** Benchmark-only code must not enter the
    public API. (Maintainer's words: "The primary use of this package is for users to be able to do
    ripple detection on their data.")
12. **Real recordings as a reference, not a truth.** The maintainer asked on 2026-09-26 that
    the plan check the package's methods on real recordings whose authors released their
    events, matching what the package computes against those events as far as the inputs can
    be verified ([phase 7](phase-7-reference-recordings.md)). Released events are another
    pipeline's output: they test implementation parity, not detection quality, and decision 2
    still governs the benchmark's truth.
13. **Simulator validation and sensitivity (2026-09-26).** Following scientific review, the
    maintainer requested rendered-measurement validation and six alternatives: coupled event
    strengths, spatial ripple variability, changing background variance, nearby gamma,
    refractory spiking and a second envelope shape. The latent-event model and fully synthetic
    scope remain the basis. [simulator-validation.md](simulator-validation.md) defines the
    measurements, source conventions, model options and report required before benchmark runs.

14. **Scoring safeguards (2026-09-27).** After an external review of phase 2, the maintainer
    chose: keep the matching rule (most pairs, then summed IoU) and score it at predeclared IoU
    levels, `MATCH_IOU_LEVELS = (0.0, 0.2, 0.5)`, with 0.0 (any overlap) the headline and the
    others a sensitivity analysis reported beside it; add `jaccard_truth_ids` (which truth
    events two methods both found) to `compare_detectors`, keeping `jaccard_true` as agreement
    of the detected intervals; report absolute boundary errors beside signed ones, each median
    with its pair count and the method's recall, and compare methods' timing on the truth
    events both recovered; keep operating curves descriptive, and choose any threshold the
    benchmark recommends on calibration replicates, reporting its performance on held-out ones.
    The levels are fixed here, before any benchmark result exists.

Defaults chosen by the planner, open to override, recorded in [Open Questions](#open-questions):
no new dependencies (outputs as `.csv.gz`, estimators in NumPy, `concurrent.futures` for
parallelism, paired bootstrap and permutation tests in place of mixed models).

## Current codebase integration points

The `literature-methods` changes are on `master` (PR #25), under this branch.
Example-owned detector functions are obsolete; line numbers cited in these files
were checked against `master` on 2026-09-26.

- `simulate_session` defaults and signals remain unchanged. Phase 1 extends simulation
  additively with the fields in [shared contracts](shared-contracts.md#simulatedsession-additions).
- The nine detectors and `registry.py` retain their public contracts.
- `src/ripple_detection/literature_methods.py` owns `Recording`, the method catalog,
  public named methods, `run_method`, output roles and stage selection. Phase 3 wraps
  these calls with benchmark input policies; it neither copies nor deletes them.
- `examples/literature_recipes.py` stays a standalone package-usage example. Phase 2
  replaces its overlap scorer with `ripple_detection.evaluate`; phase 3 preserves
  its package calls and input assembly.
- Package method tests continue to check source-derived behavior. Adapter tests
  check public-call parity, explicit requirements, inventory coverage and metadata.
- `examples/simulation_study.py` and its notebook remain the fast study until the
  phase-4 retirement condition is met. Phase 2 replaces its scorer as planned.
- New simulator/evaluation exports, documentation and CHANGELOG entries are added
  with their implementations. Benchmark-only runner/configuration code stays in examples.

## Scope and dependency policy

### Goals

- A simulator users can call to test a detector on data whose events, event types and bounds are
  known, including events a detector should *not* report.
- Evaluation functions users can call on any two event inventories, with or without a truth.
- A reproducible benchmark answering the questions in decision 7, re-runnable with one command.
- A simulator validation report based on published measurement definitions, with explicit
  assumptions and sensitivity results showing which detector conclusions depend on the model.

### Non-Goals

- Real data as ground truth, semi-synthetic data or hand labels (decision 2). Real recordings
  enter only as reference-event parity checks (decision 12, phase 7).
- Sequence capture, replay-candidate yield (decision 8).
- Moving or duplicating the public paper-method API into benchmark code (decision 10).
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

One exception, for phase 7 only: its scripts stream DANDI NWB files with `remfile` and `h5py`,
which are not declared anywhere in `pyproject.toml`. They run in an isolated environment
(`uv run --with remfile --with h5py ...`) and record the versions used; no package or test
imports them, so CI and the dependency-floors job are unaffected. GNU Octave, which runs the original MATLAB code for phase 7's original-code comparisons, is a system tool the maintainer installed (2026-10-03), not a dependency; no test runs it.

## Metrics

- Phase 1a/1b: analytic truth crossings agree with the explicit rendered modulation envelope
  within one sample at both envelope powers. Measure differences from the sampled Hilbert
  envelope rather than assuming exact agreement. Existing simulator defaults remain unchanged.
- Phase 2: on hand-built cases, every metric equals its hand-computed value exactly. The
  examples' scorers become one-to-one, so their recall and precision can fall where one event
  overlapped two windows or two events one window; the PR shows the old and new README tables side
  by side.
- Phase 3: every configured call equals its direct public method call, including
  event diagnostics and metadata; every catalog method is configured or explicitly excluded.
- Phase 4: a ready [simulator report](simulator-validation.md) precedes detector runs. The smoke
  test's runtime/output size and extrapolation to all 43 conditions (440 benchmark sessions,
  plus validation simulations) are recorded; outputs follow the
  [output schema](shared-contracts.md#benchmark-outputs).
- Phase 5/6: every summary table is regenerated from outputs by one command; every reported
  interval is a 95% bootstrap interval over sessions (replicates, across conditions), except the
  Sobol indices', which resample the sample rows because each value already averages sessions.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The simulator's choices decide the winners ("tuned to the benchmark"). | Fix source-based targets and assumed stress settings before detector results; validate rendered measurements, then report all six structural alternatives alongside reference. Limit conclusions to the tested assumptions. |
| Plausible parameter ranges yield implausible observations. | Check duration definitions, observed participation, realized rates after rejection, spatial structure and joint distributions in the simulator report. Mark unsupported properties explicitly. |
| Truth bounds are a convention (where does a ripple start?). | Use analytic envelopes at 10%, 25% and 50% of peak, and compare powers 2 and 4 at matched half-maximum widths; disclose Hilbert-envelope discrepancies and spatial delays. |
| MUA detectors judged against ripple bounds, or ripple detectors against burst bounds. | Every method is scored against every expression and the network event; headline numbers use the method's [primary expression](shared-contracts.md#primary-expression). |
| Benchmark calls drift from package methods. | Phase 3 calls the package directly and verifies input/options and output parity. Phase 6 experimental templates require separate equivalence checks. |
| Recipes that cannot be decomposed (Kay, Karlsson, Zugaro, Long, Carey, silence-bounded) break attribution. | They are fixed points in agreement analyses and excluded from the Sobol factor space; phase 6 says so. |
| Runtime grows past a workstation. | Phase 4 smoke-tests one session, measures, and extrapolates before any full run; the grid is one-factor-at-a-time plus two crossed pairs, not full factorial. |

## Rollout Strategy

Phase order: 1a, then 1b and 2 (2 needs 1a's `truth_windows` for its integration test); 3 after
2, so adapter validation uses the shared evaluator and integrated package methods; 4
needs 1a, 1b, 2 and 3; 5 needs 4; 6 needs 3, 4 and 5 (it imports phase 5's `paired_bootstrap`);
7 needs only 2 (`match_events`) and can run alongside 3-6.
Every phase must pass CI's dependency-floors job (Python 3.10, NumPy 1.24, SciPy 1.10, pandas
2.0, no matplotlib, `.github/workflows/release.yml:94-111`), which runs the whole test suite,
benchmark tests included: plotting code imports matplotlib inside the plotting functions, and no
test calls them.
Each phase is one PR onto `detector-benchmark` (itself on `master`).

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
4. **Method ownership:** settled by revised decision 10; public methods stay in the package.

## Estimated Effort

Earlier LOC estimates for phases 1a, 1b, 4 and 5 predate the simulator validation and model
alternatives and must be re-estimated during implementation. Phase 2 ~450 LOC + ~450 tests;
phase 3 uses thin call configurations and explicit input policies; phase 6 ~400 + a README section.
The previous 10–15 core-hour estimate covered 380 benchmark sessions. The grid now has 440,
plus 215 simulator-only validation sessions at default settings; measure their separate costs
before the full run. The prior memory estimate was about 0.5 GB per worker at 600 s, and phase 6
about 1 core-hour per family; both remain provisional pending smoke measurements.
