# Phase 5 — Analyses: agreement, boundaries, operating curves, robustness, rates and participation

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#agreement-statistics)

Turn the runner's outputs into the answers the benchmark exists for: which events each method
finds and misses, by type; what its false positives are; how much methods agree and how that
clusters; how much matched events overlap; signed onset and offset errors against truth at three
truth definitions and between methods, and whether methods' errors are correlated; splits and
merges; recall at matched false-positive rates; how all of it moves across conditions; and what
detection does to event rates and participation.

**Inputs to read first:**

- Phase 4's outputs for a finished run (`examples/benchmark/output/<run_name>/`) and
  `examples/benchmark/run.py`'s module docstring (output columns).
- Phase 2: `ripple_detection.evaluate` (`match_events`, `compare_detectors`, `consensus_counts`,
  `label_by_overlap`).
- Phase 1a: `truth_windows` (truth is stored parametrically; windows are recomputed per fraction).

**Contracts referenced:**

- [Benchmark outputs](shared-contracts.md#benchmark-outputs) — read only; the summaries written here go to `examples/benchmark/results/`.
- [Primary expression](shared-contracts.md#primary-expression) — headline numbers use it; appendix tables use every expression.
- [Matching and pair metrics](shared-contracts.md#matching-and-pair-metrics), [Detector comparison table](shared-contracts.md#detector-comparison-table) — the per-session computations.
- [Truth windows](shared-contracts.md#truth-windows) — fractions 0.1 (matching), 0.25, 0.5.

**Designs referenced:** [agreement statistics](designs.md#agreement-statistics),
[operating curves](designs.md#operating-curves),
[bootstrap and permutation tests](designs.md#bootstrap-and-permutation-tests),
[rates and participation](designs.md#rates-and-participation).

## Tasks

- `examples/benchmark/analyze.py`, a CLI (`--run-name`, `--workers`) that loads a run and writes
  `examples/benchmark/results/<run_name>/`: one CSV and one PNG per analysis below, plus
  `summary.md` listing each file with one sentence on what it shows. Loading helpers rebuild each
  session's event and non-event tables from `truth.csv.gz` and re-match events per session (the
  runner stores events, not pairs); re-matching is parallel over sessions.
- `paired_bootstrap` and `sign_flip_test` per the design (settings there), in `analyze.py`. Every
  interval reported is a paired bootstrap over sessions, and across conditions over replicates;
  every "A differs from B" claim carries a sign-flip p-value.
- Keep computation and plotting apart: each analysis is a function returning a DataFrame, and its
  figure a separate function that imports matplotlib inside itself. Tests call only the former (the
  dependency-floors CI job has no matplotlib).
- Analyses (methods at `default` or `published` unless stated; reference condition unless stated):
  1. **Detection profile**: recall per event type against the network truth, per method; heatmap.
  2. **False-positive classes**: unmatched events (against the primary expression) labelled by
     `label_by_overlap` against a window table of every event type's components (labelled
     `"<event_type>:<expression>"`) and every non-event (labelled by type); stacked bars per
     method, fractions with intervals.
  3. **Pairwise agreement**: `jaccard`, `jaccard_true`, `jaccard_false` matrices (mean over
     sessions with intervals); dendrogram from average linkage on `1 - jaccard`.
  4. **Consensus**: distribution of `n_methods` per true event, by type; for false positives,
     groups of overlapping unmatched events across methods (connected components of their overlap
     graph) and how many methods each group spans.
  5. **Overlap quality**: IoU, coverage and temporal precision distributions per method (matched
     pairs, primary expression).
  6. **Boundary errors against truth**: signed onset and offset errors per method at fractions
     0.1, 0.25, 0.5 (medians, IQRs, intervals; box plots). Ripple-expression and burst-expression
     errors separately for methods scored on both.
  7. **Differences between methods**: matrices of `median_onset_difference`,
     `median_offset_difference`, `fraction_a_earlier_*` (signed, a − b), and
     `onset_error_correlation` / `offset_error_correlation`.
  8. **Splits and merges**: rates per method overall and on `ripple_doublet` events.
  9. **Operating curves**: per detector and expression, recall against false positives per minute
     across its sweep; recall and median boundary errors at the target false-positive rates in
     [designs.md#operating-curves](designs.md#operating-curves), with intervals; recipes as points on the curves of the detectors sharing their primary
     expression.
  10. **Robustness**: for each factor of the grid, recall, precision, and median onset error at the
      primary expression against the factor's level, one line per method, intervals and tests
      paired by replicate (common random numbers); the two crossed pairs as heatmaps per method. Methods whose recall changes by more than 0.1 across a factor's levels
      are listed in `summary.md`.
  11. **Rates and participation**: per the design.
- **Spot checks before reporting any trend** (a trend is reported only after its underlying events
  have been looked at): for every trend written into
  `summary.md` or the README, a figure of 6 underlying events (re-simulated by seed, truth windows
  and the methods' events overlaid) in `results/<run_name>/spot_checks/`, and a line in
  `summary.md` saying what the spot check showed. Check that no trend comes from missing sessions
  (failures), empty sweeps, or a unit error (seconds vs milliseconds).
- `examples/benchmark/README.md`: a "Results" section per analysis: what the figure shows, how to
  read the sign conventions, and "what it cannot show" (simulated truth; the event types and
  non-events are the simulator's taxonomy; no sequence content).
- Commit `examples/benchmark/results/<run_name>/` (CSV and PNG only, each file under 1 MB; the
  CLI fails if one would exceed it).
- `tests/CLAUDE.md`: an item for `tests/test_benchmark_analyze.py`.

## Deliberately not in this phase

- Attribution (one-at-a-time, Sobol, Shapley) — phase 6.
- Sequence capture and replay-candidate yield — deferred by the maintainer (overview decision 8).
- Mixed-effects models or any new dependency (overview Open Question 1).
- A results notebook (overview non-goals).
- Changing detectors, recipes or the simulator in response to results. A bug found here is fixed in
  its own PR with a test; the run is repeated and the fix is named in `summary.md`.
- README headline claims beyond the one pointer phase 4 added; whether the README gains a results
  summary is the maintainer's call after reading `summary.md`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_benchmark_analyze.py::test_paired_bootstrap_by_hand` | On a hand-built per-session frame, the estimate equals the full-sample statistic and the interval brackets it; same draws for every method (paired). |
| `test_sign_flip_exact` | For differences `[1, 1, 1, 1]`, p = 2/16; for `[1, -1]`, p = 1. |
| `test_recall_at_interpolates_in_log_rate` | Hand-built curve: interpolated value by hand; NaN outside the range; zero FP rate replaced by half the resolution. |
| `test_false_positive_labels` | A hand-built session: an event over a leakage burst is labelled `spike_leakage`, one over a `burst_only` burst `burst_only:burst`, one over nothing `background`. |
| `test_profile_and_consensus_on_tiny_run` | A two-session hand-built output directory (written by a fixture): detection profile and consensus tables equal hand values. |
| `test_results_size_limit` | Writing a file over 1 MB raises. |
| manual | Every trend in `summary.md` has a spot-check figure and a sentence on it. |

## Fixtures

A fixture writing a minimal run directory (two sessions, two methods, one condition) with
hand-chosen events and truth in `tmp_path`, following the output schema. No real data.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Every interval is a paired bootstrap over sessions, every signed number states its convention, and every reported trend has its spot check.
