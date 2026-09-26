# Phase 3 — Configure benchmarks to call the installed paper methods

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#recipe-executor)

The installed `ripple_detection.literature_methods` module owns published-method
implementations and supports users' measured recordings. This phase adds benchmark
configurations that call that API. It does not move, copy or replace detector bodies.
The runner needs uniform method calls; component attribution remains phase 6.

**Prerequisite (met):** the `literature-methods` implementation is on `master`
(PR #25): `Recording`, `list_methods`, `check_method`, `run_method`,
`save_events`/`load_events` and the named methods. Use the catalog `list_methods()`
returns, not old line numbers or a frozen count of functions, as the inventory.

## Inputs and contracts

- `src/ripple_detection/literature_methods.py`: the public recording constructor,
  method catalog, named calls and dispatcher. Each method declares its requirements
  once: `list_methods()` reports them (`signals`, `cells`, `intervals`,
  `external_inputs`, `measured_options`, `sampling_frequency`, `grid`) and
  `check_method(name, recording, behavior_intervals=..., **options)` lists every
  one a call lacks, without running. `NOT_REPRODUCED` names the papers with no
  packaged method and why.
- `docs/literature/implementation.md` and the paper notes: output roles, stages,
  source choices and limits. CSV values are not executable configurations.
- `examples/literature_recipes.py`: demonstration input assembly and calls into the
  package. It remains runnable without importing benchmark code.
- `tests/test_literature_methods.py`, `tests/test_literature_recipes.py`: existing
  behavior tests and positive controls. Keep these tests and public entry points.
- [Recipe config](shared-contracts.md#recipe-config),
  [primary expression](shared-contracts.md#primary-expression), and
  [benchmark outputs](shared-contracts.md#benchmark-outputs).

## Tasks

1. Add `examples/benchmark/recipe_configs.py` with the small `RecipeConfig` contract,
   `RECIPES`, `EXCLUSIONS`, `make_recording(session, config)` and
   `run_recipe(config, recording, behavior_intervals)`. Import `Recording`, `bounds`,
   `list_methods`, `check_method` and `run_method` from the installed package.
   `run_recipe` forwards the configured method, options and the session's
   `behavior_intervals` (a per-call argument, not a `Recording` field) to
   `run_method` and preserves its DataFrame and attrs.
2. Configure each chosen method by its stable function name and explicit options.
   Resolve defaults from its public signature at execution and save the resolved
   values. Use `stage="detection"` where supported. Keep protocol variants distinct;
   do not count decoding-candidate gates or secondary labels as equivalent outputs.
   Derive DOI, paper, role and interpretation from the package catalog/results.
   `primary_expression` is an explicit benchmark decision reviewed per method.
3. Build package `Recording` objects from simulated observations, with explicit
   selections for channels, pyramidal/place cells, reference signals, sleep and
   normalization intervals, templates and any external inventory, at the rate
   the method's `sampling_frequency` requires; behavioral intervals go to each
   call instead. The benchmark
   input policy and assumptions are serialized. Prefer `Recording.from_arrays`
   so benchmark calls exercise the measured-data path. Do not infer missing method
   settings from truth windows or silently use simulation-only fallbacks. Unit-type
   and state labels supplied by the simulator must be identified as known labels;
   an external ripple inventory must come from a named detector/configuration,
   never the simulator's event truth. Exclude unsupported methods with reasons.
4. Add an inventory coverage check: every name in the integrated `list_methods()`
   catalog has a runnable configuration or an explicit exclusion explaining the
   missing input or unresolved setting; `check_method`'s messages give the
   missing inputs. A partial implementation's role stays
   visible. Missing requirements are exclusions/failures, never zero detections.
5. Record configurations and resolved result metadata in `methods.csv` as specified
   by the output contract. Separate literature-specified values from benchmark
   assumptions. Fixed thresholds and baseline choices are not tuned to results.
6. Document running a named configuration, supplying inputs, stages, roles and
   exclusions in `examples/benchmark/README.md`. Link to the package implementation
   guide for method definitions; do not duplicate parameter tables or docstrings.
   Update `tests/CLAUDE.md` for adapter tests. Preserve the standalone demo; phase 2
   owns its evaluation-function change. No package API change is required here.

## Validation

- Compare each configured call with the direct public method on the same recording
  and options, using exact DataFrame equality and metadata equality. Include all
  runnable configurations; nonempty controlled fixtures cover important branches.
- Exercise baseline/state restrictions, missing data, timestamps with large origins,
  stage selection and methods requiring explicit options. Reuse package fixtures
  where practical; compare through public APIs, not private implementation registries.
- Check options, method names, unique configuration IDs, catalog coverage and serialized
  metadata. Confirm an unknown method, missing option and missing required input
  produce explicit errors rather than empty successful results.
- Check the simulated-input adapter's unit masks, reference/interval assignments and
  lack of synthetic stand-ins when inputs are absent. Externally supplied detector
  outputs retain their own provenance. Discarded recordings must be collectable.
- The standalone demo still imports only package APIs and runs; adapter creation alone
  must not change its results. Behavior corrections belong in the package, in their
  own PR with source evidence and regression tests, not in benchmark-specific
  overrides.
- Run relevant tests and CI, including dependency floors. Lint/format benchmark
  examples explicitly. A stored snapshot can guard integration drift but cannot
  replace direct comparison or establish agreement with authors' original outputs.

## Attribution boundary

Phase 6 can define experimental component templates using public package primitives.
A template represents a named method only after checking its operation order,
normalization and native grids against the package and demonstrating event equality
on positive controls, edge cases and reference sessions. Methods that cannot be
represented stay fixed comparison points with a reason. Experimental templates
never become alternate implementations used by the main benchmark or the demo.

## Out of scope

- Moving public methods or `Recording` into `examples/`, deleting the named functions,
  or importing benchmark code from the installed package or standalone demo.
- Reimplementing all paper methods as a second declarative execution engine.
- Persistent recording caches without an explicit invalidation/lifetime design.
- Altering literature values, detector behavior, or adding replay decoding/scoring.

## Review

Before opening this phase's PR, request independent review of input policies,
public-call parity, catalog coverage, provenance and documentation. Verify that
there are no copied detector bodies or benchmark imports in the installed package.
