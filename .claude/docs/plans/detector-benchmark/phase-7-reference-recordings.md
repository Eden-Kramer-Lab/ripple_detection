# Phase 7 — Real recordings against released reference events

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [phase 2](phase-2-evaluation.md)

The simulated benchmark scores methods against a truth the simulator defines. This phase checks
the package's implementations against events their authors released for real recordings: verify
each recording's inputs, run the package method with the source's own settings, match the two
inventories, and explain every systematic difference. Released events are another pipeline's
output, not ground truth ([decision 12](overview.md#decisions-settled-with-the-maintainer-2026-09-24)):
agreement is evidence that the package reproduces the source, and a disagreement is traced to a
stage (filter, trace, normalization, threshold, bounds, post-processing) before anything changes.

Scripts and a README live in `examples/reference_recordings/`, never in the installed package.
Data are downloaded or streamed to a cache outside the repository; nothing large is committed.

**Inputs to read first:**

- [docs/literature/datasets.md](../../../../docs/literature/datasets.md) and the
  [dataset catalog](../../../../src/ripple_detection/data/literature_datasets.csv): the deposits
  and what was verified in each.
- [docs/literature/sources.md#dataset-catalog-checks](../../../../docs/literature/sources.md#dataset-catalog-checks):
  the file-level checks (hashes, table structures, stored parameters) this phase starts from.
- The paper notes: [Carey 2019](../../../../docs/literature/papers/28_Carey_2019.md),
  [Denovellis 2021](../../../../docs/literature/papers/13_Denovellis_2021.md),
  [Yang 2024](../../../../docs/literature/papers/02_Yang_2024.md) (which reuses Huszár 2022).
- `src/ripple_detection/literature_methods.py`: `carey_2019`, `denovellis_2021`,
  `Recording.from_arrays`, `check_method`, `run_method`, `save_events`.
- Phase 2's `match_events` and `EventMatching` (the only dependency on the benchmark phases).

## Sessions

Checked on 2026-09-26 (evidence in `sources.md#dataset-catalog-checks`). "To verify first" items
are tasks, not known problems.

| Reference | Recording (inputs) | Released events | Package method | To verify first |
| --- | --- | --- | --- | --- |
| Carey 2019, R050-2014-03-29 | DataLad MotivationalT session, over https: 20 `.ncs` (`CSC03a` is 20 MB), 30 `.t`, `VT1.nvt` (272 MB), `Events.nev`, `metadata.mat` (`SWRtimes`: 50 template examples; `SWRfreqs`: the `amSWR` settings; `taskvars`) | `R050-2014-03-29-candidates.mat` in vandermeerlab/papers@3aebc8e: 1654 candidates and their stored configuration (threshold 4 on the joint score rescaled to mean 0.5; at least 20 ms and 5 cells; speed below 10 in pixel units; theta z below 2) | `carey_2019` with `example_ripples` from `SWRtimes` | position units and the pixel speed threshold; that the 30 released units are the ones the 5-cell rule counted; one clock for `.ncs`, `.t`, `.nvt` and the candidates; whether the 2017 `SWRtimes` are the examples behind the 2015 candidates (compare the template spectrum with `SWRfreqs.freqs1`) |
| Denovellis 2021 (Frank lab), rat Remy, days 35-37 | Dryad 10.7272/Q61N7ZC3 (Zenodo 5587779 `Remy.tar.gz`, 3.2 GB): `remyeeg{day}-{epoch}-{tetrode}.mat` raw LFP at 1500 Hz, `remypos`, `remytask` (epochs 1, 3, 5 sleep; 2, 4 W-track), `remytetinfo` | `remyca1rippleskons{35,36,37}.mat`: per epoch, CA1 consensus events (`nstd=2`, `min_suprathresh_duration=0.015`, 24 CA1 tetrodes) with `baseline`, `std` and the full `powertrace`; no speed rule, no excluded times | `Kay_ripple_detector` and `denovellis_2021` with the speed rule off | the Frank-lab filter against the package's; how `powertrace` is built (squared or not, smoothing, square root); the bound rule; epoch and time alignment of `eegtimesvec_ref` with the LFP files |
| Huszár et al. 2022, reused by Yang 2024: DANDI 000552 v0.230630.2304 | 13 sessions have a raw recording (`*-raw_ecephys.nwb`, 5.6-184 GB, streamed in slices) and a processed file of the same subject and date | `/processing/ecephys/Ripples`: TimeIntervals with `start_time`, `stop_time`, `peaks` and a raw snippet per event (seen in two other sessions) | the detector Huszár 2022 describes, if the package has it | that the 13 paired processed files hold a `Ripples` table; the ripple channel; the detection method and its settings in Huszár 2022's Methods; the raw-to-table clock (the per-event `ripple_raw` snippets can check it) |

Not included, with the reason recorded in the README:

- Widloski 2022/2025 (Zenodo 16916108): no event start or end times are released.
- Grosmark (CRCNS hc-11, DANDI 000044), Gillespie (DANDI 000115), Shin (DANDI 000978) and Tirole
  (Dryad): no event tables in what was inspected.
- Maboudi 2018 `fig1.nel` (457 MUA epochs, 277 PBEs): include only if the object carries the
  spikes they were computed from; check first.
- CRCNS hc-14 (Harvey; ripple `.evt` files reported) and hc-18 (Drieu): need a CRCNS account.
  Add them if one is available.

## Tasks

1. **Fetch and record.** `examples/reference_recordings/fetch.py` downloads or streams each input
   into a cache directory (an environment variable, default outside the repository), checks the
   published checksums (Zenodo MD5, DANDI SHA-256, Dryad SHA-256) and writes a manifest of URLs,
   sizes and hashes. Stream DANDI NWB files with `remfile` + `h5py`; never download whole raw files.
2. **Verify inputs before detecting.** Per session, write `inputs.json`: sampling rate, units,
   channel selection (CA1 tetrodes from `tetinfo`, the ripple channel from the source), gaps, the
   speed units and conversion, and one clock for LFP, spikes, position and released events (every
   released event lies within recorded samples). A failed check stops that session and is
   recorded; it is not patched to make the run proceed.
3. **Stage parity where the source releases intermediate values.** Before comparing events,
   compare the package's trace with Remy's `powertrace` (per epoch: correlation, maximum relative
   difference, mean and SD against the stored `baseline` and `std`) and Carey's template spectrum
   with `SWRfreqs`. A difference in the trace explains event differences before any threshold
   is considered.
4. **Run the package method with the source's settings.** Settings come from the stored
   parameters in the released files or the paper. Nothing is tuned toward agreement; every
   setting the source does not establish is listed as an assumption. Save each result with
   `save_events` (detectors: the same two-file layout with the detector's name and resolved
   parameters as attrs).
5. **Compare.** `match_events(released, detected)` per session and epoch: counts, recall,
   precision, median IoU, quartiles of the signed onset and offset errors, splits and merges, and
   the unmatched events on each side; any-overlap agreement alongside for context.
6. **Explain the differences.** Plot a sample of unmatched and poorly aligned events of each side
   with the trace, the threshold and both inventories. Attribute each class of difference to a
   stage. A package bug found here is fixed in its own PR with a regression test (a small slice of
   the real data or a synthetic reproduction); a source quirk goes into the paper note; a catalog
   finding goes into the dataset CSV and `sources.md`.
7. **Report.** `examples/reference_recordings/README.md` states the inputs, settings, assumptions,
   results and explained differences; a small results CSV is committed (under 1 MB). Nothing is
   stated as agreement without the numbers behind it.

## Validation

| Check | What it establishes |
| --- | --- |
| Positive control on Remy: threshold the released `powertrace` itself at `baseline + 2 std` for at least 15 ms, with the bound rule the package uses | Events equal the released table (to one sample), or the bound rule that reproduces them is identified. This separates thresholding and bounds from filtering. |
| Smoke test: one Remy epoch and a 10-minute Carey slice first | Runtime and memory measured before full sessions; extrapolated before the full run. |
| Timestamps at the sources' own origins (Neuralynx microseconds, NWB seconds) | Tolerances scale with the timestamps; no rounding shifts a bound. |
| `fetch.py` checksum test on a small file | A changed or partial download fails instead of being used. |
| Tests for the parsing helpers (`.t`, `.nvt`, MAT tables), on files or slices small enough for CI | The readers return the documented shapes, units and clocks. |

No agreement target is set in advance; results are reported as measured.

## Out of scope

- Tuning methods or settings until the inventories agree.
- Treating released events as ground truth, or scoring methods against them in the benchmark's
  metrics.
- Replay decoding or scoring.
- Committing recordings or large outputs.

## Review

Before opening this phase's PR, request independent review of the input verification, the stage
parity, the settings and assumptions for each session, and every stated agreement or explanation.
