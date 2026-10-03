# Phase 7 — Real recordings against released reference events

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [phase 2](phase-2-evaluation.md)

The simulated benchmark scores methods against a truth the simulator defines. This phase checks
the package's implementations against events their authors released for real recordings: verify
each recording's inputs, run the package method with the source's own settings, match the two
inventories, and explain every systematic difference. Released events are another pipeline's
output, not ground truth ([decision 12](overview.md#decisions-settled-with-the-maintainer-2026-09-24)):
agreement is evidence that the package reproduces the source, and a disagreement is traced to a
stage (filter, trace, normalization, threshold, bounds, post-processing) before anything changes.

**Goal (maintainer, 2026-10-03):** establish that the package's implementations of the detectors
and paper methods are faithful to their sources. Two kinds of evidence serve it. Released events
on real recordings (the sessions below) test a method end to end where an author published both.
Comparisons with the original code (see [Original-code comparisons](#original-code-comparisons))
run the source's own code and the package on the same input; they need no released events, so
they reach the detectors no deposit covers. MS10 showed the second is the sharper test: the
source's steps, rerun, gave its 429 events exactly, and isolated the one rule (the merge cap) in
which the package departs. Sessions are chosen for distinct methods, not repeats of one.

Scripts and a README live in `examples/reference_recordings/`, never in the installed package.
Data are downloaded or streamed to a cache outside the repository; nothing large is committed.

**Inputs to read first:**

- [docs/literature/datasets.md](../../../../docs/literature/datasets.md) and the
  [dataset catalog](../../../../src/ripple_detection/data/literature_datasets.csv): the deposits
  and what was verified in each.
- [docs/literature/sources.md#dataset-catalog-checks](../../../../docs/literature/sources.md#dataset-catalog-checks):
  the file-level checks (hashes, table structures, stored parameters) this phase starts from.
- The paper notes: [Carey 2019](../../../../docs/literature/papers/28_Carey_2019.md),
  [Yang 2024](../../../../docs/literature/papers/02_Yang_2024.md) (which reuses Huszár 2022),
  [Maboudi 2018](../../../../docs/literature/papers/31_Maboudi_2018.md) and the Harvey 2023 note
  (hc-14's reuse).
- `src/ripple_detection/literature_methods.py`: `carey_2019`, `yang_2024`, `maboudi_2018`,
  `Recording.from_arrays`, `check_method`, `run_method`, `save_events`.
- `Zugaro_ripple_detector` (`detectors/_zugaro.py`, the FMAToolbox/buzcode `FindRipples` rule) and
  `detect_events_from_trace` (`detectors/_trace.py`), for the buzcode and CellExplorer sessions.
- Phase 2's `match_events` and `EventMatching` (the only dependency on the benchmark phases).

## Sessions

Checked on 2026-09-26; widened on 2026-10-02 after file-level checks of every deposit that
still rested on metadata, the CRCNS file lists (fetched with the maintainer's account) and the
Buzsáki lab databank (evidence in `sources.md#dataset-catalog-checks`; databank sessions below).
"To verify first" items are tasks, not known problems.

The databank's HTTP webshare (`buzsakilab.nyumc.org/datasets/`) redirects to the lab's index
since mid-2026, and the live copy needs a Globus login. Its event files come from Internet
Archive (Wayback) captures, which the maintainer accepted as sources on 2026-10-02: each is
used only when its archived length equals the original server length (the archive keeps only
the first 1 MiB of larger files), and the manifest records the capture URL and time. The
matching recordings are on DANDI.

| Reference | Recording (inputs) | Released events | Package method | To verify first |
| --- | --- | --- | --- | --- |
| Carey 2019, R050-2014-03-29 | DataLad MotivationalT session, over https: 20 `.ncs` (`CSC03a` is 20 MB), 30 `.t`, `VT1.nvt` (272 MB), `Events.nev`, `metadata.mat` (`SWRtimes`: 50 template examples; `SWRfreqs`: the `amSWR` settings; `taskvars`) | `R050-2014-03-29-candidates.mat` in vandermeerlab/papers@3aebc8e: 1654 candidates and their stored configuration (threshold 4 on the joint score rescaled to mean 0.5; at least 20 ms and 5 cells; speed below 10 in pixel units; theta z below 2) | `carey_2019` with `example_ripples` from `SWRtimes` | position units and the pixel speed threshold; that the 30 released units are the ones the 5-cell rule counted; one clock for `.ncs`, `.t`, `.nvt` and the candidates; whether the 2017 `SWRtimes` are the examples behind the 2015 candidates (compare the template spectrum with `SWRfreqs.freqs1`) |
| Huszár et al. 2022, reused by Yang 2024: DANDI 000552 v0.230630.2304 | 13 sessions have a raw recording (`*-raw_ecephys.nwb`, 5.6-184 GB, streamed in slices) and a processed file of the same subject and date | `/processing/ecephys/Ripples`: TimeIntervals with `start_time`, `stop_time`, `peaks` and a raw snippet per event (seen in two other sessions) | the detector Huszár 2022 describes, if the package has it | that the 13 paired processed files hold a `Ripples` table; the ripple channel; the detection method and its settings in Huszár 2022's Methods; the raw-to-table clock (the per-event `ripple_raw` snippets can check it) |
| Yang 2024 on Huszár's e15_13f1_220118 (DANDI 000552) | the same session's raw file (183.6 GB, streamed) and processed file (363 units, `SleepStates`, `Ripples`, 9502 rows) | the repository's binned inventory for that session at f1bc2fb: 4088 events on 20 ms bins; 4087 contain a Huszár ripple start | `yang_2024` with `external_ripples` from the NWB `Ripples` table | which units Yang used (the NWB has no cell types); normalization over all bins or NREM only; the package requires a ripple peak inside an event where Yang's code tests the start; the later filters behind the binned inventory; bounds known only to 20 ms |
| Girardeau et al. 2017: CRCNS hc-14, 45 sessions with `rip.evt` (Rat08, Rat10, Rat11) | per session `{session}.lfp.tar.gz` (6.5-8.2 GB; 1250 Hz, 166 channels in Rat08-20130713's `.xml`), `cat.evt` subsession bounds; or the DANDI 000061 NWB conversion, streamed | `{session}.rip.evt` in the `_clu` archive: NeuroScope start, peak and stop in ms; Rat08-20130713 has 6835 events on channel 23, 20.0-129.6 ms | `Zugaro_ripple_detector` with the Methods' thresholds and 20-130 ms limits; run only if the events prove to come from FMAToolbox's `FindRipples` (the capped merge the package follows, complementing MS10's uncapped buzcode), otherwise listed as a repeat of MS10 | the detection channel (23 in the event labels, 0- or 1-based); the Methods' thresholds and band; the non-hippocampal control channel whose coincident events were removed (the package has no such veto: compare with and without removing them); the `.lfp` against DANDI 000061's columns (one file had 160 channels where the description gives 134 or 166) |
| Petersen, databank MS10 (`Peter_MS10_170307_154746_concat`), DANDI 000059 | raw 20 kHz (377820000 x 64) and LFP 1250 Hz (23613750 x 64); the first rows of both equal the archived `.dat` and `.lfp` exactly | Wayback `ripples.events.mat` (11305 B, complete): 429 events from `bz_FindRipples`, channel 46 (0-based; Ripple tag 47), thresholds [2 5] SD, durations [50 150] ms, 120-180 Hz, stored `stdev` 202112.11 | `Zugaro_ripple_detector(low_threshold=2, high_threshold=5, minimum_inter_ripple_interval=0.05, maximum_duration=0.15, minimum_duration=0, speed_threshold=np.inf)` on column 46 filtered as the source did | **Done (2026-10-03, `results/MS10/findings.md`).** The source is petersenpeter/buzcode `bz_FindRipples` at bc3fc91 (cheby2 order 4, 20 dB, filtfilt), not buzsakilab's butter order 3; the recomputed SD matches the stored one to 4.5e-12; all 429 released events recovered with identical bounds, plus 120 package-only fragments explained entirely by FMAToolbox's merge cap, which buzcode lacks (now documented in the detector's docstring) |
| Maboudi 2018, `fig1.nel` with CRCNS hc-3 `gor01-6-7/2006-6-7_16-40-19` (2.17 GB archive) | the stored 1 kHz multiunit trace, 117 units, one unlabelled 1252 Hz LFP channel; hc-3's `.res/.clu`, `.eeg`, `.whl` | 457 MUA epochs (and 277 binned PBEs, a later inventory) | `maboudi_2018` | the stored trace already reproduces all 457 bounds (runs above the mean with a 3 SD peak, 80 ms to a maximum of 0.70-0.81 s); which hc-3 spike set rebuilds the trace (the 117 units give about 400 Hz of its 2420 Hz); the LFP channel by cross-correlation; the 1180-1250 s gap |

Optional, if time allows: Petersen's databank MS22 (`Peter_MS22_180629_110319_concat`, DANDI
000059, raw verified the same way): 1467 events from CellExplorer's `ce_FindRipples` with absolute
thresholds [307.69, 410.26] ADC units, 80-240 Hz, 20-150 ms, an EMG rule and a higher-band veto.
No packaged method takes absolute thresholds; it would be composed with `detect_events_from_trace`
(`normalization_method='none'`) plus the vetoes, and DANDI's electrodes table mislabels its shanks
(take them from the `.xml`). The databank holds more sessions of the same kinds (21 of Petersen's
22 sessions are in DANDI 000059, 7 of Girardeau's in 000061); add them after the first session of
each kind is explained.

Not included, with the reason recorded in the README:

- Denovellis 2021 (Dryad 10.7272/Q61N7ZC3; Remy's consensus ripple tables): the events come
  from the maintainer's own Frank-lab pipeline, the lineage this package's detectors grew from,
  so agreement would not be an independent check.
- Widloski 2022/2025 (Zenodo 16916108): no event start or end times are released.
- Grosmark (CRCNS hc-11, DANDI 000044), Gillespie (DANDI 000115), Shin (DANDI 000978) and Tirole
  (Dryad): no event tables in what was inspected.
- CRCNS hc-3 (beyond Maboudi's session) and pfc-2: no ripple or event files documented.
- CRCNS hc-18 (Drieu): its archives (6.0-12.4 GB) were not listed, and the released
  `BatchFindRipples.m` has its event-saving lines commented out. Stream one archive's member list
  first if Drieu's events are wanted.
- Michon 2021 (OSF), Igata 2021 (Mendeley) and the Wilson-lab archive: no event times.
- Yang 2024's `e13_26m1_210913.rippleHSE.events.mat`: spikes but no LFP, and the session is not in
  DANDI 000552; only a spikes-only HSE check would be possible.
- Tingley 2021 (DANDI 000233) and Valero 2022 (000568): ripple tables in the NWB files, but no
  stored channel or settings, and Tingley's `peak` column uses another clock. Candidates for later.
- Databank event files the Wayback archive truncated (Tingley, Grosmark, Valero, Girardeau Rat08:
  30-104 MB originals), and McKenzie and Varga sessions, whose raw files are only on Globus.
- Hand-annotated ripple datasets: deferred by the maintainer on 2026-10-02.
- Girardeau's databank Rat09-20140402 (dropped 2026-10-03): `bz_FindRipples` again, as MS10; its
  only new element is a noise veto the package does not implement, so it would add nothing about
  `Zugaro_ripple_detector`'s faithfulness.

## Original-code comparisons

Added 2026-10-03 for the goal above. For each detector and packaged paper method whose original
code is public, run that code and the package on the same inputs and compare the events with
`match_events`, stage by stage where the source exposes intermediate values (filtered signal,
normalized trace, thresholds). Inputs: simulated sessions from `simulate_session` and
`simulate_network_session` (several seeds, with the edge cases a rule distinguishes: close events
for merging, long events for ceilings, gaps, events at the record edges) and the MS10 LFP column
already cached. A difference is classified as a documented departure (the docstring lists it), an
undocumented departure (documentation fix), or a bug (its own PR with a regression test).

The original MATLAB runs in GNU Octave (installed system-wide with Homebrew on 2026-10-03, by the
maintainer's decision; not a package dependency). Where Octave cannot run a source (missing
toolbox functions), a Python transcription of the pinned source, checked line by line in review,
stands in, and the report says which was used. No CI test runs Octave; the comparison scripts
record the Octave version and source commits, and their small results are committed.

Order: first an inventory of every detector's and packaged method's original code (public or
not, pinned commit, language, whether Octave runs it, what intermediate values it exposes); then
one comparison per detector or method family, most-used first.

## Tasks

1. **Fetch and record.** `examples/reference_recordings/fetch.py` downloads or streams each input
   into a cache directory (an environment variable, default outside the repository), checks the
   published checksums (Zenodo MD5, DANDI SHA-256, Dryad SHA-256) and writes a manifest of URLs,
   sizes and hashes. Stream DANDI NWB files with `remfile` + `h5py`; never download whole raw files.
   These two are not package dependencies (see the dependency policy's phase 7 exception): the
   scripts run as `uv run --with remfile --with h5py python examples/reference_recordings/...`,
   and the manifest records their versions. CRCNS files come from `download.crcns.org` with the
   user's account, read only from `CRCNS_USERNAME` and `CRCNS_PASSWORD` and never printed or
   stored, checked against the dataset's `checksums.md5`, following the official
   `crcns-downloader` (standard library only). Wayback files are fetched by capture URL; the
   manifest records the capture time and the original length, and a file whose archived length
   differs is refused.
2. **Verify inputs before detecting.** Per session, write `inputs.json`: sampling rate, units,
   channel selection (the channel the source names), gaps, the
   speed units and conversion, and one clock for LFP, spikes, position and released events (every
   released event lies within recorded samples). A failed check stops that session and is
   recorded; it is not patched to make the run proceed.
3. **Stage parity where the source releases intermediate values.** Before comparing events,
   compare the package's template spectrum for Carey with the released `SWRfreqs`, and any
   intermediate value Huszár's files turn out to hold. A difference at an earlier stage explains
   event differences before any threshold is considered.
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

8. **Original-code inventory.** `examples/reference_recordings/original_code.md`: for each of the
   nine detectors and each packaged method family, the original code (repository and pinned
   commit, or "not public" with what was searched), language, license, whether it runs in
   Octave, and the intermediate values it exposes.
9. **Original-code comparisons**, one per detector or family from the inventory, as described
   under [Original-code comparisons](#original-code-comparisons).

## Validation

| Check | What it establishes |
| --- | --- |
| Carey stage check: the template spectrum `carey_spectral_ripple_score` builds from `SWRtimes` against the released `SWRfreqs.freqs1` | The score's template, the stage before any threshold, matches the source (or the difference is identified) before events are compared. |
| Huszár clock check: each released event's `ripple_raw` snippet against the raw recording at that event's times | The released table and the raw file share one clock and channel before events are compared. |
| Smoke test: a 10-minute Carey slice and a slice of one Huszár session first | Runtime and memory measured before full sessions; extrapolated before the full run. |
| Timestamps at the sources' own origins (Neuralynx microseconds, NWB seconds) | Tolerances scale with the timestamps; no rounding shifts a bound. |
| Original-code comparison: the source's code (Octave or a reviewed transcription) and the package on the same simulated and real inputs, per stage where exposed | The package reproduces the source, or each difference is a documented departure, a documentation gap or a bug with its own fix. |
| `fetch.py` checksum test on a small file | A changed or partial download fails instead of being used. |
| Wayback length check: each archived event file's length against the original `Content-Length` the capture records | A truncated capture (the archive keeps the first 1 MiB) is refused, not parsed. |
| Head match: the first rows of each databank session's DANDI raw and LFP arrays against the archived `.dat`/`.lfp` heads, and the DANDI column for the stored channel | The released channel index points at the same signal in the DANDI copy (DANDI drops or reorders channels in some sessions). |
| Stage check for the `bz_FindRipples` sessions: the recomputed SD of the smoothed squared ripple-band signal against the stored `stdev` | The filter and normalization match before thresholds or bounds are compared. |
| Tests for the parsing helpers (`.t`, `.nvt`, `.ncs`, MAT tables), on files or slices small enough for CI | The readers return the documented shapes, units and clocks. They use only NumPy and `scipy.io` (the Neuralynx formats are fixed binary records), so they run in every CI job; the NWB streaming code needs `remfile`/`h5py` and is checked by the smoke run, not in CI. |

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
