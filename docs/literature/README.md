# How the surveyed replay papers detect events, and what ripple_detection can reproduce

This review covers the 57 papers in the survey that ships with the package,
`src/ripple_detection/data/literature_detection_parameters.csv`, also available as
`load_literature_parameters()`. For each paper it records what the authors implemented to
detect events, and whether this package can reproduce it.

Each paper's Methods were read: 54 from local PDFs and 3 from open-access or author copies.
Citations given as "as previously described" were followed one hop, and released code was
read where the text left a deciding detail open. The notes for each paper are in
[papers/](papers/), one file per paper, named by the survey's 0-based row number. Each file
gives the paper's parameters with verbatim quotes, the citations followed, any released code,
the survey fields that disagree with the paper, and a recipe for this package.

The review was done in September 2026 by Claude Code agents, one per group of papers, working
from the paper texts. Claims about package behaviour were checked against the source. The
Carey finding below was checked directly against the paper's released candidate file. Tiers
and recipes describe the package at commit ac23a58, the unreleased development version
after 1.7.1. The corrections to the survey
CSV proposed here are on hold and have not been applied.

Tiers:

- **A**: an existing detector with arguments, plus public helpers (a `duration` filter,
  `merge_close_events`, `require_overlap`, `exclude_overlap`, `normalization_mask`).
- **B**: a short piece of user code on public functions (a custom trace plus
  `threshold_by_zscore` or `ripple_detection.core.extend_threshold_to_mean`, user-computed
  state intervals, a silence segmentation around `segment_boolean_series`).
- **C**: needs a package addition.
- **D**: outside scope (the event is defined by decoding or by sequence content).

## Counts

| Tier | Papers |
|---|---|
| A | 20 (one of them, Liu 2023, needs a radiatum channel) |
| B | 31 |
| C | 2 (Carey 2019, Igata 2021) |
| D | 4 (Widloski 2022, Widloski 2025, Kaefer 2020, Gupta 2010) |

What the dataset has to supply:

- **Radiatum sharp wave:** needed only by Liu 2023. Harvey 2023's text needs it too, but its code runs `bz_FindRipples` without it (the `Zugaro_ripple_detector` path).
- **Theta/delta or sleep scoring as a detection gate:** about 15 papers, and every one defines the state differently. Two examples: a Hilbert ratio against a fixed cutoff of 2, and k-means on a spectrogram ratio. The survey records none of the cutoffs.
- **Sorted units:** most of the spike-based papers (about 20) threshold the rate of sorted cells, often only place cells or pyramidal cells, or count participating cells. From clusterless or multiunit data those become approximations. Davidson 2009, Wu 2017, Mou 2022, Ji 2007, Bendor 2012, Gridchyn 2020 and Michon 2019/2021 use all spikes.

## Per paper

### Tier A

| Row | Paper | Event definition | Recipe | Remaining difference |
|---|---|---|---|---|
| [49](papers/49_Karlsson_2009.md) | Karlsson 2009 | Per-tetrode Hilbert envelope, 4 ms, z ≥ 3 for ≥ 15 ms on any CA1 or CA3 tetrode, back to the mean, < 2 cm/s | `Karlsson_ripple_detector(speed_threshold=2)` | Include CA3 tetrodes. The ≥ 5 place cells rule is for replay analysis |
| [46](papers/46_Carr_2012.md) | Carr 2012 | Same, CA1 only, < 4 cm/s | `Karlsson_ripple_detector` | The 1 s exclusion is measured from the other event's onset, but `exclude_close_events` measures from the end of the last kept event |
| [38](papers/38_Jadhav_2016.md) | Jadhav 2016 | Karlsson 2009 rule; SWRs within 1 s of a previous one dropped | Karlsson + `close_ripple_threshold=1.0` | Same 1 s caveat as Carr |
| [35](papers/35_Tang_2017.md) | Tang 2017 | Karlsson rule; sleep SWRs restricted to SWS | Karlsson + user SWS intervals | The 500 ms isolation rule drops both events of a pair |
| [27](papers/27_Shin_2019.md) | Shin 2019 | Karlsson rule; a 50 ms whole-event filter for analysis | Karlsson + a `duration` filter | — |
| [13](papers/13_Denovellis_2021.md) | Denovellis 2021 | Kay consensus trace, 2 SD, 15 ms (this package, 0.1.8.dev0) | `Kay_ripple_detector` defaults | That version squared the filtered signal and used a 101-tap filter. On simulated data it gives identical events |
| [14](papers/14_Gillespie_2021.md) | Gillespie 2021 | Kay 2 SD; multiunit control 3 SD over immobility | Kay; `multiunit_HSE_detector(zscore_threshold=3, normalization_mask=speed<4)` | Multiunit smoothing is unresolved between the text and the released code |
| [50](papers/50_Davidson_2009.md) | Davidson 2009 | All spikes > 100 µV, 1 ms bins, 15 ms Gaussian, peak ≥ 3 SD over STOP periods, bounds at the mean | HSE, `minimum_duration=0`, `normalization_mask=stop` | "During STOP" is unspecified (the package tests the endpoints). Within 30 s of RUN via `require_overlap` |
| [43](papers/43_Wu_2014.md) | Wu 2014 | Place-cell density, 10 ms bins, 15 ms, peak > 2 SD, whole session | HSE at `fs=100` | Reward-area restriction is user intervals |
| [40](papers/40_Silva_2015.md) | Silva 2015 | Sorted units, 10 ms, peak > 3 SD, < 5 cm/s, 100–500 ms | HSE + a `duration` filter | Bin size and normalization period not stated |
| [08](papers/08_Berners-Lee_2022.md) | Berners-Lee 2022 | Sorted pyramidal cells, 10 ms, ≥ 3 SD, 100–500 ms | HSE | The code cuts events at movement onset rather than dropping them (NaN-mask moving samples) |
| [39](papers/39_Olafsdottir_2016.md) | Ólafsdóttir 2016 | Place cells, 5 ms, > 3 SD, ≥ 40 ms, ≥ 15 % of place cells | HSE, `minimum_active_units=ceil(0.15 n)` | — |
| [32](papers/32_Olafsdottir_2017.md) | Ólafsdóttir 2017 | As 2016, plus no speed > 3 cm/s in the event, corners only | HSE + user intervals | Its ripple and theta controls are tier B |
| [33](papers/33_Wu_2017.md) | Wu 2017 | All spikes > 60 µV, unsmoothed 10 ms bins, ≥ 4 SD, 50–400 ms | HSE(`fs=100`, `smoothing_sigma=1e-4`) + a `duration` filter | — |
| [21](papers/21_Xu_2019.md) | Xu 2019 | Sorted pyramidal cells, 15 ms, 3 SD, 75–750 ms, ≥ 4 cells, ≥ 5 spikes, ≥ 10 % cells | HSE + a one-line spike filter | Normalization period not stated |
| [31](papers/31_Maboudi_2018.md) | Maboudi 2018 | All units, 20 ms, ≥ 3 SD, mean speed ≤ 5, ≥ 80 ms, ≥ 4 active pyramidal cells | HSE | `minimum_active_units` counts all columns, so the pyramidal-only count needs user code |
| [55](papers/55_Nadasdy_1999.md) | Nádasdy 1999 | 150–250 Hz RMS power summed over electrodes, 7 SD, sleep | `Roumis_ripple_detector(zscore_threshold=7)` | The RMS window and bounds come from Csicsvari 1999a (read via Wayback). Needs sleep scoring |
| [36](papers/36_Grosmark_2016.md) | Grosmark 2016 | Pyramidal rate, 15 ms, 3 SD over NREM, 50–500 ms, ≥ 5 cells, must contain an LFP ripple peak | HSE + `require_overlap` | **The LFP ripple detector is never described.** Needs sleep scoring |
| [02](papers/02_Yang_2024.md) | Yang 2024 | Grosmark 2016's text copied | As Grosmark | The code differs from the text: it z-scores the whole recording and tests the ripple start, not the peak. A likely `bz_FindRipples` (from Huszár 2022) is an inference |
| [05](papers/05_Liu_2023.md) | Liu 2023 | neurocode `DetectSWR`, with the population burst inside it (> 2 SD, bounds at the mean, 100–500 ms) | `Long_sharp_wave_ripple_detector` + HSE | **Needs radiatum.** Manual curation is not reproducible |

### Tier B

| Row | Paper | Event definition | Missing piece (user code today) |
|---|---|---|---|
| [42](papers/42_Pfeiffer_2015.md) | Pfeiffer 2015 | Mean over tetrodes of the 12.5 ms-smoothed Hilbert amplitude, > 3 SD excluding > 5 cm/s, back to the mean, 50 ms–2 s | Mean-amplitude trace. Roumis averages √(smoothed squared envelope) instead, which is close |
| [37](papers/37_Ambrose_2016.md) | Ambrose 2016 | Same, 4–7 tetrodes, near the well | Same |
| [12](papers/12_Berners-Lee_2021.md) | Berners-Lee 2021 | Same, 2 SD, 3 tetrodes | Same |
| [10](papers/10_Krause_2022.md) | Krause 2022 | Pfeiffer 2015 SWRs trimmed to the population burst (3 ms bins, > 2 spikes/s per cell, ≥ 30 ms) | Mean-amplitude trace + trim. The code uses a 10 ms HSE SD where the text says 20 |
| [45](papers/45_Pfeiffer_2013.md) | Pfeiffer 2013 | Sorted units, 10 ms, > 3 SD, ≥ 10 % units, 50 ms–2 s; bounds moved inward until the edge windows hold ≥ 2 spikes | The inward trim (~10 lines) |
| [00](papers/00_Mallory_2025.md) | Mallory 2025 | Linear track: excitatory-cell density, 12.5 ms, > 3 SD over ≤ 5 cm/s, peaks ≤ 70 ms apart merged; then decoding criteria (D) | Peak-to-peak merge. Published Methods unverified; read from the preprint and the code |
| [06](papers/06_Tirole_2022.md), [03](papers/03_HuelinGorriz_2023.md) | Tirole 2022, Huelin Gorriz 2023 | Pooled sorted and unsorted spikes, 1 ms, z ≥ 3, bounds searched ±300 ms; ≥ 100 ms, merge < 50 ms, median speed ≤ 5, ≥ 5 place cells, **ripple z ≥ 3 inside the event** | Bound search with fallback, peak-inside test, place-cell count. The code differs from the text (kernel, ripple smoothing, the 750 ms maximum not enforced) |
| [07](papers/07_Bush_2022.md) | Bush 2022 | Principal cells, 5 ms, 3 SD; merge ≤ 40 ms, drop ≤ 40 ms, then ≥ max(5, 15 %) cells, median speed ≤ 10, ≤ 0.5 s | Criteria after the merge. The detectors apply them before |
| [41](papers/41_Olafsdottir_2015.md) | Ólafsdóttir 2015 | Per template: ≥ 15 % of its cells within ≤ 300 ms, bounded by ≥ 50 ms of silence | Silence segmentation |
| [51](papers/51_Diba_2007.md) | Diba 2007 | ≥ 60 ms silence, then ≥ 5 (or 30 %) template cells in the next 300 ms | Silence-onset windows |
| [19](papers/19_Bhattarai_2020.md) | Bhattarai 2020 | SWR: 50 ms boxcar power, peak 3 SD, bounds 1 SD, > 20 ms, merge ≤ 100 ms. Replay: > 60 ms silence, ≥ 5 place cells in 300 ms, coinciding with an SWR | Two-level bounds (composable), silence windows |
| [26](papers/26_Liu_2019.md) | Liu 2019 | Sorted pyramidal spikes cut at ≥ 100 ms silence, ≥ 4 cells, 80 ms–1.2 s; SWS by theta/delta < 2 | Silence segmentation, SWS mask |
| [53](papers/53_Foster_2006.md) | Foster 2006 | Probe cells' pooled spikes split at > 50 ms gaps, ≥ 1/3 of cells, ≤ 500 ms, stopping periods | Gap split on spike times (`merge_close_events` on zero-length intervals works) |
| [54](papers/54_Lee_2002.md) | Lee 2002 | Each cell's ISI < 50 ms bursts collapsed to one spike, split at > 100 ms, SWS | Burst collapse + gap split, SWS mask |
| [52](papers/52_Ji_2007.md) | Ji 2007 | Pooled counts, 10 ms bins, σ 30 ms, ≥ T (first histogram minimum), merge gaps < G (70–90 ms), SWS; "frames" of 0.1–3 s | Data-driven T, bounds at T, SWS mask. These are not SWR-scale events |
| [47](papers/47_Bendor_2012.md) | Bendor 2012 | Davidson signal, peak z ≥ 4, bounds z ≥ 2, merge < 50 ms, ≥ 50 ms | `extend_threshold_to_mean(z >= 2, z >= 4, ...)` |
| [11](papers/11_Mou_2022.md) | Mou 2022 | All spikes, 10 ms, σ 20 ms, scaled to 0–1, peak > 0.35, bounds 0.15, merge < 30 ms | Range-scaled trace + two-level bounds |
| [24](papers/24_Chenani_2019.md) | Chenani 2019 | Place-cell rate, σ 30 ms, peak ≥ 3 SD, bounds ≥ 1 SD, ≥ 5 cells | Two-level bounds, unit count. Reward zones were chosen by eye |
| [20](papers/20_Stella_2019.md) | Stella 2019 | Morlet RMS power per electrode z-scored, max over electrodes, peak > 5, bounds 2 SD; REM excluded by theta/delta | Wavelet trace, two-level bounds, theta/delta (cutoff not stated) |
| [25](papers/25_Michon_2019.md), [15](papers/15_Michon_2021.md) | Michon 2019, 2021 | Mean envelope over 1–3 tetrodes, 15 ms, 3 s moving-median detrend, peak 8 SD / bounds 0.5 SD, merge < 20 ms, ≥ 40 ms; multiunit 4 SD / 0.5 SD, ≥ 80 ms; bursts overlapping a ripple at < 5 cm/s | Detrend, two-level bounds, `require_overlap`. Kloosterman lab |
| [17](papers/17_Gridchyn_2020.md) | Gridchyn 2020 | Online: 20 ms count ≥ 3.5 × pre-rest mean, bounds at that mean, 150 ms refractory, multiplier stepped each minute toward ~1 Hz | Rate controller. The fixed-threshold version is A with HSE and `normalization_mask=pre_rest` |
| [22](papers/22_Farooq_2019.md), [23](papers/23_Farooq_2019.md) | Farooq 2019 (Neuron, Science) | Sorted pyramidal cells, 15 ms, above 2 SD with bounds at 2 SD, 100–800 ms, ≥ 5 cells; SWS by theta/delta (two different definitions) | `extend_threshold_to_mean(z >= 2, z >= 2, ...)`, SWS mask. The Science paper's manual SWS review is not reproducible |
| [30](papers/30_Drieu_2018.md) | Drieu 2018 | Place-cell rate, 10 ms, 3 SD, back to the mean, ≤ 500 ms; SWS by k-means on theta/delta | SWS mask (the event rule is A) |
| [29](papers/29_Muessig_2019.md) | Muessig 2019 | CS cells, 10 ms, 3 SD, 100–750 ms, must overlap an SWR. SWR: 7 ms RMS on the best tetrode, > 99th percentile, 100 ms window at the peak. Rest by theta/delta | RMS and percentile SWR, peak window, rest mask |
| [34](papers/34_Yamamoto_2017.md) | Yamamoto 2017 | 140–200 Hz power > 3 SD and multiunit > 3 SD, extended to 1 SD | How the two combine and which trace sets the bounds are not stated, so this is only an approximation |
| [44](papers/44_Wikenheiser_2013.md) | Wikenheiser 2013 | 140–220 Hz power > 1 SD, 150 ms windows around supra-threshold times, joined; ≥ 3 cells, ≥ 5 spikes; theta/delta | Fixed windows, theta/delta |
| [04](papers/04_Harvey_2023.md) | Harvey 2023 | Text: DoG 80–250 Hz, 4 SD / 1 SD, ≥ 15 ms, AND a radiatum sharp wave. Code: `DetectSWR` where radiatum exists, else `bz_FindRipples` [1, 3] | Without radiatum: `Zugaro_ripple_detector(low 1, high 3, ...)` follows the code's FindRipples path |
| [56](papers/56_Kudrimoti_1999.md) | Kudrimoti 1999 | One channel, 100–300 Hz amplitude above an unreported absolute threshold for ≥ 25 ms; manual sleep scoring | The threshold value is not reported |

### Tier C

| Row | Paper | Why |
|---|---|---|
| [28](papers/28_Carey_2019.md) | Carey 2019 | The paper's own shipped candidates were made with `precand`: a spectral-template ripple score (`amSWR`, 60 ms FFT against hand-picked SWRs), times the multiunit score (geometric mean), rescaled to mean 0.5, a single threshold of 4 (8 × the mean, not a z-score), bounds at that threshold. `Carey_candidate_detector` implements the later `GenCandidateEvents` configuration instead: a Hilbert ripple score with z-scored thresholds of 1 and 3. The multiunit score, the speed and theta intervals, the 20 ms minimum and the ≥ 5 units all match. Checked against `R050-2014-03-29-candidates.mat` and `precand.m` from the paper's repository |
| [16](papers/16_Igata_2021.md) | Igata 2021 | The candidate stage is B: sorted rate, 15 ms, > 2 SD, > 4 cells, 50–2000 ms. It is followed by a GMM split on rate and multi-tetrode "Mahalanobis" ripple power, which the paper leaves under-specified |

### Tier D

| Row | Paper | Note |
|---|---|---|
| [09](papers/09_Widloski_2022.md) | Widloski 2022 | Events come from decoded bins; ripple and spike density are reference traces only |
| [01](papers/01_Widloski_2025.md) | Widloski 2025 | Replays come from decoding. The ripple and burst labels are B: the ripple is 100–220 Hz averaged over tetrodes, 2 SD for 15 ms. The code sets the merge to 0 where the text says 50 ms |
| [18](papers/18_Kaefer_2020.md) | Kaefer 2020 | All immobility is decoded. A secondary SWR detector (5 SD / 1.5 SD) is B |
| [48](papers/48_Gupta_2010.md) | Gupta 2010 | Windows are grown by a spike-order score; an SWR power gate of 2 SD applies, but how is not stated |

## Since this review: what the additions change

The tiers above describe the package at commit ac23a58. The additions made after it
are listed below, followed by the tier each tier-B or tier-C paper would now have.
Those new tiers are an assessment from the notes, not recipes run paper by paper.

| Addition | Covers |
|---|---|
| `detect_events_from_trace`: `bound_threshold`, `normalization_method="none"`, `minimum_event_duration`, `speed_rule` (with `"restrict"`), `close_event_rule="merge"` | mean-amplitude and RMS traces, events ending at k SD or at the threshold, raw or range-scaled thresholds, whole-event minimums, detection on slow samples only |
| `require_active_units`, `count_spikes_in_events` | counts, fractions and spike totals of chosen units |
| `theta_delta_ratio`, `state_intervals`, `two_cluster_threshold` | theta/delta and speed state gates, k-means sleep scoring |
| `detect_silence_bounded_events` | silence-bounded groups and windows, burst collapse |
| `peak_time` on every detector, `require_trace_peak`, `require_times_inside`, `windows_around_times` | "a ripple peak inside the burst", fixed windows around peaks or crossings |
| `merge_close_events(inclusive=, measure="peak")`, `exclude_close_events(measure_from="start")`, `require_isolation` | inclusive and peak-to-peak merges, onset-timed exclusion, isolation |
| `exclude_movement(rule=)` | every-sample, mean and median speed rules |
| `histogram_minimum_threshold` | a threshold at a distribution's first trough (Ji 2007) |
| `carey_spectral_ripple_score`, `Carey_candidate_detector(ripple_score=, threshold_method="mean")` | the rule behind Carey 2019's published candidates |
| `trim_events_to_trace`, `trim_events_to_spike_windows` | events narrowed to the population burst, to edge windows holding enough spikes, or to the first spike |
| `detect_events_from_trace(threshold=<array>, bound_search_window=, bound_threshold=(levels))` | a threshold that changes over the recording, bounds sought within a window with fallback levels |

**Now tier A** (public functions, plus traces or intervals built with a few lines of
NumPy where noted):

- **Foster/Redish mean-amplitude traces:** Pfeiffer 2015, Ambrose 2016 (the well-proximity
  intervals are the user's), Berners-Lee 2021.
- **Two-level bounds or bounds at the threshold:** Bendor 2012, Chenani 2019, Mou 2022
  (range-scaled trace), Farooq 2019 (Neuron and Science), Michon 2019 and 2021 (with a
  trace detrended by a 3 s moving median outside the package).
- **Silence-bounded:** Ólafsdóttir 2015, Diba 2007 (its "5 cells or 30%" is an OR: the
  union of two calls), Foster 2006, Lee 2002, Liu 2019, and Bhattarai 2020 (with a 50 ms
  boxcar power trace).
- **State-gated:** Drieu 2018, Muessig 2019 (a 7 ms RMS trace, a percentile threshold,
  100 ms windows around peaks), Wikenheiser 2013, Ji 2007, Stella 2019 (a wavelet power
  trace built outside the package). State definitions stay approximate: the papers'
  bands, estimators and cutoffs differ and are often unreported.
- **Others:** Bush 2022, Mallory 2025's linear-track candidates, and Carey 2019 (the
  template's example ripples were picked by hand in the paper; detected ones stand in).

- **Trims and bound searches, added later:** Krause 2022, Pfeiffer 2013 (`trim_events_to_trace`,
  `trim_events_to_spike_windows`), Tirole 2022 and Huelin Gorriz 2023 (`bound_search_window` with
  fallback levels).

**Short of tier A for reasons outside the package:**

- Gridchyn 2020: the online per-minute rate controller is not implemented; its fixed-threshold
  version is A, and a per-sample `threshold` array can carry a controller's output.
- Yamamoto 2017: the paper does not say how its ripple and multiunit criteria combine; each
  reading can be written.
- Harvey 2023, as written: needs a radiatum channel. Its code's FindRipples path is A.
- Kudrimoti 1999: its threshold is not reported.

**Still tier C:** Igata 2021, whose GMM step is under-specified.

**Tier D is unchanged:** the four decoding-defined papers.

Resulting counts: about 48 A, 4 short for the reasons above, 1 C, 4 D.

### Recipes

[examples/literature_recipes.py](../../examples/literature_recipes.py) writes each paper's rule
with the package, one function per paper, with its row, its source and where it departs from the
paper in the docstring; papers defined by decoding contribute the detection they label events
with, if any. It runs every recipe on a simulated session (running bouts with theta, rest with
delta, replay-length ripples with bursting place cells and a radiatum sharp wave) and writes the
events found, the fraction of ripples overlapped, and the events that overlap none to
`literature_recipes_results.csv`. The simulation says whether a recipe runs and behaves, not
whether it matches the paper's events. `tests/test_literature_recipes.py` runs every recipe and
checks that each surveyed paper has a recipe or a stated reason for none.

## Package additions, ranked by the papers they would move (as proposed at the review)

1. **A public trace detector with a bound level.** The shared machinery already exists privately as `detectors/_events.py::_detect_from_trace`. Exposing it would give any user-built trace the package's blocks, speed rule, duration and close-event handling, and statistics columns. Three arguments would be needed:
   - `bound_threshold`, where events end (default 0, the mean);
   - an option to skip normalization, for raw or range-scaled thresholds;
   - optionally a detrend window.

   On its own this moves about 8 tier-B papers to A:
   - the mean-amplitude traces: Pfeiffer 2015, Ambrose, Berners-Lee 2021;
   - the two-level bounds: Bendor, Michon ×2 (with the detrend), Mou (with no normalization);
   - Gridchyn's fixed-threshold version.

   With additions 2–4 as well, it moves about 20. Chenani also needs the unit count. Farooq ×2 and Stella also need the unit count and the state helper. Bhattarai also needs the unit count and silence segmentation. Ji 2007 also needs the state helper. A `bound_threshold` on the existing detectors alone would cover only the two-level cases.
2. **Counting active units over any intervals, for a chosen subset of units.** `_count_active_units` exists privately. Participation rules ("≥ 5 place cells", "≥ 15 % of pyramidal cells", "spikes on ≥ 2 tetrodes") appear in about 15 papers, at detection or at analysis. `minimum_active_units` cannot restrict the count to a subset.
3. **A theta/delta state helper.** It would return a mask and intervals, with a choice of bands, measure (Hilbert or spectral), threshold (fixed, relative to the mean, or two-cluster), minimum epoch and gap bridging. About 15 papers gate on state, all defined differently, so it enables comparisons rather than exact replication.
4. **Silence-bounded segmentation of spike trains.** Covers Diba, Ólafsdóttir 2015, Bhattarai's replay step, Liu 2019, Foster 2006 and Lee 2002, about 6 papers. Most need sorted units.
5. **`peak_time` on the threshold detectors, and a "trace peak inside each event" test.** For Tirole and Huelin Gorriz, Grosmark and Yang, and the Widloski 2025 labels. Mallory's peak-to-peak merge is related.
6. **Carey.** Accept a precomputed ripple score, and add a threshold relative to the mean, to reach the published rule.

Differences that stay after all of these:

- Several papers (Bush, Tirole, Michon) merge before applying their participation and speed criteria. The detectors apply criteria before the proximity rule.
- `merge_close_events` joins only gaps strictly below its threshold.
- Almost no paper says which samples its speed test uses.

## Problems found in this repository

- **Carey docstring.** `Carey_candidate_detector` calls itself "the candidate-event detector of Carey, Tanaka & van der Meer 2019". It reproduces the multiunit half and the state restrictions of that paper, but not its ripple score or threshold rule (see tier C above).
- **Survey CSV.** [survey_corrections.md](survey_corrections.md) lists proposed changes, each re-checked against the paper by a second reader, and the rules they follow. They are on hold and not applied. Inclusion criteria applied before decoding, such as a decoding-stage minimum cell count, stay. Values from unrelated analyses and from decoded content would be removed. Values of control detectors stay in their columns, by the maintainer's decision. The README's "Published parameter values" table is computed from this file, so applied changes would move its counts. Recurring patterns:
  - **Values from unrelated analyses:** speeds from place-field, run-segment, sleep-labelling or scorer-validation definitions (Wu 2017, Bhattarai, Harvey, Liu 2023, Yamamoto, Bendor, Ólafsdóttir 2015, Drieu).
  - **Values from control detectors:** Bush, Gillespie, Ólafsdóttir 2017 and Farooq 2019 (Neuron) record a control detector's values. These stay, and the notes should say which detector they describe.
  - **Decoded-content lengths entered as minimum durations:** Wu 2014, Stella, Carey.
  - **Grosmark 2016 and Yang 2024:** their SWR columns hold the multiunit values. The LFP ripple method is not stated in either paper.
  - **"SWR electrodes = 1" for any-tetrode designs:** Karlsson, Carr, Shin.
  - **Wrong value:** Yamamoto's band is 140–200 Hz, not 100–200.
  - **Missing values:** Maboudi's multiunit fields; Carr's 15 ms minimum; Pfeiffer 2013's 10 % units.
  - **Unsupported values:** Ambrose's 50 ms and 500 ms appear in none of the paper, the author manuscript or the supplement.
  - **"MUA" for rates of sorted cells:** Berners-Lee 2022, Pfeiffer 2013, Wu 2014, Silva, Foster 2006, Lee 2002, Diba. Left as is, since the Detection column names the trigger type.

  Each paper's notes list its row's discrepancies with quotes.

## Text versus code

Where both were checked, these papers' released code differs from their Methods:

- Tirole 2022 and Huelin Gorriz 2023
- Berners-Lee 2022 (baseline period)
- Widloski 2025 (merge threshold)
- Krause 2022 (HSE smoothing)
- Yang 2024 (normalization period, ripple-start test)
- Harvey 2023 (`DetectSWR` and `bz_FindRipples` rather than the described algorithm)

Reproductions should follow the code where it exists. Each paper's notes say which one they followed.

## Sources not fully verified

- Mallory 2025's published Methods: the Science supplement returned 403; the preprint and the paper's Zenodo code were used instead.
- Csicsvari 1999a (for Stella and Nádasdy): only partly retrieved.
- O'Neill 2008 (Muessig's state definition): paywalled, not read.
- Jackson 2006 (Redish lineage): read through a web summarizer, so its quotes are not verbatim.
- The Ambrose 2016 supplement: not retrieved.
