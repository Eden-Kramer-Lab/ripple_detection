# Literature verification: unresolved questions

Updated September 25, 2026 after the independent follow-up. The former list of
sources “not independently reopened” has been replaced by the completed
[source recheck](source_recheck.md). All listed repositories/archives were revisited
at the scope stated there. The Mallory and Bhattarai supplements supplied by the
user have now been read; missing callers and incomplete method descriptions remain
limitations, not unperformed routine checks.

The survey has 57 rows and 35 columns. All first-pass changes are accounted for;
131 retain their values and seven were deliberately refined. The cumulative
[correction ledger](parameter_corrections_2026-09-25.csv) records 236 current changes
across 49 papers. The [field-status index](parameter_verification_2026-09-25.csv)
labels every cell, including unresolved and inferred entries.

## Supplement retrieval gaps closed

Both published supplements previously listed here are now available and checked.
Bhattarai’s SI confirms 20 ms replay windows with 10 ms steps, 6 cm spatial SD,
1000 circular-position shuffles, p<0.05 and 6.56 cm reconstruction error. Its
50 ms windows are for behavioral decoding. Table S1 confirms 1342/2573 replays
(52.2%). The earlier replay-window transcription is corrected.

Mallory's user-supplied published supplement is now read: detection, decoding and
the rank-order control are checked directly. The released replay extraction and
figure-combination paths override the generic 100 ms duration to 50 ms, resolving
that concern. Ji's supplementary frame gaps and Nadasdy's cited RMS/channel method
are also verified; these are closed retrieval gaps.

The older README source-access list is now closed as well: Ambrose’s publisher
supplement, O’Neill’s Oxford-hosted article/supplement, and the primary Jackson
and Csicsvari Methods have been read. Ambrose Table S1 establishes 19.3% pooled
replay (22.3% / 15.5% by experiment); it supplies no detector-duration limits.
Ji, Michon OSF and CSV-application status statements were corrected where stale.

## Original execution details that the released evidence does not establish

| Paper(s) | Still unresolved |
|---|---|
| Mallory 2025 | Published text versus code: arena spatial SD 8 cm versus nominal 4 cm; fixed 20 cm versus arena-scaled spatial merging; Euclidean versus first-coordinate COM jumps; all cells versus excitatory cells. Track SD is 2 cm per filtering pass in code. Original session configurations/flag generation are needed to establish intended execution; the 50/100 ms default issue is resolved. See the paper note. |
| Bhattarai 2020 | Published SI does not define the instantaneous-power calculation, SWR normalization period, exact inter-ripple interval, silence population or replay event-end rule. The earlier public-code/archive audit did not establish an original detector implementation. |
| Yang 2024; Grosmark 2016 | Original independent LFP detector, parameters and caller. Population-synchrony settings cannot supply these. Yang's tagged release has empty detector parameters; Grosmark's data deposits do not supply a documented original detector pipeline. |
| Huelin Gorriz 2023 | Called extract_replay_events is missing from the complete release. Smoothing and shuffle settings are verified, but code cannot settle the published maximum-duration exclusion. |
| Mou 2022 | PBE callback referenced by the release is absent. Original population-rate normalization remains unknown. |
| Gillespie 2021 | Which preprocessing path generated the released MUA events. Paper says 15 ms; archive helper says 5 ms, and an upstream function says 15 ms. No original caller connects these. Spatial KDE bandwidth is not established by the 5 cm position-bin width. |
| Michon 2019/2021 | Original toolbox version and caller, including smoothing/detrending order. OSF imports unprovided code; original v1.2 and a later fork differ. |
| Gridchyn 2020 | Paper-specific runtime configuration. Adaptive rule and compatible example settings are checked; per-session overrides remain unknown. |
| Harvey 2023; Liu 2023 | Dependency versions and paper/session-specific callers that connect shared DetectSWR/FindRipples implementations to each dataset. Explicit caller arguments were checked where available. |
| Maboudi 2018 | Earlier 10 ms smoothing inference for one stored session could not be reproduced directly. Stored event counts/version/peak threshold were checked; paper's 20 ms linear-track Gaussian remains authoritative. |
| Ambrose 2016 | The main Methods and complete supplement do not specify detector-duration limits. Later lab-code values cannot establish the historical caller; the publication-level duration question is closed as not reported. |
| Supporting lab-code comparisons | Later Barry, Jadhav, FMAToolbox, buzcoderough, Pfeiffer, FFPhy and DataManager code does not by itself prove historical per-session settings. No conflicting later default was substituted for an unambiguous paper value. |

These are provenance gaps, not reasons to discard directly supported published
numbers. An original configuration, dependency lock, missing function, or author
clarification would be needed to close them. No author has been contacted.

## Values that remain assumptions or incomplete descriptions

- Muessig 2019: the cited O’Neill source is now read, but does not specify the
  bandwidth around Muessig’s per-session spectral peaks.
- Gridchyn/Xu secondary SWRs and Stella state scoring: the cited Csicsvari source
  is read; missing RMS/boundary or theta-ratio settings remain unreported.
- Xu 2019: 3 cm smoothing for the 1 cm linearized map is inferred from applying
  the stated similar three-bin procedure; 6 cm is explicit for the 2D map.
- Olafsdottir 2016: 10 ms decoding bins are explicit; nonoverlapping steps are
  inferred. The released fitting helper accepts the bin size as an argument.
- Farooq 2019 Science: the old approximate 5 cm reconstruction-error summary is
  not verified across developmental ages; it is qualified in the CSV.
- Unreported details (for example Yamamoto's numerical replay p-value cutoff and
  Bendor's window step) are marked as unreported, rather than supplied from a
  related paper. Figure-derived estimates are distinguished from exact numbers.

The current field ledger has nine unresolved cells: four independent LFP settings
each for Yang and Grosmark, plus Farooq Science’s cross-age error estimate. Two
more cells remain inferred: Xu’s 1D smoothing and Olafsdottir’s window step.
Other limitations above concern code provenance or parameters absent from the
survey columns, even when the corresponding published number is verified.

Fresh searches revisited earlier “no code found” claims. Their scope and results
are recorded in the source recheck; absence from those searches is not proof that
no public code exists. Full computational replication and recipe/tier equivalence
remain outside this numerical/source audit.
