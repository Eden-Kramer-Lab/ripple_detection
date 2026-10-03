# Original-code inventory

`original_code.csv` has one row for each of the following:

- each registered detector (9);
- each building block that reimplements a published rule (4): `carey_spectral_ripple_score`,
  `detect_silence_bounded_events`, `theta_delta_ratio`, `state_intervals`;
- each `list_methods()` name (86).

Columns:

- `kind`, `implementation`, `family` (rows that share one original code base);
- `public` (yes / partial / no), with `public_basis` saying what was opened or searched;
- `source_repository`, `pinned_commit` (full SHA), `source_files`, `other_sources`;
- `language`, `license`;
- `octave_feasibility`, a static judgment; every runnable entry is untested;
- `entry_point`, `intermediates`, `package_departures` (from the docstrings);
- `reference_sessions`, `notes`.

Every pin was opened, either through the GitHub API (commit, tree, files) or through the
Bitbucket API (franklab/trodes2ff_shared and kloostermannerflab/fklab-python-core). Widloski's
Zenodo record has no git commit; its file list and MD5s come from the Zenodo API.

`public` means:

- **yes**: the code the package attributes the method to is public at a pinned commit.
- **partial**: code from the source's lab implements the rule, but a step is missing from the
  release, or the code cannot be tied to the paper's events.
- **no**: nothing public was found.

Totals: 15 yes, 24 partial, 60 no.

GNU Octave 11.3.0 is installed but has no packages. Its core lacks `hilbert`, `filtfilt`,
`butter`, `cheby2`, `fir1`, `gausswin`, `sgolayfilt`, `medfilt1`, `findpeaks` (signal package);
`kmeans` and `nan*` (statistics package); `fspecial` (image package); and `datetime`,
`histcounts`, `smooth`, `table` and `string`. This was checked with `exist`; no source was run.

## Feasible comparisons, in priority order

1. **`Zugaro_ripple_detector` against FMAToolbox `FindRipples` (michael-zugaro/FMAToolbox@6bbb366).**
   - Needs Octave core only: input is an already-filtered `[t x]` matrix and FMAToolbox's Helpers
     go on the path.
   - MS10 checked buzcode's uncapped copy through a Python transcription. The capped merge the
     package follows has never been run.
   - Run `harvey_2023_no_radiatum` in the same comparison. Harvey's released caller uses
     buzcode's `bz_FindRipples`, which needs the signal package and `'EMGThresh',0`.
2. **Frank-lab family against the shared `extractevents.cpp` MEX (droumis/FFPhy@fce2048; compiles
   with `mkoctfile --mex`).**
   - A driver must feed traces directly, because the extractors read filter-framework files.
   - Covers `Karlsson_ripple_detector`, `karlsson_2009` and `carr_2012` (`extractripples` +
     `getripples`).
   - Covers `Roumis_ripple_detector` (trodes2ff_shared@e90fd3d `extractEventConsensus`, begun as
     droumis's `DR_extractkonsensus`).
   - Covers `gillespie_2021` (`AG_extractEventConsensus`, called from `Trodes_dayprocess` with
     2 SD and 15 ms) and `gillespie_2021_mua` (`extractMUAevents`).
   - Covers `Yu_ripple_detector` (`AG_extractRipplesJY`). Its threshold function is not public,
     so the threshold must be supplied.
3. **Package history in Python, no Octave needed.** Run in git worktrees; the oldest versions may
   need old pandas.
   - `denovellis_2021` and `denovellis_2021_mua`: 0.1.8.dev0 (71298f0), called as
     `replay_trajectory_paper`@f2d2b3c does.
   - `multiunit_HSE_detector`: 2a58ad7.
   - `Shvartsman_ripple_detector`: b0f39f1, the merge of PR #11.
   - `Kay_ripple_detector`: c2c1cc9. The switch to the Hilbert envelope (34958cf) is documented.
   - `Roumis_ripple_detector`: the 2017 Python version, 4444f12.
4. **Carey family against vandermeerlab@ad0bbd4 (`precand`, `amSWR`, `amMUA`, `SWRfreak`,
   `restrict`, `TSDtoIV`).**
   - These call no toolbox function and work on in-memory tsd structs. `nearest_idx3` must be
     compiled as a MEX (its `.m` file is help text only).
   - Covers `carey_2019` and `carey_spectral_ripple_score`. Pair it with the Carey R050 session
     (template stage check against `SWRfreqs`).
   - `Carey_candidate_detector` uses `GenCandidateEvents`@82ba3fe, which needs the signal
     package for `OldWizard`/`FilterLFP`. Bypass its Neuralynx loaders.
5. **DetectSWR family against neurocode `DetectSWR` (@d166a67 cited; v1.0.0 @4b33b2a for Liu and
   Harvey) and buzcode `bz_DetectSWR`@0969ddf.**
   - Covers `Long_sharp_wave_ripple_detector`, `harvey_2023_code` and `liu_2023`.
   - Needs the signal and statistics packages, NeuroScope `.lfp`/`.xml` files and session
     metadata on disk, and a stub for `datetime`.
   - MATLAB and Octave k-means differ, so compare per stage or from fixed starts.
6. **Python lab releases.**
   - `krause_2022` and `krause_2022_hse` (DrugowitschLab@cda23b7). They need a stub RatDay object.
   - `maboudi_2018` against nelpy 0.2.0 (4ce5d48) `get_mua`, `get_mua_events` and `get_PBEs`.
     This is partial: the release stores outputs, not the call.
   - `michon_2019` and `michon_2021` against fklab v1.2 `compute_envelope` and
     `detect_mountains`. Also partial: the caller is unknown.
7. **MATLAB lab releases.**
   - `tirole_2022` and `huelin_gorriz_2023`: `replay_search` is a local function. Needs the signal
     package, plus `smooth` and `histcounts`, which Octave lacks.
   - `mallory_2025` and `mallory_2025_ripples`: `findpeaks`; the artifact function is missing.
   - `berners_lee_2022`: needs `fspecial` and a session MAT file as input.
   - `yang_2024`: neurocode `find_HSE` 2021 version @0231645; needs `fspecial`.
   - `mou_2022` and `ji_2007`/`ji_2007_ripples`: lab code only.
8. **`gridchyn_2020`.** `lfp_online`@a2d9cde is C++, so a reviewed transcription of
   `IsHighSynchrony` and the update rule is the practical route.

## No public original, and why

- **The Kay trace itself.** Kay's `kk_extractconsensus`/`kk_extractconsensus2` is referenced only
  in a comment in FFPhy. Only the segmentation and the package's own history can be compared.
- **Yu's threshold estimator.** `jy_variableripthreshold_corecalculation.m` is unpublished; the
  package transliterated a private copy.
- **Methods whose papers released no detector.** The sources.md search scope (through
  2026-09-25) plus targeted GitHub searches on 2026-10-03 found none for:
  - Wilson lab: Davidson 2009, Foster 2006, Lee 2002, Bendor 2012.
  - Pfeiffer/Foster lab: Pfeiffer 2013 and 2015, Ambrose, Silva, Wu 2014, Berners-Lee 2021, the
    Maboudi open field.
  - Barry lab: Bush 2022, Ólafsdóttir 2015/2016/2017. Only later Shipley and PythonSpkAnalysis
    code exists.
  - Csicsvari lab: Kaefer, Stella, Xu, the Gridchyn ripples. `fdetswdiff` is unreleased.
  - Dragoi lab: both Farooq papers and Liu 2019.
  - Others: Muessig, Yamamoto, Igata, Bhattarai, Wikenheiser, Gupta, Diba, Nádasdy, Kudrimoti,
    Grosmark 2016, and Drieu 2018 (FMAToolbox has no Drieu caller).
  - Tang 2017 and Jadhav 2016: Frank/Jadhav-lab code (FFPhy, Jadhav-Lab-Codes) is related only.
  - Chenani 2019 and Wu 2017: the named repositories hold no detector.
  - `harvey_2023_text`: the paper's repository runs DetectSWR and FindRipples, not the path the
    text describes.
  - `detect_silence_bounded_events` and `theta_delta_ratio`: they reimplement rules with no
    public code, or a generic construction.

## Blocking or complicating

- Octave's signal, statistics and image packages are not installed.
- Most lab code reads its own file layouts: the filter framework, NeuroScope with `session.mat`,
  Neuralynx MEX loaders, DataManager structures. Each comparison needs a driver that feeds traces
  to the source's own functions; the driver is not the source.
- Two MEX files must be compiled: `extractevents.cpp` and `nearest_idx3.c`.
- trodes2ff_shared and fklab v1.2 live on Bitbucket. GitHub code search covers default branches
  only.
