# Independent source recheck — September 25, 2026

This records sources actually reopened after the first parameter audit. A complete
repository inventory supports statements that a named helper is absent; reading selected
functions does not verify every analysis in the repository. No downloaded analysis code
was executed. MAT files were inspected as data; the serialized nelpy object was parsed
without executing its embedded globals. The Zotero/Dropbox libraries were not changed.

The [correction ledger](parameter_corrections_2026-09-25.csv) records every original-to-current
CSV change. The [field-status index](parameter_verification_2026-09-25.csv) covers all 1,995
cells (57 papers × 35 columns), including unchanged values and missing entries. It is
a provenance index, not a replication claim. [Remaining questions](verification_remaining.md)
are separated from completed source retrievals.

## Field-status meanings

- `checked_paper`: supported by the reopened primary text or figure at the listed location.
- `checked_code`: supported by the relevant released code; execution/provenance limitations remain in this register.
- `derived`: unit conversion, arithmetic, or an explicitly approximate reading of a figure.
- `inferred`: a stated assumption is needed (for example, applying a similar smoothing procedure to different spatial bins).
- `unresolved`: the required source or specific value could not be independently established.
- `not_reported`: no value was established in the consulted relevant Methods; this does not prove no other source reports it.
- `not_applicable`: no corresponding criterion/quantity in the summarized method, or an empty contextual field.
- `reviewed_context`: narrative reviewed with the primary sources and the limitations here; any numerical claims inherit their individual field status.
- `checked_metadata`: bibliographic/species metadata checked against the matched source.

Ranges and qualifications stay as text. The numeric loader drops them from scalar
summaries. A checked published value can still have unresolved code provenance; those
are distinct questions. Mallory’s published supplement is now read; its remaining
text/code differences are explicit. Bhattarai’s supplied SI is also read, replacing
the unresolved labels with direct page citations or explicit unreported details.

## Versioned repositories and inspected paths

| Paper/context | Reopened repository snapshot | Result and limit |
|---|---|---|
| Yang 2024 | [winnieyangwannan/Selection-of-experience-for-memory-by-hippocampal-sharp-wave-ripples@2fa5468](https://github.com/winnieyangwannan/Selection-of-experience-for-memory-by-hippocampal-sharp-wave-ripples/tree/2fa546845defbdd750c9059e7d54489782fed57c) | Tagged release 2: complete tree, README, main.py and stored ripple_HSE MAT metadata. main.py is a template; the stored detector name is find_HSE but detectionparms is empty. MAT blob identities match the reopened snapshot. No original independent LFP caller established. |
| Yang 2024 | [winnieyangwannan/buzcode@e960920](https://github.com/winnieyangwannan/buzcode/tree/e960920bcccc4fa47140471d8bcdd1fe2c99e8a2) | Tagged release 1: bz_FindRipples and FMAToolbox FindRipples. Their differing defaults do not identify which detector generated the paper events; find_HSE is absent from the tree. |
| Huelin Gorriz 2023 | [dbendor/Nat_Com_Huelin_Gorriz_et_al@b0676a5](https://github.com/dbendor/Nat_Com_Huelin_Gorriz_et_al/tree/b0676a5f8f682faf1dee74ed63223dbeb165affa) | Complete archived tree and batch_analysis_folders, list_of_parameters, process_clusters, extract_CSC and bayesian_decoding. Batch sets 1000 shuffles. MUA nominal SD 10 ms per pass, ripple 15 ms moving average, decoder selects raw spatial maps. Called extract_replay_events is absent; retain published 750 ms cap. |
| Berners-Lee 2022 | [ABernersLee/ReplayExperienceProject_NeuronPaper@2218f37](https://github.com/ABernersLee/ReplayExperienceProject_NeuronPaper/tree/2218f37c8135f2335c0c81076b71997b46c1221f) | Publication archive: decode_spikedensity_events, detection/decoding helpers and shuffle-count analysis. Reopened helper uses 1 ms spike bins, Gaussian SD 10 ms, 3 SD, 100-500 ms and nonoverlapping 20 ms decoding. Paper also describes 5 ms steps. Shuffles assess counts of criterion-selected events. |
| Mou 2022; supporting Wu 2017 | [DaoyunJiLab/DM2021@c7327f8](https://github.com/DaoyunJiLab/DM2021/tree/c7327f8dd2d3e78c6f4c67b71c58242d52cc0b11) | Complete archived tree and DataManager callback dispatch. DataManager_V1CA1_FindMUAlevel_Callback is referenced but absent. EEG frame routines do not establish the missing PBE detector or its normalization. |
| Denovellis 2021 | [Eden-Kramer-Lab/replay_trajectory_paper@f2d2b3c](https://github.com/Eden-Kramer-Lab/replay_trajectory_paper/tree/f2d2b3cc55968fe7d94d3109621b2772cdbb2c0a) | src/load_data.py and environment.yml connect 500 Hz MUA to multiunit_HSE_detector without a smoothing override. The pinned historical package was inspected with local git: 0.1.8.dev0 / 71298f0 defaults to 15 ms. Primary caller uses raw LFP at 1500 Hz and selected valid channels. |
| Gillespie 2021 | [LorenFrankLab/Gillespie_Neuron_2021@1be6a6a](https://github.com/LorenFrankLab/Gillespie_Neuron_2021/tree/1be6a6a7dbd806154a544d6f509452617e52d8c3) | Complete v1.0 archive and downstream muadecodesv3 usage. utilities/getMUAtrace has 2 ms bins and 5 ms SD despite a 15 ms comment; no calling path connects it to the published events. Upstream trodes_extractMUAevents at e90fd3d uses 1 ms bins and 15 ms SD. Retain paper 15 ms; original event-generating path remains unresolved. |
| Gridchyn 2020 | [igridchyn/lfp_online@a2d9cde](https://github.com/igridchyn/lfp_online/tree/a2d9cde41b389d7fee47ab5ae4188788c617f81e) | LFPBuffer.cpp, Utils.cpp and configuration inventory. Threshold is expected spike count times a factor, updated by +0.5*(measured rate-target) per minute. Default target is 1 Hz. EXPERIMENTAL_assembly_inhibition.conf has initial factor 3.5, 20 ms window and 150 ms refractory at 24 kHz. This compatible example is not proven to be the paper configuration; inhibition target 0.8 is a different setting. |
| Chenani 2019 | [cleibold/ReactivationCode@c008676](https://github.com/cleibold/ReactivationCode/tree/c008676683bc07f0395ff9df863e3b3b971c6bc2) | Complete small tree; README, testsession, rankseq, checktempshuffle and related rank-statistic helpers. It analyzes supplied sequences; no original SWR detector was found in this release. |
| Carey 2019 | [vandermeerlab/papers@3aebc8e](https://github.com/vandermeerlab/papers/tree/3aebc8e454c3b863b68c20c57421e6c16a6e809b) | Resolved the actual Git LFS candidate ZIP, then read R050-2014-03-29-candidates.mat as data. 1654 candidates; preserved configuration includes threshold 4, minimum 20 ms, 5 cells, initial speed 10 cm/s, theta threshold 2, spectral window 60 ms, fs 2000 Hz, MUA kernelstd 40 samples and spike cap 2. This is one released session, not proof of every session. |
| Carey 2019 | [vandermeerlab/vandermeerlab@ad0bbd4](https://github.com/vandermeerlab/vandermeerlab/tree/ad0bbd4d01726a436b36671c0a8b2db81476e946) | Paper-pinned precand, amSWR, amMUA and restriction helpers. amSWR is template-weighted spectral scoring with a 60 ms window, not Hilbert amplitude. amMUA convolves capped per-cell activity (40 samples/2000 Hz = 20 ms SD) and subtracts a slow noise estimate. Later wrapper defaults do not override the stored configuration or later published replay restrictions. |
| Maboudi 2018 | [kemerelab/UncoveringTemporalStructureHippocampus@f86b7dc](https://github.com/kemerelab/UncoveringTemporalStructureHippocampus/tree/f86b7dcc9ebf1105d9738086a91928f8f3f575a2) | Read repository and fig1.nel through an inert pickle-opcode parser: serialized globals were not executed. Stored nelpy version 0.2.0, one session 16-40-19, 457 MUA epochs (80-696 ms; minimum peak z=3.003), 277 final binned PBEs. Original event durations and subsequent binned support are different objects. The earlier 10 ms smoothing inference was not reproduced from the stored spike trains; retain the paper's 20 ms and label that inference unresolved. |
| Mallory 2025 | [caitlinmallory/TimeCourseOrganizationOfHippocampalReplay@128513a](https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay/tree/128513a656bdacfc11a0e0dfbcaf20e2c4515520) | Followed protocol configurations, candidate extraction, replay criteria/combination, rate maps and Gaussian helper. Track: 20 ms decoding, 2 cm spatial SD per filtering pass; arena: 80 ms decoding and nominal 4 cm spatial SD in code (mvnpdf covariance 8/2). Published supplement p. 5 confirms arena SD 8 cm. load_replayEvents_cm and the arena figure-combination script override the generic 100 ms duration with 50 ms. Supplement/code spatial merge, COM-jump and cell-selection differences are detailed in the paper note. |
| Krause 2022 | [DrugowitschLab/HippocampalSWRDynamics@cda23b7](https://github.com/DrugowitschLab/HippocampalSWRDynamics/tree/cda23b7fcc8a97222238a26c15d011d3593be39b) | ripple_preprocessing.py, ratday_preprocessing.py, config.py and highsynchronyevents.py. Original SWRs are precomputed input. Population-burst trimming uses a mean per-cell rate criterion (>2 Hz), first-to-last above-threshold samples and 30 ms minimum. Secondary HSE Gaussian SD is 10 ms in code; primary decode is 3 ms/4 cm. |
| Harvey 2023 | [ryanharvey1/ripple_heterogeneity@8461a94](https://github.com/ryanharvey1/ripple_heterogeneity/tree/8461a941b3edd58f1c14f106b2229db28ba76b56) | Kenji/Girardeau FindRipples callers supplement the previously reopened replay_run and session-processing code. Select one channel by relative 100-250 Hz power; pass bounds/peak [1,3], durations [50,300] and minDuration 20. The first durations element is a merge parameter, not a 50 ms minimum. Dependency versions and all dataset-specific execution histories are not pinned. |
| Liu 2023; Harvey supporting dependency | [ayalab1/neurocode@4b33b2a](https://github.com/ayalab1/neurocode/tree/4b33b2a14f80167ee3617f8f12888072cdcaea2c) | Downloaded the full Zenodo 7819979 release, matching 4b33b2a. DetectSWR and tutorial SWRpipeline are present. No Liu-specific caller/options establish an override of the shared defaults. Paper 15/400 ms and code 25/500 ms conditions remain distinct. |
| Supporting Bush/Olafsdottir | [Barry-lab/Publication_Shipley-et-al.-Disrupted-hippocampal-replay-in-an-Alzheimer-s-mouse-model@d75b85b](https://github.com/Barry-lab/Publication_Shipley-et-al.-Disrupted-hippocampal-replay-in-an-Alzheimer-s-mouse-model/tree/d75b85b2efdd44b287b13732419078e54e158433) | Shipley replay caller and detectMUA/singleArmDecode helpers: related later pipeline defaults differ (15 ms MUA, speed 3, 100 shuffles). They do not override Bush's own Methods. |
| Supporting Barry lab | [Barry-lab/PythonSpkAnalysis@d51c15a](https://github.com/Barry-lab/PythonSpkAnalysis/tree/d51c15aac4c46847749d550b9bd3fdf7949fa1eb) | detect_ripples/instant_freq_power: 150-250 Hz, default peak 5 SD, lower bound 0.5 SD, 50 ms boxcar power. This is a later port, not a demonstrated original execution path for the surveyed papers. |
| Supporting Shin 2019 | [SynapticSage/data-pipeline@ebf36da](https://github.com/SynapticSage/data-pipeline/tree/ebf36da1014790ed5355edf2b5ef50ae13e28b35) | generateGlobalRipples: union across selected tetrodes; defaults minstd 3, minrip 1 and speed 4. Options can override these. No paper-specific invocation was established. |
| Supporting Shin/Jadhav | [JadhavLab/Jadhav-Lab-Codes-JHB@973932a](https://github.com/JadhavLab/Jadhav-Lab-Codes-JHB/tree/973932a2beb50586821fcb974cd2cd2acaa0f338) | cs_rippletimes uses minstd 3, minnum 1, velocity 4 for another task. It is related lab code; no numerical override applied to the surveyed paper. |
| Supporting Drieu 2018 | [michael-zugaro/FMAToolbox@6bbb366](https://github.com/michael-zugaro/FMAToolbox/tree/6bbb3662f7ed1ccf09c5ff4b4d233e27e17c71a6) | QuietPeriods merges brief gaps before minimum-duration filtering; BrainStates classifies theta/delta-related states. These utilities support interpretation but are not a complete versioned replay caller. |
| Supporting Grosmark 2016 | [buzsakilab/buzcoderough@f3486d9](https://github.com/buzsakilab/buzcoderough/tree/f3486d9d1e96672e980ab37530d4410e607b48b4) | Later detect_swr/DetectSWR helper uses local thresholds and different event limits. No evidence ties this later function/version to the paper's original independent LFP events. |
| Supporting Ambrose 2016 | [Brad-E-Pfeiffer/DeepSuperficialSWRs@9f6ab57](https://github.com/Brad-E-Pfeiffer/DeepSuperficialSWRs/tree/9f6ab574812d82053e58c0b2bf627ea00eeae6f5) | Full DeepSuperficialSWRs ZIP: DSRP_FIND_RIPPLE_EVENTS reads duration limits from Initial_Variables; DEEP_VS_SUPERFICIAL_RIPPLE_PARTICIPATION_ANALYSIS sets 50-500 ms. Different study, so Ambrose duration limits remain Not reported. |
| Supporting Pfeiffer/Ambrose | [Brad-Pfeiffer/MouseDevelopmentalAnalysisCode@950fc28](https://github.com/Brad-Pfeiffer/MouseDevelopmentalAnalysisCode/tree/950fc288d69d5469094cba805eca0145f1548d40) | Full mouse ZIP: KJ_FIND_RIPPLE_EVENTS uses 50-500 ms limits. This later mouse pipeline does not establish a surveyed rat paper's detector settings. |
| Supporting Pfeiffer/Ambrose | [Brad-E-Pfeiffer/ThetaForwardReverseCode@bc714a2](https://github.com/Brad-E-Pfeiffer/ThetaForwardReverseCode/tree/bc714a2f0bd9d42fdd48e24b57ebbf91c6441a60) | Full theta ZIP: IRFS_FIND_RIPPLE_EVENTS uses 50-1000 ms limits. Conflicting later lab defaults reinforce the need for original caller provenance. |
| Carr 2012; Karlsson 2009 | [droumis/FFPhy@fce2048](https://github.com/droumis/FFPhy/tree/fce2048379d942c07f79d9192265ddf44f279d88) | FFPhy extractripples, extractevents.cpp, getripples and riptriggeredspectrogram_MC. 4 ms Gaussian, global mean/SD and extension to baseline; a continuous threshold crossing must meet mindur. Caller minstd 3; selected cell-bearing tetrodes and channel union occur downstream. Exact historical per-session options remain unproven. |
| Supporting Ji 2007 | [DaoyunJiLab/DataManager@f64fcdd](https://github.com/DaoyunJiLab/DataManager/tree/f64fcdd4c03b374561d8b23cdbb5947a6b656693) | Later DataManager frame detector has 10 ms bins and configurable frame gaps, including a 60 ms default. This does not override the original figure's per-animal 70-90 ms gaps. |
| Maboudi supporting dependency | [nelpy/nelpy@1255f57](https://github.com/nelpy/nelpy/tree/1255f57a3c4cc43bfa8dd15834c4c2363218703f) | Current library utility definitions were compared with stored version metadata. Current defaults cannot establish the original preprocessing; stored fig1.nel identifies nelpy 0.2.0. |
| Supporting Michon | [KatharinaBracher/fklab-python-core@24cd9e2](https://github.com/KatharinaBracher/fklab-python-core/tree/24cd9e23deee0cf2c46299824c3616c5e94d1748) | Fork envelope/detrending routines compared with original Kloosterman v1.2. No pinned version or caller connects either ordering to the paper. See OSF/original-toolbox checks below. |

Additional paper-linked release: [Tirole 44ecf42](https://github.com/bendor-lab/Elife_Tirole_Huelin_Gorriz_2022/tree/44ecf42).
`determine_best_channel.m` selects the strongest ripple channel using relative spectral power;
`number_of_significant_replays.m` requires ripple peak ≥3 and all three tests at p<0.05
for whole events or p<0.025 for split halves. The extractor and smoothing path were
already directly read in the first pass. This does not establish every runtime override.

## Archives and recovered published sources

| Source | What was independently checked |
|---|---|
| [Ji Supplementary Fig. 12](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnn1825/MediaObjects/41593_2007_BFnn1825_MOESM12_ESM.pdf) | Two-page publisher PDF. Caption confirms hippocampal gaps 90, 70, 90, 80 ms for animals 1–4. Earlier retrieval gap closed. |
| [Csicsvari 1999a](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782375/) | Detection Methods specify 150–250 Hz, RMS per electrode summed across electrodes and a 7-SD threshold. Confirms Nadasdy's inherited multi-electrode aggregation; RMS window remains unspecified. |
| Mallory published supplement, `science.ads4760_sm.pdf` (user-supplied local copy) | 37-page Science supplement, DOI 10.1126/science.ads4760; Methods pp. 5–8 and Fig. S2 pp. 16–17 checked directly. Confirms detection/decoding values and Spearman p<0.05 control. Retrieval gap closed; seven CSV fields refined. Artifact hash recorded below. |
| [Mallory preprint v1](https://www.biorxiv.org/content/10.1101/2024.07.18.604185v1.full.pdf) | Retrieved 44 pages. Methods pp. 21–23 distinguish protocols; positional errors 2.9/3.4 cm in the main text, 400 ms behavioral windows. Code/preprint discrepancies remain documented; this is not the published supplement. |
| [Olafsdottir 2016 supplementary software](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnn.4291/MediaObjects/41593_2016_BFnn4291_MOESM24_ESM.zip) | Four MATLAB posterior/line-fitting functions, including decode_calcPosterior and lineTraj_decode. Inputs include temporal/spatial bin size; examples/comments are not paper-specific caller settings. No detector or shuffle driver supplied. |
| [Yang analysis archive](https://zenodo.org/records/10685490) and [buzcode archive](https://zenodo.org/records/10685428) | Metadata, corresponding tagged GitHub trees and selected files reopened. Analysis concept resolves to version 2 / record 10816749. Stored HSE metadata does not specify the independent LFP detector. |
| [Huelin Gorriz 10085294](https://zenodo.org/records/10085294), [Berners-Lee 6330850](https://zenodo.org/records/6330850), [Mou 5758889](https://zenodo.org/records/5758889), [Gillespie 5140706](https://zenodo.org/records/5140706), [neurocode 7819979](https://zenodo.org/records/7819979) | Release metadata and complete small code archives were downloaded; inspected paths and missing-helper conclusions are listed above. |
| [Widloski 2022 code](https://zenodo.org/records/5880582), [2025 code](https://zenodo.org/records/15199609), [2025 data](https://zenodo.org/records/16916108) | Reopened MATLAB source files and complete file inventories. Decoder/event criteria and the 2025 100 cell-ID shuffles match the published Methods. Inventory confirms required external helpers are not bundled. Data deposit metadata inspected; large raw/data arrays were not exhaustively reopened and are not used to infer detector settings. |
| [Michon OSF smzby](https://osf.io/smzby/) | API inventory and behavior/place-code notebooks reopened. The code folder is empty in the API listing; notebooks import missing code.options and other modules. No event-preprocessing caller or dependency lock found. |
| [Original Kloosterman toolbox v1.2](https://bitbucket.org/kloostermannerflab/fklab-python-core/src/f4176cd36bc73a0266fdae0df6dca60b14fee0e2/) | Original compute_envelope and core routines retrieved. This version has no median detrending in compute_envelope; the previously cited fork detrends before smoothing. Paper workflow states smoothing then detrending. Original caller/version is unresolved. |
| [Barry raw data 5566548](https://zenodo.org/records/5566548) | Metadata identifies Olafsdottir 2016, not a new Bush-specific detector release. All 1,310 ZIP-directory entries read via bounded HTTP ranges. No separately named SWR/replay/candidate event files found. This does not inspect every variable inside every MAT file. |
| Bhattarai published SI, `pnas.1912533117.sapp.pdf` (user-supplied local copy) | Title, authors and DOI match. Read Methods pp. 2–7 and inspected Table S1 image on p. 21. Replay 20 ms/10 ms step; behavioral windows 50 ms; spatial SD 6 cm; 1000 circular-position shuffles, p<0.05; error 6.56 cm and 1342/2573 replays confirmed. Nine CSV fields updated; earlier 50 ms replay transcription corrected. Retrieval gap closed. |
| [Bhattarai Figshare v2](https://doi.org/10.6084/m9.figshare.10032866.v2) | Full ZIP directory plus ReadMe/MetaData.docx: beh.mat, events.mat, lfp.mat, spikes.mat, VT.mat and MetaData.docx. Metadata defines events.mat as trial/delay/reward timing and LFP as one tetrode trace. No deposited detector code or documented SWR/replay event table. |
| [CRCNShc-11](https://crcns.org/data-sets/hc/hc-11/about-hc-11) and [DANDI 000044 v0.250624.0426](https://dandiarchive.org/dandiset/000044/0.250624.0426) | CRCNS description and DANDI asset catalog reopened. Inspected HDF5 metadata via small byte ranges in all eight NWB files: intervals contains only epochs; analysis is empty; processing contains behavior/ecephys. No SWR/replay interval table found. Did not download all raw arrays or reproduce original SWRs. |
| [Additional Jadhav-data analysis](https://github.com/edeno/Jadhav-2016-Data-Analysis/tree/0b5d4d584dacce75fed9a6ad1c21658773cf2ca1) | Fresh search found this repository; README explicitly identifies the Jadhav dataset. Parameters and ripple-detection code reopened. This is an analysis of those data, not evidence that its settings generated the original paper's events. |

## Final source-access closures

The former list at the end of the literature README was revisited directly.

| Source | Completed check | What remains |
|---|---|---|
| [Ambrose 2016 supplement](https://ars.els-cdn.com/content/image/1-s2.0-S0896627316304639-mmc1.pdf) | Full 12-page publisher PDF; Table S1 visually inspected and counts summed. Four CSV fields updated. | No published detector-duration limits; related later code cannot supply original settings. |
| [O’Neill 2008 article and supplement](https://www.mrcbndu.ox.ac.uk/sites/default/files/pdfs/oneill2008natureneurosci.pdf) | 14-page Oxford-hosted copy, especially supplementary field/state Methods on p. 13. Muessig citation retrieval closed. | Muessig’s peak-centered spectral bandwidth remains unspecified. |
| [Jackson 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674885/) | Primary LFP Methods read directly. Earlier reliance on a page summary is superseded in Wikenheiser/Bhattarai/Gupta notes. | Downstream papers’ unstated implementation details are not supplied by this citation. |
| [Csicsvari 1999](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782375/) | Primary Methods reread; stale inaccessible-source claims removed from Gridchyn, Xu and Stella notes. | Numerical theta-ratio cutoff, ripple RMS window and exact event boundaries remain unstated. |

The Ji “not reverified,” Michon “OSF not opened,” and CSV “changes on hold”
statements were also stale; the current documentation reflects the completed checks.
These closures do not certify historical pipeline replication.

Selected downloaded-artifact hashes are in [source_artifacts_2026-09-25.csv](source_artifacts_2026-09-25.csv).
Raw PDFs, private library paths and large data archives are intentionally outside the repository.

## Fresh code-availability searches

On September 25, fresh author/year/replay/GitHub or exact-title/code searches were run
for rows 12, 18, 20–23, 26, 29, 34–35, 38–44, 47–48, 50–51 and 53–56.
The paper's availability statements were also reread. These searches recovered the
Olafsdottir supplementary software and the Jadhav-data analysis above. They did not
establish an additional original detector caller for the remaining rows. This is a
bounded search result, not proof that no code exists. Related later lab releases are
already distinguished in the table. Berners-Lee 2021, Kaefer and Yamamoto explicitly
offer data/code on request. No author was contacted.

## Integrity and scope

All 138 first-pass correction cells are accounted for: 131 retain the same value and
seven were deliberately refined (Mallory and Bhattarai source/protocol context,
Denovellis MUA width/context, Ji frame-gap/context and Nadasdy channel context). No earlier edit
was silently dropped. The follow-up, including the supplied supplements and the Ambrose supplement,
changed 105 cells across 32 papers, including notes and missing-value qualifications. Relative
to the pre-audit CSV, the cumulative ledger records 236 changed cells across 49 papers.
The cumulative ledger compares the pre-audit and final CSV; the follow-up history
compares the first-pass snapshot with the current CSV, including the supplied supplements and later source closures.
Bhattarai’s restored original 20 ms replay window and 6.56 cm error no longer
contribute to the cumulative difference count; the paper note records why.

Checks preserve 57 rows, 35 columns, DOI identities and row order. Published README
statistics are recomputed from the final CSV. The field ledger records the evidence
for the current snapshot; future CSV changes need corresponding ledger updates.
Recipe equivalence, historical tier assignments and full reproduction of authors'
analyses are separate tasks and are not certified by this parameter audit.

The [follow-up change history](parameter_recheck_changes_2026-09-25.csv) preserves
first-pass values alongside current refinements. It is a baseline-to-current comparison,
not a chronological log of every interim edit.

Validation: 20 literature/recipe tests passed; Ruff check/format and git diff whitespace
checks passed. Independent CSV/ledger checks confirm every current difference and every
field-status value, with no missing rows, columns or changed DOI order.

Audited CSV SHA-256: `e283a054f2a652585e56a9bf1bbc3f52bcdb4397db28f798d8b45bc5879fc2a3`.
