# Literature sources

This catalog records primary papers, source versions, inspection scope and selected artifact fingerprints. Field-level citations and statuses are in [evidence.csv](evidence.csv); methods and limitations are in the linked paper notes. Sources were inspected through September 25, 2026.

Reading a repository is not a replication of its pipeline. Complete inventories support missing-file statements; selected function reads do not verify an entire analysis. Downloaded analysis code was not executed. MAT files were read as data, and the serialized nelpy artifact was parsed without executing its globals. PDFs and large/private data remain outside the repository; the local Zotero/Dropbox libraries were not modified.

## Primary papers

| Paper | DOI | Additional inspected sources |
|---|---|---|
| [Mallory 2025, Science](papers/00_Mallory_2025.md) | https://doi.org/10.1126/science.ads4760 | [mallory-code](#mallory-code), [mallory-supplement](#mallory-supplement), [mallory-preprint](#mallory-preprint) |
| [Widloski 2025, Nature Communications](papers/01_Widloski_2025.md) | https://doi.org/10.1038/s41467-025-65181-5 | [widloski-archives](#widloski-archives) |
| [Yang 2024, Science](papers/02_Yang_2024.md) | https://doi.org/10.1126/science.adk8261 | [yang-analysis](#yang-analysis), [yang-buzcode](#yang-buzcode), [yang-archives](#yang-archives) |
| [Huelin Gorriz 2023, Nature Communications](papers/03_HuelinGorriz_2023.md) | https://doi.org/10.1038/s41467-023-43939-z | [huelin-gorriz-code](#huelin-gorriz-code), [tirole-code](#tirole-code), [code-archives](#code-archives) |
| [Harvey 2023, Neuron](papers/04_Harvey_2023.md) | https://doi.org/10.1016/j.neuron.2023.04.015 | [harvey-code](#harvey-code), [neurocode](#neurocode) |
| [Liu 2023, Science](papers/05_Liu_2023.md) | https://doi.org/10.1126/science.adi8237 | [neurocode](#neurocode), [code-archives](#code-archives) |
| [Tirole 2022, eLife](papers/06_Tirole_2022.md) | https://doi.org/10.7554/eLife.79031 | [tirole-code](#tirole-code) |
| [Bush 2022, Current Biology](papers/07_Bush_2022.md) | https://doi.org/10.1016/j.cub.2021.10.033 | [shipley-code](#shipley-code), [barry-python](#barry-python), [barry-data](#barry-data) |
| [Berners-Lee 2022, Neuron](papers/08_Berners-Lee_2022.md) | https://doi.org/10.1016/j.neuron.2022.03.010 | [berners-lee-code](#berners-lee-code), [code-archives](#code-archives) |
| [Widloski 2022, Neuron](papers/09_Widloski_2022.md) | https://doi.org/10.1016/j.neuron.2022.02.002 | [widloski-archives](#widloski-archives) |
| [Krause 2022, Neuron](papers/10_Krause_2022.md) | https://doi.org/10.1016/j.neuron.2021.11.014 | [krause-code](#krause-code) |
| [Mou 2022, Neuron](papers/11_Mou_2022.md) | https://doi.org/10.1016/j.neuron.2021.12.005 | [mou-code](#mou-code), [code-archives](#code-archives) |
| [Berners-Lee 2021, Journal of Neuroscience](papers/12_Berners-Lee_2021.md) | https://doi.org/10.1523/JNEUROSCI.1158-20.2021 | See paper note for availability and inherited methods. |
| [Denovellis 2021, eLife](papers/13_Denovellis_2021.md) | https://doi.org/10.7554/eLife.64505 | [denovellis-code](#denovellis-code) |
| [Gillespie 2021, Neuron](papers/14_Gillespie_2021.md) | https://doi.org/10.1016/j.neuron.2021.07.029 | [gillespie-code](#gillespie-code), [code-archives](#code-archives) |
| [Michon 2021, Current Biology](papers/15_Michon_2021.md) | https://doi.org/10.1016/j.cub.2021.07.058 | [michon-osf](#michon-osf), [michon-toolbox](#michon-toolbox), [michon-fork](#michon-fork) |
| [Igata 2021, PNAS](papers/16_Igata_2021.md) | https://doi.org/10.1073/pnas.2011266118 | See paper note for availability and inherited methods. |
| [Gridchyn 2020, Neuron](papers/17_Gridchyn_2020.md) | https://doi.org/10.1016/j.neuron.2020.01.021 | [gridchyn-code](#gridchyn-code), [csicsvari-1999](#csicsvari-1999) |
| [Kaefer 2020, Neuron](papers/18_Kaefer_2020.md) | https://doi.org/10.1016/j.neuron.2020.01.015 | See paper note for availability and inherited methods. |
| [Bhattarai 2020, PNAS](papers/19_Bhattarai_2020.md) | https://doi.org/10.1073/pnas.1912533117 | [bhattarai-supplement](#bhattarai-supplement), [bhattarai-data](#bhattarai-data), [jackson-2006](#jackson-2006) |
| [Stella 2019, Neuron](papers/20_Stella_2019.md) | https://doi.org/10.1016/j.neuron.2019.01.052 | [csicsvari-1999](#csicsvari-1999) |
| [Xu 2019, Neuron](papers/21_Xu_2019.md) | https://doi.org/10.1016/j.neuron.2018.11.015 | [csicsvari-1999](#csicsvari-1999) |
| [Farooq 2019, Neuron](papers/22_Farooq_2019.md) | https://doi.org/10.1016/j.neuron.2019.05.040 | See paper note for availability and inherited methods. |
| [Farooq 2019, Science](papers/23_Farooq_2019.md) | https://doi.org/10.1126/science.aav0502 | See paper note for availability and inherited methods. |
| [Chenani 2019, Nature Communications](papers/24_Chenani_2019.md) | https://doi.org/10.1038/s41467-019-09280-0 | [chenani-code](#chenani-code) |
| [Michon 2019, Current Biology](papers/25_Michon_2019.md) | https://doi.org/10.1016/j.cub.2019.03.048 | [michon-toolbox](#michon-toolbox), [michon-fork](#michon-fork) |
| [Liu 2019, Hippocampus](papers/26_Liu_2019.md) | https://doi.org/10.1002/hipo.23034 | See paper note for availability and inherited methods. |
| [Shin 2019, Neuron](papers/27_Shin_2019.md) | https://doi.org/10.1016/j.neuron.2019.09.012 | [shin-code](#shin-code), [jadhav-code](#jadhav-code) |
| [Carey 2019, Nature Neuroscience](papers/28_Carey_2019.md) | https://doi.org/10.1038/s41593-019-0464-6 | [carey-data](#carey-data), [carey-code](#carey-code) |
| [Muessig 2019, Current Biology](papers/29_Muessig_2019.md) | https://doi.org/10.1016/j.cub.2019.01.005 | [oneill-2008](#oneill-2008) |
| [Drieu 2018, Science](papers/30_Drieu_2018.md) | https://doi.org/10.1126/science.aat2952 | [fmatoolbox](#fmatoolbox) |
| [Maboudi 2018, eLife](papers/31_Maboudi_2018.md) | https://doi.org/10.7554/eLife.34467 | [maboudi-code](#maboudi-code), [nelpy](#nelpy) |
| [Ólafsdóttir 2017, Neuron](papers/32_Olafsdottir_2017.md) | https://doi.org/10.1016/j.neuron.2017.09.035 | [shipley-code](#shipley-code), [barry-python](#barry-python) |
| [Wu 2017, Nature Neuroscience](papers/33_Wu_2017.md) | https://doi.org/10.1038/nn.4507 | [mou-code](#mou-code) |
| [Yamamoto 2017, Neuron](papers/34_Yamamoto_2017.md) | https://doi.org/10.1016/j.neuron.2017.09.017 | See paper note for availability and inherited methods. |
| [Tang 2017, Journal of Neuroscience](papers/35_Tang_2017.md) | https://doi.org/10.1523/JNEUROSCI.2291-17.2017 | See paper note for availability and inherited methods. |
| [Grosmark 2016, Science](papers/36_Grosmark_2016.md) | https://doi.org/10.1126/science.aad1935 | [buzcoderough](#buzcoderough), [grosmark-data](#grosmark-data) |
| [Ambrose 2016, Neuron](papers/37_Ambrose_2016.md) | https://doi.org/10.1016/j.neuron.2016.07.047 | [ambrose-supplement](#ambrose-supplement), [deep-superficial-code](#deep-superficial-code), [mouse-development-code](#mouse-development-code), [theta-code](#theta-code) |
| [Jadhav 2016, Neuron](papers/38_Jadhav_2016.md) | https://doi.org/10.1016/j.neuron.2016.02.010 | [jadhav-code](#jadhav-code), [jadhav-data-analysis](#jadhav-data-analysis) |
| [Ólafsdóttir 2016, Nature Neuroscience](papers/39_Olafsdottir_2016.md) | https://doi.org/10.1038/nn.4291 | [olafsdottir-software](#olafsdottir-software), [barry-data](#barry-data) |
| [Silva 2015, Nature Neuroscience](papers/40_Silva_2015.md) | https://doi.org/10.1038/nn.4151 | See paper note for availability and inherited methods. |
| [Ólafsdóttir 2015, eLife](papers/41_Olafsdottir_2015.md) | https://doi.org/10.7554/eLife.06063 | [barry-python](#barry-python) |
| [Pfeiffer 2015, Science](papers/42_Pfeiffer_2015.md) | https://doi.org/10.1126/science.aaa9633 | [deep-superficial-code](#deep-superficial-code), [mouse-development-code](#mouse-development-code), [theta-code](#theta-code) |
| [Wu 2014, Journal of Neuroscience](papers/43_Wu_2014.md) | https://doi.org/10.1523/JNEUROSCI.3414-13.2014 | See paper note for availability and inherited methods. |
| [Wikenheiser 2013, Hippocampus](papers/44_Wikenheiser_2013.md) | https://doi.org/10.1002/hipo.22049 | [jackson-2006](#jackson-2006) |
| [Pfeiffer 2013, Nature](papers/45_Pfeiffer_2013.md) | https://doi.org/10.1038/nature12112 | [mouse-development-code](#mouse-development-code), [theta-code](#theta-code) |
| [Carr 2012, Neuron](papers/46_Carr_2012.md) | https://doi.org/10.1016/j.neuron.2012.06.014 | [ffphy](#ffphy) |
| [Bendor 2012, Nature Neuroscience](papers/47_Bendor_2012.md) | https://doi.org/10.1038/nn.3203 | See paper note for availability and inherited methods. |
| [Gupta 2010, Neuron](papers/48_Gupta_2010.md) | https://doi.org/10.1016/j.neuron.2010.01.034 | [jackson-2006](#jackson-2006) |
| [Karlsson 2009, Nature Neuroscience](papers/49_Karlsson_2009.md) | https://doi.org/10.1038/nn.2344 | [ffphy](#ffphy) |
| [Davidson 2009, Neuron](papers/50_Davidson_2009.md) | https://doi.org/10.1016/j.neuron.2009.07.027 | See paper note for availability and inherited methods. |
| [Diba 2007, Nature Neuroscience](papers/51_Diba_2007.md) | https://doi.org/10.1038/nn1961 | See paper note for availability and inherited methods. |
| [Ji 2007, Nature Neuroscience](papers/52_Ji_2007.md) | https://doi.org/10.1038/nn1825 | [ji-figure-12](#ji-figure-12), [ji-code](#ji-code) |
| [Foster 2006, Nature](papers/53_Foster_2006.md) | https://doi.org/10.1038/nature04587 | See paper note for availability and inherited methods. |
| [Lee 2002, Neuron](papers/54_Lee_2002.md) | https://doi.org/10.1016/S0896-6273(02)01096-6 | See paper note for availability and inherited methods. |
| [Nádasdy 1999, Journal of Neuroscience](papers/55_Nadasdy_1999.md) | https://doi.org/10.1523/JNEUROSCI.19-21-09497.1999 | [csicsvari-1999](#csicsvari-1999) |
| [Kudrimoti 1999, Journal of Neuroscience](papers/56_Kudrimoti_1999.md) | https://doi.org/10.1523/JNEUROSCI.19-10-04090.1999 | See paper note for availability and inherited methods. |

## Repositories

### yang-analysis

Yang 2024: [winnieyangwannan/Selection-of-experience-for-memory-by-hippocampal-sharp-wave-ripples@2fa5468](https://github.com/winnieyangwannan/Selection-of-experience-for-memory-by-hippocampal-sharp-wave-ripples/tree/2fa546845defbdd750c9059e7d54489782fed57c)

Tagged release 2: complete tree, README, main.py and stored ripple_HSE MAT metadata. main.py is a template; the stored detector name is find_HSE but detectionparms is empty. MAT blob identities match the inspected snapshot. No original independent LFP caller established.

### yang-buzcode

Yang 2024: [winnieyangwannan/buzcode@e960920](https://github.com/winnieyangwannan/buzcode/tree/e960920bcccc4fa47140471d8bcdd1fe2c99e8a2)

Tagged release 1: bz_FindRipples and FMAToolbox FindRipples. Their differing defaults do not identify which detector generated the paper events; find_HSE is absent from the tree.

### huelin-gorriz-code

Huelin Gorriz 2023: [dbendor/Nat_Com_Huelin_Gorriz_et_al@b0676a5](https://github.com/dbendor/Nat_Com_Huelin_Gorriz_et_al/tree/b0676a5f8f682faf1dee74ed63223dbeb165affa)

Complete archived tree and batch_analysis_folders, list_of_parameters, process_clusters, extract_CSC and bayesian_decoding. Batch sets 1000 shuffles. MUA nominal SD 10 ms per pass, ripple 15 ms moving average, decoder selects raw spatial maps. Called extract_replay_events is absent; retain published 750 ms cap.

### berners-lee-code

Berners-Lee 2022: [ABernersLee/ReplayExperienceProject_NeuronPaper@2218f37](https://github.com/ABernersLee/ReplayExperienceProject_NeuronPaper/tree/2218f37c8135f2335c0c81076b71997b46c1221f)

Publication archive: decode_spikedensity_events, detection/decoding helpers and shuffle-count analysis. Inspected helper uses 1 ms spike bins, Gaussian SD 10 ms, 3 SD, 100-500 ms and nonoverlapping 20 ms decoding. Paper also describes 5 ms steps. Shuffles assess counts of criterion-selected events.

### mou-code

Mou 2022; supporting Wu 2017: [DaoyunJiLab/DM2021@c7327f8](https://github.com/DaoyunJiLab/DM2021/tree/c7327f8dd2d3e78c6f4c67b71c58242d52cc0b11)

Complete archived tree and DataManager callback dispatch. DataManager_V1CA1_FindMUAlevel_Callback is referenced but absent. EEG frame routines do not establish the missing PBE detector or its normalization.

### denovellis-code

Denovellis 2021: [Eden-Kramer-Lab/replay_trajectory_paper@f2d2b3c](https://github.com/Eden-Kramer-Lab/replay_trajectory_paper/tree/f2d2b3cc55968fe7d94d3109621b2772cdbb2c0a)

src/load_data.py and environment.yml connect 500 Hz MUA to multiunit_HSE_detector without a smoothing override. The pinned historical package was inspected with local git: 0.1.8.dev0 / 71298f0 defaults to 15 ms. Primary caller uses raw LFP at 1500 Hz and selected valid channels.

### gillespie-code

Gillespie 2021: [LorenFrankLab/Gillespie_Neuron_2021@1be6a6a](https://github.com/LorenFrankLab/Gillespie_Neuron_2021/tree/1be6a6a7dbd806154a544d6f509452617e52d8c3)

Complete v1.0 archive and downstream muadecodesv3 usage. utilities/getMUAtrace has 2 ms bins and 5 ms SD despite a 15 ms comment; no calling path connects it to the published events. Upstream trodes_extractMUAevents at e90fd3d uses 1 ms bins and 15 ms SD. Retain paper 15 ms; original event-generating path remains unresolved.

### gridchyn-code

Gridchyn 2020: [igridchyn/lfp_online@a2d9cde](https://github.com/igridchyn/lfp_online/tree/a2d9cde41b389d7fee47ab5ae4188788c617f81e)

LFPBuffer.cpp, Utils.cpp and configuration inventory. Threshold is expected spike count times a factor, updated by +0.5*(measured rate-target) per minute. Default target is 1 Hz. EXPERIMENTAL_assembly_inhibition.conf has initial factor 3.5, 20 ms window and 150 ms refractory at 24 kHz. This compatible example is not proven to be the paper configuration; inhibition target 0.8 is a different setting.

Also inspected at the same commit: [LFPPipeline.cpp](https://github.com/igridchyn/lfp_online/blob/a2d9cde41b389d7fee47ab5ae4188788c617f81e/lfp_online/LFPPipeline.cpp#L173), [PackageExractorProcessor.cpp](https://github.com/igridchyn/lfp_online/blob/a2d9cde41b389d7fee47ab5ae4188788c617f81e/lfp_online/PackageExractorProcessor.cpp#L192) and [LPTTriggerProcessor.cpp](https://github.com/igridchyn/lfp_online/blob/a2d9cde41b389d7fee47ab5ae4188788c617f81e/lfp_online/LPTTriggerProcessor.cpp#L326). The pipeline processes buffered chunks; the trigger logic can accumulate further evidence before making an inhibition decision.

### chenani-code

Chenani 2019: [cleibold/ReactivationCode@c008676](https://github.com/cleibold/ReactivationCode/tree/c008676683bc07f0395ff9df863e3b3b971c6bc2)

Complete small tree; README, testsession, rankseq, checktempshuffle and related rank-statistic helpers. It analyzes supplied sequences; no original SWR detector was found in this release.

### carey-data

Carey 2019: [vandermeerlab/papers@3aebc8e](https://github.com/vandermeerlab/papers/tree/3aebc8e454c3b863b68c20c57421e6c16a6e809b)

Resolved the actual Git LFS candidate ZIP, then read R050-2014-03-29-candidates.mat as data. 1654 candidates; preserved configuration includes threshold 4, minimum 20 ms, 5 cells, initial speed threshold 10 (stored units; see Carey note), theta threshold 2, spectral window 60 ms, fs 2000 Hz, MUA kernelstd 40 samples and spike cap 2. This is one released session, not proof of every session.

### carey-code

Carey 2019: [vandermeerlab/vandermeerlab@ad0bbd4](https://github.com/vandermeerlab/vandermeerlab/tree/ad0bbd4d01726a436b36671c0a8b2db81476e946)

Paper-pinned precand, amSWR, amMUA and restriction helpers. amSWR is template-weighted spectral scoring with a 60 ms window, not Hilbert amplitude. amMUA convolves capped per-cell activity (40 samples/2000 Hz = 20 ms SD) and subtracts a slow noise estimate. Later wrapper defaults do not override the stored configuration or later published replay restrictions.

### maboudi-code

Maboudi 2018: [kemerelab/UncoveringTemporalStructureHippocampus@f86b7dc](https://github.com/kemerelab/UncoveringTemporalStructureHippocampus/tree/f86b7dcc9ebf1105d9738086a91928f8f3f575a2)

Read repository and fig1.nel through an inert pickle-opcode parser: serialized globals were not executed. Stored nelpy version 0.2.0, one session 16-40-19, 457 MUA epochs (80-696 ms; minimum peak z=3.003), 277 final binned PBEs. Original event durations and subsequent binned support are different objects. The stored spike trains do not independently establish a 10 ms smoothing kernel; the paper's 20 ms remains authoritative.

### mallory-code

Mallory 2025: [caitlinmallory/TimeCourseOrganizationOfHippocampalReplay@128513a](https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay/tree/128513a656bdacfc11a0e0dfbcaf20e2c4515520)

Release: [v1.0.0](https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay/tree/v1.0.0), archived as [Zenodo 14237298](https://doi.org/10.5281/zenodo.14237298).

Followed protocol configurations, candidate extraction, replay criteria/combination, rate maps and Gaussian helper. Track: 20 ms decoding, 2 cm spatial SD per filtering pass; arena: 80 ms decoding and nominal 4 cm spatial SD in code (mvnpdf covariance 8/2). Published supplement p. 5 confirms arena SD 8 cm. load_replayEvents_cm and the arena figure-combination script override the generic 100 ms duration with 50 ms. Supplement/code spatial merge, COM-jump and cell-selection differences are detailed in the paper note.

### krause-code

Krause 2022: [DrugowitschLab/HippocampalSWRDynamics@cda23b7](https://github.com/DrugowitschLab/HippocampalSWRDynamics/tree/cda23b7fcc8a97222238a26c15d011d3593be39b)

ripple_preprocessing.py, ratday_preprocessing.py, config.py, highsynchronyevents.py and utils.py. The reopened binning helper excludes bins ending exactly at the SWR end; trimming retains the unbinned remainder in its end coordinate. Original SWRs are precomputed input. Population-burst trimming uses a mean per-cell rate criterion (>2 Hz), first-to-last above-threshold samples and 30 ms minimum. Secondary HSE Gaussian SD is 10 ms in code; primary decode is 3 ms/4 cm.

### harvey-code

Harvey 2023: [ryanharvey1/ripple_heterogeneity@8461a94](https://github.com/ryanharvey1/ripple_heterogeneity/tree/8461a941b3edd58f1c14f106b2229db28ba76b56)

Kenji/Girardeau FindRipples callers supplement the inspected replay_run and session-processing code. Select one channel by relative 100-250 Hz power; pass bounds/peak [1,3], durations [50,300] and minDuration 20. The first durations element is a merge parameter, not a 50 ms minimum. Dependency versions and all dataset-specific execution histories are not pinned.

### neurocode

Liu 2023; Harvey supporting dependency: [ayalab1/neurocode@4b33b2a](https://github.com/ayalab1/neurocode/tree/4b33b2a14f80167ee3617f8f12888072cdcaea2c)

Downloaded the full Zenodo 7819979 release, matching 4b33b2a. DetectSWR and tutorial SWRpipeline are present. No Liu-specific caller/options establish an override of the shared defaults. Paper 15/400 ms and code 25/500 ms conditions remain distinct.

### shipley-code

Supporting Bush/Olafsdottir: [Barry-lab/Publication_Shipley-et-al.-Disrupted-hippocampal-replay-in-an-Alzheimer-s-mouse-model@d75b85b](https://github.com/Barry-lab/Publication_Shipley-et-al.-Disrupted-hippocampal-replay-in-an-Alzheimer-s-mouse-model/tree/d75b85b2efdd44b287b13732419078e54e158433)

Shipley replay caller and detectMUA/singleArmDecode helpers: related later pipeline defaults differ (15 ms MUA, speed 3, 100 shuffles). They do not override Bush's own Methods.

### barry-python

Supporting Barry lab: [Barry-lab/PythonSpkAnalysis@d51c15a](https://github.com/Barry-lab/PythonSpkAnalysis/tree/d51c15aac4c46847749d550b9bd3fdf7949fa1eb)

detect_ripples/instant_freq_power: 150-250 Hz, default peak 5 SD, lower bound 0.5 SD, 50 ms boxcar power. This is a later port, not a demonstrated original execution path for the surveyed papers.

### shin-code

Supporting Shin 2019: [SynapticSage/data-pipeline@ebf36da](https://github.com/SynapticSage/data-pipeline/tree/ebf36da1014790ed5355edf2b5ef50ae13e28b35)

generateGlobalRipples: union across selected tetrodes; defaults minstd 3, minrip 1 and speed 4. Options can override these. No paper-specific invocation was established.

### jadhav-code

Supporting Shin/Jadhav: [JadhavLab/Jadhav-Lab-Codes-JHB@973932a](https://github.com/JadhavLab/Jadhav-Lab-Codes-JHB/tree/973932a2beb50586821fcb974cd2cd2acaa0f338)

cs_rippletimes uses minstd 3, minnum 1, velocity 4 for another task. It is related lab code; no numerical override applied to the surveyed paper.

### fmatoolbox

Supporting Drieu 2018: [michael-zugaro/FMAToolbox@6bbb366](https://github.com/michael-zugaro/FMAToolbox/tree/6bbb3662f7ed1ccf09c5ff4b4d233e27e17c71a6)

QuietPeriods merges brief gaps before minimum-duration filtering; BrainStates classifies theta/delta-related states. These utilities support interpretation but are not a complete versioned replay caller.

### buzcoderough

Supporting Grosmark 2016: [buzsakilab/buzcoderough@f3486d9](https://github.com/buzsakilab/buzcoderough/tree/f3486d9d1e96672e980ab37530d4410e607b48b4)

Later detect_swr/DetectSWR helper uses local thresholds and different event limits. No evidence ties this later function/version to the paper's original independent LFP events.

### deep-superficial-code

Supporting Ambrose 2016: [Brad-E-Pfeiffer/DeepSuperficialSWRs@9f6ab57](https://github.com/Brad-E-Pfeiffer/DeepSuperficialSWRs/tree/9f6ab574812d82053e58c0b2bf627ea00eeae6f5)

Full DeepSuperficialSWRs ZIP: DSRP_FIND_RIPPLE_EVENTS reads duration limits from Initial_Variables; DEEP_VS_SUPERFICIAL_RIPPLE_PARTICIPATION_ANALYSIS sets 50-500 ms. Different study, so Ambrose duration limits remain Not reported.

### mouse-development-code

Supporting Pfeiffer/Ambrose: [Brad-Pfeiffer/MouseDevelopmentalAnalysisCode@950fc28](https://github.com/Brad-Pfeiffer/MouseDevelopmentalAnalysisCode/tree/950fc288d69d5469094cba805eca0145f1548d40)

Full mouse ZIP: KJ_FIND_RIPPLE_EVENTS uses 50-500 ms limits. This later mouse pipeline does not establish a surveyed rat paper's detector settings.

### theta-code

Supporting Pfeiffer/Ambrose: [Brad-E-Pfeiffer/ThetaForwardReverseCode@bc714a2](https://github.com/Brad-E-Pfeiffer/ThetaForwardReverseCode/tree/bc714a2f0bd9d42fdd48e24b57ebbf91c6441a60)

Full theta ZIP: IRFS_FIND_RIPPLE_EVENTS uses 50-1000 ms limits. Conflicting later lab defaults reinforce the need for original caller provenance.

### ffphy

Carr 2012; Karlsson 2009: [droumis/FFPhy@fce2048](https://github.com/droumis/FFPhy/tree/fce2048379d942c07f79d9192265ddf44f279d88)

FFPhy extractripples, extractevents.cpp, getripples and riptriggeredspectrogram_MC. 4 ms Gaussian, global mean/SD and extension to baseline; a continuous threshold crossing must meet mindur. Caller minstd 3; selected cell-bearing tetrodes and channel union occur downstream. Exact historical per-session options remain unproven.

### ji-code

Supporting Ji 2007: [DaoyunJiLab/DataManager@f64fcdd](https://github.com/DaoyunJiLab/DataManager/tree/f64fcdd4c03b374561d8b23cdbb5947a6b656693)

Later DataManager frame detector has 10 ms bins and configurable frame gaps, including a 60 ms default. This does not override the original figure's per-animal 70-90 ms gaps.

### nelpy

Maboudi supporting dependency: [nelpy/nelpy@1255f57](https://github.com/nelpy/nelpy/tree/1255f57a3c4cc43bfa8dd15834c4c2363218703f)

Current library utility definitions were compared with stored version metadata. Current defaults cannot establish the original preprocessing; stored fig1.nel identifies nelpy 0.2.0.

### michon-fork

Supporting Michon: [KatharinaBracher/fklab-python-core@24cd9e2](https://github.com/KatharinaBracher/fklab-python-core/tree/24cd9e23deee0cf2c46299824c3616c5e94d1748)

Fork envelope/detrending routines compared with original Kloosterman v1.2. No pinned version or caller connects either ordering to the paper. See OSF/original-toolbox checks below.

### tirole-code

Tirole 2022: [bendor-lab/Elife_Tirole_Huelin_Gorriz_2022](https://github.com/bendor-lab/Elife_Tirole_Huelin_Gorriz_2022/tree/44ecf4275c2a7ca33dda6b67f11b4e854c3123e9)

`determine_best_channel.m` selects the strongest ripple channel by relative spectral power. `number_of_significant_replays.m` requires ripple peak ≥3 and all three tests at p<0.05 for whole events or p<0.025 for split halves. The extractor and smoothing paths were read. This does not establish every runtime override.

## Archives and supporting publications

### ji-figure-12

[Ji Supplementary Fig. 12](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnn1825/MediaObjects/41593_2007_BFnn1825_MOESM12_ESM.pdf)

Two-page publisher PDF. Caption confirms hippocampal gaps 90, 70, 90, 80 ms for animals 1–4.

### csicsvari-1999

[Csicsvari 1999a](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782375/)

Detection Methods specify 150–250 Hz, RMS per electrode summed across electrodes and a 7-SD threshold. Confirms Nadasdy's inherited multi-electrode aggregation; RMS window remains unspecified.

### mallory-supplement

Mallory published supplement, `science.ads4760_sm.pdf` (user-supplied local copy)

37-page Science supplement, DOI 10.1126/science.ads4760; Methods pp. 5–8 and Fig. S2 pp. 16–17 checked directly. Confirms detection/decoding values and Spearman p<0.05 control.

### mallory-preprint

[Mallory preprint v1](https://www.biorxiv.org/content/10.1101/2024.07.18.604185v1.full.pdf)

Retrieved 44 pages. Methods pp. 21–23 distinguish protocols; positional errors 2.9/3.4 cm in the main text, 400 ms behavioral windows. Code/preprint discrepancies remain documented; this is not the published supplement.

### olafsdottir-software

[Olafsdottir 2016 supplementary software](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnn.4291/MediaObjects/41593_2016_BFnn4291_MOESM24_ESM.zip)

Four MATLAB posterior/line-fitting functions, including decode_calcPosterior and lineTraj_decode. Inputs include temporal/spatial bin size; examples/comments are not paper-specific caller settings. No detector or shuffle driver supplied.

### yang-archives

[Yang analysis archive](https://zenodo.org/records/10685490) and [buzcode archive](https://zenodo.org/records/10685428)

Metadata, corresponding tagged GitHub trees and selected files inspected. Analysis concept resolves to version 2 / record 10816749. Stored HSE metadata does not specify the independent LFP detector.

### code-archives

[Huelin Gorriz 10085294](https://zenodo.org/records/10085294), [Berners-Lee 6330850](https://zenodo.org/records/6330850), [Mou 5758889](https://zenodo.org/records/5758889), [Gillespie 5140706](https://zenodo.org/records/5140706), [neurocode 7819979](https://zenodo.org/records/7819979)

Release metadata and complete small code archives were downloaded; inspected paths and missing-helper conclusions are listed above.

### widloski-archives

[Widloski 2022 code](https://zenodo.org/records/5880582), [2025 code](https://zenodo.org/records/15199609), [2025 data](https://zenodo.org/records/16916108)

Inspected MATLAB source files and complete file inventories. Decoder/event criteria and the 2025 100 cell-ID shuffles match the published Methods. Inventory confirms required external helpers are not bundled. Data deposit metadata inspected; large raw/data arrays were not exhaustively inspected and are not used to infer detector settings.

### michon-osf

[Michon OSF smzby](https://osf.io/smzby/)

API inventory and behavior/place-code notebooks inspected. The code folder is empty in the API listing; notebooks import missing code.options and other modules. No event-preprocessing caller or dependency lock found.

### michon-toolbox

[Original Kloosterman toolbox v1.2](https://bitbucket.org/kloostermannerflab/fklab-python-core/src/f4176cd36bc73a0266fdae0df6dca60b14fee0e2/)

Original compute_envelope and core routines retrieved. This version has no median detrending in compute_envelope; the inspected fork detrends before smoothing. Paper workflow states smoothing then detrending. Original caller/version is unresolved.

### barry-data

[Barry raw data 5566548](https://zenodo.org/records/5566548)

Metadata identifies Olafsdottir 2016, not a new Bush-specific detector release. All 1,310 ZIP-directory entries read via bounded HTTP ranges. No separately named SWR/replay/candidate event files found. This does not inspect every variable inside every MAT file.

### bhattarai-supplement

Bhattarai published SI, `pnas.1912533117.sapp.pdf` (user-supplied local copy)

Title, authors and DOI match. Read Methods pp. 2–7 and inspected Table S1 image on p. 21. Replay 20 ms/10 ms step; behavioral windows 50 ms; spatial SD 6 cm; 1000 circular-position shuffles, p<0.05; error 6.56 cm and 1342/2573 replays confirmed.

### bhattarai-data

[Bhattarai Figshare v2](https://doi.org/10.6084/m9.figshare.10032866.v2)

Full ZIP directory plus ReadMe/MetaData.docx: beh.mat, events.mat, lfp.mat, spikes.mat, VT.mat and MetaData.docx. Metadata defines events.mat as trial/delay/reward timing and LFP as one tetrode trace. No deposited detector code or documented SWR/replay event table.

### grosmark-data

[CRCNShc-11](https://crcns.org/data-sets/hc/hc-11/about-hc-11) and [DANDI 000044 v0.250624.0426](https://dandiarchive.org/dandiset/000044/0.250624.0426)

CRCNS description and DANDI asset catalog inspected. Inspected HDF5 metadata via small byte ranges in all eight NWB files: intervals contains only epochs; analysis is empty; processing contains behavior/ecephys. No SWR/replay interval table found. Did not download all raw arrays or reproduce original SWRs.

### jadhav-data-analysis

[Additional Jadhav-data analysis](https://github.com/edeno/Jadhav-2016-Data-Analysis/tree/0b5d4d584dacce75fed9a6ad1c21658773cf2ca1)

README explicitly identifies the Jadhav dataset. Parameters and ripple-detection code inspected. This is an analysis of those data, not evidence that its settings generated the original paper's events.

### ambrose-supplement

[Ambrose 2016 supplement](https://ars.els-cdn.com/content/image/1-s2.0-S0896627316304639-mmc1.pdf)

Full 12-page publisher PDF; Table S1 visually inspected and counts summed. No published detector-duration limits; related later code cannot supply original settings.

### oneill-2008

[O’Neill 2008 article and supplement](https://www.mrcbndu.ox.ac.uk/sites/default/files/pdfs/oneill2008natureneurosci.pdf)

14-page Oxford-hosted copy, especially supplementary field/state Methods on p. 13. Muessig’s peak-centered spectral bandwidth remains unspecified.

### jackson-2006

[Jackson 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674885/)

Primary LFP Methods read directly. Supports the inherited-method discussion in the Wikenheiser, Bhattarai and Gupta notes. Downstream papers’ unstated implementation details are not supplied by this citation.

## Other cited locations

These links are retained from the paper notes for provenance. Their role and inspection limits are given in those notes; an unpinned landing page does not establish a historical implementation.

| Location | Paper notes |
|---|---|
| http://bitbucket.org/kloostermannerflab | [15_Michon_2021](papers/15_Michon_2021.md) |
| http://fmatoolbox.sourceforge.net | [30_Drieu_2018](papers/30_Drieu_2018.md) |
| http://github.com/vandermeerlab/papers | [28_Carey_2019](papers/28_Carey_2019.md) |
| http://www.bitbucket.org/kloostermannerflab | [25_Michon_2019](papers/25_Michon_2019.md) |
| http://zugarolab.net/wp-content/uploads/Drieu2018.pdf | [30_Drieu_2018](papers/30_Drieu_2018.md) |
| https://danbush.co.uk/CurrentBiology2021.pdf | [07_Bush_2022](papers/07_Bush_2022.md) |
| https://bitbucket.org/franklab/trodes2ff_shared/commits/e90fd3d | [14_Gillespie_2021](papers/14_Gillespie_2021.md) |
| https://crcns.org/files/data/hc-11/crcns_hc-11_data_description.pdf | [36_Grosmark_2016](papers/36_Grosmark_2016.md) |
| https://doi.org/10.1038/s41467-026-72252-8 | [01_Widloski_2025](papers/01_Widloski_2025.md) |
| https://doi.org/10.5281/zenodo.10085294 | [03_HuelinGorriz_2023](papers/03_HuelinGorriz_2023.md) |
| https://doi.org/10.5281/zenodo.10685428 | [02_Yang_2024](papers/02_Yang_2024.md) |
| https://doi.org/10.5281/zenodo.10685490 | [02_Yang_2024](papers/02_Yang_2024.md) |
| https://doi.org/10.5281/zenodo.14237298 | [00_Mallory_2025](papers/00_Mallory_2025.md) |
| https://doi.org/10.5281/zenodo.5566548 | [07_Bush_2022](papers/07_Bush_2022.md) |
| https://doi.org/10.5281/zenodo.7819979 | [05_Liu_2023](papers/05_Liu_2023.md) |
| https://dx.doi.org/10.17632/4xk5w69yr5.1 | [16_Igata_2021](papers/16_Igata_2021.md) |
| https://github.com/DaoyunJiLab/DM2021.git | [11_Mou_2022](papers/11_Mou_2022.md) |
| https://github.com/DrugowitschLab/HippocampalSWRDynamics/tree/v1.0 | [10_Krause_2022](papers/10_Krause_2022.md) |
| https://github.com/Eden-Kramer-Lab/ripple_detection | [13_Denovellis_2021](papers/13_Denovellis_2021.md) |
| https://github.com/JadhavLab/PrefrontalRipples/tree/72f0f3e | [27_Shin_2019](papers/27_Shin_2019.md) |
| https://github.com/ryanharvey1/ripple_heterogeneity/tree/7b368a9 | [04_Harvey_2023](papers/04_Harvey_2023.md) |
| https://github.com/zivadinac/jcl/tree/5e9fe5f | [18_Kaefer_2020](papers/18_Kaefer_2020.md), [20_Stella_2019](papers/20_Stella_2019.md) |
| https://github.com/zugarolab/FMAToolbox/tree/978749c | [30_Drieu_2018](papers/30_Drieu_2018.md) |
| https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7817193/supplementaryFiles | [16_Igata_2021](papers/16_Igata_2021.md) |
| https://www.jneurosci.org/content/jneuro/19/1/274.full.pdf | [55_Nadasdy_1999](papers/55_Nadasdy_1999.md) |
| https://www.nature.com/articles/s41467-025-65181-5 | [01_Widloski_2025](papers/01_Widloski_2025.md) |
| https://www.nature.com/articles/s41467-026-72252-8 | [01_Widloski_2025](papers/01_Widloski_2025.md) |
| https://zenodo.org/record/5140706 | [14_Gillespie_2021](papers/14_Gillespie_2021.md) |
| https://zenodo.org/record/5566548 | [07_Bush_2022](papers/07_Bush_2022.md) |

## Artifact fingerprints

SHA-256 and byte counts identify the inspected copies. `*_listing.txt` and `dandi_internal_check.json` are inspection records, not hashes of the complete remote datasets. The two user-supplied supplements are identified by filename, DOI and fingerprint, without machine-specific paths.

| Artifact | Source | SHA-256 | Bytes |
|---|---|---|---:|
| `ji_fig12.pdf` | [ji-figure-12](#ji-figure-12) | `f65b749ec03c33b1c07078ad9d1b7c0ce53deb4be12dadbbaf141ba55fdfb759` | 70874 |
| `olaf2016_software.zip` | [olafsdottir-software](#olafsdottir-software) | `b7f22464c3381c7932db2e268d652c3ece59dc1b9611d1549fe02f0b49344c37` | 8784 |
| `mallory_biorxiv.pdf` | [mallory-preprint](#mallory-preprint) | `fee6af4c179b3e8a37e9e3ac1459c3c0db313e828e6352334b54134528e6c777` | 4511799 |
| `carey_candidates/R050-2014-03-29-candidates.mat` | [carey-data](#carey-data) | `77a209e6bfe8dd857e2ddde41f755d7076f396bb5ac7eaad35c3ca30650cab98` | 38120231 |
| `maboudi/data/fig1.nel` | [maboudi-code](#maboudi-code) | `cec48bd1a313f06f2a0612d0c3d4a4ee637e1ba0ee355e000ac4ba5cd7de5d3a` | 56105728 |
| `barry_zip_listing.txt` | [barry-data](#barry-data) | `9e2ebfaa95684370d79da216375a6be06cfdc829e5d338238b7bbb34176d2ebe` | 59125 |
| `figshare_zip_listing.txt` | [bhattarai-data](#bhattarai-data) | `21c72b0e4cd3f30bab363905b955bdbc9f29ac98fc5d7cd13ffc7787defc5a77` | 58 |
| `dandi_internal_check.json` | [grosmark-data](#grosmark-data) | `df232ee64bf4e2f6741fecbc9bb939e574d7d1e95317d0a7757006b4ef4e9aa6` | 2635 |
| `science.ads4760_sm.pdf` | [mallory-supplement](#mallory-supplement) | `11de2fe56bfcd149e24dd81d2b02c3d29156c70f3f46c26bb1cbe1050f2bea59` | 6996578 |
| `mallory_128513a/compute_allSequences_NaNseparated_merge.m` | [mallory-code](#mallory-code) | `5cd12bdc244539914a00cf16c5b728c51b56154cad433d628691f8b34a6fbb23` | 1467 |
| `mallory_128513a/compute_filtering_binDecoding_cm.m` | [mallory-code](#mallory-code) | `335e28f76c1353533f5eded9e77579dca508c7c32d2bd616f8eb5e71b759ced5` | 3612 |
| `mallory_128513a/spearman_median.m` | [mallory-code](#mallory-code) | `cf2138cd59d61a2c158e1330214bf9125351406a6c7545106ff36e4fc832c935` | 771 |
| `pnas.1912533117.sapp.pdf` | [bhattarai-supplement](#bhattarai-supplement) | `201e03d62b4dc63126c24bd2257217e9900dc47d098f359c11799bce0285ff6b` | 3405508 |
| `oneill2008_combined.pdf` | [oneill-2008](#oneill-2008) | `90e9f3486607e034abf59b092fb8870d13e2920b99900f45f10d6e1b3c8e2ee6` | 4629607 |
| `ambrose2016_publisher_supplement.pdf` | [ambrose-supplement](#ambrose-supplement) | `6eb04d129fe724456322db4c3944faa0e1727da7fea487ce292389aada0ccdff` | 1158744 |
| `jackson2006.html` | [jackson-2006](#jackson-2006) | `bc453fc2789b7bbbd0e45badbf6c745b66deaa6ce50e8d64a4cec8f1b2449838` | 250423 |
| `csicsvari1999a.html` | [csicsvari-1999](#csicsvari-1999) | `9a91d9616ba16c0806dfb763106b1bee7182670e5bd1ccb574e2dcfc8145fd8e` | 231069 |
| `krause_cda23b7/replay_structure/utils.py` | [krause-code](#krause-code) | `690712cd995404ad386e77b085f3720e37547ae4cdb995871140ce57798d9536` | 13946 |
| `krause_cda23b7/replay_structure/ripple_preprocessing.py` | [krause-code](#krause-code) | `8529b4378c98f5bf4f87c0dad095570af5e1d915cbdd99187b0ac5cdb1d0007a` | 7541 |

## Code-availability search scope

Searches through September 25, 2026 covered GitHub repositories/code and author/lab accounts, Zenodo, Figshare, Bitbucket, CRCNS and DANDI, together with the papers’ availability statements. Author/year/replay or exact-title/code searches covered Berners-Lee 2021, Kaefer, Stella, Xu, both Farooq papers, Liu 2019, Muessig, Yamamoto, Tang, Jadhav, the three Ólafsdóttir papers, Silva, Pfeiffer 2015, Wu 2014, Wikenheiser, Bendor, Gupta, Davidson, Diba, Foster, Lee, Nádasdy and Kudrimoti. The supplementary Ólafsdóttir software and secondary Jadhav-data analysis are cataloged above. These searches did not establish further original detector callers. This is a bounded search result, not proof that no public code exists. Berners-Lee 2021, Kaefer and Yamamoto offer data/code on request. No author was contacted.
