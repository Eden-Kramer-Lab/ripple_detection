# Literature parameter audit — September 25, 2026

The shipped survey has been corrected in 236 fields across 49 of its 57 rows.
This count includes contextual notes and previously omitted control-detector values;
it is not a count of 236 incorrect numerical thresholds.
The [field-by-field correction ledger](parameter_corrections_2026-09-25.csv) gives
old and new values, the reason, DOI and a link to the detailed paper note.
The README statistics were recomputed from the corrected CSV.

The [independent source recheck](source_recheck.md) records the completed follow-up:
original GitHub/Zenodo releases, OSF, supporting lab code, archived event metadata,
and decoding/significance checks. The [field-status index](parameter_verification_2026-09-25.csv)
covers every CSV cell. The [remaining questions](verification_remaining.md) now contain
only unresolved sources, unpublished execution details and interpretation limits.
All 138 first-pass correction cells are accounted for: 131 unchanged, seven deliberately
refined. The follow-up changed 105 cells in 32 papers; notes and qualifications are
included in those counts. See the [follow-up history](parameter_recheck_changes_2026-09-25.csv).
The Bhattarai SI check reduced the cumulative count from 233 to 232 by restoring
two original values and adding one trajectory clarification. The later Ambrose
supplement check adds four changes, bringing the current total to 236. These are
deliberate refinements, not missing CSV edits.

The local Dropbox/Zotero collection supplied PDFs for 55 papers. Sources were matched
using DOI metadata and titles; linked files in `Dropbox/Papers` and attachments in
`Zotero/storage` were read without changing the library. The published Widloski 2025
HTML and an author-hosted Bush PDF supplied the other two papers. Full supplements
were checked where available, including the newly retrieved Igata SI Appendix and
Drieu author copy. The user-supplied Mallory Science supplement (37 pages) and
Bhattarai SI Appendix (22 pages) were subsequently read. Mallory was reconciled
with the pinned release; Bhattarai replay bins are restored to 20 ms with 10 ms steps.
Existing paper notes were also reviewed. This is a source audit of detection and decoding values, including shuffle counts, significance tests,
window sizes and reported reconstruction/replay fractions. It does not certify
that every protocol variant or analysis has been computationally replicated.
The source gaps below remain explicit.

## Material corrections

| Paper | Correction | Source location |
|---|---|---|
| Carey 2019 | Candidate minimum is 20 ms and 5 active units; 50 ms / 4 units describe decoded sequences. | Methods, SWR event detection, PDF p. 12; paper note also records the released candidate files. |
| Yang 2024; Grosmark 2016 | 3 SD and 15 ms describe population synchrony. The independent LFP detector's threshold/band/channel count cannot be inferred from them. | Yang supplement p. 11; Grosmark supplement, combined PDF p. 9. |
| Grosmark 2016 | Initial synchrony lasts 50–500 ms; replay analysis requires at least 100 ms. | Supplement, Bayesian Replay Analysis. |
| Harvey 2023 | Released replay code requires 80 ms; its main DetectSWR path uses a 2.5 local-SD peak threshold. The paper's 400 ms cap concerns the sharp wave. | `replay_run.py` at 8461a94; `update_session_data.m` at 7b368a9; neurocode `DetectSWR.m` at 4b33b2a. Variant sessions retain separate values in notes. |
| Tirole 2022; Huelin Gorriz 2023 | Released code uses a 10 ms Gaussian per pass, a 15 ms ripple moving average; Tirole additionally has no maximum-duration exclusion. | Release 44ecf42: `process_clusters.m`, `list_of_parameters.m`, `extract_CSC.m`, `extract_replay_events.m`. Text says 5 ms, 100 ms and 750 ms. Huelin Gorriz has the same smoothing code, but its event extractor is missing, so its published 750 ms cap is retained. |
| Krause 2022 | Secondary HSE smoothing is 10 ms SD in released code; text states 20 ms. | Release v1.0, `replay_structure/highsynchronyevents.py`, lines 48–50. |
| Maboudi 2018 | Restore linear-track MUA peak 3 SD, Gaussian SD 20 ms, 4 pyramidal cells and 80 ms minimum. | Methods, Population burst events, PDF p. 16. Open-field variant is identified separately. |
| Yamamoto 2017 | Ripple detection band is 140–200 Hz. Retain 70 ms for grouping ripple peaks; it is not an inter-event boundary gap. | Methods, Candidate Ripple and Replay Event Detection / Ripple Burst Analysis, PDF p. 15. |
| Ambrose 2016 | 50–500 ms limits remain `Not reported` after the full supplement check. Record 4–7 sampled tetrodes. Replace rough replay fraction with 1147/5948 = 19.3% pooled, separating 22.3% / 15.5% by experiment. | Main Methods p. 12; publisher supplement Table S1 p. 9, Tables S2–S3 pp. 10–11. Later lab code does not settle original duration settings. |
| Igata 2021 | Distinguish sequential events (absolute weighted correlation ≥0.5) from trajectory events (shuffle p<0.05). | SI Appendix pp. 10–11. Detection thresholds, smoothing widths and decoding bins were confirmed in pp. 6–8. |
| Bhattarai 2020 | Restore 20 ms replay bins; add explicit 10 ms steps. Verify 1000 circular-position shuffles, p<0.05, Gaussian spatial SD 6 cm and error 6.56 cm. Distinguish R² significance from weighted-correlation direction/strength. | Published SI pp. 3–6; Table S1 p. 21 confirms 1342/2573 replays and mean error. The earlier audit mistakenly assigned 50 ms behavioral bins to replay. |
| Mallory 2025 | Confirm spike-density and secondary ripple peak 3 SD / Gaussian SD 12.5 ms; distinguish track/arena settings and add the Spearman p<0.05 control. Preserve arena SD 8 cm in text versus nominal 4 cm in code. Replay code overrides the 100 ms default with 50 ms. | Published supplement pp. 5–8, Fig. S2 pp. 16–17; release v1.0.0 candidate, replay-extraction/combination and spatial-filter helpers. See the paper note for remaining cell-selection and spatial-criterion differences. |

Place-field construction and unrelated behavioral cutoffs were removed from the event-speed
column (Harvey, Liu 2023, Bhattarai, Drieu, Wu 2017, Yamamoto and Ólafsdóttir 2015).
Liu 2019's main sleep-state cutoff is 1 cm/s; its awake variant is 2 cm/s.
Decoded-content minima in Stella and Wu 2014 were moved out of detector-duration
summaries and retained in notes. Diba's 300 ms candidate window and Ólafsdóttir
2015's 300 ms limit were restored. Foster's participation fraction is exactly one-third,
not 33%.

Secondary detector parameters remain in numeric columns, following the repository's
recorded convention. Missing values were restored for Widloski 2025, Denovellis,
Kaefer, Farooq 2019 Science, Chenani, Drieu, Ólafsdóttir 2017, Wu 2014, Diba and Ji.
For Widloski 2025, the original 100 cell-ID shuffles and p<0.05 replay criterion are
correct: these are distinct from the separate 100-snippet tests for ripple/burst absence.
The April 2026 author correction changes references, not those parameters.

## Interpretation rules

- A numeric entry can describe a primary detector, an inclusion criterion or a
  secondary analysis. `Detection Notes` identifies variants and ordering. A row is
  not necessarily one executable detector configuration.
- A paper's own released implementation sets a value when the relevant execution
  path is clear; disagreements with text are preserved. Later or merely related
  lab code does not establish the paper's value.
- `#N/A` represents no applicable criterion in the summarized method. `Not reported`
  denotes a method parameter for which the consulted primary description gives no
  number. A source that could not be rechecked is labeled as such, not silently
  promoted to a verified number.
- Gaussian SD, moving-average width, RMS window and spectral window are different
  quantities. Tirole's 41-point Gaussian has nominal SD 10 ms per pass; after forward/backward filtering its finite-kernel SD is about 12.58 ms (the untruncated approximation would be 14.14 ms). These widths are not interchangeable
  `smoothing_sigma` arguments. Stella's spatial basis width is also not a Gaussian SD.
- Event grouping can use a gap between pooled spikes (Foster/Lee), a gap between
  event boundaries or a peak-to-peak interval (Yamamoto/Mallory). The README groups
  these descriptively and does not claim they are the same operation.
- Electrode counts describe channels sampled/combined, not the minimum number that
  must cross threshold. In several studies any one of multiple tetrodes can trigger.
- Fractions, ranges, strict inequalities and non-z-scored thresholds remain text.
  The loader converts these to missing in numeric columns. For example, Ji's ripple
  peak is 7 times voltage SD without mean subtraction, not a 7-SD z-score.

## Source limitations after the independent recheck

Ji Supplementary Figure 12 was recovered from the publisher and confirms the
70–90 ms hippocampal frame gaps. The cited Csicsvari 1999a Methods were reopened
and confirm multi-electrode summed RMS for Nadasdy. Those two retrieval gaps are closed.

The Mallory published supplement retrieval gap is closed. Its Methods confirm
the text/code spatial-smoothing difference; the replay caller resolves the earlier
50/100 ms duration concern by overriding the default to 50 ms. Spatial merging,
COM-jump and cell-selection differences remain documented. Bhattarai's SI retrieval
gap is also closed: 20 ms replay bins, 10 ms steps, 1000 circular-position shuffles,
p<0.05, 6 cm spatial SD and 6.56 cm reconstruction error are directly verified.
Its 50 ms windows describe behavioral reconstruction. Original detector callers
and dependency versions are still missing for several papers; a related toolbox default is not a substitute. The detailed, bounded
[remaining-source register](verification_remaining.md) lists these questions.

The older source-access list is now also closed: Ambrose’s complete supplement,
O’Neill’s article with supplementary Methods, and direct Jackson/Csicsvari Methods
are checked. Unreported details remain unreported. See the final source-closure
entries in [source recheck](source_recheck.md).

Additional material changes include 1000 shuffles and the released smoothing widths
for Huelin Gorriz; 15 ms MUA SD for Denovellis; 5000 shuffles for Drieu and Maboudi;
10 ms bins and 100 field-rotation shuffles for Olafsdottir 2016; adaptive rather than
fixed windows for Kaefer and Gupta; and separate per-event versus aggregate-count
significance for Berners-Lee, Silva and Farooq. Kaefer's old approximate 45% measured
SWR overlap among replays, not replay among candidates. Tang's replay fractions now
retain separate awake/post-sleep/pre-sleep denominators. Figure estimates and
inferences are labeled in the field-status index.

These limits mean the survey should not be described as having every number fully
verified. Older discrepancy lists remain historical; each changed paper note now
has a dated independent-recheck disposition. Recipe/tier classifications are a
separate review and were not revalidated by this data correction.

## Primary online sources used

- [Widloski 2025 published Methods](https://www.nature.com/articles/s41467-025-65181-5)
  and [author correction](https://www.nature.com/articles/s41467-026-72252-8).
- [Bush author PDF](https://danbush.co.uk/CurrentBiology2021.pdf).
- [Drieu author PDF including supplementary materials](http://zugarolab.net/wp-content/uploads/Drieu2018.pdf).
- [Igata SI archive from Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7817193/supplementaryFiles).
- [Mallory paper release v1.0.0](https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay/tree/v1.0.0).
- [Tirole paper release 44ecf42](https://github.com/bendor-lab/Elife_Tirole_Huelin_Gorriz_2022/tree/44ecf42).
- [Krause paper release v1.0](https://github.com/DrugowitschLab/HippocampalSWRDynamics/tree/v1.0).
- [Harvey replay code 8461a94](https://github.com/ryanharvey1/ripple_heterogeneity/tree/8461a94),
  [session processing 7b368a9](https://github.com/ryanharvey1/ripple_heterogeneity/tree/7b368a9)
  and [neurocode 4b33b2a](https://github.com/ayalab1/neurocode/tree/4b33b2a).
- [Carey code pinned by the paper](https://github.com/vandermeerlab/vandermeerlab/tree/ad0bbd4d01726a436b36671c0a8b2db81476e946).

## Local source inventory

The entries below identify source availability, not blanket verification of every
field. Supplementary source gaps are listed above. Zotero keys identify attachment
records; no library files were changed or copied into the repository.

| Row | Paper note | Local PDF | Zotero attachment key(s) |
|---|---|---|---|
| 00 | [Mallory 2025](papers/00_Mallory_2025.md) | Main PDF and user-supplied published supplement available | V86BPFP4 (main text) |
| 01 | [Widloski 2025](papers/01_Widloski_2025.md) | Online source used | — |
| 02 | [Yang 2024](papers/02_Yang_2024.md) | Available | SC2CCZWK |
| 03 | [Huelin Gorriz 2023](papers/03_HuelinGorriz_2023.md) | Available | 7S7FSHHU, KCARB3RB |
| 04 | [Harvey 2023](papers/04_Harvey_2023.md) | Available | CMD4SGMX |
| 05 | [Liu 2023](papers/05_Liu_2023.md) | Available | 2WRJMAFL, QMPVC6DS |
| 06 | [Tirole 2022](papers/06_Tirole_2022.md) | Available | BW865B4T |
| 07 | [Bush 2022](papers/07_Bush_2022.md) | Online source used | — |
| 08 | [Berners-Lee 2022](papers/08_Berners-Lee_2022.md) | Available | MMFRCI7Q |
| 09 | [Widloski 2022](papers/09_Widloski_2022.md) | Available | NIPY8PH6 |
| 10 | [Krause 2022](papers/10_Krause_2022.md) | Available | 28E982LI |
| 11 | [Mou 2022](papers/11_Mou_2022.md) | Available | VXB82JGQ |
| 12 | [Berners-Lee 2021](papers/12_Berners-Lee_2021.md) | Available | WXL9QE3F |
| 13 | [Denovellis 2021](papers/13_Denovellis_2021.md) | Available | 7NN7AHHF |
| 14 | [Gillespie 2021](papers/14_Gillespie_2021.md) | Available | LBWA2KLI |
| 15 | [Michon 2021](papers/15_Michon_2021.md) | Available | GMQYGZG2 |
| 16 | [Igata 2021](papers/16_Igata_2021.md) | Available | 5QZMXC3W |
| 17 | [Gridchyn 2020](papers/17_Gridchyn_2020.md) | Available | 9VCS8YBA |
| 18 | [Kaefer 2020](papers/18_Kaefer_2020.md) | Available | 87B4REBU |
| 19 | [Bhattarai 2020](papers/19_Bhattarai_2020.md) | Main PDF and user-supplied published SI available | 9QPGUMM5, N5MKVAV3 |
| 20 | [Stella 2019](papers/20_Stella_2019.md) | Available | BBTNJ3AF |
| 21 | [Xu 2019](papers/21_Xu_2019.md) | Available | G3A9AM6N |
| 22 | [Farooq 2019](papers/22_Farooq_2019.md) | Available | U3SBHG2J |
| 23 | [Farooq 2019](papers/23_Farooq_2019.md) | Available | 3UD3JJCA |
| 24 | [Chenani 2019](papers/24_Chenani_2019.md) | Available | G2V5QTMR, YYS346JZ |
| 25 | [Michon 2019](papers/25_Michon_2019.md) | Available | JUJIHZMR |
| 26 | [Liu 2019](papers/26_Liu_2019.md) | Available | 9FQJPCH7 |
| 27 | [Shin 2019](papers/27_Shin_2019.md) | Available | 9GV7U4TC |
| 28 | [Carey 2019](papers/28_Carey_2019.md) | Available | ECEQYURX, WAMGX3KK |
| 29 | [Muessig 2019](papers/29_Muessig_2019.md) | Available | MJAC9B9W |
| 30 | [Drieu 2018](papers/30_Drieu_2018.md) | Available | SFACWY73 |
| 31 | [Maboudi 2018](papers/31_Maboudi_2018.md) | Available | NHPXIAM9 |
| 32 | [Ólafsdóttir 2017](papers/32_Olafsdottir_2017.md) | Available | TKX9ULZ4, V4VUR67B |
| 33 | [Wu 2017](papers/33_Wu_2017.md) | Available | IA628G3V |
| 34 | [Yamamoto 2017](papers/34_Yamamoto_2017.md) | Available | 5QLGWIZD, EK36V925, HVZKUJRJ |
| 35 | [Tang 2017](papers/35_Tang_2017.md) | Available | 642KY564, LSE7H476 |
| 36 | [Grosmark 2016](papers/36_Grosmark_2016.md) | Available | RM2HHBZT |
| 37 | [Ambrose 2016](papers/37_Ambrose_2016.md) | Main PDF and publisher supplement available | T9AN39CQ (main text) |
| 38 | [Jadhav 2016](papers/38_Jadhav_2016.md) | Available | B77CW7E2 |
| 39 | [Ólafsdóttir 2016](papers/39_Olafsdottir_2016.md) | Available | 38FRHB97, A4KRQ99F, SG6JJVMN |
| 40 | [Silva 2015](papers/40_Silva_2015.md) | Available | 8EK57QXA |
| 41 | [Ólafsdóttir 2015](papers/41_Olafsdottir_2015.md) | Available | — |
| 42 | [Pfeiffer 2015](papers/42_Pfeiffer_2015.md) | Available | RAHN2BCH |
| 43 | [Wu 2014](papers/43_Wu_2014.md) | Available | WGWVTVNP |
| 44 | [Wikenheiser 2013](papers/44_Wikenheiser_2013.md) | Available | 6NUAGTZ2 |
| 45 | [Pfeiffer 2013](papers/45_Pfeiffer_2013.md) | Available | IUEWAHWC |
| 46 | [Carr 2012](papers/46_Carr_2012.md) | Available | 8GWMPU85 |
| 47 | [Bendor 2012](papers/47_Bendor_2012.md) | Available | NM7KGUSU |
| 48 | [Gupta 2010](papers/48_Gupta_2010.md) | Available | BQ9PCUUD |
| 49 | [Karlsson 2009](papers/49_Karlsson_2009.md) | Available | N73SCPVZ |
| 50 | [Davidson 2009](papers/50_Davidson_2009.md) | Available | 8ZVHGWPK |
| 51 | [Diba 2007](papers/51_Diba_2007.md) | Available | USGNIGSH |
| 52 | [Ji 2007](papers/52_Ji_2007.md) | Available | UW4N62MY |
| 53 | [Foster 2006](papers/53_Foster_2006.md) | Available | 3QFH59DJ |
| 54 | [Lee 2002](papers/54_Lee_2002.md) | Available | 9KUEQMRT |
| 55 | [Nádasdy 1999](papers/55_Nadasdy_1999.md) | Available | XVY2LJTB |
| 56 | [Kudrimoti 1999](papers/56_Kudrimoti_1999.md) | Available | GT3YRQ3L |
