# Bhattarai 2020 — Distinct effects of reward and navigation history on hippocampal forward and reverse replays

Sources: published main text from the matched Dropbox/Zotero PDF; published SI
Appendix `pnas.1912533117.sapp.pdf`, supplied locally by the user and independently
read on September 25, 2026. The 22-page supplement cover gives the exact title,
authors (Bhattarai, Lee and Jung) and DOI 10.1073/pnas.1912533117. Methods pp. 2–7
were read, and the image of Table S1 on p. 21 was inspected directly. This closes
the earlier SI retrieval gap and supersedes the earlier unverified transcription.

Supplement size: 3,405,508 bytes. SHA-256:
`201e03d62b4dc63126c24bd2257217e9900dc47d098f359c11799bce0285ff6b`.

Trigger: SWR+MUA (silence-bounded place-cell bursts that must coincide with a detected SWR; the SWRs themselves require ≥ 5 place cells)

## Method as implemented

SWR detection (SI Appendix, Materials and Methods, "SWR events", SI p. 4):
- "LFP signals obtained from one channel each from two tetrodes carrying well-isolated unit signals were filtered between 100 and 250 Hz. Instantaneous LFP power was calculated at each time point (sampling frequency, 2000 Hz), smoothed with a boxcar filter (width, 50 ms) and averaged." How "instantaneous power" was computed (squared Hilbert amplitude, squared signal) is not stated; the averaging is across the two channels (inference from the sentence). 15 tetrodes in CA1 on one hemisphere per animal (SI p. 3).
- Threshold and bounds: "Points of LFP power exceeding three SD above the mean were identified as candidate SWR events. The beginning and end of each candidate SWR event were determined at one SD above the mean." The period for mean and SD is not stated (inference: whole session).
- Duration and merging: "Candidate SWR events with ≤ 20 ms durations were excluded and two events with a ≤ 100 ms inter-ripple interval were merged (2)." Ref. 2 = Jackson, Johnson & Redish 2006. Whether the interval is end-to-start or peak-to-peak is not stated.
- Cells: "to further eliminate false positive events, we included those candidate SWR events coincident with spikes of at least five different place cells (from any block of a given session) in the analysis (3)." Ref. 3 = Grosmark & Buzsáki 2016.
- Maximum duration: not stated. Speed: not stated for SWR detection. Theta: a theta-power measure exists ("filtered between 4 and 10 Hz. Instantaneous theta power was calculated at each time point and z-scored", SI p. 4) but is not used as a detection criterion in the text I found.

Replay candidates (SI "Replay events", SI p. 4; main text p. 692):
- "A candidate replay event consisted of five or more place cells from a given block (determined using forced-choice and correct free-choice trials) activated together within a 300-ms window with at least one coincident SWR event and preceded by a > 60 ms period of silence (4)." Ref. 4 = Diba & Buzsáki 2007. Main text: "We identified a candidate replay event as a set of spikes emitted by at least 5 different place cells from a given block preceded by a >60-ms silent period and with at least one coincident SWR".
- Whose silence (block place cells, all place cells, all units) is not stated. How the event end is set is not stated: the reported "mean (±SD) duration was 151.9 ± 61.5 ms" (main text) implies the event is not the full 300 ms window (inference: first to last spike within it).
- Speed: none stated for detection. Tuning curves exclude "periods of immobility (speed < 4 cm/s)" (SI "Bayesian decoding"), an encoding-model criterion, not detection. The main text notes "The majority of SWRs were observed in the reward zone" (p. 692).
- 2,573 candidates for the current trajectory; significance by circular-shift shuffle of decoded positions ("P < 0.05 (n = 1342; 52.2% of candidate replay events"), analysis.

Later analysis restrictions: replay significance uses 1000 circular time shifts of decoded positions and linear-regression R² (p<0.05). Weighted correlation measures replay strength/direction. Replay bins are 20 ms with 10 ms steps; 50 ms bins are for behavioral reconstruction. Session sets with median absolute reconstruction error >20 cm are excluded (SI pp. 5–6).

## Inherited from
- Diba & Buzsáki 2007 (ref. 4; findings/51_Diba_2007.md, Supplementary Methods): "When the animal's speed was ≤ 10 cm s–1 ... pre-play/replay events were detected by searching for silent periods ≥ 60 ms. If ≥ 5 or ≥ 30 percent of place-cells (whichever was greater) from a template fired within the next 300 ms, an event was recorded." Bhattarai keeps the 60 ms silence, 300 ms window and 5 cells, drops the 30% alternative and does not state Diba's speed ≤ 10 cm/s, and adds the coincident-SWR requirement.
- Jackson et al. 2006 (ref. 2, for the ≤ 100 ms merge; [primary Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674885/), directly reopened September 25, 2026): "Threshold crossings <20 ms were removed; the remaining events were concatenated if <100 ms apart." Bhattarai uses the same numbers with ≤.
- Grosmark & Buzsáki 2016 (ref. 3, for the ≥ 5 cells; findings/36_Grosmark_2016.md): events "in which at least five distinct pyramidal cells each fired at least one spike".

## Code
No code link. Data: "The raw data is deposited in Figshare with identifier doi.org/10.6084/m9.figshare.10032866.v2" (main text).

### Code search, September 2026

- **Own data.** [Figshare 10.6084/m9.figshare.10032866.v2](https://doi.org/10.6084/m9.figshare.10032866.v2), DataSet.zip: behaviour, events (trial timing only, per MetaData.docx), one tetrode's LFP, spikes and video tracking. No deposited detector code or documented SWR/replay event table was established from the archive inventory and metadata. The earlier public-code searches did not establish an original detector repository; this is a bounded search result.

## Historical survey CSV discrepancies

These refer to the original survey; the current disposition is recorded below.
- Animal Speed (cm/s): CSV 4; the paper states no speed criterion for SWR or replay detection. The 4 cm/s is the immobility exclusion for tuning curves ("excluding periods of immobility (speed < 4 cm/s)").
- Detection "SWR": the replay trigger is a silence-bounded place-cell burst (≥ 5 block place cells within 300 ms after > 60 ms silence) that must coincide with an SWR; "SWR, MUA" would be closer.
- Detection Notes omit the 300 ms window and the ≥ 5 place-cell requirement on the SWRs themselves; SWR smooth 50 is a boxcar width, not a Gaussian.
- Other fields match (3 SD, 1 SD bounds, 5 cells, 2 electrodes, 100–250 Hz, > 20 ms, merge ≤ 100 ms).

## Package mapping
Tier: B     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe:
```python
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d
from ripple_detection import (filter_ripple_band, get_envelope, normalize_signal,
                              segment_boolean_series, merge_close_events, require_overlap)
from ripple_detection.core import extend_threshold_to_mean
# lfps: (n_time, 2) one wire from each of two cell-layer tetrodes, fs = 2000
power = get_envelope(filter_ripple_band(lfps, fs, band=(100.0, 250.0))) ** 2      # "instantaneous power"
power = uniform_filter1d(power, size=int(round(0.050 * fs)), axis=0).mean(axis=1)  # 50 ms boxcar, mean of 2
z = normalize_signal(power)
swr = np.asarray(extend_threshold_to_mean(z >= 1.0, z >= 3.0, time, 0.0))           # peak > 3 SD, bounds at 1 SD
swr = swr[(swr[:, 1] - swr[:, 0]) > 0.020]
swr = merge_close_events(swr, 0.100)                                               # "<= 100 ms" (package: < 100)
# user code: keep SWRs with spikes from >= 5 distinct place cells inside

# replay candidates (user code around segment_boolean_series; place_spikes (n_time, n_cells) at 1 ms)
silent = pd.Series(pop_spikes.sum(axis=1) == 0, index=t_ms)                        # population not stated
silences = segment_boolean_series(silent, minimum_duration=0.060)
starts = np.array([end for _, end in silences]) + 0.001
windows = np.column_stack([starts, starts + 0.300])
# keep windows with >= 5 block place cells active; bounds -> first/last spike (inferred); then
# candidates = require_overlap(windows, swr)
```
Smoke-tested (SWR part, silence segmentation) on simulate_session data.
Remaining deviations:
- "Instantaneous power" computed here as the squared Hilbert amplitude; the paper does not say how. The 50 ms boxcar is scipy, not a package smoother (the package smooths with a Gaussian).
- Mean/SD period not stated (whole session assumed); thresholds `>=` vs "exceeding".
- Merge "≤ 100 ms" vs `merge_close_events` strictly `<`; interval definition not stated.
- SWR place-cell count, the silence-then-300 ms window, the ≥ 5 block-cell count and the event end are user code; silence population and event end are not stated.
- No speed criterion is applied, because none is stated (Diba 2007 used ≤ 10 cm/s).
Smallest package addition (if C): n/a. To make it A: a silence-onset window helper (as for Diba 2007), a boxcar option for smoothing, and a spike-participation filter for any event table (`minimum_active_units` usable on LFP detectors).

## Published supplement recheck — September 25, 2026

The user-supplied SI corrects the earlier audit's replay-window transcription:
**20 ms with 10 ms steps** is explicit at the end of p. 5. The earlier paragraph's
50 ms window describes behavioral reconstruction. The original survey's 20 ms
entry was correct and is restored. The 50 ms behavioral value remains in
`Reconst. Error bin (ms)`.

| CSV quantity | Verified value and scope | SI page |
|---|---|---|
| SWR threshold/bounds | Peak >3 SD; boundaries at 1 SD | 4 |
| SWR smoothing | 50 ms boxcar of instantaneous power | 4 |
| Channels and band | One channel from each of two tetrodes; 100–250 Hz | 4 |
| SWR duration/grouping | Exclude ≤20 ms events; merge ≤100 ms inter-ripple intervals | 4 |
| Cell participation | ≥5 place cells from any block for SWRs; ≥5 from the given block for replay candidates | 4–5 |
| Replay candidate | 300 ms search window after >60 ms silence; must coincide with an SWR | 4–5 |
| Detection speed/maximum SWR duration | No value stated; 4 cm/s is a template-construction cutoff and 300 ms is a replay search window | 4–6 |
| Replay temporal bins | 20 ms windows, 10 ms step | 5 |
| Spatial templates | 2 cm bins, Gaussian SD 6 cm; exclude speed <4 cm/s | 3, 5–6 |
| Significance | Linear regression of posterior-mode decoded positions against time-bin number; R² compared with 1000 circular time shifts of decoded positions; p<0.05 | 5 |
| Replay direction/strength | Weighted correlation using only posterior bins with probability >0.01 | 6 |
| Behavioral reconstruction | 50 ms windows | 5 |
| Reconstruction error | 6.56±0.46 cm, mean±SEM across analyzed session sets; exclude median absolute errors >20 cm (3 of 28 sets) | 6; Table S1, 21 |
| Replay fraction | 1342/2573 = 52.157%, reported as 52.2% | 5; Table S1, 21 |

The shuffle changes the temporal ordering of decoded positions before refitting
R². It is not a weighted-correlation shuffle or a cell-identity shuffle. Weighted
correlation has a different role, so the `Trajectory` field now distinguishes both.

The 25 session-set errors printed in Table S1 average 6.5636 cm, confirming 6.56.
Their sample-SD SEM is 0.4664 cm; population-SD SEM is 0.4570 cm. The Methods
report 0.46 cm without specifying SD normalization, so the published SEM is retained
and is not claimed to be an exact sample-SEM reproduction from the rounded table.
The candidate and replay columns independently sum to 2573 and 1342.

Nine CSV fields were updated: detection notes, replay bin and step, trajectory,
decoding notes, reconstruction error, shuffle method/count and significance.
The error value returns from `6.56 (not reverified)` to `6.56`; the existing
6 cm spatial smoothing and 52.2% replay fraction are now directly confirmed.
All 35 cell citations/statuses were updated in the
[field-status index](../parameter_verification_2026-09-25.csv). Other paper rows,
including the Mallory supplement corrections, are unchanged.

The [cumulative correction ledger](../parameter_corrections_2026-09-25.csv)
compares original and current values. Restored original values (20 ms replay bins
and 6.56 cm error) therefore no longer appear as corrections; their deliberate
restoration is documented here, not treated as lost edits. The
[follow-up ledger](../parameter_recheck_changes_2026-09-25.csv) likewise compares
the first-pass baseline with the current values.

Still unspecified in the published Methods: the instantaneous-power calculation,
the period used for SWR mean/SD, the precise inter-ripple interval definition,
which cell population defines silence, and the replay event-end rule. Original
detector code was not established in the earlier archive/search audit. These are
implementation limits, not a remaining supplement retrieval gap.
