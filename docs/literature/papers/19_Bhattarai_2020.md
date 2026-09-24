# Bhattarai 2020 — Distinct effects of reward and navigation history on hippocampal forward and reverse replays
Source: main text the extracted text (Dropbox PDF; methods deferred to the SI). SI Appendix: pnas.1912533117.sapp.pdf from PMC6955321 (https://pmc.ncbi.nlm.nih.gov/articles/instance/6955321/bin/pnas.1912533117.sapp.pdf, fetched through a browser because of PMC's download challenge), converted to scratchpad/code/bh_si.txt. Title verified: yes (Bhattarai, Lee & Jung, PNAS 117:689–697), both files.
Trigger: SWR+MUA (silence-bounded place-cell bursts that must coincide with a detected SWR; the SWRs themselves require ≥ 5 place cells)

## Method as implemented

SWR detection (SI Appendix, Materials and Methods, "SWR events", SI p. 4):
- "LFP signals obtained from one channel each from two tetrodes carrying well-isolated unit signals were filtered between 100 and 250 Hz. Instantaneous LFP power was calculated at each time point (sampling frequency, 2000 Hz), smoothed with a boxcar filter (width, 50 ms) and averaged." How "instantaneous power" was computed (squared Hilbert amplitude, squared signal) is not stated; the averaging is across the two channels (inference from the sentence). 15 tetrodes per hemisphere in CA1 (SI p. 3).
- Threshold and bounds: "Points of LFP power exceeding three SD above the mean were identified as candidate SWR events. The beginning and end of each candidate SWR event were determined at one SD above the mean." The period for mean and SD is not stated (inference: whole session).
- Duration and merging: "Candidate SWR events with ≤ 20 ms durations were excluded and two events with a ≤ 100 ms inter-ripple interval were merged (2)." Ref. 2 = Jackson, Johnson & Redish 2006. Whether the interval is end-to-start or peak-to-peak is not stated.
- Cells: "to further eliminate false positive events, we included those candidate SWR events coincident with spikes of at least five different place cells (from any block of a given session) in the analysis (3)." Ref. 3 = Grosmark & Buzsáki 2016.
- Maximum duration: not stated. Speed: not stated for SWR detection. Theta: a theta-power measure exists ("filtered between 4 and 10 Hz. Instantaneous theta power was calculated at each time point and z-scored", SI p. 4) but is not used as a detection criterion in the text I found.

Replay candidates (SI "Replay events", SI p. 4; main text p. 692):
- "A candidate replay event consisted of five or more place cells from a given block (determined using forced-choice and correct free-choice trials) activated together within a 300-ms window with at least one coincident SWR event and preceded by a > 60 ms period of silence (4)." Ref. 4 = Diba & Buzsáki 2007. Main text: "We identified a candidate replay event as a set of spikes emitted by at least 5 different place cells from a given block preceded by a >60-ms silent period and with at least one coincident SWR".
- Whose silence (block place cells, all place cells, all units) is not stated. How the event end is set is not stated: the reported "mean (±SD) duration was 151.9 ± 61.5 ms" (main text) implies the event is not the full 300 ms window (inference: first to last spike within it).
- Speed: none stated for detection. Tuning curves exclude "periods of immobility (speed < 4 cm/s)" (SI "Bayesian decoding"), an encoding-model criterion, not detection. The main text notes "The majority of SWRs were observed in the reward zone" (p. 692).
- 2,573 candidates for the current trajectory; significance by circular-shift shuffle of decoded positions ("P < 0.05 (n = 1342; 52.2% of candidate replay events"), analysis.

Later analysis restrictions (not detection): replay significance (1000 circular shuffles of decoded positions, R²), weighted correlation for direction, sessions with decoding error > 20 cm excluded, 50 ms decoding bins.

## Inherited from
- Diba & Buzsáki 2007 (ref. 4; findings/51_Diba_2007.md, Supplementary Methods): "When the animal's speed was ≤ 10 cm s–1 ... pre-play/replay events were detected by searching for silent periods ≥ 60 ms. If ≥ 5 or ≥ 30 percent of place-cells (whichever was greater) from a template fired within the next 300 ms, an event was recorded." Bhattarai keeps the 60 ms silence, 300 ms window and 5 cells, drops the 30% alternative and does not state Diba's speed ≤ 10 cm/s, and adds the coincident-SWR requirement.
- Jackson et al. 2006 (ref. 2, for the ≤ 100 ms merge; PMC6674885 via WebFetch): "Threshold crossings <20 ms were removed; the remaining events were concatenated if <100 ms apart." Bhattarai uses the same numbers with ≤.
- Grosmark & Buzsáki 2016 (ref. 3, for the ≥ 5 cells; findings/36_Grosmark_2016.md): events "in which at least five distinct pyramidal cells each fired at least one spike".

## Code
No code link. Data: "The raw data is deposited in Figshare with identifier doi.org/10.6084/m9.figshare.10032866.v2" (main text).

## Survey CSV discrepancies
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
