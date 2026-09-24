# Chenani 2019 — Hippocampal CA1 replay becomes less prominent but more rigid without inputs from medial entorhinal cortex
Source: the extracted text (pdftotext of the Dropbox PDF, Nature Communications 10:1341). The layout is two-column, so I read the columns separately. Title verified: yes
Trigger: MUA (place-cell population bursts of sorted units)

## Method as implemented
Detection criteria (place-cell bursts, the events used for sequence and replay analysis):
- Units: sorted place cells only, identified by a two-way ANOVA on position × running direction. "Place-selective firing rates along with the direction of each run (leftward/rightward) were then put under a two-way ANOVA test in order to find spatially modulated and directional selective units. Those were identified as place cells." (Methods "Place cell identification and burst rates", p. 11)
- Rate and smoothing: "Place cell rates were calculated as the total number of spikes of all place cells per time bin of 1 ms convolved with a Gaussian kernel (σ = 30 ms)." (p. 11)
- Two-threshold rule: "A pace [sic] cell burst was defined as the time span while the place cell rate remained at least one standard deviation (SD) above its average during a period when the peak firing rate reached at least three SD." (p. 11). The period for the average and SD is not stated ("its average").
- Participation: "Only place cell bursts with five or more active cells were considered for the analysis." (p. 11). Main text: "place cell bursts (consisting of at least five place cells during periods when the population activity exceeded three standard deviations over the mean) during PRE, RUN, and POST epochs" (Results, p. 4).
- Periods, stated for RATES (and, by the definition of rest periods, for which RUN bursts count; the latter is inferred): "Burst rates were calculated as number of place cell bursts divided by immobility time in an epoch (PRE, RUN, and POST). For RUN epochs only the time that the animal spent in the reward zone was taken into account. Reward zones were defined individually for each session based on visual judgment of the spatial distribution of running speeds." (p. 11). Also "Rest periods during PRE and POST epochs were defined as periods while the animal was placed in a Plexiglas enclosure in a familiar room. Rest periods during RUN epochs were defined when the animal was at the reward site at either end of the track." (General data organization, p. 11)
- No speed threshold, duration limit, merging, theta or sleep-stage criterion is stated for bursts.
- Epoch-level analysis restriction: "Epochs with 20 or less place cell bursts were excluded from further analysis." (p. 11)

Secondary HFE detection (Fig. 5, SWR vs fast-gamma bursts), a separate event type:
- Channels: "In each session the least noisy and most stable channels were selected for further analysis. Successively, the selected LFP signal were whitened using a second order autoregressive (AR, 2) model" (LFP analysis and HFEs, p. 11). How many channels, and how they were combined, is not stated. The LFP is "one channel of each tetrode ... 1–450 Hz" (p. 10).
- "Candidate events were detected using a threshold on the absolute value of Hilbert transform (smoothed using a Gaussian kernel with σ = 12 ms) of the band passed (100–250 Hz) LFP. Peaks higher than three SD were recorded as candidate events with event duration defined as times where the absolute value of the Hilbert transform remained above one SD." (p. 11)
- Then classification: "Power spectra of the whitened signals of all candidate HFE events were determined using multitaper method ... Power vectors were projected to PC space and clustering was performed on the first two PCs using different clustering algorithms ... (MiniBatchKMeans, SpectralClustering, Ward, Birch) ... we accept the most stable partitioning" (p. 11), into SWR (150–200 Hz peak) and FGB clusters (Results, p. 7).

## Inherited from
Nothing is deferred for the burst detection. The SSI (rank-order) method cites refs 10 and 13 (sequence analysis, not detection).

## Code
https://github.com/cleibold/ReactivationCode ("The custom MATLAB core routines for sequence analysis", p. 11). I opened it: it contains only rank-order / SSI scripts (testsession.m, rankseq.m, checktempseq.m, ...). There is NO burst or HFE detection code, so nothing further can be resolved from it.

## Survey CSV discrepancies
No discrepancies. MUA Z 3, smooth 30 ms, Min cells 5, and the notes ("Place cell rate peak reached 3 SD, boundaries where it fell below 1 SD, and 5 or more active cells / Reward zones defined by visual judgment of the speed distribution") all agree.
Omission: the sorted, ANOVA-selected place cells only (not all units). The HFE detector (100–250 Hz, σ 12 ms, 3 SD / 1 SD, AR(2) whitening, then spectral clustering) is a separate analysis and correctly not in the SWR columns.

## Package mapping
Tier: B     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe:
```python
from ripple_detection import gaussian_smooth, normalize_signal, exclude_overlap
from ripple_detection.core import extend_threshold_to_mean
# place_cells: (n_time, n_place_cells) spike counts in 1 ms bins (fs = 1000)
rate = gaussian_smooth(place_cells.sum(axis=1) * fs, 0.030, fs)
z = normalize_signal(rate, normalization_mask=epoch_mask)    # "its average": period not stated
bursts = np.array(extend_threshold_to_mean(z >= 1, z >= 3, time, minimum_duration=0.0)).reshape(-1, 2)
i0, i1 = np.searchsorted(time, bursts[:, 0]), np.searchsorted(time, bursts[:, 1]) + 1
n_active = np.array([(place_cells[a:b].sum(axis=0) > 0).sum() for a, b in zip(i0, i1)])
bursts = bursts[n_active >= 5]
bursts_run = exclude_overlap(bursts_in_run, outside_reward_zone_intervals)   # user-defined reward zones
```
Checked: runs on simulate_session data (22 of 22 detections matched ground truth in a 60 s synthetic test; a smoke test only).
Secondary HFE candidates (Tier B for the candidates): AR(2) whitening in user code (statsmodels), `filter_ripple_band(..., band=(100, 250))`, `gaussian_smooth(get_envelope(x), 0.012, fs)`, z-score, then `extend_threshold_to_mean(z > 1, z > 3, time, 0.0)`. The SWR/FGB spectral clustering is outside the package (D for the classification step).
Remaining deviations:
- multiunit_HSE_detector extends to the mean (z = 0), not to 1 SD, so it cannot be used directly. The two-mask call is the whole difference.
- Active-cell count: user code. multiunit_HSE_detector's minimum_active_units uses the same "≥1 spike in the event" definition, but only inside that detector.
- Normalization period not stated (per epoch is the natural reading). Choose normalization_mask accordingly, e.g. per PRE/RUN/POST epoch, or run separately per epoch.
- Reward zones come from visual judgment of the speed distribution. Supply them as intervals; they are not reproducible from the paper.
- Whether a threshold on raw place-cell counts at 1 ms vs another sampling rate changes results: bin at 1 ms to match.
- extend_threshold_to_mean does not split at gaps. Run it per contiguous block.
Smallest package addition (if C): n/a. To make it A: a `bound_threshold` (extension level in SD) and `minimum_active_units` usable with a user-supplied trace, or `bound_threshold` on multiunit_HSE_detector. Then `multiunit_HSE_detector(place_cells, zscore_threshold=3, bound_threshold=1, smoothing_sigma=0.03, minimum_active_units=5, minimum_duration=0, speed_threshold=np.inf)` would be the whole recipe.
