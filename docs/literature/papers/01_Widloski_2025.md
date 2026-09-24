# Widloski 2025 — Replay without sharp wave ripples in a spatial memory task
Source: https://www.nature.com/articles/s41467-025-65181-5 (open access; HTML saved and converted to a local copy). Author Correction https://doi.org/10.1038/s41467-026-72252-8 (17 Apr 2026) changes only reference numbering ("refs. 6, 7, 17, 23, ... were incorrect"); the current HTML reflects it. Code: Zenodo 10.5281/zenodo.15199609 (files in a local copy). title verified: yes (Widloski & Foster, Nature Communications, 2025)
Trigger: decoding (ripples and bursts are detected only to LABEL replays as with or without ripple/burst)

## Method as implemented
Event (replay) detection is decoding-based (Methods, "Replay detection"):
- "The Bayesian decoder was applied to spikes within a sliding window of 80 ms duration (shifted in 5 ms increments) over the entire session from all place cells found in the session. Time bins were kept for further analysis based on three criteria: rat speed (νrat < 5 cm/s ...), posterior spread (m < 10 cm), and posterior COM jumps size (δ < 20 cm). We defined a candidate replay as a set of temporally contiguous bins satisfying the above criteria ... Neighboring sequences were merged if the spatial and temporal gap between them was 20 cm and 50 ms, respectively."
- "A candidate replay (merged or not) was denoted a replay if: (1) its duration was greater than 100 ms, and (2) it's spatial dispersion D was greater than 12 cm ... and (3) it passed a place cell-ID shuffle test: Each event was re-decoded using shuffled place-cell IDs 100 times ... Replays were required to have p-values for each measure less than 0.05."
- Inherited explicitly: "We briefly outline the procedure for detecting replays, which has been previously published [23 = Widloski & Foster 2022]." Criterion (3), the shuffle test, is new relative to 2022.

Ripple and burst detection, used to label replays (Methods, "Ripple and burst event detection"). Implementable (see mapping).
- Ripple power: "Sharp wave-ripple amplitude (denoted as 'ripple power') was computed for each tetrode by band-pass filtering the LFP on one its four channels in the 100 to 220 Hz range and extracting the magnitude of the Hilbert transform." Up to 64 tetrodes in CA1, bilateral. Which channel of each tetrode: not stated.
- Spike density: "Population spike density was computed by first summing the total number of spikes from all clusters within a session (i.e., clusters with noise overlap < 0.03, isolation > 0.95, peak SNR > 1.5) in 1 ms non-overlapping time bins." That is sorted, well-isolated clusters, not only place cells.
- Smoothing, normalization, baseline: "Both the ripple amplitude and spike density were smoothed through convolution with a Gaussian kernel (80 ms SD, 1000 ms kernel size) and z-scored, unless when ripple power was averaged across tetrodes, in which case z-scoring occurred after averaging. The mean and standard deviations used for z-scoring were computed from stopping periods only (i.e., rat speed <5 cm/s)."
- Artifacts (manual): "Peak events in stopping period data were visually inspected to make sure that z-scoring was not biased by 'noise' events (e.g., chewing artifacts, implant collisions, scratching, etc.)."
- Ripple events: "Candidate ripple events were defined as when the z-scored ripple power (on individual tetrodes or averaged across tetrodes, depending on the analysis) peaked above 2 standard deviations and lasted for at least 15 ms. Event start and end times were defined as when the ripple power returned to the mean. Adjacent ripple events were merged if the time boundaries were less than 50 ms apart." Results: "we defined ripple events as when the ripple power exceeded 2 standard deviations (SD) for at least 15 ms [45 = Kay et al. 2016], where ripple power was averaged across all tetrodes [22 = Pfeiffer & Foster 2013]."
- Burst events: "Candidate burst events were defined as when the z-scored population spike density peaked above 3 standard deviations and lasted for at least 50 ms [22]. Event start and end times were defined as when the spike density returned to the mean." No merge rule is stated for bursts.
- Labelling (analysis): "A given replay was determined to be ripple-less if it (1) contained no ripple events ... (by 'contain', we mean that the peak ripple power time was not within the replay) and (2) passed a ripple power shuffle test requiring that the peak ripple power within the replay not exceed the 95th percentile of a shuffle distribution comprised of ripple power peaks taken from 100 random equal-length snippets of LFP across stopping periods but outside of other replay times." Bursts are labelled the same way. Analysis was restricted to replays during reward consumption (smoothed speed < 1 cm/s at the rewarded well).
- Brain state, radiatum sharp wave: not used. Theta power is reported as a descriptive control only.

Code observations (`plot_replayEvents_rippleStats_allSessions.m`; the detection helper `compute_events_times` is not in the deposit):
- Line 48: `eventMergeThr = 0;%0.05;`. The labelling code passes merge = 0 to `compute_events_times`, so the 50 ms merge in the text appears disabled in this script.
- Lines 50–52: `eventZscoreThr_rippleEvent = 2; eventDurationThr_rippleEvent = 0.015;` and `eventZscoreThr_spikeDensityEvent = 3; eventDurationThr_spikeDensityEvent = 0.05;`. These match the text.
- Lines 66–70 and 146–149: mean and SD are computed over immobility intervals (speed <= speedThr), matching "stopping periods only".
- Line 301: spike density is smoothed with `setUp_gaussFilt([1 1000],0.1/spikeDensityStepSize)`. The 1000-sample kernel matches "1000 ms kernel size". In the lab's `setUp_gaussFilt` (copy in the Mallory 2025 repo) the second argument is the `mvnpdf` covariance, so SD = sqrt(100 bins²) = 10 ms, not the stated 80 ms. This is an inference from reading the code, not run. The ripple-power smoothing is precomputed outside the deposit.
- Line 96: raw-LFP deflection detection (`findpeaks(abs(diff(LFP_raw)))`, peaks >= 500, ±1 s windows) builds `times_remove`, which is an artifact rule not mentioned in the text. Whether it feeds the detection is not visible in this file.
- Whether "lasted for at least 15 ms" applies to the above-threshold run or to the mean-to-mean event is not resolvable here (`compute_events_times` is absent). The cited source for the ripple rule, Kay et al. 2016, applies it to the time above threshold, which is what the package's Kay semantics implement. The cited source for the burst rule, Pfeiffer & Foster 2013, applies 50 ms to the mean-to-mean bounds ("boundaries less than 50 ms ... apart were excluded", the extracted text).

## Inherited from
- Replay detection: Widloski & Foster 2022 (manifest 09; see that finding), plus a new cell-ID shuffle test.
- Ripple rule "2 SD for at least 15 ms": Kay et al. 2016 (ref 45; not in the manifest; this is the rule the package's Kay_ripple_detector implements, with `minimum_duration` on the above-threshold run). Averaging across tetrodes follows Pfeiffer & Foster 2013 (ref 22, manifest 45), which averaged smoothed envelopes across tetrodes, used a 3 SD threshold, and used only velocity < 5 periods.
- Burst rule "3 SD, >= 50 ms": Pfeiffer & Foster 2013. There: 1 ms histogram of all clustered units at velocity < 5 cm/s, Gaussian SD 10 ms, peak > mean + 3 SD, bounds at the mean, events with boundaries < 50 ms or > 2,000 ms apart excluded, and >= 10% of units participating. Widloski 2025 keeps only the 3 SD and 50 ms parts.

## Code
Zenodo 10.5281/zenodo.15199609 (MATLAB analysis and plot scripts; the helper functions are not deposited). Data: Zenodo 10.5281/zenodo.16916108.

## Survey CSV discrepancies
- Detection Notes "candidate = continuous epoch >=100 ms of smoothly changing decoded position": in the paper, a candidate replay is any set of contiguous passing bins, and "duration was greater than 100 ms" is a replay criterion, strictly greater. Minor.
- Detection Notes omit replay criterion (3), the place-cell-ID shuffle test (100 shuffles, p < 0.05 for both mean posterior spread and jump size).
- Detection Notes "ripples (2 SD, >=15 ms) and bursts (3 SD, >=50 ms) label events after detection": correct but incomplete. The ripple band is 100–220 Hz, both traces use Gaussian 80 ms SD smoothing and a stopping-period (speed < 5) z-score baseline, ripples are merged if < 50 ms apart (per text; code merge = 0), and ripple-less/burst-less also requires a shuffle test.
- Numeric fields (speed 5, min duration 100, combine 50) match. Because ripples and bursts do not define events, SWR/MUA fields at N/A are defensible.

## Package mapping
Tier: D     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Event detection is D because replays are defined by the decoded posterior (spread, jump, dispersion, cell-ID shuffle). The ripple/burst LABELLING step is implementable, tier B:
```python
import numpy as np
from ripple_detection import (filter_ripple_band, get_envelope, gaussian_smooth,
    normalize_signal, threshold_by_zscore, merge_close_events,
    get_multiunit_population_firing_rate)
stopping = speed < 5
# ripple power, averaged across tetrodes (one channel per tetrode), z-scored after averaging
filt = filter_ripple_band(lfp_one_ch_per_tetrode, fs, band=(100, 220))
rp = gaussian_smooth(get_envelope(filt), 0.080, fs).mean(axis=1)
z_rp = normalize_signal(rp, normalization_mask=stopping)
rip = np.array(threshold_by_zscore(z_rp, time, minimum_duration=0.015, zscore_threshold=2))
rip = merge_close_events(rip, 0.050)
# per-tetrode variant: loop over columns, z-score each column over stopping periods
# bursts: all well-isolated clusters, 1 ms bins
sd = get_multiunit_population_firing_rate(spikes_1ms, 1000.0, smoothing_sigma=0.080)
z_sd = normalize_signal(sd, normalization_mask=stopping_1ms)
bur = np.array(threshold_by_zscore(z_sd, time_1ms, minimum_duration=0.0, zscore_threshold=3))
bur = bur[(bur[:, 1] - bur[:, 0]) >= 0.050]      # 50 ms on mean-to-mean bounds (Pfeiffer 2013 reading)
# label: replay is ripple-less if no ripple PEAK time falls inside it (+ shuffle test, user code)
```
Close tier-A equivalents: `Roumis_ripple_detector(time, filt, speed, fs, speed_threshold=np.inf, minimum_duration=0.015, zscore_threshold=2, smoothing_sigma=0.080, normalization_mask=stopping)` followed by `merge_close_events(ev, 0.05)`. And `multiunit_HSE_detector(time_1ms, spikes_1ms, speed_1ms, 1000, speed_threshold=np.inf, minimum_duration=0.0, zscore_threshold=3, smoothing_sigma=0.080, normalization_mask=stopping_1ms)`, then duration >= 0.05.
Remaining deviations (labelling step):
- Roumis uses sqrt(smoothed squared envelope). The paper averages smoothed envelopes. The B recipe is exact.
- The 15 ms / 50 ms semantics (above-threshold run vs whole event) are ambiguous in the text; the recipe follows the cited sources.
- Merge: the text says 50 ms, the code says 0.
- The spike-density smoothing SD in the code may be 10 ms rather than 80 ms (inference).
- "Contain" means the event's peak time lies within the replay. The package has no peak-time column for these detectors (`peak_time` exists only on Zugaro and Long), so peak times need user code on the z trace; require_overlap would test interval overlap instead.
- Ripple-power and spike-density shuffle tests and manual artifact inspection are user code or manual.
- Filter design is unstated in the paper; the package uses a remez FIR.
Smallest package addition (if C): n/a for the D verdict. A `peak_time` column (time of max z) on the threshold detectors would make the "peak within replay" labelling direct.
