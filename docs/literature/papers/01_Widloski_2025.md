# Widloski 2025 — Replay without sharp wave ripples in a spatial memory task
Source: https://www.nature.com/articles/s41467-025-65181-5 (open access; HTML saved and converted to a local copy). Author Correction https://doi.org/10.1038/s41467-026-72252-8 (17 Apr 2026) changes only reference numbering ("refs. 6, 7, 17, 23, ... were incorrect"); the current HTML reflects it. Code: Zenodo 10.5281/zenodo.15199609 (files in a local copy). title matched (Widloski & Foster, Nature Communications, 2025)
Trigger: decoding (ripples and bursts are detected only to LABEL replays as with or without ripple/burst)

[Paper](https://doi.org/10.1038/s41467-025-65181-5) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [widloski-archives](../sources.md#widloski-archives).

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

- Replay detection: Widloski & Foster 2022 ([paper note](09_Widloski_2022.md)), plus a new cell-ID shuffle test.

- Ripple rule "2 SD for at least 15 ms": Kay et al. 2016 (ref 45; not one of the 57 surveyed papers; this is the rule the package's Kay_ripple_detector implements, with `minimum_duration` on the above-threshold run). Averaging across tetrodes follows Pfeiffer & Foster 2013 (ref 22, [paper note](45_Pfeiffer_2013.md)), which averaged smoothed envelopes across tetrodes, used a 3 SD threshold, and used only velocity < 5 periods.

- Burst rule "3 SD, >= 50 ms": Pfeiffer & Foster 2013. There: 1 ms histogram of all clustered units at velocity < 5 cm/s, Gaussian SD 10 ms, peak > mean + 3 SD, bounds at the mean, events with boundaries < 50 ms or > 2,000 ms apart excluded, and >= 10% of units participating. Widloski 2025 keeps only the 3 SD and 50 ms parts.

## Code

Zenodo 10.5281/zenodo.15199609 (MATLAB analysis and plot scripts; the helper functions are not deposited). Data: Zenodo 10.5281/zenodo.16916108.

## Analysis and interpretation

Memory-less Bayesian decoding with a uniform prior

## Uncertainties

Required external helpers are not bundled in the code deposit. Raw/data arrays were not exhaustively inspected. The text’s 50 ms ripple merge and code’s zero merge remain distinct.

## Package mapping

Executable example: `widloski_2025` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.
