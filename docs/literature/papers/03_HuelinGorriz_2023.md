# Huelin Gorriz 2023 — The role of experience in prioritizing hippocampal replay
Source: the extracted text (pdftotext of the Dropbox PDF), plus the authors' Zenodo code and the related Tirole 2022 pipeline (see Code); title matched (Huelin Gorriz, Takigawa & Bendor, Nature Communications 14:8157)
Trigger: SWR+MUA (an MUA burst is the event; ripple-band z > 3 is required)

[Paper](https://doi.org/10.1038/s41467-023-43939-z) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [huelin-gorriz-code](../sources.md#huelin-gorriz-code), [tirole-code](../sources.md#tirole-code), [code-archives](../sources.md#code-archives).

## Method as implemented

Detection. Quotes are from Methods, "Detection of replay event" (p. 12), unless noted.

- Lineage: "Replay trajectories were decoded using a Bayesian decoding algorithm, as described in Tirole et al. (2022)27." Results (p. 3): "Candidate replay events were selected based on a minimum duration >= 100 ms, and a z-score threshold of 3 for the smoothed multi-unit activity across all channels27".

- Both thresholds are required: "Detection of candidate sharp-wave ripple (SWR) associated replay events was based on the thresholds set on both multi-unit activity (MUA) and ripple-band power."

- MUA: "MUA was first binned into 1 ms steps and smoothed with a Gaussian Kernel (sigma = 5 ms). Only MUA bursts with a maximum duration of 300 ms and z-scored activity over 3 were included."

- Ripple: "Next, the ripple-band filtered LFP signal was smoothed with a 0.1 s moving average filter, and calculated the amplitude of ripple-band filtered signal using the Hilbert transform. The candidate replay event was required to pass ripple threshold set at z-score of 3."

- Channel: one channel. Methods, "Local field potential analysis" (p. 11): "The PSD was used to identify the channels with higher power for theta (4–12 Hz) and ripple (125–300 Hz) oscillations ... The LFP of the selected channels was down-sampled from 30 kHz to 1 kHz and band-passed filtered in forward and reverse directions ... (MATLAB command filtfilt)."

- Speed, participation, duration: "Candidate replay events passing both thresholds were next speed-filtered (above 5 cm/s), and discarded if the events involved less than 5 different units active or if their duration was below 100 ms or above 750 ms. Therefore, each event should contain at least five 20 ms time bins for decoding." "Speed-filtered (above 5 cm/s)" means events above 5 cm/s were removed. The Tirole text and the code both keep median speed ≤ 5.

- Merge: "Events detected within 50 ms of each other were combined."

- Code: this paper's repository calls `extract_replay_events` (the batch scripts) but does not contain it. The Tirole 2022 extractor is a related implementation; its identity with the missing function is not established. Its `list_of_parameters.m` has the same detection values (`min_zscore=0`, `max_zscore=3`, `max_search_length=300`, `min_event_duration=0.1`, `max_event_duration = 0.75`). `process_clusters.m` and `extract_CSC.m` are byte-identical to the Tirole repository (diff is empty). The related Tirole extractor does the following (the trace helpers are present in both releases):
  - MUA: all sorted and multi-unit spikes. 1 ms histogram, gausswin(41,2) (10 ms SD) applied with filtfilt, z-scored over the whole recording.
  - Threshold: z ≥ 3. Bounds at z < 0, searched ±300 ms from the burst's first supra-threshold sample with an adaptive fallback.
  - Duration: ≥ 100 ms before merging, then merge < 50 ms.
  - Speed: median speed ≤ 5.
  - Participation: ≥ 5 good place cells.
  - Ripple: 125–300 Hz `fir1`, |Hilbert| then a 15-sample (15 ms) moving average, z over the whole recording. The peak inside the event must be ≥ 3 (`replay.ripple_peak >= ripple_zscore_threshold` in this repository's `Replay analysis/decoding/number_of_significant_replays.m`).
  - Maximum 750 ms: not enforced by the Tirole extractor; the missing Huelin Gorriz extractor leaves this runtime condition unknown.

- The text order is smooth, then Hilbert ("smoothed with a 0.1 s moving average filter, and calculated the amplitude ... using the Hilbert transform"). The code does Hilbert, then smooths over 15 samples.

Later analysis restrictions (not detection):

- Sleep versus rest labels: "Replay events during POST1 and POST2 were classified as sleep replay if they occurred when animals' mean moving speed within a 1-min time bin was lower than 4 cm/s accompanied by transient periods of high multi-unit activity (z-score greater than 0), otherwise they were classified as rest replay."

- Putative sleep (Methods, "Putative sleep quantification", p. 11): "periods of immobility (windows of 60 s with velocity lower than 4 cm/s) accompanied by transient periods of high multi-unit activity (z-score greater than 0) ... only the most active units (top 1/3 units in terms of total spike counts) were used ... both the velocity and MUA threshold were visually checked for each data and session and corrected if needed". A theta/delta check was used only as a comparison.

- Main analyses use "the first 30 min of cumulative sleep".

- "Awake SWR events" in the regressions are the candidate events before decoding: "For some analyses that examined the relationship between awake SWR events and sleep replay, candidate SWR events before decoding were used." In the code these are the candidates passing the ripple threshold (`pre_ripple_threshold_index`).

- Sessions with a re-exposure decoding error > 15 cm were discarded. Whole events require all three shuffle tests at p<0.05; split halves require all three at p<0.025, with 1000 shuffles per method.

## Inherited from

Tirole et al. 2022 (ref 27), see [paper note](06_Tirole_2022.md); followed one hop through its code. The released trace helpers and shuffle settings can be inspected directly; the missing event extractor remains a provenance limit.

## Code

- **Own release.** [Zenodo 10.5281/zenodo.10085294](https://doi.org/10.5281/zenodo.10085294) ([dbendor/Nat_Com_Huelin_Gorriz_et_al @b0676a5](https://github.com/dbendor/Nat_Com_Huelin_Gorriz_et_al/tree/b0676a5), v1.0.1). It calls the detection pipeline but does not contain its implementation.

- **Related lab implementation.** [bendor-lab/Elife_Tirole_Huelin_Gorriz_2022 @44ecf42](https://github.com/bendor-lab/Elife_Tirole_Huelin_Gorriz_2022/tree/44ecf42), `Pipeline/Extract replay and Bayesian Decoding/extract_replay_events.m`. Its values support comparison, but do not prove the contents of the missing function.

## Analysis and interpretation

Whole-event replay must pass all three shuffle tests at p<0.05; a split half must pass all three at p<0.025. Shuffles are 1000 per method.

The published maximum duration is 750 ms. The released preprocessing uses nominal MUA Gaussian SD 10 ms per pass (finite-kernel effective SD about 12.58 ms) and a 15 ms ripple moving average; the text states 5 ms and 100 ms, respectively. The complete release omits `extract_replay_events`, so the related Tirole extractor cannot establish whether this paper enforced the maximum.

## Uncertainties

`extract_replay_events` is absent from the complete release. Its original implementation is needed to establish runtime duration handling. The example recipe uses the related Tirole path and omits the published 750 ms cap; it is an approximation, not evidence for changing the CSV.

## Package mapping

Executable example: `huelin_gorriz_2023` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.
