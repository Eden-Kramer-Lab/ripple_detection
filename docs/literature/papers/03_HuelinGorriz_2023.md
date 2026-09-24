# Huelin Gorriz 2023 — The role of experience in prioritizing hippocampal replay
Source: the extracted text (pdftotext of the Dropbox PDF), plus the authors' Zenodo code and the Tirole 2022 pipeline it calls (see Code); title verified: yes (Huelin Gorriz, Takigawa & Bendor, Nature Communications 14:8157)
Trigger: SWR+MUA (an MUA burst is the event; ripple-band z > 3 is required)

## Method as implemented

Detection. Quotes are from Methods, "Detection of replay event" (p. 12), unless noted.
- Lineage: "Replay trajectories were decoded using a Bayesian decoding algorithm, as described in Tirole et al. (2022)27." Results (p. 3): "Candidate replay events were selected based on a minimum duration >= 100 ms, and a z-score threshold of 3 for the smoothed multi-unit activity across all channels27".
- Both thresholds are required: "Detection of candidate sharp-wave ripple (SWR) associated replay events was based on the thresholds set on both multi-unit activity (MUA) and ripple-band power."
- MUA: "MUA was first binned into 1 ms steps and smoothed with a Gaussian Kernel (sigma = 5 ms). Only MUA bursts with a maximum duration of 300 ms and z-scored activity over 3 were included."
- Ripple: "Next, the ripple-band filtered LFP signal was smoothed with a 0.1 s moving average filter, and calculated the amplitude of ripple-band filtered signal using the Hilbert transform. The candidate replay event was required to pass ripple threshold set at z-score of 3."
- Channel: one channel. Methods, "Local field potential analysis" (p. 11): "The PSD was used to identify the channels with higher power for theta (4–12 Hz) and ripple (125–300 Hz) oscillations ... The LFP of the selected channels was down-sampled from 30 kHz to 1 kHz and band-passed filtered in forward and reverse directions ... (MATLAB command filtfilt)."
- Speed, participation, duration: "Candidate replay events passing both thresholds were next speed-filtered (above 5 cm/s), and discarded if the events involved less than 5 different units active or if their duration was below 100 ms or above 750 ms. Therefore, each event should contain at least five 20 ms time bins for decoding." "Speed-filtered (above 5 cm/s)" means events above 5 cm/s were removed. The Tirole text and the code both keep median speed ≤ 5.
- Merge: "Events detected within 50 ms of each other were combined."
- Code: this paper's repository calls `extract_replay_events` (`batch_analysis.m`, line 76) but does not contain it. The detection code is therefore the Tirole 2022 pipeline. Its `list_of_parameters.m` has the same detection values (`min_zscore=0`, `max_zscore=3`, `max_search_length=300`, `min_event_duration=0.1`, `max_event_duration = 0.75`). `process_clusters.m` and `extract_CSC.m` are byte-identical to the Tirole repository (diff is empty). What the code actually does, per the Tirole findings (06_Tirole_2022.md):
  - MUA: all sorted and multi-unit spikes. 1 ms histogram, gausswin(41,2) (10 ms SD) applied with filtfilt, z-scored over the whole recording.
  - Threshold: z ≥ 3. Bounds at z < 0, searched ±300 ms from the burst's first supra-threshold sample with an adaptive fallback.
  - Duration: ≥ 100 ms before merging, then merge < 50 ms.
  - Speed: median speed ≤ 5.
  - Participation: ≥ 5 good place cells.
  - Ripple: 125–300 Hz `fir1`, |Hilbert| then a 15-sample (15 ms) moving average, z over the whole recording. The peak inside the event must be ≥ 3 (`replay.ripple_peak >= ripple_zscore_threshold` in this repository's `Replay analysis/decoding/number_of_significant_replays.m`).
  - Maximum 750 ms: not enforced by the code.
- The text order is smooth, then Hilbert ("smoothed with a 0.1 s moving average filter, and calculated the amplitude ... using the Hilbert transform"). The code does Hilbert, then smooths over 15 samples.

Later analysis restrictions (not detection):
- Sleep versus rest labels: "Replay events during POST1 and POST2 were classified as sleep replay if they occurred when animals' mean moving speed within a 1-min time bin was lower than 4 cm/s accompanied by transient periods of high multi-unit activity (z-score greater than 0), otherwise they were classified as rest replay."
- Putative sleep (Methods, "Putative sleep quantification", p. 11): "periods of immobility (windows of 60 s with velocity lower than 4 cm/s) accompanied by transient periods of high multi-unit activity (z-score greater than 0) ... only the most active units (top 1/3 units in terms of total spike counts) were used ... both the velocity and MUA threshold were visually checked for each data and session and corrected if needed". A theta/delta check was used only as a comparison.
- Main analyses use "the first 30 min of cumulative sleep".
- "Awake SWR events" in the regressions are the candidate events before decoding: "For some analyses that examined the relationship between awake SWR events and sleep replay, candidate SWR events before decoding were used." In the code these are the candidates passing the ripple threshold (`pre_ripple_threshold_index`).
- Sessions with a re-exposure decoding error > 15 cm were discarded. Significance requires p < 0.05 on three shuffles.

## Inherited from
Tirole et al. 2022 (ref 27), which is in the manifest (06_Tirole_2022); followed one hop through its code. Every tier-deciding detail comes from that code; see 06_Tirole_2022.md.

## Code
- Zenodo 10.5281/zenodo.10085294 (dbendor/Nat_Com_Huelin_Gorriz_et_al v1.0.1, GitHub https://github.com/dbendor/Nat_Com_Huelin_Gorriz_et_al/tree/v1.0.1). Downloaded; commit b0676a5 inside the zip.
- It contains no replay-detection function. Detection is https://github.com/bendor-lab/Elife_Tirole_Huelin_Gorriz_2022 `Pipeline/Extract replay and Bayesian Decoding/extract_replay_events.m` (commit 44ecf42).

## Survey CSV discrepancies
Row 3.
- SWR electrodes (#): CSV "#N/A". The paper and code use one channel, the one with the highest ripple-band power (`bestCSC_ripple`); the value should be 1.
- The other values match the paper text (3, 3, 5 cm/s, 5 ms, 100 ms, 5, 125–300 Hz, 100, 750, 50). The text differs from the code in three places:
  - MUA smoothing: 5 ms in the text, versus a 10 ms SD Gaussian via filtfilt in the code.
  - SWR smoothing: 100 ms in the text, versus 15 ms in the code.
  - 750 ms maximum: not enforced in the code.
- Min. Cells "5" counts place cells with a field on either track, not any unit. The text says "units".
- Detection Notes "MUA and SWR" is right (AND).

## Package mapping
Tier: B     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n for detection (y for the sleep/rest labels, which use the lab's own speed + MUA rule, not LFP scoring)
Recipe: identical to 06_Tirole_2022.md (smoke-tested there):
```python
hse = multiunit_HSE_detector(time, all_spikes, speed, 1000, zscore_threshold=3.0, minimum_duration=0.0,
                             smoothing_sigma=0.005, speed_threshold=np.inf)
ev = hse.loc[hse.duration >= 0.100, ["start_time", "end_time"]].to_numpy()
ev = merge_close_events(ev, 0.050)
ev = exclude_movement_by_majority(ev, speed, time, speed_threshold=5.0)
ev = ev[n_active_units(place_cell_spikes, time, ev) >= 5]                  # user code
ev = ev[peak_within(ripple_z_15ms_boxcar, time, ev) >= 3.0]                # user code
```
The sleep label for analysis would be user code: 60 s windows with mean speed < 4 cm/s and top-third-unit MUA z > 0. The package's `exclude_overlap` / `require_overlap` can then apply those intervals.
Remaining deviations: as in Tirole 2022:
- ±300 ms capped search with adaptive 0.25/0.5 bounds.
- Kernel: 5 ms (text) versus ≈14 ms effective (code).
- Boxcar versus Gaussian for the ripple envelope; `fir1` order 34 versus remez.
- Peak-inside-event ripple test and place-cell participation both need user code.
- Median versus majority speed rule on ties.
Smallest package addition (if C): n/a (B). As for Tirole, public `count_active_units` and `peak_within` / `require_peak` helpers would bring it close to A.
