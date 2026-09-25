# Denovellis 2021 — Hippocampal replay of experience at real-world speeds
Source: the extracted text (pdftotext of the Dropbox PDF); title verified: yes (Denovellis EL, Gillespie AK, Coulter ME, Sosa M, Chung JE, Eden UT, Frank LM, eLife 10:e64505, doi:10.7554/eLife.64505)
Trigger: SWR (with a multiunit-HSE control)

## Method as implemented
SWR DETECTION (Materials and methods, "SWR detection", p. 23 of 33):
- "Sharp wave ripples were detected using the same method as in Kay et al., 2016. Each CA1 LFP was obtained by downsampling the original 30 kHz electrical potential to 1.5 kHz and bandpass filtering between 0.5 Hz and 400 Hz. This was further bandpass filtered for the ripple band (150–250 Hz), squared, and then summed across tetrodes—forming a single population trace over time. This trace was smoothed with a Gaussian with a 4 ms standard deviation and the square root of this trace was taken to get an estimate of the population ripple band power." → Kay consensus trace (sum over tetrodes of the squared FILTERED signal, smoothed, square root).
- Normalization, threshold, minimum duration, speed: "Candidate SWR times were found by z-scoring the population power trace of an entire recording session and finding times when the z-score exceeded two standard deviations for a minimum of 15 ms and the speed of the animal was less than 4 cm/s." The 15 ms applies to the above-threshold run.
- Bounds: "The SWR times were then extended before and after the threshold crossings to include the time until the population trace returned to the mean value."
- "The code used for ripple detection can be found at https://github.com/Eden-Kramer-Lab/ripple_detection (Denovellis, 2021b)." (The reference-list entry "Denovellis 2021b" is actually loren_frank_data_processing, Zenodo 10.5281/zenodo.5523666 — a citation mix-up in the paper.)
- Maximum duration: not stated. Close-event rule: not stated.

ANALYSIS restriction:
- "We only analyzed SWRs with spikes from at least two tetrodes." (p. 23)

MULTIUNIT CONTROL (Materials and methods, "Identifying events of high multiunit activity", p. 27):
- "We identified times of high multiunit activity when the animal was immobile as a control analysis. Our approach was similar to Davidson et al., 2009. High multiunit periods were identified as times when the z-scored multiunit population spiking activity was greater than two standard deviations for at least 15 ms and the animal was moving at speeds less than 4 cm/s."
- Multiunit = threshold crossings > 60 µV on any wire of a tetrode (p. 22). Smoothing, normalization period and bounds of the multiunit trace: not stated in the text (see Code).

## Inherited from
- Kay et al. 2016 (Dropbox PDF → a local copy), "SWR detection": "LFPs from all available CA1 cell layer tetrodes were filtered between 150–250 Hz, then squared and summed across tetrodes. This sum was smoothed with a Gaussian kernel (σ = 4 ms) and the square root of the smoothed sum was analysed. SWRs were detected when the signal exceeded 2 s.d. of the recording epoch mean for at least 15 ms." Kay also required >= 3 CA1 cell-layer recordings and head speed < 4 cm/s.
- Davidson et al. 2009 for the multiunit control.

## Code
Analysis repo named in the paper: https://github.com/Eden-Kramer-Lab/replay_trajectory_paper (cloned to a local copy, HEAD f2d2b3c, 2021-11-03). Opened because the text's "squared" disagrees with the package's envelope-based Kay trace.
- `environment.yml` pins `ripple_detection == 0.1.8.dev0` (this repo's tag 0.1.8.dev0, commit 71298f0, 2018-07-16).
- `src/load_data.py::get_ripple_times` calls `Kay_ripple_detector(time, ripple_lfps.values, speed.values, 1500, zscore_threshold=2.0, close_ripple_threshold=np.timedelta64(0, 'ms'), minimum_duration=np.timedelta64(15, 'ms'))` on the RAW LFPs (0.1.8.dev0 filtered internally). Tetrodes: those with `validripple == 1` in the tetrode table if annotated, otherwise all CA1, CA2 and CA3 tetrodes (`brain_areas=['CA1','CA2','CA3']`) — the text says CA1.
- ripple_detection 0.1.8.dev0 (`git show 0.1.8.dev0:ripple_detection/detectors.py`, `core.py`): `filter_ripple_band` = `remez` equiripple FIR, 101 taps, 25 Hz transition bands, `filtfilt`; Kay trace = `np.sum(filtered ** 2)` of the FILTERED signal (no Hilbert envelope; the envelope was added on 2020-10-14, commit 34958cf) → `gaussian_smooth(0.004, truncate=8)` → `sqrt` → scipy `zscore` over all non-NaN samples → runs >= 2 with `end_time >= start_time + 15 ms` extended to z >= 0 → `exclude_movement`: speed at the start AND end sample <= 4. NaN samples were dropped and concatenated. So the code matches the paper text (squared filtered signal), not the current package (squared envelope).
- Multiunit control (`load_data`): 2 ms grid (`SAMPLING_FREQUENCY = 500`), per-tetrode spike indicators, `multiunit_HSE_detector(time, multiunit_spikes, speed, 500, minimum_duration=15 ms, zscore_threshold=2.0, close_event_threshold=0)`; in 0.1.8.dev0 the population rate = mean over tetrodes x fs, Gaussian `smoothing_sigma=0.015` (default), scipy z-score over the whole epoch, bounds at the mean, endpoint speed rule <= 4.

## Survey CSV discrepancies
- MUA Z-score Thresh. (STD): CSV "#N/A"; paper runs a multiunit control with "greater than two standard deviations for at least 15 ms" → 2 (control analysis). Detection Notes "SWR" could say "MUA done as control" as the Gillespie 2021 row does.
- MUA smooth (ms): not stated in the text; code used the 15 ms default. "#N/A" is defensible for a text-only survey.
SWR fields: no discrepancies (2 SD, 4 cm/s, 4 ms, >1 electrodes, 150–250 Hz, 15 ms).

## Package mapping
Tier: A     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe:
```python
from ripple_detection import filter_ripple_band, Kay_ripple_detector, multiunit_HSE_detector
filtered = filter_ripple_band(ca1_lfps, sampling_frequency=1500)   # one channel per ripple tetrode
swrs = Kay_ripple_detector(time, filtered, speed, 1500)            # defaults = the paper:
# zscore_threshold=2.0, minimum_duration=0.015, smoothing_sigma=0.004, speed_threshold=4.0,
# z-scored over the whole session
# analysis: keep SWRs with multiunit spikes on >= 2 tetrodes -- user code
# control: per-tetrode spike indicators on a 2 ms grid
hse = multiunit_HSE_detector(time_2ms, tetrode_spike_indicators, speed_2ms, 500,
                             zscore_threshold=2.0, minimum_duration=0.015, smoothing_sigma=0.015)
```
Exact replication of the run code's trace (squared filtered signal) is B: compute `np.sqrt(gaussian_smooth((filtered ** 2).sum(axis=1), 0.004, 1500))`, z-score with `normalize_signal`, then `threshold_by_zscore(..., minimum_duration=0.015, zscore_threshold=2.0)` and `exclude_movement(events, speed, time, 4.0)`.
Remaining deviations:
- Trace: the paper and the code actually run (0.1.8.dev0) square the FILTERED signal; the current `Kay_ripple_detector` squares the Hilbert envelope. The package docstring (`get_Kay_ripple_consensus_trace`) says the two differ by a factor sqrt(2) after 4 ms smoothing. Checked on simulated data (simulate_session, 60 s, 1500 Hz, 23 ripples, rng=1): traces correlate at r = 0.99999999, median ratio 1.41421, and z >= 2 / 15 ms detection gives the same 25 events with 0 ms bound differences. Effectively equivalent.
- Filter: 0.1.8.dev0 used a 101-tap remez design with 25 Hz transitions; the current package at 1500 Hz uses the shipped 318-tap Frank-lab kernel by default, or a 155-tap remez with `transition_width=25.0` — neither is the 101-tap filter.
- Minimum run: 0.1.8.dev0 required a first-to-last-sample span >= 15 ms (24 samples at 1500 Hz; 9 at 500 Hz); current `minimum_sample_count` gives 23 (1500 Hz) and 8 (500 Hz). One sample.
- Channel set: the code used `validripple` tetrodes or CA1+CA2+CA3 tetrodes; the text says CA1. User's choice of columns.
- NaN handling: 0.1.8.dev0 dropped NaNs and concatenated; current package splits into blocks and flags clipped events. Identical when there are no NaNs.
- Speed: endpoint rule <= 4 in both code versions (matches); text says "less than 4 cm/s".
- MUA population rate: 0.1.8.dev0 averaged over tetrodes, current sums; z-scoring removes the constant factor.
- ">= 2 tetrodes with spikes" is user code (no public helper counting units/tetrodes in LFP-detected events).
Smallest package addition (if C): n/a

## Independent parameter recheck — September 25, 2026

The current CSV supersedes the historical discrepancy list below. Independent checks and
source limitations are indexed in [the source recheck](../source_recheck.md) and
[the complete field-status ledger](../parameter_verification_2026-09-25.csv).

- **MUA smooth (ms)**: `15`. replay_trajectory_paper f2d2b3c, src/load_data.py and environment.yml; pinned ripple_detection 0.1.8.dev0 / local 71298f0 detectors.py default smoothing_sigma=0.015.

Current detection notes: SWR / only SWRs with spikes on >= 2 tetrodes analyzed / MUA done as control (2 SD, >= 15 ms) / The original caller bins MUA at 500 Hz and calls multiunit_HSE_detector with z=2 and minimum duration 15 ms, without overriding the pinned package's Gaussian SD 15 ms.
