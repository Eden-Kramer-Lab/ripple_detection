# Gridchyn 2020 — Assembly-Specific Disruption of Hippocampal Replay Leads to Selective Memory Deficit
Source: the extracted text (pdftotext of the Dropbox PDF, Neuron 106); title verified: yes
Trigger: MUA (online high-synchrony events, HSE)

## Method as implemented
Detection criteria (online, closed loop, in the rest/sleep-box sessions):
- Signal: detected spikes (not sorted units) from all tetrodes in fixed 20 ms windows. "high synchrony events (HSE) were detected using multiunit activity (MUA) in the time windows of fixed length (20 ms). Based on the pre-rest recording, the mean number of spikes in all tetrodes in the 20 ms time windows was calculated." (Real-time decoding, p. e2)
- Threshold, a multiple of the mean (not an SD): "HSE detection was triggered when the number of spikes in a time window exceeded a threshold of 3.5 times the baseline mean established in the pre-rest session." (p. e3)
- Bounds at the baseline mean: "Moments in time, when the synchrony level first reached the baseline level (i.e., mean spike number) before and after the peak were defined as the beginning and the end of the HSE correspondingly." (p. e3)
- Rate-targeted adaptive threshold: "The HSE detection threshold was then adaptively recalculated using the Newton-Raphson method every minute based on the actual number of detected HSEs so that the effective HSE detection rate would be near 1 Hz." (p. e3). Also in Results: "This threshold was initially set to 3.5 times the mean spike numbers measured in 20-ms windows during the pre-rest session, but the threshold was dynamically adjusted to achieve an approximate 1-Hz detection rate." (p. 3)
- Onset: "The beginning of the HSE was then adjusted to the time of the first spike in the HSE window." (p. e3)
- Refractory period: "In both cases, detection refractory period of 150 ms was applied to avoid multiple detections of the same HSE as well as spiking ''rebound'' after the light pulse ceases." (p. e3)
- No speed, theta, sleep-stage, duration or unit-count criterion is stated. Theta absence is an observation, not a criterion: "The majority (93.7%) of the HSEs were detected in the absence of theta oscillations, during slow-wave sleep and immobile waking periods (Figure S3A)" (p. 3).
- What happens next (decoding confidence decides the light pulse) is the manipulation, not detection.

Secondary SWR detection (analysis only: cross-correlation with HSEs, SWR counts):
- "SWR detection was performed as previously described (Csicsvari et al., 1999). Local field potentials were band-pass filtered (150–250 Hz), and a reference signal (from a channel that did not contain ripple oscillations) was subtracted to eliminate common-mode noise (such as muscle artifacts). The power (root mean square) of the filtered signal was calculated for each electrode and summed across electrodes designated as being in the CA1 pyramidal cell layer. The SWR detection threshold (6 SD above baseline) was always set in the pre-rest session, and the same threshold used throughout." (LFP analysis, p. e4)
- The RMS window, bounds and duration limits are not stated.
- Theta and delta: RMS of 2–4 / 6–10 Hz band-passed signal "in the 60 s intervals" (p. e4), used only for analysis (Fig. S3A).

## Inherited from
- The SWR detection cites Csicsvari et al. 1999, J Neurosci 19:274–287 ("Oscillatory coupling ..."). Not retrievable here (403/CAPTCHA on jneurosci.org, PMC, Europe PMC, academia.edu), so its RMS window and bound rule are UNKNOWN. This affects only the secondary SWR detector.
- The real-time decoding is based on Kloosterman et al. 2014 (clusterless KDE decoding). That is the manipulation, not detection.

## Code
https://github.com/igridchyn/lfp_online (listed in the Key Resources table p. e1, and "The real-time decoding software is available online", p. e6). I opened it for the tier-deciding adaptive rule:
- `LFPBuffer::IsHighSynchrony()` (lfp_online/LFPBuffer.cpp): `return (high_synchrony_tetrode_spikes_ >= sync_spikes_window_ * high_synchrony_factor_);`. It counts spikes on the configured "synchrony tetrodes" in a SLIDING window of `pop.vec.win.len.ms` ending at the current sample. `sync_spikes_window_` is the sum over those tetrodes of (estimated firing rate × window length).
- The "Newton-Raphson" update is a fixed-gain proportional step (`Utils::NewtonSolver::Update`: `current_x_ += (target_f_ - f_value_) * alpha_;`), constructed as `NewtonSolver(TARGET_SYNC_RATE, 24000*60, -0.5, high_synchrony_factor_)`. So every minute (24000×60 samples), factor ← factor + 0.5 × (observed rate − target rate).
- The refractory period is `lpt.trigger.cooldown`, counted from the last trigger (onset to onset).
- The example config in the repo (Res/synchrony_detection.conf: factor 4.5, target 0.5 Hz, adjustment off, cooldown 1200 samples) is NOT the paper's setting. The paper states 3.5, ~1 Hz and 150 ms.
- In the code the baseline firing rates are estimated once, from the spikes buffered before a start-up delay (`estimate_firing_rates`). The paper says pre-rest.

## Survey CSV discrepancies
No value discrepancies: Detection MUA, and the notes "3.5x the baseline mean spike count, not a SD multiple / adaptively adjusted to ~1 event/s; 150 ms detection refractory" all agree with the paper.
Omissions: the 20 ms counting window (MUA smooth could read "20 ms boxcar"), bounds at the baseline mean, baseline from the pre-rest session, and the offline SWR detector at 6 SD (analysis only, correctly not in the SWR columns).

## Package mapping
Tier: B     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe (offline replication of the online rule):
```python
from ripple_detection.core import extend_threshold_to_mean
counts = multiunit.sum(axis=1)                               # all detected spikes, all tetrodes, per sample
win = int(round(0.020 * fs))
count20 = np.convolve(counts, np.ones(win))[: len(time)]     # causal sliding 20 ms count
baseline = count20[pre_rest].mean()
factor, threshold = 3.5, np.empty_like(count20)
for minute in minutes:                                       # sequential proportional controller
    threshold[minute] = factor * baseline
    n = n_onsets_with_150ms_refractory(count20[minute] >= threshold[minute])
    factor = max(factor + 0.5 * (n / 60.0 - 1.0), 1.0 + 1e-9)   # must stay above the mean (see below)
hse = extend_threshold_to_mean(count20 > baseline, count20 >= threshold, time, minimum_duration=0.0)
# then onset-to-onset 150 ms refractory, and start := first spike in the window (user loops)
```
Checked: runs on simulate_session multiunit (26 of 27 events matched ground truth in a 60 s synthetic test; a smoke test only).
Static approximation (Tier A, if the adaptive part is dropped): because 3.5 × mean equals z = 2.5·μ/σ, use `multiunit_HSE_detector(time, multiunit, speed, fs, zscore_threshold=2.5 * mu / sd, normalization_mask=pre_rest, minimum_duration=0.0, speed_threshold=np.inf, smoothing_sigma=0.0058)`, where μ and σ are the pre-rest mean and SD of the same smoothed trace. The z = 0 bound is then exactly "back to the baseline mean". Deviation: a Gaussian (σ ≈ 20 ms/√12) instead of a 20 ms boxcar, and a fixed threshold.
Secondary SWR (Tier A): `Roumis_ripple_detector(time, filter_ripple_band(lfp_ca1 - ref[:, None], fs), speed, fs, zscore_threshold=6, normalization_mask=pre_rest, speed_threshold=np.inf, minimum_duration=0.0)`. A sum of per-electrode RMS is a constant times Roumis's mean of sqrt(smoothed squared envelope), and a z-score ignores the constant. Unknowns: the RMS window (Gaussian vs boxcar), and the bounds (Roumis extends to the mean; the Csicsvari rule is unknown).
Remaining deviations:
- extend_threshold_to_mean works on any boolean masks, so a per-minute (time-varying) threshold is already expressible. BUT it raises ValueError if any above-threshold run lies outside an above-mean run, so the adaptive factor must be clamped above 1.
- The rate controller is user code. The paper's "Newton-Raphson" is in the code a fixed-gain step (gain 0.5 per Hz, updated every 60 s). An offline, non-causal alternative (one factor chosen to give 1 Hz over the session) is simpler but not what was done.
- The 150 ms refractory runs onset to onset. exclude_close_events measures from the END of the last kept event to the next START, so it is not identical. Extension to the mean already collapses repeat crossings within one above-mean run.
- Sliding 20 ms count (online) vs any binning you choose offline. The code sums spikes over the configured "synchrony tetrodes"; the paper says "all tetrodes".
- Onset adjusted to the first spike: user code.
Smallest package addition (if C): not required for B. To make it A: an HSE option for a boxcar count window, and a `target_event_rate` threshold search (static or per-block). The "multiple of the baseline mean" is already reachable through zscore_threshold and normalization_mask as shown above.
