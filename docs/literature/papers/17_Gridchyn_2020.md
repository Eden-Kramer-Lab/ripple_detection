# Gridchyn 2020 — Assembly-Specific Disruption of Hippocampal Replay Leads to Selective Memory Deficit
Source: the extracted text (pdftotext of the Dropbox PDF, Neuron 106); title matched
Trigger: MUA (online high-synchrony events, HSE)

[Paper](https://doi.org/10.1016/j.neuron.2020.01.021) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [gridchyn-code](../sources.md#gridchyn-code), [csicsvari-1999](../sources.md#csicsvari-1999).

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

- The cited [Csicsvari 1999 Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782375/) do not specify the ripple RMS window or exact event-boundary rule. The 0.2 ms RMS window elsewhere in that source is for spike detection, not ripples.

- The real-time decoding is based on Kloosterman et al. 2014 (clusterless KDE decoding). That is the manipulation, not detection.

## Code

https://github.com/igridchyn/lfp_online (listed in the Key Resources table p. e1, and "The real-time decoding software is available online", p. e6). The adaptive rule is implemented as follows:

- `LFPBuffer::IsHighSynchrony()` (lfp_online/LFPBuffer.cpp): `return (high_synchrony_tetrode_spikes_ >= sync_spikes_window_ * high_synchrony_factor_);`. It counts spikes on the configured "synchrony tetrodes" in a SLIDING window of `pop.vec.win.len.ms` ending at the current sample. `sync_spikes_window_` is the sum over those tetrodes of (estimated firing rate × window length).

- The "Newton-Raphson" update is a fixed-gain proportional step (`Utils::NewtonSolver::Update`: `current_x_ += (target_f_ - f_value_) * alpha_;`), constructed as `NewtonSolver(TARGET_SYNC_RATE, 24000*60, -0.5, high_synchrony_factor_)`. So every minute (24000×60 samples), factor ← factor + 0.5 × (observed rate − target rate).

- The refractory period is `lpt.trigger.cooldown`, counted from the last trigger (onset to onset).

- The example config in the repo (Res/synchrony_detection.conf: factor 4.5, target 0.5 Hz, adjustment off, cooldown 1200 samples) is NOT the paper's setting. The paper states 3.5, ~1 Hz and 150 ms.

- In the code the baseline firing rates are estimated once, from the spikes buffered before a start-up delay (`estimate_firing_rates`). The paper says pre-rest.

`EXPERIMENTAL_assembly_inhibition.conf` supplies a compatible example with factor 3.5, 20 ms window and 150 ms refractory at 24 kHz. The adaptive target defaults to 1 Hz; inhibition target 0.8 is a separate setting. This example does not establish the paper’s runtime configuration.

## Analysis and interpretation

Real-time

## Uncertainties

The adaptive rule and compatible example configuration are inspected, but paper-specific runtime overrides are unknown. The cited Csicsvari source does not supply the missing secondary-ripple RMS window or exact bounds.

## Package mapping

Executable example: `gridchyn_2020` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.
