# Berners-Lee 2022 — Hippocampal replays appear after a single experience and incorporate greater detail with more experience
Source: the extracted text (pdftotext of Dropbox Neuron 2022 PDF, incl. STAR Methods); code from Zenodo 10.5281/zenodo.6330850 (downloaded to a local copy). title verified: yes (Berners-Lee, Feng, Silva, Wu, Ambrose, Pfeiffer, Foster; Neuron 110)
Trigger: MUA (strictly: population spike density of sorted putative pyramidal cells, not unsorted multiunit)

## Method as implemented
Detection (STAR Methods, "Identifying candidate events", p. e2):
- Signal: "We identified candidate events as HP putative pyramidal neuron spike density events."
- Binning / smoothing: "Spikes were binned into 1 ms bins and then smoothed with a gaussian filter (radius = 100 ms, sigma = 10 ms)."
- Speed: "For run epochs, we removed any spiking occurring when the rat was running (>5 cm/sec)."
- Threshold and baseline: "We identified events that exceeded 3 standard deviations of the mean spike rate across the whole session (we defined the mean and standard deviation separately for each epoch)."
- Bounds: "Events were clipped at the start and end when they returned to the mean."
- Duration, applied to the mean-to-mean event: "We removed events with durations less than 100 ms or more than 500 ms." Results: "Candidate replay events (lasting 100–500 ms)" (p. 2).
- Merging close events, cell-participation minimum, brain state, LFP criterion: not stated.
- Code, `5_helper_functions/decode_spikedensity_events.m`, lines 92–128 (settles the ambiguous details):
  - `Filter=fspecial('gaussian',[100 1],10); % ref Brad Pfeiffer 2013`: 100-sample window, SD 10 samples = 10 ms at 1 ms bins.
  - `sMean=mean(Spike_Density(abs(Spike_Density(:,3))<VelThresh,1)); sPeak=sMean+3*std(...)` with VelThresh = 5. The mean and SD are computed over samples with speed < 5 cm/s only, not over the whole session with running spikes zeroed.
  - `Check=diff([0 ; Spike_Density(:,1)>sMean & abs(Spike_Density(:,3))<VelThresh ; 0])`: an event is a contiguous run where density > mean AND speed < 5, so events are cut at movement onset.
  - `DurationBound=[0.1 .5]; target=find(EE-SS>DurationBound(1)/Time_Bin_Size & EE-SS<DurationBound(2)/Time_Bin_Size)`: duration strictly between 100 and 500 ms.
  - `if Spike_Density(SS:EE-1,1)<sPeak, delete`: a MATLAB `if` on a vector is true only when every element is true, so an event is deleted only when no sample reaches sPeak. In effect: peak >= mean + 3 SD.
  - The density histogram is built from all rows of `spikedata`. Interneurons (`hpinterneurons`) are excluded explicitly only when building place fields, so whether they are in the density depends on the contents of `spikedata`, which this file does not show.

Analysis (not detection):
- Decoding (20 ms bins, 5 ms step or non-overlapping; whole-session or lap-by-lap fields). Replay = |weighted correlation| and max-jump thresholds, with a significance matrix against 5,000 shuffles (Silva et al. 2015 criteria; weighted correlation > 0.6 and max jump < 0.4 for the "green box").

## Inherited from
Detection is stated in full. The code comment cites Pfeiffer & Foster 2013 (manifest 45) for the kernel. Pfeiffer 2013: "A histogram (1-ms bins) of all clustered units for times when the rat's velocity was less than 5 cm s-1 was smoothed (Gaussian kernel, standard deviation of 10 ms). Population events were defined as peaks in the smoothed histogram greater than the mean + 3 standard deviations. Start and end boundaries ... where the smoothed histogram crossed the mean ... Candidate events in which fewer than 10% of the clustered units participated or with boundaries less than 50 ms or greater than 2,000 ms apart were excluded" (the extracted text, "Sequential event analysis"). Berners-Lee 2022 drops the 10% participation rule and uses 100–500 ms.

## Code
Zenodo DOI 10.5281/zenodo.6330850 (GitHub ABernersLee/ReplayExperienceProject_NeuronPaper, tag Publication). Detection is in `5_helper_functions/decode_spikedensity_events.m`.

## Survey CSV discrepancies
- Detection "MUA" / notes "MUA": the paper uses sorted putative pyramidal neurons ("HP putative pyramidal neuron spike density events"), not unsorted multiunit. Terminology only; everything else is consistent.
- MUA 3 SD, speed 5, MUA smooth 10 ms, min 100 ms, max 500 ms and combine N/A all match. (Detail: the code takes the mean and SD over speed < 5 cm/s samples; the paper says "across the whole session ... separately for each epoch".)

## Package mapping
Tier: A     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe:
```python
import numpy as np
from ripple_detection import multiunit_HSE_detector
# spikes: (n_time, n_pyramidal_units) spike counts at 1 kHz (1 ms bins)
still = speed < 5
ev = multiunit_HSE_detector(
    time, spikes, speed, 1000.0,
    speed_threshold=5.0,            # endpoint rule
    minimum_duration=0.0,           # peak criterion only: >= 1 sample at/above 3 SD
    zscore_threshold=3.0,
    smoothing_sigma=0.010,
    normalization_mask=still,       # code: mean/SD over speed < 5 samples
    maximum_duration=0.5,
)
ev = ev[(ev.duration > 0.1) & (ev.duration < 0.5)]
```
Code-exact variant (B): `rate = get_multiunit_population_firing_rate(spikes, 1000, 0.010)`; `z = normalize_signal(rate, normalization_mask=still)`; `z[~still] = np.nan` (events then end at movement onset, as in the code); `threshold_by_zscore(z, time, 0.0, 3)`; keep 0.1 < duration < 0.5.
Remaining deviations:
- Speed handling: the package's endpoint rule (speed at the first and last sample <= 5) drops events that reach movement. The code truncates them at movement onset and keeps them if they are still 100–500 ms long. The B variant reproduces the truncation.
- Kernel support: the package truncates the Gaussian at 8 SD. The code's `fspecial('gaussian',[100 1],10)` truncates at ±5 SD (100 samples). This effect is negligible.
- Threshold: the package uses ">= 3 SD" and extends to ">= mean". The code uses "> mean" for the bounds and ">= sPeak" for the peak. These differ only on exact ties.
- The package's `maximum_duration` keeps events <= 0.5 s. The code requires < 0.5 s strictly. Apply the post-filter above.
- The "whole session vs speed < 5" baseline ambiguity between the text and the code (the code's behaviour is used above).
- Units: whether interneurons contribute to the density depends on `spikedata`. The paper says pyramidal.
Smallest package addition (if C): n/a

## Independent parameter recheck — September 25, 2026

The current CSV supersedes the historical discrepancy list below. Independent checks and
source limitations are indexed in [the source recheck](../source_recheck.md) and
[the complete field-status ledger](../parameter_verification_2026-09-25.csv).

- **Time bin step (ms)**: `5 or 20`. Berners-Lee 2022 Methods, PDF pp. 17-18; Zenodo 6330850 Publication release, decode_spikedensity_events.m.
- **Shuffle Method**: `Time-bin shuffle (test of event counts)`. Berners-Lee 2022 Methods, PDF pp. 17-18; Zenodo 6330850 Publication release, decode_spikedensity_events.m.
- **Sig. Thresh.**: `Criteria-based events; p<0.05 on counts`. Berners-Lee 2022 Methods, PDF pp. 17-18; Zenodo 6330850 Publication release, decode_spikedensity_events.m.

Current decoding notes: Primary event criteria are abs(weighted correlation)>0.6 and maximum jump<0.4 of track length. Paper describes 20 ms windows both in 5 ms steps and without overlap; the reopened spike-density helper uses nonoverlapping 20 ms windows. The 5000 time shuffles test counts of qualifying events, not a p<0.05 test for each replay.
