# Maboudi 2018 — Uncovering temporal structure in hippocampal output patterns
Source: the extracted text (pdftotext of the Dropbox PDF); title matched
Trigger: MUA (population burst events from pooled single + multi-units; no LFP criterion)

[Paper](https://doi.org/10.7554/eLife.34467) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [maboudi-code](../sources.md#maboudi-code), [nelpy](../sources.md#nelpy).

## Method as implemented

Materials and methods, "Population burst events" (p. 16), linear-track data (Diba & Buzsáki 2007 recordings):

- Signal: "a SDF was calculated by counting the total number of spikes across all recorded single and multi-units in non-overlapping 1 ms time bins."

- Smoothing: "The SDF was then smoothed using a Gaussian kernel (20 ms standard deviation, 60 ms half-width)."

- Normalization/threshold: "Candidate events were identified as time windows with a peak SDF of at least three standard deviations above the mean calculated over all the session." (Whole session, including running.)

- Bounds: "The boundaries of each event were set to time points of crossing the mean, preceding and following the peak."

- Speed (detection criterion, mean over the event): "Events during which animals were moving (average movement speed of >5 cm/s) were excluded from all further analyses".

- Inclusion for analysis (applied to every analysis, so effectively part of the event set): "we then binned each PBE into 20 ms (non-sliding) time bins. Spikes from putative interneurons (mean firing rate when moving >10 Hz) were excluded, as were events with duration less than four time bins or with fewer than four active pyramidal cells." (≥80 ms; ≥4 active pyramidal cells, counted after dropping interneuron spikes.)

- Results (p. 3): "We did not add any other restrictions on behavior, LFP, or the participation of place cells." SWRs are described as accompanying most PBEs but are not a criterion. "While we identified active behavior using a speed criterion, we found similar results when we instead used a theta-state detection approach (not shown)."

- Open-field data: "we used the previously reported criteria (Pfeiffer and Foster, 2013) for identifying PBEs prior to binning (10 ms standard deviation kernel, minimum of 10% of units active, duration between 50 ms and 2000 ms)."

- W-maze (Karlsson) and post-task sleep datasets: PBE criteria and sleep scoring not stated separately.

- ANALYSIS, not detection: HMM on 20 ms bins; Bayesian replay with column-cycle shuffles.

## Inherited from

Linear track: nothing deferred (fully specified). Open field: Pfeiffer & Foster 2013 ([paper note](45_Pfeiffer_2013.md); the paper restates its criteria; note Pfeiffer's text there says the stopped-period histogram uses speed <5 cm/s and a 10 ms SD kernel).

## Code

The [paper release](../sources.md#maboudi-code) supplies `data/fig1.nel`, a serialized nelpy 0.2.0 object for session 16-40-19. Inert parsing established 457 MUA epochs lasting 80–696 ms, minimum peak z=3.003, and 277 final binned PBEs. Original event epochs and final binned support are different objects. The release does not independently establish the smoothing kernel or a maximum-duration setting. The published linear-track Gaussian SD is 20 ms.

The open-field analysis (`Figure6.ipynb`) consumes Pfeiffer’s `BradRippleStartEndTimes`. Current nelpy defaults are a supporting comparison, not evidence of the original 0.2.0 preprocessing.

## Analysis and interpretation

Both HMM model-congruence and Bayesian replay scores use 5000 surrogates per event. HMM shuffle permutes off-diagonal transitions within rows; Bayesian shuffle circularly shifts each posterior column. Significance thresholds are varied for ROC analyses; the pooled Bayesian comparison uses 99%, with HMM threshold adjusted to match its event fraction. Temporal/time-swap/Poisson surrogates are model-training controls.

## Uncertainties

The released session does not independently establish a 10 ms smoothing kernel or a 750 ms maximum. The published 20 ms linear-track Gaussian remains authoritative. Stored MUA epochs and subsequent binned PBE support are different objects and should not be treated as interchangeable durations.

## Package mapping

Executable example: `maboudi_2018` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.
