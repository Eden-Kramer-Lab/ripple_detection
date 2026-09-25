# Widloski 2022 — Flexible rerouting of hippocampal replay sequences around changing barriers in the absence of global place field remapping
Source: the extracted text (pdftotext of Dropbox Neuron 2022 PDF incl. STAR Methods and supplement legends); code from Zenodo 10.5281/zenodo.5880582 (`load_replayEvents.m` read). title matched (Widloski & Foster; Neuron 110, 1547–1558)
Trigger: decoding

[Paper](https://doi.org/10.1016/j.neuron.2022.02.002) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [widloski-archives](../sources.md#widloski-archives).

## Method as implemented

Detection is entirely decoding-based (STAR Methods, "Replay detection and analysis", p. e5):

- Rationale: "Classical approaches to extracting replay start with identifying population burst or sharp-wave ripple events ... In practice, we have found that many replay-like events during immobility periods were unaccompanied by large ripples or population bursts (Figure S2B). Thus, we developed a 'bottom-up' procedure for replay extraction that wasn't predicated on the existence of such events."

- Decoding: "the Bayesian decoder was applied to spikes within a sliding window of 80 ms duration (shifted in 5 ms increments) over the entire session."

- Bin criteria: "Time bins were kept for further analysis based on three criteria: rat speed (vrat < 5 cm/s; rat speed was computed at the center of each time bin via linear interpolation), posterior spread (m<10 cm), and posterior COM jumps size (d < 20 cm)."

- Subsequences and merging: "We defined a subsequence as a set of temporally contiguous bins satisfying the above criteria ... neighboring subsequences were merged if the spatial and temporal gap between them was 20 cm and 50 ms, respectively."

- Duration: "A subsequence (merged or not) was denoted a candidate sequence if its duration was greater than 100 ms."

- Dispersion: "A candidate sequence was defined as a replay if its dispersion was greater than 12 cm."

- Main text: "During stopping periods in the task, candidate events were identified as continuous epochs lasting at least 100 ms where the decoded position changed smoothly" (p. 1549).

- Ripple / burst detection: NONE is used to detect, gate or label events. SWR amplitude and spike density are computed only as reference traces (STAR Methods, "Spike density and sharp wave-ripple amplitude", p. e4): "Population spike density was computed by first summing the total number of spikes from all clusters in 1 ms non-overlapping time bins. Sharp wave-ripple amplitude was computed by band-pass filtering the LFP in the 120 to 170 Hz range and then extracting the amplitude envelope via a Hilbert transform. Both the spike density and ripple amplitude were smoothed through convolution with a Gaussian kernel (80 ms SD) and z-scored." These traces are used for Figure S2: "for reference, spike density and sharp wave-ripple (SWR) amplitude ... The dashed line in the fourth row indicates the classical threshold (set to 3) for detecting replay candidate events according to the peak z-scored spike density." They are also used for per-replay peak z values ("the z-scored peak spike density and, below that, the z-scored peak ripple amplitude of the replay", Fig. S3 legend). Which LFP channel, and which period was used for z-scoring: not stated.

- Brain state, cell participation, artifact rules: not stated.

Code: `load_replayEvents.m` lines 11–49 carry the same thresholds (`sequence_posteriorSpreadThr = 10`, `sequence_delxThr = 20`, `sequence_deltThr = 0.05`, `sequence_jumpThr = 20`, `sequence_durationThr = 0.1`). A per-bin spike-density threshold parameter `sequence_spikeDensityThr` is passed to `compute_filtering_binDecoding`; its value in the commented defaults is `-inf`, i.e. disabled. Line 27 smooths the spike density with `setUp_gaussFilt([1 500], windowSizeDecoding_replay/spikeDensityStepSize)` and then `zscore` over the whole trace. `setUp_gaussFilt` (as found in the Mallory 2025 repo from the same lab) passes that value to `mvnpdf` as the covariance, which would give SD about sqrt(80) ≈ 9 ms rather than 80 ms. This is an inference from reading the code, not run, and it affects only the reference trace.

## Inherited from

None for detection; the method is new here. Tetrode placement cites Pfeiffer & Foster 2013 for surgery only.

## Code

Zenodo 10.5281/zenodo.5880582 (MATLAB; `load_replayEvents.m`, `compute_imageSpread.m`, `compute_sequenceDispersion.m`, `compute_sequenceJumps.m`). The helpers it calls (`compute_filtering_binDecoding`, `compute_allSequences_NaNseparated_merge`) are not in the deposit.

## Analysis and interpretation

demarcate events based on
posterior spread < 10 cm
posterior jump size < 20 cm

Behavioral reconstruction error uses 250 ms windows (Methods, PDF p. 19).

## Reproduction limits

The method-specific qualifications above apply. The executable recipe documents its assumptions about unstated parameters, state scoring and simulation stand-ins.

## Package mapping

Replay is defined by decoding, outside this package’s detector scope. [literature_methods.py](../../../src/ripple_detection/literature_methods.py) records the reason in `NOT_REPRODUCED`.
