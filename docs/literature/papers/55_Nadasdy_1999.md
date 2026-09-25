# Nádasdy 1999 — Replay and Time Compression of Recurring Spike Sequences in the Hippocampus
Source: the extracted text (pdftotext of the Dropbox PDF); title matched
Cited method source: [Csicsvari 1999a](../sources.md#csicsvari-1999), primary Methods.
Trigger: SWR (but the replay/sequence analysis itself is NOT gated by events; see below)

[Paper](https://doi.org/10.1523/JNEUROSCI.19-21-09497.1999) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [csicsvari-1999](../sources.md#csicsvari-1999).

## Method as implemented

- Signal and band (Methods, p. 9498): "For the extraction of sharp-wave (SPW) ripple events during sleep, the wide-band recorded data were bandpass filtered digitally (150–250 Hz)."

- Envelope/power: "The power (root mean square) of the filtered signal was calculated, and the beginning, peak, and end of individual ripple episodes were determined." RMS window length: not stated.

- Threshold: "The threshold for ripple detection was set to 7 SDs above the background mean power (Csicsvari et al., 1999)." What "background" is (whole session, sleep only): not stated.

- Channels: tetrodes or silicon arrays "implanted in the CA1 pyramidal layer"; which/how many channels enter the ripple power: not stated in this paper (see Inherited).

- Event bounds: "beginning, peak, and end ... were determined" — rule not stated. Minimum/maximum duration, merging: not stated.

- Brain state: SPW-Rs extracted "during sleep" (home-cage sleep sessions 1 and 3). Theta: "u epochs were detected by calculating the ratio of the u (5–10 Hz) and d (2–4 Hz) frequency bands in 2.0 sec windows. A Hamming window was used" (u = theta, d = delta in the extracted text). How slow-wave sleep was delimited beyond this ratio: not stated.

- Speed: none (sleep and wheel running; no speed criterion stated).

- ANALYSIS, not detection: the sequence search (template matching, joint probability maps) runs over the whole parallel spike trains, not inside detected SPW-Rs: "Shuffling was performed across spike trains for these tests because the spike trains contained both u and non-u epochs" (Results, p. 9502). SPW detection is used for participation probability (Fig. 3) and for EEG correlates.

## Inherited from

Csicsvari et al. 1999a ("Oscillatory coupling ...", J Neurosci 19:274–287), Methods, "Detection of SPWs, ripples, and theta patterns":

- "the wide-band recorded data were digitally band-pass filtered (150–250 Hz; Fig. 1). The power (root mean square) of the filtered signal was calculated for each electrode and summed across electrodes to reduce variability. During SPW-ripple episodes, the power substantially increased which enabled us to determine the beginning, peak, and end of individual ripple episodes. The threshold for ripple detection was set to 7 SDs above the background mean power. Epochs with <4 SD above the background mean power were designated no-SPW periods."

- Theta: θ/δ (5–10 / 2–4 Hz) ratio in 2 s windows; "The exact beginning and end of theta epochs during slow-wave sleep–REM sleep transitions sometimes were adjusted manually."

- Still not stated there: RMS window length, bound rule, duration limits.

- Related but NOT the cited paper: Csicsvari et al. 1999b (J Neurosci 19:RC20; a local copy) gives an RMS window of "1.6 msec", per-electrode RMS summed, detection ≥2 SD and bounds at <1 SD, theta periods excluded. Nádasdy cites only 1999a, so these values are context, not evidence of what Nádasdy used.

## Code

None linked.

### Related code

- Searched GitHub (code, repositories, and the lab's and authors' accounts), Zenodo, Figshare and CRCNS/DANDI: no original detector caller established in these searches.

## Analysis and interpretation

Template/sequence analysis uses multiple resolutions (5, 6.7 and 10 ms); 10 ms is not a universal Bayesian decoding window. Methods describe p<0.01; Figure 6 describes p<=0.01 against 100 surrogates.

## Uncertainties

The cited Csicsvari 1999a Methods establish summed per-electrode RMS, but do not specify the RMS window, event-boundary rule or normalization period. Numerical theta-ratio thresholds and complete sleep staging remain unspecified.

## Package mapping

Executable example: `nadasdy_1999` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.
