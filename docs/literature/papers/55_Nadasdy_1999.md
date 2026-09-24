# Nádasdy 1999 — Replay and Time Compression of Recurring Spike Sequences in the Hippocampus
Source: the extracted text (pdftotext of the Dropbox PDF); title verified: yes
Cited method source: Csicsvari et al. 1999a, J Neurosci 19:274, via Wayback copy of https://www.jneurosci.org/content/jneuro/19/1/274.full.pdf (saved as a local copy; jneurosci.org and PMC were Cloudflare/captcha-blocked)
Trigger: SWR (but the replay/sequence analysis itself is NOT gated by events; see below)

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

## Survey CSV discrepancies
No discrepancies in the filled fields (Detection SWR, 7 SD, 150–250 Hz). "SWR electrodes" is #N/A; per the cited Csicsvari 1999a the power is summed over all pyramidal-layer electrodes (count per session not stated). Detection Notes could say "RMS power summed over electrodes, 7 SD; sleep only".

## Package mapping
Tier: A     Needs radiatum: n   Needs theta: y (θ/δ ratio marks theta epochs)   Needs sleep scoring: y (events "during sleep"; SWS delimitation not stated)
Recipe:
```python
filtered = filter_ripple_band(lfps, fs, band=(150, 250))        # (n_time, n_pyr_channels)
events = Roumis_ripple_detector(                                 # mean over channels of per-channel
    time, filtered, speed, fs,                                   # sqrt(smoothed squared envelope) =
    speed_threshold=np.inf,                                      # "RMS per electrode, summed" up to a
    zscore_threshold=7.0,                                        # constant that z-scoring removes
    minimum_duration=0.0,                                        # no minimum stated
    smoothing_sigma=0.001,                                       # RMS window not stated; short
    normalization_mask=is_sleep,                                 # if "background" = sleep (unstated)
)
events = exclude_overlap(events, theta_epochs)                   # user-computed θ/δ epochs
```
Remaining deviations:
- Hilbert envelope + Gaussian smoothing vs sliding-window RMS of the filtered signal (window unstated).
- Bounds: package extends to the mean (z = 0); paper's bound rule not stated.
- Package needs ≥1 sample at or above 7 SD ("at or above" vs "above").
- Roumis averages channels with equal weight; "summed" is the same after z-scoring, but a dead/noisy channel weighs the same either way.
- "Background" for mean/SD unspecified; theta/delta and sleep scoring are user-supplied.
Smallest package addition (if C): n/a.
