# Muessig 2019 — Coordinated Emergence of Hippocampal Replay and Theta Sequences during Post-natal Development
Source: the extracted text (pdftotext of the Dropbox PDF; Current Biology 29, 834–840.e1–e4, 2019; Muessig, Lasek, Varsavsky, Cacucci, Wills). Supplemental figures (Fig. S1, which characterizes rest/SWR/MUA events) are not in the text and were not consulted. Title verified: yes
Trigger: SWR+MUA (MUA bursts kept only if they overlap an SWR)

[Paper](https://doi.org/10.1016/j.cub.2019.01.005) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [oneill-2008](../sources.md#oneill-2008).

## Method as implemented

All from STAR Methods, "Detection of slow-wave sleep, sharp-wave ripples and multi-unit activity bursts", p. e2–e3, unless noted.

- **State gate ("rest"; restricts all analyses):** "The brain states slow-wave sleep (SWS), rapid-eye movement sleep (REM) and awake movement were defined following [22]. A multitaper power spectral density estimate of the hippocampal local field potential (LFP) was derived for 1.6 s windows, overlapping by 0.8 s (MATLAB function ‘pmtm’). From this, power in the delta and theta bands were calculated in each window. As theta frequency changes during development [18], theta and delta peak frequencies were calculated for each session, defined as the peak frequency of the fast Fourier transform of the LFP, in the bands 5-11Hz (theta) and 1.5-4Hz (delta). Mean running speed for each 1.6 s bin was also estimated. In the absence of EMG recordings, we could not unequivocally discriminate between slow wave sleep and quiet immmobility, we therefore restricted all analyses to epochs termed ‘rest’. Rest was defined as epochs with running speed < 2.5cm/s, and theta/delta power ratio < 2 and waking movement as theta/delta power ratio > 2 and speed > 2.5cm/s."
  - Ratio: theta power / delta power per 1.6 s window (0.8 s step, multitaper), each band centered on a per-session peak frequency found within 5–11 Hz and 1.5–4 Hz. The width of the band around each peak is not stated. Threshold 2 (absolute), AND speed < 2.5 cm/s.
  - [22] = O'Neill et al. 2008 Nat Neurosci, available in the [Oxford-hosted article and supplement](https://www.mrcbndu.ox.ac.uk/sites/default/files/pdfs/oneill2008natureneurosci.pdf).
  - Role: "restricted all analyses to epochs termed ‘rest’". The events analyzed lie in rest (a detection-time or analysis-time restriction; the text does not separate them). The detection statistics are not said to be computed over rest only (see the SWR threshold below).

- **SWR detection:** "Sharp-wave ripples were detected by first filtering the LFP in the band 100-250Hz. The instantaneous power of the filtered LFP was then estimated by calculating the root mean square over 7ms intervals (MATLAB function ‘envelope’ with option ‘rms’). From all LFPs across tetrodes in the CA1 layer, the LFP whose power estimate had the highest standard deviation was then used to define ripple events, as 100ms windows around the peak power, whenever the power was greater than the 99th percentile of all powers in the trial (approximately equal to 4 standard deviations above the mean)."
  - Filter type: not stated. Envelope: moving RMS over 7 ms. One channel: the CA1 tetrode whose RMS power has the highest SD. Threshold: the 99th percentile over the whole trial (not z-scored; "≈ 4 SD" is the authors' approximation). Bounds: a **fixed 100 ms window** around the power peak (centered, inferred from "around"). No min/max duration beyond the fixed window.

- **MUA bursts:** "Multi-unit activity (MUA) bursts were defined by binning all spikes from CS cells into 1ms bins and smoothing the resulting binned spike train with a Gaussian kernel (s.d. 10ms). MUA events were then defined as crossing of a threshold defined as 3 standard deviations above the mean of the smoothed spike train, with a duration from 100-750ms."
  - Signal: pooled sorted complex-spike (putative pyramidal) cells (CS criteria and TINT manual isolation, txt 785–791). Bounds: "crossing of a threshold", read literally as the above-3 SD period. Extension to the mean is not mentioned. Duration 100–750 ms applies to the MUA event. The mean/SD period is not stated ("the smoothed spike train", presumably the trial).

- **Conjunction:** "Only MUA bursts which temporally overlapped (even in part) with SWR events were included in the replay analysis." Stated as a replay-analysis inclusion rule. In effect it is the event definition: Results say "we first defined ensemble spiking events as bursts of multi-unit activity (MUA) that coincided with SWRs" (txt ~206–208). Decoding spans the MUA burst: "spanning the duration of the MUA burst" (txt 867). So the event bounds are the MUA bounds.

- **Awake (RUN) events:** "SWR/MUA joint events during the RUN trial were defined using the same criteria as those in sleep trials. Only data from non-locomotory epochs during RUN were included in further analyses, these were defined using the same criteria as rest during sleep trials with the exception that the limit for running speed was set to < 1cm/s." (txt 884–886).

- Cell participation: none per event. Session inclusion: "Replay analysis was only applied to CS cell ensembles in which > 25 CS cells fired > 75 spikes during RUN" (txt 881; a session-level criterion).

- Merging / close events / artifact rejection: not stated.

Analysis-only:

- The reactivation (pairwise co-firing) analysis uses "all spikes occurring in rest windows, during SWS epochs" (txt 854). Separate from replay events.

## Inherited from

O'Neill 2008 Supplementary Methods (combined PDF p. 13) specify 1600 ms spectral windows, 800 ms steps, Thomson multitaper, and manual state identification from theta/delta versus speed. It does not specify Muessig's bandwidth around per-session peaks. Muessig's own speed <2.5 cm/s and theta/delta <2 rules remain authoritative.

## Code

None linked in the text.

### Related code

- Searched GitHub (code, repositories, and the lab's and authors' accounts), Zenodo, Figshare and CRCNS/DANDI: no original detector caller established in these searches. The WillsCacucciLab repositories and `LaurenzMuessig`'s hold no ripple, MUA or rest-state code.

## Analysis and interpretation

The <10 cm median reconstruction error is an ensemble-inclusion criterion, not the mean error achieved across all ensembles.

## Uncertainties

O’Neill’s cited field/state Methods were read, but do not specify the bandwidth around Muessig’s per-session spectral peaks. The <10 cm median error is an ensemble-inclusion criterion.

## Package mapping

Packaged primary method: `muessig_2019` in [literature_methods.py](../../../src/ripple_detection/literature_methods.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

Additional inventories in the same module: `muessig_2019_ripples`. See their docstrings for required settings and output stages.

For measured inputs, `muessig_2019` uses supplied `sleep_intervals` as the eligible
rest/non-locomotory epochs and retains events wholly inside them. These intervals
must already satisfy the selected trial's published state criteria: mean speed
over 1.6 s windows stepped by 0.8 s, below 2.5 cm/s for rest trials or 1 cm/s for
RUN, together with the theta/delta criterion. A brief instantaneous speed excursion
does not automatically invalidate an otherwise eligible state window.
`sample_speed_veto=True` optionally adds the earlier package implementation's
stricter rule on every native-grid speed sample within the event. This extra veto
is not specified by the paper. Simulation without supplied intervals retains the
documented approximate state proxy.
