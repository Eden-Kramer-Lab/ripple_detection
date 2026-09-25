# Karlsson 2009 — Awake replay of remote experiences in the hippocampus
Source: the extracted text (pdftotext of the Dropbox PDF, main text + Online Methods + Supplementary Information); title verified: yes (Karlsson MP & Frank LM, Nat Neurosci 12:913–918, doi:10.1038/nn.2344)
Trigger: SWR

## Method as implemented
DETECTION (Online Methods, first page, right column):
- Signal and channels: "SWRs were identified on the basis of peaks in the LFP recorded from one channel from each tetrode in the CA3 and CA1 cell layers." Note CA3 as well as CA1 tetrodes; sorted units are not used for detection. LFP sampling rate not stated in this paper (Carr 2012, same rats/system, says 1.5 kHz).
- Filter / envelope / smoothing: "The raw LFP data were bandpass-filtered between 150–250 Hz, and the SWR envelope was determined using a Hilbert transform. The envelope was smoothed with a Gaussian (4-ms s.d.)." Filter type not stated.
- Threshold, minimum duration, combination: "We initially identified SWR events as sets of times when the smoothed envelope stayed above 3 s.d. of the mean for at least 15 ms on at least one tetrode." So: per-tetrode z-scoring, per-tetrode threshold, ANY tetrode triggers. The 15 ms applies to the run above 3 s.d. (the "threshold crossing event"), not to the whole event.
- Normalization period: "3 s.d. of the mean": the period over which mean and s.d. are computed is not stated.
- Bounds: "We defined the entire SWR as including times immediately before and after that threshold crossing event during which the envelope exceeded the mean."
- Merging across channels: "Overlapping SWRs were combined across tetrodes, so many events extended beyond the SWR seen on a single tetrode."
- Maximum duration: not stated. Merging/dropping of close (non-overlapping) events: not stated.
- Speed: main text p. 914: "To avoid confusing replay events and sequential firing during movement-related phase precession, we examined SWRs that occurred when rats were moving less than 2 cm s−1." Supplementary Information: "we used a 2 cm / second maximum velocity cutoff for all main analyses to exclude periods associated with movement". Which sample(s) of the event the speed is tested at: not stated. The phrasing ("we examined", "for all main analyses") reads as an analysis restriction rather than part of the detector; the paper does not say which.
- Brain state: none for detection.

ANALYSIS restrictions (not detection):
- Candidate replay events (main text p. 914; Online Methods): "Candidate replay events were defined as SWRs during which at least five place cells from the replayed environment fired at least one spike each." Place cell = peak rate >= 3 Hz outside SWRs.
- Awake vs quiescent split by immobility time (main text p. 914): awake = "times when rats had been immobile no more than 5 s"; quiescent = "immobile for 5 s or more".
- Theta/delta ratio (Online Methods; Supplementary): theta 6–12 Hz and delta 0.5–4 Hz Hilbert envelopes, 1-s s.d. Gaussian, ratio on the CA3 tetrode with highest variance; used only to describe the state at SWR times (Supplementary Fig. 15), not to select SWRs.
- Supplementary "Lack of remote replay outside of SWRs" control uses a 2 s.d. SWR threshold to exclude spikes; not the main detector.

## Inherited from
- Online Methods: "A distinct set of analyses of the data used in this study and the associated methods have been presented previously [20]" = Karlsson & Frank 2008, J Neurosci 28:14271 (Dropbox PDF, converted to a local copy). It gives the same detector verbatim, adding the kernel width: "The envelope was smoothed with a Gaussian with a SD of 4 ms and a width of 32 ms. SWRs were defined as contiguous periods when the smoothed SWR envelope stayed above 3 SDs of the mean for at least 15 ms on at least one tetrode." (p. 14273–14274). A 32 ms width at 4 ms SD is a truncation at +/-4 SD.
- The package's own `Karlsson_ripple_detector` cites this paper as its source (src/ripple_detection/detectors/_lfp.py, docstring references).

## Code
No code link in the paper (analysis in custom Matlab). A later Frank-lab code snapshot was checked below.

### Code search, September 2026

- **Lab convention.** [droumis/FFPhy @fce2048](https://github.com/droumis/FFPhy/tree/fce2048), `Functions/getripples.m` lines 29 and 116: with no cell filter it uses every tetrode with cells (CA1 and CA3) and counts an event on any one. It cannot be dated to 2009, and no code by Karlsson was found. CRCNS hc-6 lists no ripple files.

## Survey CSV discrepancies
- SWR electrodes (#): CSV "1"; paper uses one channel from EACH CA3 and CA1 tetrode, event on "at least one tetrode" (30-tetrode drive) → should be ">1" (the CSV may mean "one channel per tetrode", but other rows use ">1" for the same design).
- Min. Cells (#): CSV "5" matches the paper, but it is the candidate-replay (analysis) criterion ("at least five place cells from the replayed environment"), not an SWR detection criterion.
- Animal Speed: CSV "2" matches ("moving less than 2 cm s−1"); whether it is detection or analysis is not stated.
- Detection Notes: could record "any-tetrode, CA3+CA1 tetrodes, overlapping events merged".
Otherwise no discrepancies (3 SD, 4 ms smoothing, 150–250 Hz, 15 ms, no max, no combine threshold all match).

## Package mapping
Tier: A     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe:
```python
from ripple_detection import filter_ripple_band, Karlsson_ripple_detector
# lfps: (n_time, n_tetrodes), one channel per CA1 AND CA3 cell-layer tetrode, 1500 Hz
filtered = filter_ripple_band(lfps, sampling_frequency=1500)   # shipped Frank-lab kernel at 1500 Hz
events = Karlsson_ripple_detector(
    time, filtered, speed, 1500,
    zscore_threshold=3.0, minimum_duration=0.015, smoothing_sigma=0.004,
    speed_threshold=2.0,
)
# candidate replay (analysis): >= 5 place cells with >= 1 spike in [start_time, end_time] -- user code
```
Remaining deviations:
- Speed: package keeps an event if speed at its first and last sample is <= 2 cm/s; the paper says "< 2 cm/s" and does not say which samples (or whether it filters events or times).
- Threshold comparison: package ">=" 3 SD and bounds at z >= 0; paper "stayed above" / "exceeded the mean" (strict). Negligible.
- Smoothing kernel: package Gaussian truncated at 8 SD (64 ms support at 4 ms SD; `gaussian_smooth(truncate=8)`, not exposed by the detector); Karlsson & Frank 2008 used a 32 ms-wide kernel (+/-4 SD). Negligible numerically.
- Normalization period: package uses all valid samples of the recording passed in (optionally `normalization_mask`); the paper does not state the period.
- Minimum run: package counts samples (`minimum_sample_count`, 23 samples at 1500 Hz for 15 ms, rounded half up); paper says 15 ms. Negligible.
- Filter: paper does not state the filter; package at 1500 Hz uses the shipped Frank-lab 150–250 Hz kernel (plausibly the same lineage; not verified).
- Merging: package merges overlapping AND touching per-channel events (`merge_overlapping_ranges`); paper says "Overlapping SWRs were combined". Same in practice.
- The 5-place-cell candidate criterion has no public helper for LFP detectors (only `multiunit_HSE_detector(minimum_active_units=...)` counts units, and it detects from spikes); it is a few lines of user code.
Smallest package addition (if C): n/a
