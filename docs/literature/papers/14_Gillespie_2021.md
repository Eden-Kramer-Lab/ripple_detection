# Gillespie 2021 — Hippocampal replay reflects specific past experiences rather than a plan for subsequent choice
Source: the extracted text (pdftotext of the Dropbox PDF); title verified: yes (Gillespie AK, Astudillo Maya DA, Denovellis EL, ..., Roumis DK, Eden UT, Frank LM, Neuron 109:3149–3163, doi:10.1016/j.neuron.2021.07.029)
Trigger: SWR (MUA-based detection as a control)

## Method as implemented
SWR DETECTION (STAR Methods, "SWR detection", p. e3):
- "SWRs were detected using a consensus method based on the envelope of the ripple filtered (150-250 Hz) trace, smoothed and combined across all CA1 cell layer tetrodes (Kay et al., 2016)." → Kay consensus trace over all CA1 cell-layer tetrodes. Smoothing width not stated here (Kay 2016: 4 ms).
- Threshold, normalization, minimum duration: "Events were detected as deviations in the consensus trace exceeding 2 SD above total session baseline (mean) for at least 15 ms." The 15 ms applies to the above-threshold run; mean/SD over the whole session.
- Bounds: "Each event start was defined as the time at which the consensus trace first crossed session baseline before the threshold crossing; event end was the time at which the trace returned to session baseline after the event."
- Speed: "SWRs were only detected during immobility (velocity < 4 cm/s)." Which samples: not stated in the text (see Code: start and end).
- LFP: one channel per tetrode, 0.1–300 Hz, referenced to a corpus callosum tetrode (p. e2); sampling rate not stated. Spikes: > 100 µV on any channel.
- Maximum duration, close-event rule: not stated.
- Main text p. 4: "we identified SWRs using a permissive threshold to ensure that we were examining a large fraction of the actual replay events".

MUA DETECTION (control; STAR Methods, same paragraph):
- "For multiunit activity (MUA) event detection, a histogram of spike counts was constructed using 1 ms bins; all spikes > 100 mV [µV] on tetrodes in CA1 cell layer were included. The MUA trace was smoothed with a Gaussian kernel (15 ms SD), and the mean and standard deviation of this trace during immobility periods (< 4 cm/s) were calculated (Davidson et al., 2009). Deviations in the multiunit trace exceeding 3 SD above the mean during immobility periods were considered MUA events; event start and end were defined as the times before and after the event at which the trace returned to the mean immobility MUA rate."
- MUA minimum duration: not stated. MUA speed criterion: events "during immobility periods" (samples not stated).
- Main text p. 4: "we verified that our conclusions were unchanged by using a multiunit activity-based event detection strategy (Davidson et al., 2009; Pfeiffer and Foster, 2013) instead of a SWR-based one".

ANALYSIS (not detection): SWR amplitude = "the maximum threshold (in units of SD) at which the event would still be detected"; tetrode engagement = fraction of CA1 tetrodes with spikes in the event; replay content from a clusterless state-space decoder. No cell-count or duration restriction on which SWRs are decoded is stated.

## Inherited from
Kay et al. 2016 (a local copy): CA1 tetrodes filtered 150–250 Hz, "squared and summed across tetrodes", Gaussian σ = 4 ms, square root, > 2 s.d. of the epoch mean for >= 15 ms, head speed < 4 cm/s. Note Kay squares the filtered signal, Gillespie says "based on the envelope". Davidson et al. 2009 for the MUA approach (immobility-normalized population rate).

## Code
Paper: "All code used for analysis of this data is publicly available at https://zenodo.org/record/5140706" = LorenFrankLab/Gillespie_Neuron_2021 v1.0 (downloaded to a local copy). Data: DANDI 000115. README: processing used Frank-lab Matlab repos trodes2ff_shared and filterframework_shared (not in the archive).
- SWR detection itself is NOT in the archive: scripts load precomputed `ca1rippleskons` structures (`dfs_makeFFripdecodes.m`, `plot_behavior_example.m`), with fields `starttime`, `endtime`, `maxthresh`. So envelope-vs-squared-signal and kernel details cannot be verified from this code.
- Speed rule, verified (`dfa_makeFFripdecodes.m` line 98, with `timefilter = {'ag_get2dstate', '($immobility == 1)','immobility_velocity',4,'immobility_buffer',0}` in `dfs_makeFFripdecodes.m`): `valrips = ~isExcluded(starttime, excludeperiods) & ~isExcluded(endtime, excludeperiods) & ...` → an SWR is kept if its START and END times are in immobility (velocity threshold 4), i.e. the package's endpoint rule, applied when building the decoded-SWR set. Events outside the decoded posterior time range are also dropped. (`ag_get2dstate` itself is not in the archive, so "< 4" vs "<= 4" is unverified.)
- MUA: `utilities/getMUAtrace.m` bins pooled CA1 spikes in 2 ms bins and smooths with `gaussian(smoothingwidth/timebin, 100)` where `smoothingwidth = .005` and the comment says "15ms kernel width" — i.e. 5 ms SD, not the "1 ms bins ... 15 ms SD" of the paper. But this function is not called anywhere in the archive, and the MUA events (`muaripples` / `muadecodesv3`) were produced upstream, so which smoothing produced the published MUA events is unknown.

## Survey CSV discrepancies
- SWR smooth (ms): CSV "4"; not stated in Gillespie ("smoothed"), inherited from Kay 2016 (4 ms). Correct by inheritance.
- Min. Duration (ms): CSV "15" is correct for SWRs; the MUA minimum duration is not stated (the column cannot hold both).
- MUA smooth (ms): CSV "15" matches the text; the archive's (uncalled) `getMUAtrace.m` uses 5 ms SD — unresolved.
Otherwise no discrepancies (SWR 2 SD, MUA 3 SD, 4 cm/s, >1 electrodes, 150–250 Hz, no min cells, no max, no combine).

## Package mapping
Tier: A     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe:
```python
import numpy as np
from ripple_detection import filter_ripple_band, Kay_ripple_detector, multiunit_HSE_detector
filtered = filter_ripple_band(ca1_lfps, sampling_frequency=fs)   # one channel per CA1 cell-layer tetrode
swrs = Kay_ripple_detector(time, filtered, speed, fs,
                           zscore_threshold=2.0, minimum_duration=0.015,
                           smoothing_sigma=0.004, speed_threshold=4.0)   # whole-session z-score (default)
# MUA control: spike counts (n_time_1ms, n_ca1_tetrodes) on a 1 ms grid, speed resampled to it
immobile = speed_1ms < 4.0
mua = multiunit_HSE_detector(time_1ms, counts_1ms, speed_1ms, 1000,
                             zscore_threshold=3.0, smoothing_sigma=0.015,
                             normalization_mask=immobile,   # mean/SD during immobility; bounds at immobility mean
                             minimum_duration=0.0,          # none stated (0 -> one sample)
                             speed_threshold=4.0)
```
Remaining deviations:
- SWR speed: package endpoint rule (<= 4 at first and last sample) matches the archived code's start/end test; the text says "< 4 cm/s".
- Kay trace: package squares the Hilbert envelope; Gillespie says "based on the envelope" (consistent), Kay 2016 squares the filtered signal; the upstream Matlab code is not available to check. The two traces differ by a factor sqrt(2) after 4 ms smoothing (simulated check in 13_Denovellis_2021.md: identical events), so this does not matter.
- Smoothing width (4 ms) is inherited, not stated.
- MUA minimum duration not stated; `minimum_duration=0.0` (one sample) or the package default 0.015 are both guesses. MUA smoothing: text 15 ms SD vs archive utility 5 ms SD (unresolved).
- MUA population rate: package sums per-unit columns then smooths; the paper pools spikes across tetrodes into one histogram — identical (sum of counts). The package z-scores with `normalization_mask`, whose mean is the immobility mean, so "returned to the mean immobility MUA rate" is z = 0 as in the package.
- MUA speed rule: endpoint rule; text "during immobility periods" (samples not stated).
Smallest package addition (if C): n/a
