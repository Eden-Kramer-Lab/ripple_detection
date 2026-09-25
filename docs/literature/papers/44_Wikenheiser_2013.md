# Wikenheiser 2013 — The balance of forward and backward hippocampal sequences shifts across behavioral states
Source: the extracted text (pdftotext of the Dropbox PDF, the NIH author manuscript, PMC3774294); title verified: yes (Wikenheiser & Redish, Hippocampus 23:22–29). Page numbers below are the manuscript's.
Trigger: SWR (ripple-power threshold crossings, each expanded to a fixed 150 ms window)

## Method as implemented

Detection (Methods, "Bayesian decoding of ripple events and theta cycles", p. 4):
- Signal, filter, power: "To detect ripple events, the LFP recorded from the pyramidal cell layer was band-pass filtered at ripple frequency (140–220 hz) and the power in this band was estimated via the Hilbert transform." Number of channels and whether channels were combined: not stated (the singular "the LFP" suggests one channel; inference). Whether "power" is the Hilbert amplitude or its square: not stated. Smoothing: not stated. Filter type: not stated. 12-tetrode hyperdrives, CA1 (p. 3).
- Threshold and window: "As in previous studies (e.g. Karlsson and Frank 2009), candidate ripple events were defined as 150 ms windows centered on times when ripple power exceeded a threshold (one standard deviation the baseline value)." The word "above" appears to be missing; the "baseline" (period for mean and SD) is not defined. Whether the window is centred on each supra-threshold sample, on each upward crossing, or on the peak is not stated.
- Rationale: "We chose a lower ripple threshold than some previous studies to err on the side of decoding many putative events and used a bootstrapping procedure ... to assess the significance of each event's sequence content."
- Merging: "Overlapping events were concatenated."
- Cells: "Only events in which at least 3 neurons fired a total of at least 5 spikes were included for analysis."
- Sleep (pre- and post-run rest): "To isolate times of sleep during pre- and post- periods, only events surrounded by at least 30 s of little or no motion (movement speed < 2 cm/s) and mean theta-delta ratio < 0 were included." Epoch definition: "Data were restricted to times when the theta-delta ratio was below 0 and movement speed was < 2 cm/s" (p. 3).
- Awake (run-LIA): "To ensure that theta sequences were not misclassified as ripple events within the run-LIA epoch, candidate events were included only if the rat's average speed during the event was < 2 cm/s and the average theta-delta ratio during the event was < 0." Epoch: "The run-LIA epoch was defined as times when the z-scored ratio of theta to delta (2–4 Hz) oscillatory power (Csicsvari et al., 1999; Jackson et al., 2006) fell below 0. The run-theta epoch consisted of times when the z-scored theta-delta ratio exceeded 0.5" (p. 3). Theta band "(6–10 Hz)" (Abstract).
- Minimum and maximum duration: none beyond the fixed 150 ms window (merged windows can be longer).
- Result: "Of 24,240 candidate ripples (across all sessions), 6796 (28%) were deemed significant" (p. 5).

Later analysis restrictions (not detection): decoding in 10 ms windows, "only time steps containing at least one spike were decoded"; significance by bootstrap on the cumulative sum of decoded-position differences. Theta-cycle sequences (fissure LFP; theta-delta ratio > 0.5 SD; ≥ 3 cells with ≥ 5 spikes) are a separate analysis, not SWR events.

## Inherited from
- "As in previous studies (e.g. Karlsson and Frank 2009)" is attached to the ripple-power thresholding. Karlsson & Frank 2009 used a 3 SD threshold on a 4 ms-smoothed Hilbert envelope with bounds at the mean (see findings/49_Karlsson_2009.md); it has no 150 ms window. So only the general approach is inherited; the window and the 1 SD threshold are this paper's.
- Theta-delta ratio: Jackson, Johnson & Redish 2006 (J Neurosci 26:12415; PMC6674885, read via WebFetch, so the quotes are as returned by the page summarizer): theta (6–10 Hz) and delta (2–4 Hz) Hilbert amplitudes "averaged across traces", then non-theta periods from the log-transformed theta/delta ratio relative to the session mean. Jackson 2006 detected SWRs differently (100–250 Hz, log amplitude > 2.5σ, crossings < 20 ms removed, events concatenated if < 100 ms apart); Wikenheiser does not cite it for ripple detection.
- Preprocessing "as described previously (Jackson et al., 2006; Wikenheiser and Redish, 2011)".

## Code
No code link.

### Code search, September 2026

- Searched GitHub (code, repositories, and the lab's and authors' accounts), Zenodo, Figshare and CRCNS/DANDI: no released code, same-lab code or event files. The Redish lab's public code is MClust; `awikenheiser` and `kkeus/Wikenheiser-Lab-Group` hold Open Ephys and tsd tools, with no ripple windowing.

## Survey CSV discrepancies
No value contradicts the paper (1 SD, 2 cm/s, 3 cells, 140–220 Hz, 150 ms window, theta/delta ratio). Notes:
- Detection Notes "150 ms window centered on crossing": the paper says "centered on times when ripple power exceeded a threshold", which does not say crossing; also add "overlapping windows concatenated" and "≥ 5 spikes total".
- Min. Cells 3 omits the "total of at least 5 spikes" half of the criterion.
- Speed 2 cm/s is applied two ways (mean over the event < 2 in run-LIA; ≥ 30 s of < 2 cm/s surrounding the event in rest); SWR electrodes "#N/A" is fair (not stated).

## Package mapping
Tier: B     Needs radiatum: n   Needs theta: y (theta/delta ratio)   Needs sleep scoring: y (a proxy: theta/delta < 0 and ≥ 30 s at < 2 cm/s, no true scoring)
Recipe:
```python
import numpy as np, pandas as pd
from ripple_detection import (filter_ripple_band, get_envelope, normalize_signal,
                              segment_boolean_series, merge_close_events)
# lfp: (n_time,) one CA1 pyramidal-layer channel; units: (n_time, n_units) spike counts
power = get_envelope(filter_ripple_band(lfp[:, None], fs, band=(140.0, 220.0)))[:, 0]  # or **2; smoothing not stated
z = normalize_signal(power)                                  # "baseline" undefined: whole session here
runs = np.asarray(segment_boolean_series(pd.Series(z > 1.0, index=time), minimum_duration=0.0))
# reading 1: window around each supra-threshold sample  -> [run_start - 0.075, run_end + 0.075]
# reading 2: window around each upward crossing         -> [run_start - 0.075, run_start + 0.075]
windows = np.column_stack([runs[:, 0] - 0.075, runs[:, 1] + 0.075])
events = merge_close_events(windows, 0.0)                    # "Overlapping events were concatenated"
# user code: >= 3 units with spikes and >= 5 spikes total in the window
# user code: theta/delta = z-scored log(theta 6-10 Hz amplitude / delta 2-4 Hz amplitude),
#            each band via a scipy band-pass (butter + sosfiltfilt) + get_envelope, averaged across channels
#            (filter_ripple_band's default 25 Hz transition makes it unsuitable for 2-4 and 6-10 Hz)
# rest epochs: keep events with speed < 2 and theta/delta < 0 over the surrounding 30 s (rolling test)
# run-LIA: keep events with mean speed < 2 and mean theta/delta < 0 inside the event
```
Remaining deviations:
- Fixed 150 ms window around threshold times is user code (a known gap), and which "times" it centres on is ambiguous.
- "Power" amplitude vs squared, smoothing, channel count and the "baseline" for the SD are not stated.
- Cell/spike criterion (≥ 3 units and ≥ 5 spikes) is user code; no detector here applies it (HSE's `minimum_active_units` only counts units).
- Theta/delta ratio and the 30 s surrounding-immobility test are user code (low-frequency filtering needs scipy directly); the package has only Carey's theta exclusion. The theta/delta channel is not stated.
- Speed: mean over the event (run-LIA) or a 30 s surround (rest), not the package's endpoint rule; computed in user code.
- `filter_ripple_band` at 140–220 Hz designs an equiripple FIR; the paper's filter is not stated.
Smallest package addition (if C): n/a. A fixed-window option (window of given length around each threshold crossing or peak) and a theta/delta-ratio helper would make it A apart from the spike-count and 30 s-surround criteria.
