# Ólafsdóttir 2015 — Hippocampal place cells construct reward related sequences through unexplored space
Source: the extracted text (pdftotext of the Dropbox PDF); title verified: yes (Ólafsdóttir, Barry, Saleem, Hassabis & Spiers, eLife 4:e06063)
Trigger: other (sorted place-cell spiking events: participation within ≤300 ms, bounded by silence; no rate z-score, no ripple)

## Method as implemented

Detection ("spiking events"):
- Rule: "For each rest period, times where at least 15% of cells from a given template fired within 300 ms and were bound by at least 50 ms of silence were selected as 'spiking events' (for R1838, which had a lower cell yield than the other rats, a minimum of 4 cells were required to be active). If a single cell fired more than one spike within this period, the first spike was counted and other spikes disregarded." (Methods, "Preplay analysis", p. 12)
- Main text restates it (p. 1): "During a rest period before RUN1 (REST1) and after GOAL-CUE (REST2), spiking events—periods of 300 ms or less, where at least 15% of cells were active (Foster and Wilson, 2006; Diba and Buzsaki, 2007)—were analysed." So 300 ms is an upper limit on the event ("300 ms or less").
- Denominator: the cells of **a given template**. Events are therefore found per template. There are four arm templates (UCA, DCA, UUA, DUA) built from RUN2 ratemaps, plus stem templates from RUN1 for a follow-up.
- Template membership (Methods, "Template generation", p. 12): "Cells whose peak firing rate in the linearised ratemap were below 0.5 Hz, had less than five contiguous bins with rates above the mean firing rate of the cell, or whose spatial correlation between the first and second half of the RUN2 session was less than 0.3 were excluded."
- Silence: whose silence counts is not stated. Inference: the template cells' pooled spike train, following Foster & Wilson 2006 (see Inherited from).
- Signal: sorted single units (place cells) from CA1 tetrodes. No multiunit, no smoothing, no z-score.
- Brain state and speed for rest events: none stated. The rest enclosure periods are used whole. "During this period the animals' quiescence was assessed based on speed estimates" (Methods, p. 11) describes session handling, not an event criterion.
- Ripple: not a detection criterion. It is used after the fact to validate events: "These spiking events were associated with significantly higher power in the ripple spectrum (80–250 Hz) than other comparable periods" (p. 1).
  - The LFP measure (Methods, "Local field potential analysis", p. 15): "the LFP was band-pass filtered between 80 and 250 Hz ... An instantaneous measure of power was found by taking the squared complex modulus of the signal at each time point. This measure was then down sampled to 50 Hz ... and finally was smoothed with a boxcar filter of width 0.1 s."

Later analysis restrictions (not detection):
- GOAL-CUE (on-track) events only: "we only considered events recorded when an animal's velocity was below 10 cm/s and the animal was located within 20 cm (10% of track length) of the barrier." (p. 13)
- Bayesian decoding only: "posterior probability matrices were produced for events with ≥7 active cells using 5 ms non-overlapping time windows" (p. 14)
- Down-sampling controls equate template sizes and event counts (p. 13).

## Inherited from
The paper cites Foster & Wilson 2006 and Diba & Buzsáki 2007 for the event definition. I followed one hop, using the Dropbox PDFs:
- Foster & Wilson 2006 (Nature 440:680), Methods "Spike-train analysis": "A spike train was constituted from all spikes (from all cells in the probe sequence) that occurred during stopping periods ... This spike train was then broken between every pair of successive spikes separated by more than 50 ms, to form a large set of proto-events. Those proto-events in which at least one-third of the cells in the probe sequence fired at least one spike were then selected as events. The few events longer than 500 ms in duration were rejected".
  - This is where the "50 ms of silence" comes from: the pooled train of the probe-sequence (template) cells is split at gaps > 50 ms.
- Diba & Buzsáki 2007 (Nat Neurosci 10:1241): "when ≥30% or ≥5 of the place cells, whichever was greater, fired in 300-ms windows that were preceded by ≥60-ms silence".
  - This is the likely source of the "within 300 ms" window.
- Ólafsdóttir 2015 combines the two with 15% participation. Which reading was implemented is not stated. Reading (a): a silence-bounded segment of the template train, ≤ 300 ms long. Reading (b): a 300 ms window after ≥ 50 ms of silence.

## Code
None linked in the paper.

## Survey CSV discrepancies
Row 41.
- Animal Speed (cm/s): CSV "10". The rest-period events have no speed criterion. 10 cm/s applies only to the GOAL-CUE analysis ("velocity was below 10 cm/s and the animal was located within 20 cm ... of the barrier").
- Min. Cells (#): CSV "7". Detection requires "at least 15% of cells from a given template" (≥ 4 cells for R1838). "≥7 active cells" is only the inclusion rule for Bayesian decoding.
- Max Duration (ms): CSV "#N/A". The paper says events are "periods of 300 ms or less", so 300.
- Detection: CSV "MUA" is misleading. There is no MUA rate or z-score: the rule is a participation count on the template cells' sorted spikes, bounded by silence. Detection Notes ("If 15% of cells within 300 ms") omit that the denominator is the template's cells and that detection is per template.

## Package mapping
Tier: B (borderline C)     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe (reading (a); smoke-tested on `simulate_session` data at 1 kHz):
```python
import numpy as np, pandas as pd
from ripple_detection.core import segment_boolean_series

fs = 1000.0
# template_spikes: (n_time, n_template_cells) 1 ms counts for ONE template, REST period only
pooled = template_spikes.sum(axis=1)
silence = np.array(segment_boolean_series(pd.Series(pooled == 0, index=time), minimum_duration=0.050))
events = np.column_stack([silence[:-1, 1] + 1 / fs, silence[1:, 0] - 1 / fs])  # spans between silences
events = events[(events[:, 1] - events[:, 0]) <= 0.300]                           # "300 ms or less"
min_cells = int(np.ceil(0.15 * template_spikes.shape[1]))                         # 4 for R1838
events = events[n_active_units(template_spikes, time, events) >= min_cells]       # user code
# repeat for each of the four templates
```
Remaining deviations:
- The implemented reading of "fired within 300 ms and were bound by at least 50 ms of silence" is unknown: (a) above, or (b) the Diba-style 300 ms window after 60 ms of silence. Whose silence counts is also unknown: template cells (assumed here) or all cells.
- Silence of "at least 50 ms": `segment_boolean_series` counts samples (50 at 1 kHz). Foster & Wilson split at gaps "more than 50 ms", a strict inequality differing by one sample.
- Rounding of 15% to a cell count is not stated.
- Detection is per template, so the same spikes can form events for several templates. The package has no notion of this; each template is its own run.
- No detector covers this. The only package piece used is `segment_boolean_series`, and the rest is numpy. That is why it is borderline C.
Smallest package addition (if C): `silence_bounded_events(spikes, time, minimum_silence=0.05, maximum_duration=0.3, minimum_active_units=None, minimum_active_fraction=None)`. It would split the pooled train of the given units at silent gaps, keep segments within the duration limit, and apply the participation test. Together with a public `count_active_units`, it would also serve Foster & Wilson 2006 and Diba & Buzsáki 2007.
