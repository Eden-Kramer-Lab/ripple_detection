# Denovellis 2021 — Hippocampal replay of experience at real-world speeds
Source: the extracted text (pdftotext of the Dropbox PDF); title matched (Denovellis EL, Gillespie AK, Coulter ME, Sosa M, Chung JE, Eden UT, Frank LM, eLife 10:e64505, doi:10.7554/eLife.64505)
Trigger: SWR (with a multiunit-HSE control)

[Paper](https://doi.org/10.7554/eLife.64505) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [denovellis-code](../sources.md#denovellis-code).

## Method as implemented

SWR DETECTION (Materials and methods, "SWR detection", p. 23 of 33):

- "Sharp wave ripples were detected using the same method as in Kay et al., 2016. Each CA1 LFP was obtained by downsampling the original 30 kHz electrical potential to 1.5 kHz and bandpass filtering between 0.5 Hz and 400 Hz. This was further bandpass filtered for the ripple band (150–250 Hz), squared, and then summed across tetrodes—forming a single population trace over time. This trace was smoothed with a Gaussian with a 4 ms standard deviation and the square root of this trace was taken to get an estimate of the population ripple band power." → Kay consensus trace (sum over tetrodes of the squared FILTERED signal, smoothed, square root).

- Normalization, threshold, minimum duration, speed: "Candidate SWR times were found by z-scoring the population power trace of an entire recording session and finding times when the z-score exceeded two standard deviations for a minimum of 15 ms and the speed of the animal was less than 4 cm/s." The 15 ms applies to the above-threshold run.

- Bounds: "The SWR times were then extended before and after the threshold crossings to include the time until the population trace returned to the mean value."

- "The code used for ripple detection can be found at https://github.com/Eden-Kramer-Lab/ripple_detection (Denovellis, 2021b)." (The reference-list entry "Denovellis 2021b" is actually loren_frank_data_processing, Zenodo 10.5281/zenodo.5523666 — a citation mix-up in the paper.)

- Maximum duration: not stated. Close-event rule: not stated.

ANALYSIS restriction:

- "We only analyzed SWRs with spikes from at least two tetrodes." (p. 23)

MULTIUNIT CONTROL (Materials and methods, "Identifying events of high multiunit activity", p. 27):

- "We identified times of high multiunit activity when the animal was immobile as a control analysis. Our approach was similar to Davidson et al., 2009. High multiunit periods were identified as times when the z-scored multiunit population spiking activity was greater than two standard deviations for at least 15 ms and the animal was moving at speeds less than 4 cm/s."

- Multiunit = threshold crossings > 60 µV on any wire of a tetrode (p. 22). Smoothing, normalization period and bounds of the multiunit trace: not stated in the text (see Code).

## Inherited from

- Kay et al. 2016 (Dropbox PDF → a local copy), "SWR detection": "LFPs from all available CA1 cell layer tetrodes were filtered between 150–250 Hz, then squared and summed across tetrodes. This sum was smoothed with a Gaussian kernel (σ = 4 ms) and the square root of the smoothed sum was analysed. SWRs were detected when the signal exceeded 2 s.d. of the recording epoch mean for at least 15 ms." Kay also required >= 3 CA1 cell-layer recordings and head speed < 4 cm/s.

- Davidson et al. 2009 for the multiunit control.

## Code

Analysis repo named in the paper: https://github.com/Eden-Kramer-Lab/replay_trajectory_paper (cloned to a local copy, HEAD f2d2b3c, 2021-11-03). Opened because the text's "squared" disagrees with the package's envelope-based Kay trace.

- `environment.yml` pins `ripple_detection == 0.1.8.dev0` (this repo's tag 0.1.8.dev0, commit 71298f0, 2018-07-16).

- `src/load_data.py::get_ripple_times` calls `Kay_ripple_detector(time, ripple_lfps.values, speed.values, 1500, zscore_threshold=2.0, close_ripple_threshold=np.timedelta64(0, 'ms'), minimum_duration=np.timedelta64(15, 'ms'))` on the RAW LFPs (0.1.8.dev0 filtered internally). Tetrodes: those with `validripple == 1` in the tetrode table if annotated, otherwise all CA1, CA2 and CA3 tetrodes (`brain_areas=['CA1','CA2','CA3']`) — the text says CA1.

- ripple_detection 0.1.8.dev0 (`git show 0.1.8.dev0:ripple_detection/detectors.py`, `core.py`): `filter_ripple_band` = `remez` equiripple FIR, 101 taps, 25 Hz transition bands, `filtfilt`; Kay trace = `np.sum(filtered ** 2)` of the FILTERED signal (no Hilbert envelope; the envelope was added on 2020-10-14, commit 34958cf) → `gaussian_smooth(0.004, truncate=8)` → `sqrt` → scipy `zscore` over all non-NaN samples → runs >= 2 with `end_time >= start_time + 15 ms` extended to z >= 0 → `exclude_movement`: speed at the start AND end sample <= 4. NaN samples were dropped and concatenated. So the code matches the paper text (squared filtered signal), not the current package (squared envelope).

- Multiunit control (`load_data`): 2 ms grid (`SAMPLING_FREQUENCY = 500`), per-tetrode spike indicators, `multiunit_HSE_detector(time, multiunit_spikes, speed, 500, minimum_duration=15 ms, zscore_threshold=2.0, close_event_threshold=0)`; in 0.1.8.dev0 the population rate = mean over tetrodes x fs, Gaussian `smoothing_sigma=0.015` (default), scipy z-score over the whole epoch, bounds at the mean, endpoint speed rule <= 4.

## Reproduction limits

The method-specific qualifications above apply. The executable recipe documents its assumptions about unstated parameters, state scoring and simulation stand-ins.

## Package mapping

Executable example: `denovellis_2021` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.
