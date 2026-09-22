"""The Carey, Tanaka & van der Meer joint ripple-power and multiunit candidate detector."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal import butter, filtfilt

from ripple_detection.core import (
    BoolArray,
    FloatArray,
    IntArray,
    _boolean_run_bounds,
    _unit_area_gaussian,
    get_envelope,
    normalize_signal,
    sample_count_within,
)
from ripple_detection.detectors._blocks import (
    _drop_short_blocks,
    _valid_blocks,
)
from ripple_detection.detectors._events import (
    _count_active_units,
    _exclude_long_events,
    _get_event_stats,
)
from ripple_detection.detectors._validation import (
    _validate_detector_inputs,
    _validate_duration_limits,
)


def _state_intervals(
    is_in_state: BoolArray, time: FloatArray, merge_gap: float, minimum_length: float
) -> IntArray:
    """Contiguous runs of a state, merged across gaps shorter than ``merge_gap``
    and dropped when shorter than ``minimum_length`` (vandermeerlab ``TSDtoIV``).
    Returns ``[start_index, stop_index]`` rows, inclusive."""
    bounds = _boolean_run_bounds(is_in_state)
    if len(bounds) == 0:
        return np.empty((0, 2), dtype=int)
    starts, stops = bounds[:, 0], bounds[:, 1] - 1
    if len(starts) > 1:
        gaps = time[starts[1:]] - time[stops[:-1]]
        merge = gaps < merge_gap
        keep_start = np.concatenate([[True], ~merge])
        keep_stop = np.concatenate([~merge, [True]])
        starts, stops = starts[keep_start], stops[keep_stop]
    long_enough = (time[stops] - time[starts]) > minimum_length  # TSDtoIV: strict
    return np.column_stack([starts[long_enough], stops[long_enough]])


def _contained_in_intervals(event_bounds: IntArray, intervals: IntArray) -> BoolArray:
    """True for each ``[start, stop]`` event lying inside some interval (vandermeerlab ``restrict``)."""
    if len(event_bounds) == 0:
        return np.empty(0, dtype=bool)
    if len(intervals) == 0:
        return np.zeros(len(event_bounds), dtype=bool)
    inside = (event_bounds[:, [0]] >= intervals[:, 0]) & (
        event_bounds[:, [1]] <= intervals[:, 1]
    )
    return np.asarray(inside.any(axis=1), dtype=bool)


def Carey_candidate_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    multiunit: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    *,
    speed_threshold: float = 4.0,
    low_threshold: float = 1.0,
    high_threshold: float = 3.0,
    minimum_duration: float = 0.020,
    minimum_active_units: int = 5,
    ripple_smoothing_sigma: float = 0.010,
    spike_kernel_sigma: float = 0.020,
    spike_cap: float = 2.0,
    baseline_sigma: float = 0.125,
    baseline_cap: float = 4.0,
    theta_lfp: ArrayLike | None = None,
    theta_band: tuple[float, float] = (6.0, 10.0),
    theta_threshold: float = 2.0,
    state_merge_gap: float = 0.050,
    state_minimum_length: float = 0.050,
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect candidate replay events from ripple power and multiunit activity jointly.

    The candidate-event detector of Carey, Tanaka & van der Meer 2019 [1]_
    (vandermeerlab ``GenCandidateEvents`` with its Hilbert ripple score
    ``OldWizard`` and multiunit score ``amMUA`` by Elyot Grant and A. Carey)
    [2]_. A ripple score and a multiunit score are combined as their
    **geometric mean**, so an event needs both a ripple and a population
    burst, then z-scored and segmented with two thresholds. Reimplemented
    from the code as read.

    - **Ripple score**: Hilbert envelope of the ripple-band signal, averaged
      across channels, smoothed with a Gaussian (10 ms SD, +/-3 SD), rescaled
      to mean 1.
    - **Multiunit score**: each unit's spike train smoothed with a unit-area
      Gaussian (20 ms SD, +/-5 SD) and capped at the peak that ``spike_cap``
      coincident spikes would give, so no single unit dominates; summed over
      units; a slow baseline (the sum capped at ``baseline_cap`` units'
      worth, smoothed with a 125 ms SD Gaussian) and one unit's cap are
      subtracted; divided by the mean and floored at zero.
    - **Joint score**: ``sqrt(ripple * multiunit)``, z-scored. The original
      first rescales it to mean 0.5, a positive factor the z-score removes,
      so that step is omitted. This combination is asymmetric. The multiunit score is
      floored at zero, so a ripple without a population burst cannot be a
      candidate. The ripple score is an envelope rescaled to mean 1 and is
      never zero, so a burst without a ripple can be. The joint score is
      therefore closer to "burst, weighted by ripple power" than to a
      symmetric conjunction. A candidate is a run strictly above
      ``low_threshold`` whose maximum is strictly above ``high_threshold``.
      Its sample count must meet ``minimum_duration`` under the package's
      duration rule.
    - **State**: a candidate is kept only if it lies entirely inside a
      low-speed interval (speed at or below ``speed_threshold``, runs merged across
      gaps under ``state_merge_gap`` and dropped under
      ``state_minimum_length``) and, when ``theta_lfp`` is given, inside a
      low-theta interval (z-scored theta-band envelope below
      ``theta_threshold``, same interval rules), and has at least
      ``minimum_active_units`` units with a spike inside it.

    The original works in samples at 2 kHz (kernel SDs of 40 and 250 samples,
    +/-60-sample ripple smoothing); the defaults here are those values in
    seconds. Its speed limit is 10 pixels/s in tracking units; the default
    here is the package's 4 cm/s. Missing samples (NaN in the LFP, the
    spikes, speed or ``theta_lfp``, or a gap in ``time``) split the recording
    into blocks; the scores, the segmentation and the state intervals run
    within each block, so no candidate spans a gap, and a candidate cut off by
    one is flagged in ``clipped_start`` and ``clipped_end``. With
    ``theta_lfp`` given, a block shorter than the theta filter's pad length
    (16 samples) is treated as missing, with a warning.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in seconds.
    filtered_lfps : array_like, shape (n_time, n_channels)
        Ripple-band-filtered LFP; the original uses 140-250 Hz on one channel.
    multiunit : array_like, shape (n_time, n_units)
        Spike counts (or indicators) per sample per unit; clusterless marks
        per tetrode work, with the per-unit cap then applying per tetrode.
    speed : array_like, shape (n_time,)
        Animal's running speed in cm/s.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Speed at or below which the animal is considered stopped. Default is 4.0.
    low_threshold, high_threshold : float, optional
        Boundary and peak thresholds on the z-scored joint score, the same
        two-threshold rule as ``Zugaro_ripple_detector``. Defaults 1 and 3
        (the original's ``DetectorThreshold`` and ``DetectorThreshold2``).
    minimum_duration : float, optional
        Minimum candidate duration in seconds, applied as an inclusive
        round-half-up sample count (``sample_count_within``); the original's
        ``RemoveIV`` compared elapsed time strictly. Default 0.020.
    minimum_active_units : int, optional
        Minimum number of units with a spike inside the candidate. Default 5.
    ripple_smoothing_sigma : float, optional
        Gaussian standard deviation in seconds of the ripple-score smoothing.
        Default 0.010.
    spike_kernel_sigma : float, optional
        Gaussian standard deviation in seconds of each unit's spike kernel.
        Default 0.020.
    spike_cap : float, optional
        Per-unit cap in coincident spikes. Default 2.
    baseline_sigma : float, optional
        Gaussian standard deviation in seconds of the slow baseline. Default
        0.125.
    baseline_cap : float, optional
        Baseline cap in units' worth. Default 4.
    theta_lfp : array_like, shape (n_time,), optional
        Raw LFP of a theta channel; when given, candidates during elevated
        theta are excluded. Default None (no theta exclusion).
    theta_band : tuple of (float, float), optional
        Theta pass-band in Hz, Butterworth of total order 4 (order 2 per edge, as MATLAB's `fdesign` 'N' counts it). Default (6, 10).
    theta_threshold : float, optional
        Theta-envelope z-score at or above which a period is excluded. Default 2.
    state_merge_gap, state_minimum_length : float, optional
        Interval rules for the low-speed and low-theta periods, in seconds.
        Defaults 0.050 and 0.050 (vandermeerlab ``TSDtoIV`` defaults).

    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.
    Returns
    -------
    candidate_times : pd.DataFrame
        One row per candidate, indexed by ``event_number``, with the package's
        statistics computed on the z-scored joint score, the speed statistics,
        and ``n_active_units``.

    References
    ----------
    .. [1] Carey, A. A., Tanaka, Y., & van der Meer, M. A. A. (2019). Reward
       revaluation biases hippocampal replay content away from the preferred
       outcome. Nature Neuroscience, 22(9), 1450-1459.
       doi:10.1038/s41593-019-0464-6
    .. [2] van der Meer lab, ``code-matlab/tasks/Alyssa_Tmaze/GenCandidateEvents.m``
       with ``beta/OldWizard.m``, ``beta/amMUA.m``, and ``beta/TSDtoIV2.m``.
       https://github.com/vandermeerlab/vandermeerlab/blob/82ba3fe29cc3912575b32a0fcdaaa1c4fe097231/code-matlab/tasks/Alyssa_Tmaze/GenCandidateEvents.m

    """
    _validate_duration_limits(minimum_duration, maximum_duration)
    multiunit = np.asarray(multiunit, dtype=float)
    if multiunit.ndim != 2:
        msg = f"multiunit must be a 2D array of shape (n_time, n_units), got shape {multiunit.shape}."
        raise ValueError(msg)
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    n_time = len(time)
    if multiunit.shape[0] != n_time:
        msg = f"Array length mismatch: multiunit has {multiunit.shape[0]} samples but time has {n_time}."
        raise ValueError(msg)
    signals = [filtered_lfps, multiunit, speed]
    theta_signal: FloatArray | None = None
    theta_filter = None
    if theta_lfp is not None:
        theta_signal = np.asarray(theta_lfp, dtype=float)
        if theta_signal.shape != (n_time,):
            msg = f"theta_lfp must have shape ({n_time},), got {theta_signal.shape}."
            raise ValueError(msg)
        signals.append(theta_signal)
        theta_filter = butter(
            2, np.asarray(theta_band) / (0.5 * sampling_frequency), btype="bandpass"
        )
    is_valid, blocks = _valid_blocks(time, *signals)
    if theta_filter is not None:
        # filtfilt needs strictly more samples than its default pad length
        padlen = 3 * max(len(theta_filter[0]), len(theta_filter[1]))
        blocks = _drop_short_blocks(blocks, is_valid, padlen + 1, "the theta filter")

    # ripple score (OldWizard, 'amplitude', 'wizard' kernel), rescaled to mean 1
    ripple_score = np.full(n_time, np.nan)
    for start, stop in blocks:
        envelope = get_envelope(filtered_lfps[start:stop]).mean(axis=1)
        ripple_score[start:stop] = gaussian_filter1d(
            envelope,
            ripple_smoothing_sigma * sampling_frequency,
            truncate=3.0,
            mode="constant",
        )
    ripple_score = ripple_score / np.nanmean(ripple_score)

    # multiunit score (amMUA)
    sigma_samples = spike_kernel_sigma * sampling_frequency
    spike_kernel = _unit_area_gaussian(sigma_samples, 5.0)
    cap = spike_cap / (sigma_samples * np.sqrt(2.0 * np.pi))
    baseline_kernel = _unit_area_gaussian(baseline_sigma * sampling_frequency, 12.0)
    summed = np.full(n_time, np.nan)
    baseline = np.full(n_time, np.nan)
    for start, stop in blocks:
        # convolve1d with zero padding equals np.convolve(..., "same") for these
        # odd symmetric kernels, and also works on a block shorter than the kernel
        block_sum = np.zeros(stop - start)
        for unit in multiunit[start:stop].T:
            block_sum += np.minimum(convolve1d(unit, spike_kernel, mode="constant"), cap)
        summed[start:stop] = block_sum
        baseline[start:stop] = convolve1d(
            np.minimum(baseline_cap * cap, block_sum), baseline_kernel, mode="constant"
        )
    mean_summed = np.nanmean(summed)
    if mean_summed <= 0:
        msg = "multiunit contains no spikes; cannot form a multiunit score."
        raise ValueError(msg)
    multiunit_score = np.maximum(0.0, (summed - baseline - cap) / mean_summed)
    if not np.any(multiunit_score > 0):
        msg = (
            "The multiunit score never rises above its baseline, so no candidate is "
            "possible: the population never fires more than spike_cap coincident "
            "spikes per unit above its slow rate. Check that multiunit holds spikes at "
            "sampling_frequency, not a rate, and that baseline_cap and spike_cap fit it."
        )
        raise ValueError(msg)

    joint = np.sqrt(ripple_score * multiunit_score)
    zscored = normalize_signal(joint)

    # two-threshold segmentation (TSDtoIV2) within each block: runs above the
    # edge, kept if the peak is above
    candidate_runs: list[tuple[int, int]] = []
    for start, stop in blocks:
        block_z = zscored[start:stop]
        for run_start, run_stop in _boolean_run_bounds(block_z > low_threshold):
            if block_z[run_start:run_stop].max() > high_threshold:
                candidate_runs.append((start + run_start, start + run_stop - 1))
    candidates = np.asarray(candidate_runs, dtype=int).reshape(-1, 2)
    if len(candidates):
        n_samples = candidates[:, 1] - candidates[:, 0] + 1
        candidates = candidates[sample_count_within(n_samples, time, minimum_duration)]

    # state restriction: contained in a low-speed (and low-theta) interval
    def _intervals(is_in_state: BoolArray) -> IntArray:
        return np.concatenate(
            [np.empty((0, 2), dtype=int)]
            + [
                start
                + _state_intervals(
                    is_in_state[start:stop],
                    time[start:stop],
                    state_merge_gap,
                    state_minimum_length,
                )
                for start, stop in blocks
            ]
        )

    if len(candidates):
        candidates = candidates[
            _contained_in_intervals(candidates, _intervals(speed <= speed_threshold))
        ]
    if len(candidates) and theta_filter is not None and theta_signal is not None:
        theta_envelope = np.full(n_time, np.nan)
        for start, stop in blocks:
            theta_envelope[start:stop] = get_envelope(
                filtfilt(*theta_filter, theta_signal[start:stop])
            )
        low_theta = _intervals(normalize_signal(theta_envelope) < theta_threshold)
        candidates = candidates[_contained_in_intervals(candidates, low_theta)]

    # minimum number of active units
    n_active = _count_active_units(multiunit, candidates)
    if len(candidates):
        keep = n_active >= minimum_active_units
        candidates, n_active = candidates[keep], n_active[keep]

    event_times = (
        np.column_stack([time[candidates[:, 0]], time[candidates[:, 1]]])
        if len(candidates)
        else np.empty((0, 2))
    )
    event_times, keep = _exclude_long_events(event_times, time, maximum_duration)
    n_active = n_active[keep]
    events = _get_event_stats(
        event_times, time, zscored, speed, minimum_duration=minimum_duration, blocks=blocks
    )
    events["n_active_units"] = n_active
    return events
