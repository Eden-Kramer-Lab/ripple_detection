"""The Carey, Tanaka & van der Meer joint ripple-power and multiunit candidate detector."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal import butter, oaconvolve, sosfiltfilt

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    BoolArray,
    FloatArray,
    IntArray,
    _boolean_run_bounds,
    _is_immobile,
    _unit_area_gaussian,
    get_envelope,
    normalize_signal,
    sample_count_within,
)
from ripple_detection.detectors._blocks import (
    _contiguous_valid_blocks,
    _drop_short_blocks,
    _mask_invalid,
    _reject_flat_channels,
    _valid_blocks,
)
from ripple_detection.detectors._events import (
    _count_active_units,
    _exclude_long_events,
    _get_event_stats,
)
from ripple_detection.detectors._validation import (
    _check_band,
    _check_gap,
    _check_minimum_active_units,
    _check_positive,
    _check_smoothing_sigma,
    _check_thresholds,
    _validate_detector_inputs,
    _validate_duration_limits,
    _validate_multiunit,
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


_SPIKE_COST_RATIO = 20_000
"""Adding the kernel at one spike costs about as much as 20,000 samples times
taps of direct convolution (3.6 us against 0.18 ns, measured at 1500 Hz)."""


def _convolve_spikes(counts: FloatArray, kernel: FloatArray) -> FloatArray:
    """``convolve1d(counts, kernel, mode="constant")`` for an odd symmetric kernel.

    A spike train is mostly zeros, so when adding the kernel at each spike
    is cheaper than convolving every sample, that is what this does: seven
    times faster for 100 units at a few hertz, sampled at 1500 Hz. The sum
    at a sample reached by two spikes is taken in a different order than the
    direct convolution's, so the two agree to rounding (1e-16), not bit for
    bit. Dense trains are convolved directly.
    """
    spikes = np.flatnonzero(counts)
    if len(spikes) * _SPIKE_COST_RATIO >= len(counts) * len(kernel):
        return np.asarray(convolve1d(counts, kernel, mode="constant"), dtype=float)
    radius = len(kernel) // 2
    padded = np.zeros(len(counts) + 2 * radius)
    for spike in spikes:
        padded[spike : spike + len(kernel)] += counts[spike] * kernel
    return padded[radius : radius + len(counts)]


def _theta_envelope(
    theta_lfp: FloatArray,
    time: FloatArray,
    sampling_frequency: float,
    theta_band: tuple[float, float],
) -> FloatArray:
    """Theta-band envelope of the theta channel, NaN where it cannot be formed.

    The original filters the whole theta recording at once (vandermeerlab
    ``FilterLFP``: a fourth-order Butterworth band-pass applied with
    ``filtfilt``, then the Hilbert envelope in ``LFPpower``). Here the channel
    is filtered over each of its own runs of finite samples, split also at
    gaps in ``time``, and not over the blocks the detector's other inputs
    define, so a NaN in the LFP or the spikes does not restart the filter. A run
    with no more samples than the filter's pad length cannot be filtered and
    is NaN, which makes its samples missing for the detector, with a warning.
    """
    sos = butter(
        2, np.asarray(theta_band) / (0.5 * sampling_frequency), btype="bandpass", output="sos"
    )
    padlen = 3 * (2 * len(sos) + 1)  # sosfiltfilt's default pad; a run needs more than this
    finite = np.isfinite(theta_lfp)
    envelope = np.full(theta_lfp.size, np.nan)
    if not np.any(finite):
        return envelope  # every sample missing; _valid_blocks reports it
    runs = _drop_short_blocks(
        _contiguous_valid_blocks(finite, time), finite, padlen + 1, "the theta filter"
    )
    for start, stop in runs:
        envelope[start:stop] = get_envelope(sosfiltfilt(sos, theta_lfp[start:stop]))
    return envelope


@explain_call_errors
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
    spikes or ``theta_lfp``, or a gap in ``time``) split the recording into
    blocks; the scores, the segmentation and the state intervals run within
    each block, so no candidate spans a gap, and a candidate cut off by one is
    flagged in ``clipped_start`` and ``clipped_end``. A NaN in speed is an
    unknown speed: it splits no block and is not low speed, so it breaks a
    low-speed interval unless the interval merge (``state_merge_gap``) bridges
    it. With ``theta_lfp`` given, the theta channel is filtered over each of
    its own runs of finite samples, as the original filtered the whole
    recording, so a NaN in the spikes does not restart the theta filter; a
    run of
    theta samples no longer than the filter's pad length (15) cannot be
    filtered and is treated as missing, with a warning. The filter's
    transient lasts about three time constants, 0.4 s for a 6-10 Hz band, so
    the theta state within that distance of a theta gap or the record edges
    is unreliable; the original has the same transient at its record edges.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in seconds.
    filtered_lfps : array_like, shape (n_time, n_channels)
        Ripple-band-filtered LFP; the original uses 140-250 Hz on one channel.
    multiunit : array_like, shape (n_time, n_units)
        Spike counts (or indicators) per sample per unit, non-negative whole
        numbers, not a rate; clusterless marks per tetrode work, with the
        per-unit cap then applying per tetrode.
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
        theta are excluded, as the original's theta restriction does. It
        needs a channel the other inputs do not, so it is optional here.
        Default None (no theta exclusion).
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
    _check_thresholds("low_threshold", low_threshold, "high_threshold", high_threshold)
    _check_smoothing_sigma(
        ripple_smoothing_sigma=ripple_smoothing_sigma,
        spike_kernel_sigma=spike_kernel_sigma,
        baseline_sigma=baseline_sigma,
    )
    _check_positive(spike_cap=spike_cap, baseline_cap=baseline_cap)
    _check_gap(state_merge_gap=state_merge_gap, state_minimum_length=state_minimum_length)
    if not np.isfinite(theta_threshold):
        msg = f"theta_threshold must be finite, got {theta_threshold}."
        raise ValueError(msg)
    multiunit = np.asarray(multiunit, dtype=float)
    _validate_multiunit(multiunit)
    _check_minimum_active_units(minimum_active_units, multiunit.shape[1])
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    if theta_lfp is not None:
        _check_band("theta_band", theta_band, sampling_frequency)
    n_time = len(time)
    if multiunit.shape[0] != n_time:
        msg = f"Array length mismatch: multiunit has {multiunit.shape[0]} samples but time has {n_time}."
        raise ValueError(msg)
    signals = [filtered_lfps, multiunit]
    theta_envelope: FloatArray | None = None
    if theta_lfp is not None:
        theta_signal = np.asarray(theta_lfp, dtype=float)
        if theta_signal.shape != (n_time,):
            msg = f"theta_lfp must have shape ({n_time},), got {theta_signal.shape}."
            raise ValueError(msg)
        theta_envelope = _theta_envelope(theta_signal, time, sampling_frequency, theta_band)
        signals.append(theta_envelope)
    is_valid, blocks = _valid_blocks(time, *signals, minimum_duration=minimum_duration)
    _reject_flat_channels(filtered_lfps, blocks, "filtered_lfps")

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
        # zero padding at the block edges, as np.convolve(..., "same") pads, for
        # these odd symmetric kernels; a block shorter than a kernel works too
        block_sum = np.zeros(stop - start)
        for unit in multiunit[start:stop].T:
            block_sum += np.minimum(_convolve_spikes(unit, spike_kernel), cap)
        summed[start:stop] = block_sum
        # FFT convolution: the 250 ms baseline kernel has thousands of taps,
        # and the capped sum is dense (equal to convolve1d to rounding, 1e-15)
        baseline[start:stop] = oaconvolve(
            np.minimum(baseline_cap * cap, block_sum), baseline_kernel, mode="same"
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
            _contained_in_intervals(
                candidates, _intervals(_is_immobile(speed, speed_threshold))
            )
        ]
    if len(candidates) and theta_envelope is not None:
        theta_z = normalize_signal(_mask_invalid(theta_envelope, is_valid))
        candidates = candidates[
            _contained_in_intervals(candidates, _intervals(theta_z < theta_threshold))
        ]

    # minimum number of active units
    n_active = _count_active_units(multiunit, candidates)
    keep = n_active >= minimum_active_units
    candidates, n_active = candidates[keep], n_active[keep]

    event_times = np.column_stack([time[candidates[:, 0]], time[candidates[:, 1]]])
    event_times, keep = _exclude_long_events(event_times, time, maximum_duration)
    n_active = n_active[keep]
    events = _get_event_stats(
        event_times, time, zscored, speed, minimum_duration=minimum_duration, blocks=blocks
    )
    events["n_active_units"] = n_active
    return events
