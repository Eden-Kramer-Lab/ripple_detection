"""The FMAToolbox FindRipples two-threshold detector."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ripple_detection.core import (
    BoolArray,
    FloatArray,
    _boolean_run_bounds,
    _is_immobile_at_endpoints,
    normalize_signal,
    sample_count_within,
)
from ripple_detection.detectors._blocks import (
    _drop_short_blocks,
    _normalization_mask_over_valid,
    _reject_flat_channels,
    _valid_blocks,
)
from ripple_detection.detectors._events import (
    _get_event_stats,
)
from ripple_detection.detectors._validation import (
    _check_non_negative,
    _check_thresholds,
    _check_whole_number,
    _validate_detector_inputs,
    _validate_duration_limits,
)


def _zugaro_smoothing_window(sampling_frequency: float) -> int:
    """Moving-average length of the FindRipples power trace: 11 samples at 1250 Hz,
    scaled with the rate and kept odd so the filter is zero-phase."""
    window = round(sampling_frequency / 1250.0 * 11.0)
    return window + 1 if window % 2 == 0 else window


def _two_threshold_events(
    zscored: FloatArray,
    time: FloatArray,
    low_threshold: float,
    high_threshold: float,
    minimum_inter_ripple_interval: float,
    minimum_duration: float,
    maximum_duration: float | None,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Segment a normalized power trace with the FindRipples two-threshold rule.

    Follows the segmentation rule of FMAToolbox ``FindRipples``:

    1. Candidate events are runs strictly above ``low_threshold``. An event
       starts at the last sample *at or below* the threshold before the run and
       ends at the run's last sample, as the original's ``diff``-based
       crossing search does. A run touching the block's first or last sample
       lacks one crossing; the original discards it, this package keeps it
       and flags it ``clipped_start`` or ``clipped_end``.
    2. Consecutive candidates are merged, one neighbor per pass, while the
       gap between them is under ``minimum_inter_ripple_interval`` and the
       merged span is under ``maximum_duration``. This is deliberately not
       :func:`~ripple_detection.core.merge_close_events`: that helper works on
       times with a tolerance at the boundary and an inclusive span cap, while
       FindRipples compares sample indices with strict inequalities. Keeping
       this loop is what keeps the detector faithful to the algorithm it is
       named for.
    3. A candidate is kept only if its maximum is strictly above
       ``high_threshold``.
    4. Candidates whose sample count (first to last sample, inclusive) is
       below ``minimum_duration`` or above ``maximum_duration`` are dropped.
       The package's duration rule (``sample_count_within``: round-half-up
       sample counts, inclusive limits) replaces the original's elapsed-time
       comparison; the two differ by one sample at an exact limit.

    Parameters
    ----------
    zscored : ndarray, shape (n_time,)
        Normalized smoothed power for one contiguous block.
    time : ndarray, shape (n_time,)
        Sample timestamps in seconds.
    low_threshold, high_threshold : float
        Boundary and peak thresholds in standard deviations.
    minimum_inter_ripple_interval, minimum_duration : float
        In seconds.
    maximum_duration : float or None
        In seconds; None removes the ceiling from the merge and from step 4.

    Returns
    -------
    event_times : ndarray, shape (n_events, 2)
        ``[start_time, end_time]`` per event.
    peak_times : ndarray, shape (n_events,)
        Time of the maximum of ``zscored`` within each event.
    clipped : ndarray of bool, shape (n_events, 2)
        Whether the event lacks its rising or its falling crossing, that is,
        its run began on the block's first sample or ended on its last. An
        event whose run began on the second sample has its crossing on the
        first and is not clipped, although it starts there.

    """
    zscored = np.asarray(zscored, dtype=float)
    time = np.asarray(time, dtype=float)
    empty = (np.empty((0, 2)), np.empty(0), np.empty((0, 2), dtype=bool))
    runs = _boolean_run_bounds(zscored > low_threshold)
    if len(runs) == 0:
        return empty
    # an event spans from the last sample at or below the low threshold before
    # the run to the run's last sample; a run that begins on the block's first
    # sample starts there instead, and is clipped
    events = np.column_stack([np.maximum(runs[:, 0] - 1, 0), runs[:, 1] - 1])
    clipped = np.column_stack([runs[:, 0] == 0, runs[:, 1] == len(zscored)])

    while len(events) > 1:
        gap = time[events[1:, 0]] - time[events[:-1, 1]]
        merged_span = time[events[1:, 1]] - time[events[:-1, 0]]
        to_merge = gap < minimum_inter_ripple_interval
        if maximum_duration is not None:
            to_merge &= merged_span < maximum_duration
        if not np.any(to_merge):
            break
        padded = np.concatenate([[False], to_merge])
        run_starts = np.flatnonzero(~padded[:-1] & padded[1:])
        events[run_starts, 1] = events[run_starts + 1, 1]
        clipped[run_starts, 1] = clipped[run_starts + 1, 1]
        events = np.delete(events, run_starts + 1, axis=0)
        clipped = np.delete(clipped, run_starts + 1, axis=0)

    peaks = np.array(
        [start + int(np.argmax(zscored[start : stop + 1])) for start, stop in events],
        dtype=int,
    )
    keep = zscored[peaks] > high_threshold
    n_samples = events[:, 1] - events[:, 0] + 1
    keep &= sample_count_within(n_samples, time, minimum_duration, maximum_duration)
    events, peaks, clipped = events[keep], peaks[keep], clipped[keep]
    return np.column_stack([time[events[:, 0]], time[events[:, 1]]]), time[peaks], clipped


def Zugaro_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    *,
    speed_threshold: float = 4.0,
    low_threshold: float = 2.0,
    high_threshold: float = 5.0,
    minimum_inter_ripple_interval: float = 0.030,
    minimum_duration: float = 0.020,
    maximum_duration: float | None = 0.100,
    smoothing_window: int | None = None,
    normalization_mask: ArrayLike | None = None,
) -> pd.DataFrame:
    """Detect ripples with the FMAToolbox ``FindRipples`` two-threshold algorithm.

    The algorithm of Hajime Hirase as implemented by Michaël Zugaro in
    FMAToolbox [1]_ (``FindRipples``), carried into buzcode as
    ``bz_FindRipples`` [2]_ and into neurocode [3]_. The ripple-band signal is
    squared, summed across channels, smoothed with a short moving average and
    z-scored. A **low** threshold bounds each event, and an event is kept only
    if its **peak** exceeds a **high** threshold. Neighboring events closer
    than a minimum interval are merged, and events outside a duration range
    are discarded. Csicsvari et al. 1999 [4]_ describe the summed rms-power
    thresholding this rule descends from.

    The same two-threshold rule appears independently in the van der Meer lab's
    ``getSWR`` (vandermeerlab, ``code-matlab/tasks/Replay_Analysis/getSWR.m``):
    140-200 Hz, the Hilbert envelope rather than the squared signal, boundaries
    at 2 SD, merge within 20 ms applied before a 20 ms minimum, peak above
    5 SD. That variant is not implemented separately. It is this detector with
    ``low_threshold=2``, ``high_threshold=5``,
    ``minimum_inter_ripple_interval=0.02``, ``minimum_duration=0.02`` and no
    maximum, apart from using the envelope rather than the squared signal.

    This is a reimplementation from the algorithm, not a transcription. The
    original is GPL-3. Four departures, each documented per parameter below.
    The package's endpoint speed rule is applied. The peak is the maximum of
    the normalized power, not the trough of a single filtered channel.

    Missing samples are handled block-wise, as in every detector here, so
    smoothing and segmentation never cross a gap; a NaN in ``speed`` is an
    unknown speed, which fails the endpoint rule but splits no block. A run that touches a gap or
    the record edge lacks one of its two crossings; the original drops it,
    this package keeps it and flags it in ``clipped_start`` or
    ``clipped_end``, so ``events[~(events.clipped_start | events.clipped_end)]``
    reproduces the original rule. A block shorter than the smoothing window is
    treated as missing, with a warning.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in seconds.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals already bandpass filtered to the ripple band. FMAToolbox
        documents 100-200 Hz input, buzcode filters 130-200 Hz, neurocode
        80-250 Hz; this package's ``filter_ripple_band`` gives 150-250 Hz.
        Channels are squared and summed, as the original does. NaN marks
        missing samples.
    speed : array_like, shape (n_time,)
        Animal's running speed in cm/s.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        An event is kept only if the speed at its first and last sample is at
        or below this value (``exclude_movement``). **Not part of the original
        algorithm**, which has no speed criterion; pass ``np.inf`` to disable.
        Default is 4.0.
    low_threshold : float, optional
        Boundary threshold in standard deviations. Default is 2.0 (FMAToolbox
        and buzcode; neurocode's copy uses 0.5).
    high_threshold : float, optional
        Peak threshold in standard deviations. Default is 5.0 (FMAToolbox and
        buzcode; neurocode's copy uses 2.5).
    minimum_inter_ripple_interval : float, optional
        Events separated by less than this are merged, provided the merged
        event stays under ``maximum_duration``. Default is 0.030 s
        (FMAToolbox; neurocode uses 0.050).
    minimum_duration, maximum_duration : float, optional
        Events shorter or longer than these are discarded. Defaults are
        0.020 and 0.100 s (FMAToolbox; neurocode uses 0.025 and 0.500).
        The 100 ms ceiling is FMAToolbox's, and is shorter than the ceilings
        most of the replay literature uses. Raise it for a limit typical of
        that literature rather than of this algorithm's original settings, or
        pass ``maximum_duration=None`` for no ceiling, as the other detectors
        default to.
    smoothing_window : int, optional
        Moving-average length in samples. Default is the original's 11
        samples at 1250 Hz scaled to ``sampling_frequency`` and kept odd. A
        supplied value must be a positive odd integer.
    normalization_mask : array_like, shape (n_time,), optional
        Samples used for the z-score statistics (the original's ``restrict``).

    Returns
    -------
    ripple_times : pd.DataFrame
        One row per event, indexed by ``event_number``, with the columns of
        the other detectors plus ``peak_time``, the time of the maximum
        normalized power within the event.

    References
    ----------
    .. [1] Zugaro, M. FMAToolbox, ``Analyses/FindRipples.m`` (initial algorithm
       by H. Hirase). The repository has no license file; the file headers
       state GPL-3 or later.
       https://github.com/michael-zugaro/FMAToolbox/blob/6bbb3662f7ed1ccf09c5ff4b4d233e27e17c71a6/Analyses/FindRipples.m
    .. [2] Buzsáki lab, buzcode, ``analysis/SharpWaveRipples/bz_FindRipples.m``
       (edited by D. Tingley, 2017), GPL-3.
       https://github.com/buzsakilab/buzcode/blob/0969ddf7f55ccaca8c71969bee4b21f310840047/analysis/SharpWaveRipples/bz_FindRipples.m
    .. [3] AYA lab, neurocode, ``SharpWaveRipples/FindRipples.m`` (no license
       file), doi:10.5281/zenodo.7819979
       https://github.com/ayalab1/neurocode/blob/d166a67ffb73096d8d11b14be6693d96ad63e4ed/SharpWaveRipples/FindRipples.m
    .. [4] Csicsvari, J., Hirase, H., Czurkó, A., Mamiya, A., & Buzsáki, G.
       (1999). Oscillatory coupling of hippocampal pyramidal cells and
       interneurons in the behaving rat. Journal of Neuroscience, 19(1),
       274-287. doi:10.1523/JNEUROSCI.19-01-00274.1999

    """
    _validate_duration_limits(minimum_duration, maximum_duration)
    _check_thresholds("low_threshold", low_threshold, "high_threshold", high_threshold)
    _check_non_negative(minimum_inter_ripple_interval=minimum_inter_ripple_interval)
    if smoothing_window is not None:
        _check_whole_number("smoothing_window", smoothing_window, 1)
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    window = (
        _zugaro_smoothing_window(sampling_frequency)
        if smoothing_window is None
        else int(smoothing_window)
    )
    if window < 1 or window % 2 == 0:
        msg = f"smoothing_window must be a positive odd integer, got {window}."
        raise ValueError(msg)
    is_valid, blocks = _valid_blocks(time, filtered_lfps, minimum_duration=minimum_duration)
    _reject_flat_channels(filtered_lfps, is_valid, "filtered_lfps")
    blocks = _drop_short_blocks(blocks, is_valid, window, "the smoothing window")

    kernel = np.ones(window) / window
    power = np.sum(filtered_lfps**2, axis=1)
    smoothed = np.full(len(time), np.nan)
    for start, stop in blocks:
        smoothed[start:stop] = np.convolve(power[start:stop], kernel, mode="same")

    mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
    normalized = normalize_signal(smoothed, normalization_mask=mask)

    event_time_blocks: list[FloatArray] = [np.empty((0, 2))]
    peak_time_blocks: list[FloatArray] = [np.empty(0)]
    clipped_blocks: list[BoolArray] = [np.empty((0, 2), dtype=bool)]
    for start, stop in blocks:
        block_events, block_peaks, block_clipped = _two_threshold_events(
            normalized[start:stop],
            time[start:stop],
            low_threshold,
            high_threshold,
            minimum_inter_ripple_interval,
            minimum_duration,
            maximum_duration,
        )
        event_time_blocks.append(block_events)
        peak_time_blocks.append(block_peaks)
        clipped_blocks.append(block_clipped)
    event_times: FloatArray = np.concatenate(event_time_blocks)
    peak_times: FloatArray = np.concatenate(peak_time_blocks)
    clipped_flags: BoolArray = np.concatenate(clipped_blocks)

    keep = _is_immobile_at_endpoints(event_times, speed, time, speed_threshold)
    event_times, peak_times, clipped_flags = (
        event_times[keep],
        peak_times[keep],
        clipped_flags[keep],
    )

    # clipped means "lacks a crossing" here, which the segmentation knows and
    # the event's position alone does not: a run beginning on the block's
    # second sample starts on its first yet has its crossing there
    events = _get_event_stats(
        event_times,
        time,
        normalized,
        speed,
        minimum_duration=minimum_duration,
        blocks=blocks,
        clipped=clipped_flags,
    )
    events["peak_time"] = peak_times
    return events
