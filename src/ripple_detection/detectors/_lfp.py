"""The envelope-based ripple detectors on ripple-band LFP: Kay, Karlsson, Roumis,
Shvartsman and Yu, with their consensus traces."""

from itertools import chain
from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    BoolArray,
    FloatArray,
    IntArray,
    NormalizationMethod,
    _check_non_negative,
    _is_immobile,
    _is_immobile_at_endpoints,
    _is_immobile_by_majority,
    _normalization_statistics,
    _runs_extended_to_mean,
    estimate_noise_threshold,
    gaussian_smooth,
    get_envelope,
    merge_overlapping_ranges,
    merge_overlapping_ranges_track_participation,
    minimum_sample_count,
    normalize_signal,
    normalize_signal_manually,
)
from ripple_detection.detectors._blocks import (
    _contiguous_valid_blocks,
    _normalization_mask_over_valid,
    _reject_flat_channels,
    _smoothed_envelope,
    _threshold_blocks,
    _valid_blocks,
)
from ripple_detection.detectors._events import (
    _detect_from_trace,
    _finish_events,
    _get_event_stats,
)
from ripple_detection.detectors._validation import (
    _check_finite_non_negative,
    _check_gap,
    _check_smoothing_sigma,
    _check_whole_number,
    _validate_detector_inputs,
    _validate_duration_limits,
    _validate_lfp_dimensions,
    _warn_if_not_ripple_band,
)


def get_Kay_ripple_consensus_trace(
    ripple_filtered_lfps: ArrayLike,
    sampling_frequency: float,
    smoothing_sigma: float = 0.004,
    *,
    time: ArrayLike | None = None,
) -> FloatArray:
    """Compute Kay consensus trace from multi-channel ripple-filtered LFPs.

    Combines multiple LFP channels into a single consensus trace, following
    Kay et al. 2016: ``sqrt(gaussian_smooth(sum(envelope ** 2)))``. The
    smoothing sits between the sum and the square root. The paper's text
    squares the filtered signal itself, not its Hilbert envelope; the two
    traces differ by a constant factor of sqrt(2) once the 4 ms smoothing has
    removed the doubled-frequency term, so the events are the same after the
    z-score. The envelope is what the Frank lab code uses.

    Rows holding NaN or infinity in any channel are missing and returned as
    NaN, and each contiguous run of valid rows is processed on its own, so no
    envelope or smoothing window spans a gap.

    Parameters
    ----------
    ripple_filtered_lfps : array_like, shape (n_time, n_channels)
        Bandpass filtered LFP signals in the ripple band (150-250 Hz). Input with most of its
        power below 100 Hz, as raw LFP and ADC counts have, warns.
    sampling_frequency : float
        Sampling rate in Hz.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in seconds.
        Default is 0.004 (4 ms).
    time : array_like, shape (n_time,), optional
        Sample timestamps in seconds, used to split at gaps in the timestamps
        (a step larger than 1.5 times the median step) as well as at missing
        samples. Keyword only. Default is None, which
        splits at missing samples only.

    Returns
    -------
    consensus_trace : ndarray, shape (n_time,)
        ``sqrt(gaussian_smooth(sum(envelope ** 2)))`` per sample.

    Raises
    ------
    ValueError
        If ``ripple_filtered_lfps`` is not 2-D, ``time`` is not one timestamp
        per sample, or no sample is finite in every channel.

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., &
       Frank, L. M. (2016). A hippocampal network for spatial coding during
       immobility and sleep. Nature, 531(7593), 185-190.
       doi:10.1038/nature17144

    """
    # Cast to float so integer input is not truncated and the squared envelope
    # cannot overflow before the square root.
    ripple_filtered_lfps = np.asarray(ripple_filtered_lfps, dtype=float)
    _validate_lfp_dimensions(ripple_filtered_lfps)
    time_array = _consensus_time(time, ripple_filtered_lfps.shape)
    is_valid = np.all(np.isfinite(ripple_filtered_lfps), axis=1)
    if not np.any(is_valid):
        msg = "No sample has finite values in every channel."
        raise ValueError(msg)
    _warn_if_not_ripple_band(ripple_filtered_lfps, sampling_frequency, "ripple_filtered_lfps")
    blocks = _contiguous_valid_blocks(is_valid, time_array)
    return _kay_consensus(ripple_filtered_lfps, blocks, sampling_frequency, smoothing_sigma)


def _consensus_time(time: ArrayLike | None, lfp_shape: tuple[int, ...]) -> FloatArray | None:
    """The optional timestamps of a consensus trace as a float array of one
    timestamp per row of an LFP of shape ``lfp_shape``, or None."""
    if time is None:
        return None
    time_array = np.asarray(time, dtype=float)
    if time_array.shape != lfp_shape[:1]:
        msg = (
            f"time has shape {time_array.shape} but filtered_lfps has {lfp_shape[0]} samples."
        )
        if time_array.shape == lfp_shape[1:]:
            msg += (
                f" filtered_lfps has shape {lfp_shape}, which looks transposed: transpose "
                "it to (n_time, n_channels), time down the rows (pass .T)."
            )
        raise ValueError(msg)
    return time_array


def _kay_consensus(
    filtered_lfps: FloatArray,
    blocks: list[tuple[int, int]],
    sampling_frequency: float,
    smoothing_sigma: float,
) -> FloatArray:
    """``sqrt(gaussian_smooth(sum(envelope ** 2)))`` within each block, NaN
    outside every block. The detector passes its own blocks, so the LFP is
    neither copied nor searched for gaps a second time."""
    smoothed = np.full(len(filtered_lfps), np.nan)
    for start, stop in blocks:
        summed_power = np.sum(get_envelope(filtered_lfps[start:stop]) ** 2, axis=1)
        smoothed[start:stop] = gaussian_smooth(
            summed_power, smoothing_sigma, sampling_frequency
        )
    return np.asarray(np.sqrt(smoothed), dtype=float)


def get_Yu_ripple_consensus_trace(
    ripple_filtered_lfps: ArrayLike,
    sampling_frequency: float,
    smoothing_sigma: float = 0.004,
    zscore_per_channel: bool = True,
    *,
    time: ArrayLike | None = None,
) -> FloatArray:
    """Compute the Yu et al. 2017 consensus trace: median of per-tetrode envelopes.

    Each channel's ripple-band envelope is smoothed with a Gaussian kernel and,
    by default, z-scored over the whole recording; the consensus is the median
    across channels at each sample. A median rather than a sum keeps one
    tetrode from dominating the trace.

    Envelope and smoothing run separately inside each maximal contiguous block
    of valid samples, so missing data never bleeds across a gap; the per-channel
    z-score statistics are pooled over all valid samples. Rows with a non-finite
    value in any channel are returned as NaN.

    Parameters
    ----------
    ripple_filtered_lfps : array_like, shape (n_time, n_channels)
        Bandpass filtered LFP signals in the ripple band (150-250 Hz). Input with most of its
        power below 100 Hz, as raw LFP and ADC counts have, warns.
    sampling_frequency : float
        Sampling rate in Hz.
    smoothing_sigma : float, optional
        Standard deviation of the Gaussian smoothing kernel in seconds, applied
        per channel before aggregation. Default is 0.004 (4 ms).
    zscore_per_channel : bool, optional
        If True (default), z-score each channel's smoothed envelope (sample
        standard deviation, ``ddof=1``) over all valid samples before taking
        the median, as the original lab implementation does. If False, take
        the median of the raw smoothed envelopes.
    time : array_like, shape (n_time,), optional
        Sample timestamps in seconds. When given, a step larger than 1.5
        times the median step also ends a block, so disjoint intervals that were
        concatenated are not smoothed across. Default is None, which declares
        a regular sample grid.

    Returns
    -------
    consensus_trace : ndarray, shape (n_time,)
        Median across channels of the smoothed (and z-scored) envelopes, with
        NaN wherever any channel was non-finite.

    Raises
    ------
    ValueError
        If the input is not 2-D, if ``time`` does not match its length, if no
        valid samples exist, or if a channel's standard deviation over the
        valid samples is zero or non-finite when z-scoring is enabled.

    References
    ----------
    .. [1] Yu, J. Y., Kay, K., Liu, D. F., Grossrubatscher, I., Loback, A.,
       Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., & Frank, L. M.
       (2017). Distinct hippocampal-cortical memory representations for
       experiences associated with movement versus immobility. eLife, 6,
       e27621. doi:10.7554/eLife.27621

    """
    ripple_filtered_lfps = np.asarray(ripple_filtered_lfps, dtype=float)
    _validate_lfp_dimensions(ripple_filtered_lfps)
    time_array = _consensus_time(time, ripple_filtered_lfps.shape)
    is_valid = np.all(np.isfinite(ripple_filtered_lfps), axis=1)
    if not np.any(is_valid):
        msg = "No sample has finite values in every channel."
        raise ValueError(msg)
    _warn_if_not_ripple_band(ripple_filtered_lfps, sampling_frequency, "ripple_filtered_lfps")
    return _yu_consensus(
        ripple_filtered_lfps,
        is_valid,
        _contiguous_valid_blocks(is_valid, time_array),
        sampling_frequency,
        smoothing_sigma,
        zscore_per_channel,
    )


def _yu_consensus(
    filtered_lfps: FloatArray,
    is_valid: BoolArray,
    blocks: list[tuple[int, int]],
    sampling_frequency: float,
    smoothing_sigma: float,
    zscore_per_channel: bool,
) -> FloatArray:
    """The median across channels of the smoothed, optionally z-scored,
    envelopes within ``blocks``; NaN outside them. The z-score statistics
    pool the ``is_valid`` samples."""
    smoothed = _smoothed_envelope(filtered_lfps, blocks, sampling_frequency, smoothing_sigma)
    if zscore_per_channel:
        valid_rows = smoothed[is_valid]
        mean = valid_rows.mean(axis=0, keepdims=True)
        std = (
            valid_rows.std(axis=0, ddof=1, keepdims=True)
            if valid_rows.shape[0] > 1
            else np.full((1, smoothed.shape[1]), np.nan)
        )
        bad = ~np.isfinite(std) | (std <= 0)
        if np.any(bad):
            msg = (
                "Cannot z-score channels with zero or undefined standard deviation "
                f"over the valid samples: channel indices {np.flatnonzero(bad).tolist()}."
            )
            raise ValueError(msg)
        smoothed = (smoothed - mean) / std

    consensus_trace = np.full(len(filtered_lfps), np.nan)
    consensus_trace[is_valid] = np.median(smoothed[is_valid], axis=1)
    return consensus_trace


def _extract_Yu_ripple_events(
    trace: FloatArray,
    time: FloatArray,
    minimum_duration: float,
    threshold: float,
) -> tuple[FloatArray, IntArray]:
    """Extract events from one contiguous block of a mean-zero consensus trace.

    A run of consecutive samples at or above ``threshold`` qualifies when it
    holds at least ``minimum_sample_count(time, minimum_duration)`` samples.
    Each qualifying run is extended to the run of samples at or above zero,
    the immobility mean, that contains it, as ``threshold_by_zscore`` does. One event is emitted per containing
    run. This is the sample-count convention of the Frank lab
    ``extractevents`` routine, which the Yu et al. 2017 detector used
    (``DFFunctions/extractevents.cpp`` in
    https://github.com/droumis/FFPhy/tree/fce2048/DFFunctions).

    Parameters
    ----------
    trace : ndarray, shape (n_time,)
        Consensus trace normalized so the immobility mean is zero, for one
        contiguous block with no missing samples.
    time : ndarray, shape (n_time,)
        Native timestamps of the block's samples, in seconds.
    minimum_duration : float
        Minimum time the trace must stay at or above ``threshold``, in
        seconds; converted to a sample count with round-half-up.
    threshold : float
        Detection threshold in the trace's normalized units; must be finite
        and strictly positive.

    Returns
    -------
    event_times : ndarray, shape (n_events, 2)
        ``[start_time, end_time]`` of each event: the native timestamps of the
        first and last samples of the containing run at or above zero.
    n_suprathreshold_samples : ndarray of int, shape (n_events,)
        Sample count of the longest qualifying run inside each event.

    Raises
    ------
    ValueError
        If ``threshold`` is not finite or not strictly positive, since the
        threshold-then-mean-crossing rule is undefined otherwise.

    """
    if not np.isfinite(threshold) or threshold <= 0:
        msg = f"threshold must be finite and strictly above the zero mean, got {threshold}."
        raise ValueError(msg)
    trace = np.asarray(trace, dtype=float)
    time = np.asarray(time, dtype=float)
    bounds, n_suprathreshold = _runs_extended_to_mean(
        trace >= 0, trace >= threshold, minimum_sample_count(time, minimum_duration)
    )
    event_times = np.column_stack([time[bounds[:, 0]], time[bounds[:, 1] - 1]])
    return np.asarray(event_times, dtype=float).reshape(-1, 2), n_suprathreshold


def _check_threshold_parameters(
    *,
    speed_threshold: float,
    minimum_duration: float,
    maximum_duration: float | None,
    close_ripple_threshold: float,
    smoothing_sigma: float,
    zscore_threshold: float | None = None,
) -> None:
    """The checks on the envelope detectors' tunables that need no data:
    types, ranges, and durations that look like milliseconds. Yu has no
    ``zscore_threshold``. ``DetectorSpec.check_parameters`` runs them too."""
    _check_non_negative(speed_threshold=speed_threshold)
    _validate_duration_limits(minimum_duration, maximum_duration)
    if zscore_threshold is not None:
        _check_finite_non_negative(zscore_threshold=zscore_threshold)
    _check_gap(close_ripple_threshold=close_ripple_threshold)
    _check_smoothing_sigma(smoothing_sigma=smoothing_sigma)


def _check_shvartsman_parameters(
    *,
    speed_threshold: float,
    minimum_duration: float,
    maximum_duration: float | None,
    zscore_threshold: float,
    close_ripple_threshold: float,
    smoothing_sigma: float,
    normalization_method: str,
    normalization_mask: object,
    channel_baselines: object,
    channel_deviations: object,
    minimum_participating_channels: int | None,
    minimum_participating_fraction: float | None,
) -> None:
    """``_check_threshold_parameters`` and the normalization and
    participation options of ``Shvartsman_ripple_detector``: which go
    together, and their ranges."""
    if normalization_method not in ("zscore", "median_mad", "manual"):
        msg = (
            "normalization_method must be 'zscore', 'median_mad' or 'manual', "
            f"got {normalization_method!r}."
        )
        raise ValueError(msg)
    if normalization_method == "manual":
        if channel_baselines is None or channel_deviations is None:
            msg = (
                "normalization_method='manual' needs channel_baselines and "
                "channel_deviations, one entry per channel."
            )
            raise ValueError(msg)
        if normalization_mask is not None:
            msg = (
                "normalization_mask has no meaning with normalization_method='manual': "
                "the statistics are the ones supplied. Drop one or the other."
            )
            raise ValueError(msg)
    elif channel_baselines is not None or channel_deviations is not None:
        msg = (
            "channel_baselines and channel_deviations apply only with "
            f"normalization_method='manual', not {normalization_method!r}."
        )
        raise ValueError(msg)
    if (
        minimum_participating_channels is not None
        and minimum_participating_fraction is not None
    ):
        msg = (
            "Give minimum_participating_channels or minimum_participating_fraction, not both."
        )
        raise ValueError(msg)
    if minimum_participating_channels is not None:
        _check_whole_number(
            "minimum_participating_channels", minimum_participating_channels, 0
        )
    if minimum_participating_fraction is not None and not (
        0.0 <= minimum_participating_fraction <= 1.0
    ):
        msg = (
            f"minimum_participating_fraction must lie in [0, 1], got "
            f"{minimum_participating_fraction}."
        )
        raise ValueError(msg)
    _check_threshold_parameters(
        speed_threshold=speed_threshold,
        minimum_duration=minimum_duration,
        maximum_duration=maximum_duration,
        close_ripple_threshold=close_ripple_threshold,
        smoothing_sigma=smoothing_sigma,
        zscore_threshold=zscore_threshold,
    )


@explain_call_errors
def Shvartsman_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    *,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 3.0,
    smoothing_sigma: float = 0.004,
    close_ripple_threshold: float = 0.0,
    normalization_method: NormalizationMethod | Literal["manual"] = "zscore",
    normalization_mask: ArrayLike | None = None,
    channel_baselines: ArrayLike | None = None,
    channel_deviations: ArrayLike | None = None,
    minimum_participating_channels: int | None = None,
    minimum_participating_fraction: float | None = None,
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripples on each channel, keeping events that enough
    channels share.

    An event is kept when at least ``minimum_participating_channels`` channels
    (default 2), or ``minimum_participating_fraction`` of the channels, detect
    it. This sits between the Kay detector, which builds one consensus trace,
    and the Karlsson detector, which keeps a ripple from any single channel.
    Requiring several channels makes it less sensitive than Karlsson to noise
    on one channel.

    It also accepts a baseline and a deviation per channel, through
    ``normalization_method="manual"`` with ``channel_baselines`` and
    ``channel_deviations``, in place of statistics computed from the data.
    Statistics from a whole recording day rather than one epoch matter for
    sleep sessions; see ``normalize_signal_manually``.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in **seconds**.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals **already bandpass filtered** to ripple band (150-250 Hz).
        Must be pre-filtered using `filter_ripple_band()` before calling this detector.
        Input with most of its power below 100 Hz, as raw LFP and ADC counts
        have, warns.
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point in **cm/s**.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (in cm/s) for ripple detection. An event is kept only if
        at least half of its samples have speed at or below this value
        (``exclude_movement_by_majority``); movement over up to half of an
        event does not exclude it. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, pass ``np.inf``, which also
        keeps events whose speed is unknown (NaN).
    minimum_duration : float, optional
        Minimum time above threshold in **seconds**. Default is 0.015 (15
        milliseconds). The signal must stay at or above ``zscore_threshold`` for
        at least ``minimum_sample_count(time, minimum_duration)`` consecutive
        samples, rounded half up from the median timestamp step (23 at 1500 Hz
        and 15 ms). The 15 ms is Karlsson & Frank 2009's; the rounding is the
        Frank lab ``extractevents`` convention. The event is then extended to
        the surrounding mean-crossings, so the reported ``duration`` is
        typically longer.
        It is the time above threshold, not the whole event's duration; for a
        minimum on the whole event, which most published minimums mean, see
        ``detect_events_from_trace(minimum_event_duration=)``.
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    zscore_threshold : float, optional
        Detection sensitivity threshold in standard deviations above mean.
        Default is 3.0, the per-tetrode threshold of Karlsson & Frank 2009.
        Lower values detect more events.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in **seconds**.
        Default is 0.004 (4 ms). Rarely needs adjustment; increase for
        noisier data.
    close_ripple_threshold : float, optional
        Minimum time in **seconds** between ripples. Events closer than this
        are excluded -- the later event is dropped, not merged. Default is 0.0
        (no exclusion). Set to 0.05-0.1 s to drop closely-spaced events.
    normalization_method : {'zscore', 'median_mad', 'manual'}, optional
        Method for normalizing each channel. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains
        outliers. 'manual' normalizes each channel with the supplied
        `channel_baselines` and `channel_deviations` instead of statistics
        computed from the data (``normalize_signal_manually``).
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask selecting samples used to compute normalization statistics.
        For example, use `speed <= speed_threshold` to compute statistics only
        during immobility. Default is None (use all data). Has no meaning with
        ``normalization_method='manual'`` and raises if given with it.
    channel_baselines : array_like, shape (n_channels,), optional
        Baseline (center) value per channel. Required with, and only allowed
        with, ``normalization_method='manual'``.
    channel_deviations : array_like, shape (n_channels,), optional
        Deviation (scale) value per channel, on the scale of a standard
        deviation (multiply a MAD by 1.4826 first). Required with, and only
        allowed with, ``normalization_method='manual'``. A zero or NaN entry
        raises; drop that channel before detecting.
    minimum_participating_channels : int, optional
        Number of channels that must detect a ripple in the merged event for
        it to be kept. Default is 2 when neither participation argument is
        given; 0 imposes no criterion. Each distinct channel with a detected
        ripple anywhere in the merged event counts once, including channels
        connected through a chain of overlapping ripples.
        Raises when it exceeds the number of channels, since no event could be kept.
    minimum_participating_fraction : float, optional
        The same criterion as a fraction of the channels in `filtered_lfps`,
        in [0, 1]; 1.0 requires every channel. Give one of the two arguments,
        not both. Default is None.

    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with detected ripples and comprehensive statistics (see
        Kay_ripple_detector for the shared columns). This detector additionally
        returns ``participants`` (sorted tuple of every channel whose ripple appears
        anywhere in the event), ``n_participants`` (``len(participants)``), and
        ``frac_participants`` (``n_participants`` / total channels). Participation
        is measured on each channel's full zero-crossing-extended ripple (the same
        extent as the event boundaries). The per-event z-score statistics are
        averaged over the ``participants`` union.

        Returns empty DataFrame if no ripples detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 3.0 to 2.0)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your data contains ripple oscillations (150-250 Hz)

    Notes
    -----
    Missing samples: a NaN or infinite value in any channel of
    ``filtered_lfps`` marks that sample missing, and a step in ``time``
    larger than 1.5 times its median step ends a block as a missing sample
    does. Every detector in the package splits the valid samples into these
    blocks and runs every step within one, so nothing is computed across a
    gap and no event spans one; an event cut off by a gap or by the
    recording edge is kept and flagged in ``clipped_start`` and
    ``clipped_end``. Here a block too short for an event of
    ``minimum_duration`` is treated as missing, with a warning, and no block
    left raises. A NaN in ``speed`` is an unknown speed, not a missing
    sample: it splits no block, and the majority rule counts only the
    samples whose speed is known, so an event with no known speed fails it.
    See the README's "Choosing a detector" table for how the detectors'
    conventions differ.

    References
    ----------
    Unpublished variant contributed by Gabrielle Shvartsman (2026); it has no
    paper of its own. The participation rule requires
    ``minimum_participating_channels`` channels (default 2) to detect the
    ripple, so at the default a single-channel input raises; pass
    ``minimum_participating_channels=1`` for one channel.

    Examples
    --------
    >>> from ripple_detection import filter_ripple_band
    >>> from ripple_detection.simulate import simulate_session, simulate_time
    >>> time = simulate_time(45_000, 1500)  # 30 s at 1500 Hz
    >>> session = simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], rng=0)
    >>> filtered_lfps = filter_ripple_band(session.lfps, sampling_frequency=1500)
    >>> events = Shvartsman_ripple_detector(
    ...     time, filtered_lfps, session.speed, 1500, minimum_participating_channels=2
    ... )
    >>> bool((events.n_participants >= 2).all())
    True

    """
    _check_shvartsman_parameters(
        speed_threshold=speed_threshold,
        minimum_duration=minimum_duration,
        maximum_duration=maximum_duration,
        zscore_threshold=zscore_threshold,
        close_ripple_threshold=close_ripple_threshold,
        smoothing_sigma=smoothing_sigma,
        normalization_method=normalization_method,
        normalization_mask=normalization_mask,
        channel_baselines=channel_baselines,
        channel_deviations=channel_deviations,
        minimum_participating_channels=minimum_participating_channels,
        minimum_participating_fraction=minimum_participating_fraction,
    )
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    required_channels = (
        2 if minimum_participating_channels is None else minimum_participating_channels
    )
    if minimum_participating_fraction is None and required_channels > filtered_lfps.shape[1]:
        msg = (
            f"minimum_participating_channels is {required_channels} but filtered_lfps has "
            f"{filtered_lfps.shape[1]} channel(s), so no event could be kept. Pass "
            f"minimum_participating_channels={filtered_lfps.shape[1]} or fewer, or more channels."
        )
        raise ValueError(msg)
    is_valid, blocks = _valid_blocks(time, filtered_lfps, minimum_duration=minimum_duration)
    _reject_flat_channels(filtered_lfps, blocks, "filtered_lfps")
    _warn_if_not_ripple_band(filtered_lfps, sampling_frequency)

    smoothed = _smoothed_envelope(filtered_lfps, blocks, sampling_frequency, smoothing_sigma)
    if normalization_method == "manual":
        normalized = normalize_signal_manually(
            smoothed, cast(ArrayLike, channel_baselines), cast(ArrayLike, channel_deviations)
        )
    else:
        mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
        normalized = normalize_signal(
            smoothed, method=normalization_method, normalization_mask=mask
        )

    candidate_ripple_times = [
        _threshold_blocks(channel, time, blocks, minimum_duration, zscore_threshold)
        for channel in normalized.T
    ]
    # Merge each channel's mean-crossing-extended intervals and retain the union
    # of contributing channels, preserving the original participation rule.
    merged_candidates = merge_overlapping_ranges_track_participation(candidate_ripple_times)

    n_elecs = normalized.shape[1]
    if minimum_participating_fraction is not None:
        # round so that 25 channels at 0.28 ask for 7, not the 7.000000000000001 of
        # floating-point multiplication, which would demand 8
        n_elecs_thresh = round(n_elecs * minimum_participating_fraction, 9)
    else:
        n_elecs_thresh = required_channels
    participation_mask = (
        np.asarray([len(interval[2]) for interval in merged_candidates]) >= n_elecs_thresh
    )
    candidate_bounds = np.asarray(
        merged_candidates[participation_mask, :2], dtype=float
    ).reshape(-1, 2)
    keep = _is_immobile_by_majority(candidate_bounds, speed, time, speed_threshold, 0.5)
    ripple_times, kept = _finish_events(
        candidate_bounds, keep, time, close_ripple_threshold, maximum_duration
    )
    # sorted tuples rather than sets: deterministic, hashable, and they survive
    # JSON, CSV and a DynamicTable
    participant_sets = merged_candidates[participation_mask, 2][kept]
    participants = np.empty(len(participant_sets), dtype=object)
    for index, channels in enumerate(participant_sets):
        participants[index] = tuple(sorted(channels))
    return _get_event_stats(
        ripple_times,
        time,
        normalized,
        speed,
        minimum_duration,
        blocks,
        participants=participants,
    )


@explain_call_errors
def Kay_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    *,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 2.0,
    smoothing_sigma: float = 0.004,
    close_ripple_threshold: float = 0.0,
    normalization_method: NormalizationMethod = "zscore",
    normalization_mask: ArrayLike | None = None,
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripple events using multi-channel consensus method.

    Implements the Kay et al. 2016 ripple detection algorithm, which combines
    multiple LFP channels into a consensus trace using sum of squared envelopes.
    Ripples are identified as periods where the z-scored consensus exceeds a
    threshold during immobility.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in **seconds**.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals **already bandpass filtered** to ripple band (150-250 Hz).
        Must be pre-filtered using `filter_ripple_band()` before calling this detector.
        Input with most of its power below 100 Hz, as raw LFP and ADC counts
        have, warns.
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point in **cm/s**.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (in cm/s) for ripple detection. An event is kept only if
        the speed at its first and last sample is at or below this value
        (``exclude_movement``); speed inside the event is not tested. Apply a
        whole-event rule afterwards if one is needed. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, pass ``np.inf``, which also
        keeps events whose speed is unknown (NaN).
    minimum_duration : float, optional
        Minimum time above threshold in **seconds**. Default is 0.015 (15
        milliseconds). The signal must stay at or above ``zscore_threshold`` for
        at least ``minimum_sample_count(time, minimum_duration)`` consecutive
        samples, rounded half up from the median timestamp step (23 at 1500 Hz
        and 15 ms). The 15 ms is Karlsson & Frank 2009's; the rounding is the
        Frank lab ``extractevents`` convention. The event is then extended to
        the surrounding mean-crossings, so the reported ``duration`` is
        typically longer.
        It is the time above threshold, not the whole event's duration; for a
        minimum on the whole event, which most published minimums mean, see
        ``detect_events_from_trace(minimum_event_duration=)``.
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    zscore_threshold : float, optional
        Detection sensitivity threshold in standard deviations above mean.
        Default is 2.0. Lower values (e.g., 1.5) detect more events but may
        include false positives. Higher values (e.g., 3.0) are more conservative.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in **seconds**.
        Default is 0.004 (4 ms). Rarely needs adjustment; increase for
        noisier data.
    close_ripple_threshold : float, optional
        Minimum time in **seconds** between ripples. Events closer than this
        are excluded -- the later event is dropped, not merged. Default is 0.0
        (no exclusion). Set to 0.05-0.1 s to drop closely-spaced events.
    normalization_method : {'zscore', 'median_mad'}, optional
        Method for normalizing the consensus trace. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains outliers.
        The median/MAD method is more resistant to extreme values.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask to specify which samples to use for computing normalization
        statistics. For example, use `speed <= speed_threshold` to compute
        statistics only during immobility. Default is None (use all data).

    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with one row per detected ripple, containing:
        - start_time, end_time, duration, n_samples
        - max_sustained_zscore: the largest z-score sustained for
          ``minimum_duration``, i.e. the highest threshold that would still
          detect the event
        - mean_zscore, median_zscore, max_zscore, min_zscore
        - area: integral of z-score
        - total_energy: integral of squared z-score
        - speed metrics: speed_at_start, speed_at_end, max/min/median/mean_speed
        - clipped_start, clipped_end: whether the event was cut off by missing
          data or the recording edge
        - peak_time: time of the consensus trace's largest value in the event

        Returns empty DataFrame if no ripples detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 2.0 to 1.5)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your data contains ripple oscillations (150-250 Hz)

    Notes
    -----
    Missing samples: a NaN or infinite value in any channel of
    ``filtered_lfps`` marks that sample missing, and a step in ``time``
    larger than 1.5 times its median step ends a block as a missing sample
    does. Every detector in the package splits the valid samples into these
    blocks and runs every step within one, so nothing is computed across a
    gap and no event spans one; an event cut off by a gap or by the
    recording edge is kept and flagged in ``clipped_start`` and
    ``clipped_end``. Here a block too short for an event of
    ``minimum_duration`` is treated as missing, with a warning, and no block
    left raises. A NaN in ``speed`` is an unknown speed, not a missing
    sample: it splits no block, and an event whose first or last sample has
    unknown speed fails the speed criterion.
    See the README's "Choosing a detector" table for how the detectors'
    conventions differ.

    Examples
    --------
    >>> from ripple_detection import filter_ripple_band
    >>> from ripple_detection.simulate import simulate_session, simulate_time
    >>> time = simulate_time(45_000, 1500)  # 30 s at 1500 Hz
    >>> session = simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], rng=0)
    >>> # filter to the ripple band first; the detector takes ripple-band LFP
    >>> filtered_lfps = filter_ripple_band(session.lfps, sampling_frequency=1500)
    >>> ripples = Kay_ripple_detector(time, filtered_lfps, session.speed, 1500)
    >>> "start_time" in ripples.columns, bool(len(ripples))
    (True, True)

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., &
       Frank, L. M. (2016). A hippocampal network for spatial coding during
       immobility and sleep. Nature, 531(7593), 185-190.
       doi:10.1038/nature17144

    """
    _check_threshold_parameters(
        speed_threshold=speed_threshold,
        minimum_duration=minimum_duration,
        maximum_duration=maximum_duration,
        close_ripple_threshold=close_ripple_threshold,
        smoothing_sigma=smoothing_sigma,
        zscore_threshold=zscore_threshold,
    )
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, filtered_lfps, minimum_duration=minimum_duration)
    _reject_flat_channels(filtered_lfps, blocks, "filtered_lfps")
    _warn_if_not_ripple_band(filtered_lfps, sampling_frequency)

    consensus = _kay_consensus(filtered_lfps, blocks, sampling_frequency, smoothing_sigma)
    return _detect_from_trace(
        consensus,
        time,
        speed,
        is_valid,
        blocks,
        minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold,
        speed_threshold=speed_threshold,
        close_event_threshold=close_ripple_threshold,
        maximum_duration=maximum_duration,
        normalization_method=normalization_method,
        normalization_mask=normalization_mask,
    )


@explain_call_errors
def Yu_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    *,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.020,
    percentile: float = 99.99,
    smoothing_sigma: float = 0.004,
    close_ripple_threshold: float = 0.0,
    normalization_mask: ArrayLike | None = None,
    zscore_per_channel: bool = True,
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripples with a data-driven noise threshold (Yu et al. 2017).

    The consensus trace is the median across tetrodes of each tetrode's
    smoothed, z-scored ripple-band envelope (``get_Yu_ripple_consensus_trace``).
    Its values during immobility are taken as noise plus a signal tail. The
    part below the mode is mirrored about the mode to estimate the noise
    distribution, and the detection threshold is the ``percentile`` of that
    mirrored distribution (``estimate_noise_threshold``). An event is a run of
    at least ``minimum_duration`` at or above the threshold, extended to where
    the trace returns to the immobility mean.

    Like every detector in the package, it splits the recording into
    contiguous blocks of valid samples, so smoothing, thresholding, and event
    extraction never cross a gap, and an event truncated by a gap or by the
    end of the recording is kept and flagged in ``clipped_start`` and
    ``clipped_end``.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in seconds.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals already bandpass filtered to the ripple band (150-250 Hz),
        e.g. with ``filter_ripple_band``. NaN marks missing samples. Input with
        most of its power below 100 Hz, as raw LFP and ADC counts have, warns.
    speed : array_like, shape (n_time,)
        Animal's running speed in cm/s.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Immobility is speed at or below this value (cm/s), the package's rule
        (the paper says "below 4 cm/s"; the two differ only at exact equality).
        It selects the noise sample for the threshold and, at event
        boundaries, which events are kept. A NaN speed is unknown: it is left
        out of the noise sample, fails the rule at an event boundary, and
        splits no block. Default is 4.0.
    minimum_duration : float, optional
        Minimum time the consensus must stay at or above the threshold, in
        seconds, applied as a sample count (round-half-up). Default is 0.020.
        It is the time above threshold, not the whole event's duration; for a
        minimum on the whole event, which most published minimums mean, see
        ``detect_events_from_trace(minimum_event_duration=)``.
    percentile : float, optional
        Percentile of the mirrored noise distribution used as the threshold.
        Default is 99.99.
    smoothing_sigma : float, optional
        Standard deviation of the per-tetrode Gaussian envelope smoothing, in
        seconds. Default is 0.004 (4 ms).
    close_ripple_threshold : float, optional
        Minimum separation between events in seconds; a later event starting
        within this time of the previous event's end is dropped. Default is
        0.0 (no exclusion).
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask selecting the noise sample instead of ``speed <=
        speed_threshold``. It sets the samples the threshold is estimated
        from and the trace is normalized to. With ``zscore_per_channel``, the
        per-channel z-score that precedes the median is pooled over every
        valid sample regardless of this mask, as the original implementation
        does.
    zscore_per_channel : bool, optional
        Z-score each tetrode's smoothed envelope before the median, as the
        original implementation does; the threshold is then estimated on that
        trace and converted to immobility-normalized units. If False, the
        median of raw envelopes is normalized to immobility first and the
        threshold is estimated on the normalized trace, the reading of the
        published text. Default is True.

    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.

    Returns
    -------
    ripple_times : pd.DataFrame
        One row per event, indexed by ``event_number``, with the columns of
        the other detectors (``start_time``, ``end_time``, ``duration``,
        ``max_sustained_zscore``, z-score and speed statistics, ``clipped_start``,
        ``clipped_end`` and ``peak_time``) plus ``n_suprathreshold_samples`` (longest run at or
        above the threshold) and ``detection_threshold_zscore`` (the threshold
        in the normalized units the statistics are reported in).

    Raises
    ------
    ValueError
        If inputs are malformed, no valid immobility samples exist, the noise
        distribution cannot be resolved (see ``estimate_noise_threshold``), or
        the estimated threshold does not lie above the immobility mean.

    Notes
    -----
    ``max_sustained_zscore`` uses the same sample-count duration convention as event
    selection, so an event's ``max_sustained_zscore`` is never undefined.

    References
    ----------
    .. [1] Yu, J. Y., Kay, K., Liu, D. F., Grossrubatscher, I., Loback, A.,
       Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., & Frank, L. M.
       (2017). Distinct hippocampal-cortical memory representations for
       experiences associated with movement versus immobility. eLife, 6,
       e27621. doi:10.7554/eLife.27621

    Examples
    --------
    >>> from ripple_detection import filter_ripple_band
    >>> from ripple_detection.simulate import simulate_session, simulate_time
    >>> time = simulate_time(45_000, 1500)  # 30 s at 1500 Hz
    >>> session = simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], rng=0)
    >>> filtered_lfps = filter_ripple_band(session.lfps, sampling_frequency=1500)
    >>> events = Yu_ripple_detector(time, filtered_lfps, session.speed, 1500)
    >>> bool((events.detection_threshold_zscore > 0).all())  # estimated per call
    True

    """
    _check_threshold_parameters(
        speed_threshold=speed_threshold,
        minimum_duration=minimum_duration,
        maximum_duration=maximum_duration,
        close_ripple_threshold=close_ripple_threshold,
        smoothing_sigma=smoothing_sigma,
    )
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, filtered_lfps, minimum_duration=minimum_duration)
    _reject_flat_channels(filtered_lfps, blocks, "filtered_lfps")
    _warn_if_not_ripple_band(filtered_lfps, sampling_frequency)

    consensus = _yu_consensus(
        filtered_lfps,
        is_valid,
        blocks,
        sampling_frequency,
        smoothing_sigma,
        zscore_per_channel,
    )

    if normalization_mask is None:
        # _is_immobile, so speed_threshold=np.inf takes every valid sample as
        # noise, unknown speed included, as it turns the speed rule off elsewhere
        normalization_mask = _is_immobile(speed, speed_threshold)
        if not np.any(normalization_mask & is_valid):
            msg = (
                "No sample with valid LFP has speed at or below speed_threshold "
                f"({speed_threshold} cm/s), so there is no immobility noise to estimate "
                "the threshold from. Pass normalization_mask to choose the noise sample."
            )
            raise ValueError(msg)
    noise_mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
    noise_values = consensus[noise_mask]
    # the statistics the trace is normalized by, so a threshold estimated in
    # the raw units converts with exactly the center and scale the trace used
    center, scale = _normalization_statistics(consensus, noise_mask, "zscore")
    normalized = np.asarray((consensus - center) / scale, dtype=float)
    if zscore_per_channel:
        # The original estimates on the median of per-tetrode z-scores, whose
        # units the histogram grid assumes; convert the result to the
        # immobility-normalized units the events are extracted in.
        threshold = estimate_noise_threshold(noise_values, percentile=percentile)
        threshold_zscore = float((threshold - center.item()) / scale.item())
    else:
        # A raw median of envelopes is not in the grid's units, so follow the
        # paper's text instead: normalize to immobility, then estimate.
        threshold_zscore = float(
            estimate_noise_threshold(normalized[noise_mask], percentile=percentile)
        )
    if not np.isfinite(threshold_zscore) or threshold_zscore <= 0:
        msg = (
            f"Estimated threshold ({threshold_zscore:.4f} SD) does not lie above the "
            "immobility mean; the detection rule is undefined. The mirrored histogram "
            "reaches past the mean only when the immobility trace has some spread of its "
            "own, and here it has almost none next to the ripples: the grid's 0.01 SD "
            "bins cannot resolve its distribution. This happens on simulated brown noise, "
            "which has almost no ripple-band power, and on recordings with very little "
            "ripple-band background. `noise_threshold_diagnostics` on the consensus trace "
            "shows the histogram."
        )
        raise ValueError(msg)

    event_time_blocks: list[FloatArray] = [np.empty((0, 2))]
    n_suprathreshold_blocks: list[IntArray] = [np.empty(0, dtype=int)]
    for start, stop in blocks:
        block_events, block_n = _extract_Yu_ripple_events(
            normalized[start:stop], time[start:stop], minimum_duration, threshold_zscore
        )
        event_time_blocks.append(block_events)
        n_suprathreshold_blocks.append(block_n)
    event_times: FloatArray = np.concatenate(event_time_blocks)
    n_suprathreshold: IntArray = np.concatenate(n_suprathreshold_blocks)

    keep = _is_immobile_at_endpoints(event_times, speed, time, speed_threshold)
    event_times, kept = _finish_events(
        event_times, keep, time, close_ripple_threshold, maximum_duration
    )
    events = _get_event_stats(event_times, time, normalized, speed, minimum_duration, blocks)
    events["n_suprathreshold_samples"] = n_suprathreshold[kept]
    events["detection_threshold_zscore"] = threshold_zscore
    return events


@explain_call_errors
def Karlsson_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    *,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 3.0,
    smoothing_sigma: float = 0.004,
    close_ripple_threshold: float = 0.0,
    normalization_method: NormalizationMethod = "zscore",
    normalization_mask: ArrayLike | None = None,
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripples using per-channel detection with merging.

    Implements the Karlsson & Frank 2009 algorithm, which detects ripples on
    each LFP channel independently, then merges overlapping events across
    channels. More sensitive to local ripples than consensus methods.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in **seconds**.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals **already bandpass filtered** to ripple band (150-250 Hz).
        Must be pre-filtered using `filter_ripple_band()` before calling this detector.
        Input with most of its power below 100 Hz, as raw LFP and ADC counts
        have, warns.
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point in **cm/s**.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (in cm/s) for ripple detection. An event is kept only if
        the speed at its first and last sample is at or below this value
        (``exclude_movement``); speed inside the event is not tested. Apply a
        whole-event rule afterwards if one is needed. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, pass ``np.inf``, which also
        keeps events whose speed is unknown (NaN).
    minimum_duration : float, optional
        Minimum time above threshold in **seconds**. Default is 0.015 (15
        milliseconds). The signal must stay at or above ``zscore_threshold`` for
        at least ``minimum_sample_count(time, minimum_duration)`` consecutive
        samples, rounded half up from the median timestamp step (23 at 1500 Hz
        and 15 ms). The 15 ms is Karlsson & Frank 2009's; the rounding is the
        Frank lab ``extractevents`` convention. The event is then extended to
        the surrounding mean-crossings, so the reported ``duration`` is
        typically longer.
        It is the time above threshold, not the whole event's duration; for a
        minimum on the whole event, which most published minimums mean, see
        ``detect_events_from_trace(minimum_event_duration=)``.
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    zscore_threshold : float, optional
        Detection sensitivity threshold in standard deviations above mean.
        Default is 3.0, the per-tetrode threshold of Karlsson & Frank 2009.
        Lower values detect more events.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in **seconds**.
        Default is 0.004 (4 ms). Rarely needs adjustment; increase for
        noisier data.
    close_ripple_threshold : float, optional
        Minimum time in **seconds** between ripples. Events closer than this
        are excluded -- the later event is dropped, not merged. Default is 0.0
        (no exclusion). Set to 0.05-0.1 s to drop closely-spaced events.
    normalization_method : {'zscore', 'median_mad'}, optional
        Method for normalizing each channel. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains outliers.
        The median/MAD method is more resistant to extreme values.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask to specify which samples to use for computing normalization
        statistics. For example, use `speed <= speed_threshold` to compute
        statistics only during immobility. Default is None (use all data).

    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with detected ripples and comprehensive statistics (see
        Kay_ripple_detector for column descriptions). The z-score statistics
        (``max_sustained_zscore``, ``mean_zscore``, ``max_zscore``, ...) are computed on
        the elementwise maximum across channels of the per-channel z-scores,
        i.e. the strongest tetrode at each sample, so ``max_sustained_zscore`` is at
        least ``zscore_threshold`` for every event.

        Returns empty DataFrame if no ripples detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 3.0 to 2.0)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your data contains ripple oscillations (150-250 Hz)

    Notes
    -----
    Missing samples: a NaN or infinite value in any channel of
    ``filtered_lfps`` marks that sample missing, and a step in ``time``
    larger than 1.5 times its median step ends a block as a missing sample
    does. Every detector in the package splits the valid samples into these
    blocks and runs every step within one, so nothing is computed across a
    gap and no event spans one; an event cut off by a gap or by the
    recording edge is kept and flagged in ``clipped_start`` and
    ``clipped_end``. Here a block too short for an event of
    ``minimum_duration`` is treated as missing, with a warning, and no block
    left raises. A NaN in ``speed`` is an unknown speed, not a missing
    sample: it splits no block, and an event whose first or last sample has
    unknown speed fails the speed criterion.
    See the README's "Choosing a detector" table for how the detectors'
    conventions differ.

    References
    ----------
    .. [1] Karlsson, M. P., & Frank, L. M. (2009). Awake replay of remote
       experiences in the hippocampus. Nature Neuroscience, 12(7), 913-918.
       doi:10.1038/nn.2344

    Examples
    --------
    >>> from ripple_detection import filter_ripple_band
    >>> from ripple_detection.simulate import simulate_session, simulate_time
    >>> time = simulate_time(45_000, 1500)  # 30 s at 1500 Hz
    >>> session = simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], rng=0)
    >>> filtered_lfps = filter_ripple_band(session.lfps, sampling_frequency=1500)
    >>> events = Karlsson_ripple_detector(time, filtered_lfps, session.speed, 1500)
    >>> events.index.name, bool(len(events))
    ('event_number', True)

    """
    _check_threshold_parameters(
        speed_threshold=speed_threshold,
        minimum_duration=minimum_duration,
        maximum_duration=maximum_duration,
        close_ripple_threshold=close_ripple_threshold,
        smoothing_sigma=smoothing_sigma,
        zscore_threshold=zscore_threshold,
    )
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, filtered_lfps, minimum_duration=minimum_duration)
    _reject_flat_channels(filtered_lfps, blocks, "filtered_lfps")
    _warn_if_not_ripple_band(filtered_lfps, sampling_frequency)

    smoothed = _smoothed_envelope(filtered_lfps, blocks, sampling_frequency, smoothing_sigma)
    mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
    normalized = normalize_signal(
        smoothed, method=normalization_method, normalization_mask=mask
    )
    candidate_ripple_times = np.asarray(
        list(
            merge_overlapping_ranges(
                chain.from_iterable(
                    _threshold_blocks(
                        channel, time, blocks, minimum_duration, zscore_threshold
                    )
                    for channel in normalized.T
                )
            )
        ),
        dtype=float,
    ).reshape(-1, 2)
    keep = _is_immobile_at_endpoints(candidate_ripple_times, speed, time, speed_threshold)
    ripple_times, _ = _finish_events(
        candidate_ripple_times, keep, time, close_ripple_threshold, maximum_duration
    )
    # statistics on the strongest channel at each sample, so an event that one
    # channel triggered cannot report a sub-threshold max_sustained_zscore
    return _get_event_stats(
        ripple_times, time, normalized.max(axis=1), speed, minimum_duration, blocks
    )


@explain_call_errors
def Roumis_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    *,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 2.0,
    smoothing_sigma: float = 0.004,
    close_ripple_threshold: float = 0.0,
    normalization_method: NormalizationMethod = "zscore",
    normalization_mask: ArrayLike | None = None,
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripples using averaged square-root envelope method.

    Variant detection method that averages the square-root of squared envelopes
    across channels. Provides a balanced approach between Kay (consensus) and
    Karlsson (per-channel) methods.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in **seconds**.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals **already bandpass filtered** to ripple band (150-250 Hz).
        Must be pre-filtered using `filter_ripple_band()` before calling this detector.
        Input with most of its power below 100 Hz, as raw LFP and ADC counts
        have, warns.
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point in **cm/s**.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (in cm/s) for ripple detection. An event is kept only if
        the speed at its first and last sample is at or below this value
        (``exclude_movement``); speed inside the event is not tested. Apply a
        whole-event rule afterwards if one is needed. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, pass ``np.inf``, which also
        keeps events whose speed is unknown (NaN).
    minimum_duration : float, optional
        Minimum time above threshold in **seconds**. Default is 0.015 (15
        milliseconds). The signal must stay at or above ``zscore_threshold`` for
        at least ``minimum_sample_count(time, minimum_duration)`` consecutive
        samples, rounded half up from the median timestamp step (23 at 1500 Hz
        and 15 ms). The 15 ms is Karlsson & Frank 2009's; the rounding is the
        Frank lab ``extractevents`` convention. The event is then extended to
        the surrounding mean-crossings, so the reported ``duration`` is
        typically longer.
        It is the time above threshold, not the whole event's duration; for a
        minimum on the whole event, which most published minimums mean, see
        ``detect_events_from_trace(minimum_event_duration=)``.
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    zscore_threshold : float, optional
        Detection sensitivity threshold in standard deviations above mean.
        Default is 2.0. Lower values (e.g., 1.5) detect more events but may
        include false positives. Higher values (e.g., 3.0) are more conservative.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in **seconds**.
        Default is 0.004 (4 ms). Rarely needs adjustment; increase for
        noisier data.
    close_ripple_threshold : float, optional
        Minimum time in **seconds** between ripples. Events closer than this
        are excluded -- the later event is dropped, not merged. Default is 0.0
        (no exclusion). Set to 0.05-0.1 s to drop closely-spaced events.
    normalization_method : {'zscore', 'median_mad'}, optional
        Method for normalizing the combined trace. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains outliers.
        The median/MAD method is more resistant to extreme values.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask to specify which samples to use for computing normalization
        statistics. For example, use `speed <= speed_threshold` to compute
        statistics only during immobility. Default is None (use all data).

    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with detected ripples and comprehensive statistics (see
        Kay_ripple_detector for column descriptions).

        Returns empty DataFrame if no ripples detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 2.0 to 1.5)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your data contains ripple oscillations (150-250 Hz)

    Notes
    -----
    Missing samples: a NaN or infinite value in any channel of
    ``filtered_lfps`` marks that sample missing, and a step in ``time``
    larger than 1.5 times its median step ends a block as a missing sample
    does. Every detector in the package splits the valid samples into these
    blocks and runs every step within one, so nothing is computed across a
    gap and no event spans one; an event cut off by a gap or by the
    recording edge is kept and flagged in ``clipped_start`` and
    ``clipped_end``. Here a block too short for an event of
    ``minimum_duration`` is treated as missing, with a warning, and no block
    left raises. A NaN in ``speed`` is an unknown speed, not a missing
    sample: it splits no block, and an event whose first or last sample has
    unknown speed fails the speed criterion.
    See the README's "Choosing a detector" table for how the detectors'
    conventions differ.

    References
    ----------
    Unpublished Frank-lab variant contributed by Demetris Roumis (2017); it has
    no paper of its own. It averages across channels the square root of each
    channel's smoothed squared envelope, then z-scores, between Kay's
    consensus trace and Karlsson's per-channel rule.

    Examples
    --------
    >>> from ripple_detection import filter_ripple_band
    >>> from ripple_detection.simulate import simulate_session, simulate_time
    >>> time = simulate_time(45_000, 1500)  # 30 s at 1500 Hz
    >>> session = simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], rng=0)
    >>> filtered_lfps = filter_ripple_band(session.lfps, sampling_frequency=1500)
    >>> events = Roumis_ripple_detector(time, filtered_lfps, session.speed, 1500)
    >>> events.index.name, bool(len(events))
    ('event_number', True)

    """
    _check_threshold_parameters(
        speed_threshold=speed_threshold,
        minimum_duration=minimum_duration,
        maximum_duration=maximum_duration,
        close_ripple_threshold=close_ripple_threshold,
        smoothing_sigma=smoothing_sigma,
        zscore_threshold=zscore_threshold,
    )
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, filtered_lfps, minimum_duration=minimum_duration)
    _reject_flat_channels(filtered_lfps, blocks, "filtered_lfps")
    _warn_if_not_ripple_band(filtered_lfps, sampling_frequency)

    smoothed_power = _smoothed_envelope(
        filtered_lfps, blocks, sampling_frequency, smoothing_sigma, square=True
    )
    combined = np.mean(np.sqrt(smoothed_power), axis=1)
    return _detect_from_trace(
        combined,
        time,
        speed,
        is_valid,
        blocks,
        minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold,
        speed_threshold=speed_threshold,
        close_event_threshold=close_ripple_threshold,
        maximum_duration=maximum_duration,
        normalization_method=normalization_method,
        normalization_mask=normalization_mask,
    )
