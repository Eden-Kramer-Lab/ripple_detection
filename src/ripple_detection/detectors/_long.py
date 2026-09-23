"""J. D. Long II's two-channel sharp-wave ripple detector on raw LFP."""

from itertools import pairwise

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.cluster.vq import kmeans2
from scipy.ndimage import convolve1d

from ripple_detection.core import (
    FloatArray,
    _is_immobile_at_endpoints,
    _unit_area_gaussian,
    minimum_sample_count,
    normalize_signal,
)
from ripple_detection.detectors._blocks import (
    _drop_short_blocks,
    _reject_flat_channels,
    _valid_blocks,
)
from ripple_detection.detectors._events import (
    _get_event_stats,
)
from ripple_detection.detectors._validation import (
    _check_band,
    _check_gap,
    _check_positive,
    _check_thresholds,
    _validate_detector_inputs,
    _validate_duration_limits,
)


def _gaussian_lowpass_fir(
    cutoff: float, sampling_frequency: float, n_sd: float = 6.0
) -> FloatArray:
    """Unit-area Gaussian low-pass kernel with standard deviation
    ``fs / (2 pi cutoff)`` samples, truncated at ``n_sd`` standard deviations
    (Eran Stark's ``makegausslpfir``)."""
    sigma_samples = sampling_frequency / (2.0 * np.pi * cutoff)
    return _unit_area_gaussian(sigma_samples, max(n_sd, 3.0))


def _firfilt(x: FloatArray, kernel: FloatArray) -> FloatArray:
    """Zero-phase FIR filtering along axis 0 with the ends reflected.

    A centered convolution with the signal mirrored at both ends, which is what
    Eran Stark's ``firfilt`` (mirror-pad, causal filter, crop the delay)
    computes for the odd symmetric kernels used here.
    """
    return np.asarray(
        convolve1d(np.asarray(x, dtype=float), kernel, axis=0, mode="reflect"), dtype=float
    )


def _difference_of_gaussians_band(
    x: FloatArray, band: tuple[float, float], sampling_frequency: float
) -> FloatArray:
    """Band-pass as the difference of two Gaussian low-passes: low-pass at the
    band's upper edge, minus a low-pass of that at the band's lower edge."""
    low_passed = _firfilt(x, _gaussian_lowpass_fir(band[1], sampling_frequency))
    slow = _firfilt(low_passed, _gaussian_lowpass_fir(band[0], sampling_frequency))
    return low_passed - slow


def _matlab_percentile(values: FloatArray, percent: float) -> float:
    """MATLAB ``prctile``: linear interpolation between order statistics placed
    at percentiles 100 (k - 0.5) / n (NumPy's ``hazen`` method)."""
    return float(np.percentile(values, percent, method="hazen"))


def Long_sharp_wave_ripple_detector(
    time: ArrayLike,
    raw_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    *,
    speed_threshold: float = 4.0,
    sharp_wave_band: tuple[float, float] = (2.0, 50.0),
    ripple_band: tuple[float, float] = (80.0, 250.0),
    sharp_wave_percentile: float = 10.0,
    ripple_power_percentile: float = 50.0,
    window_size: float = 0.040,
    local_window: float = 5.0,
    sharp_wave_thresholds: tuple[float, float] = (0.5, 2.5),
    ripple_thresholds: tuple[float, float] = (0.5, 2.5),
    minimum_separation: float = 0.050,
    minimum_sharp_wave_duration: float = 0.020,
    maximum_sharp_wave_duration: float = 0.500,
    minimum_ripple_duration: float = 0.025,
    random_state: int | np.random.Generator | None = 0,
) -> pd.DataFrame:
    """Detect sharp-wave ripples from a pyramidal-layer and a stratum radiatum channel.

    John D. Long II's two-channel detector (buzcode ``bz_DetectSWR`` [1]_,
    converted to buzcode by Andrea Navas-Olive; filtering routines adapted
    from Eran Stark's ``detect_hfos``; carried into AYA-lab neurocode as
    ``DetectSWR`` [2]_). It needs two channels: one that records the ripple,
    in or just above the CA1 pyramidal layer, and a deeper one that records
    the sharp wave in stratum radiatum. It cannot run on a single layer.

    **Unlike the other detectors, this one takes raw, unfiltered LFP**, shape
    ``(n_time, 2)`` with the ripple channel first, because it filters both
    bands itself. Handing it ripple-band data raises nothing and returns
    nonsense, and no check on the array can tell the two apart for every
    recording, so the caller must know which it holds. The sharp-wave feature is the ripple channel minus the
    radiatum channel after a 2-50 Hz difference-of-Gaussians band-pass; the
    ripple feature is the smoothed rectified 80-250 Hz band of the
    common-average-referenced pair, maximum over the two channels. In each
    non-overlapping 40 ms block the sharp-wave maximum and the ripple maximum
    within +/-20 ms of it form a feature pair; two-cluster k-means on those
    pairs separates sharp-wave ripples (the smaller cluster) from the rest,
    and candidates must exceed the 10th sharp-wave percentile of the SWR
    cluster and the 50th ripple-power percentile of the other. Because that
    sharp-wave cut is taken from the SWR cluster itself, the detector discards
    roughly the weakest tenth of its own events by construction. Each candidate
    is then tested against **local** statistics over +/-5 s: its sharp wave and
    ripple power must each be at least median + 2.5 SD, its boundaries are
    where the sharp wave falls back below median + 0.5 SD, candidates closer
    than 50 ms to the previous candidate are dropped, and duration limits
    apply. Event bounds are the sharp-wave bounds.

    Reimplemented from the algorithm as read. The source states no license.
    It departs from the original in five ways:

    1. ``random_state`` seeds the k-means, where MATLAB's is unseeded.
    2. A candidate whose local window holds no sample below the boundary
       threshold is rejected, where the original errors.
    3. The package's endpoint speed rule is applied afterwards; a NaN in
       ``speed`` is an unknown speed, which fails it at an endpoint but
       splits no block.
    4. Missing samples (NaN in ``raw_lfps``, or a gap in ``time``) split the
       recording into blocks: the filters and candidate windows run within
       each block, the k-means pools the candidates of every block, and a
       candidate within ``local_window`` of a block edge is not evaluated,
       as the original does at the record edges. A block shorter than the
       sharp-wave low-pass kernel is treated as missing, with a warning.
       ``clipped_start`` and ``clipped_end`` are therefore False in practice;
       the one case that can set one is a sharp-wave boundary falling
       exactly on the edge of the local window.
    5. The duration limits are the package's round-half-up sample counts
       (``sample_count_within``), where the original floors
       ``duration * rate``.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in seconds.
    raw_lfps : array_like, shape (n_time, 2)
        **Raw** LFP: column 0 the ripple (pyramidal-layer) channel, column 1
        the sharp-wave (stratum radiatum) channel. NaN marks missing samples.
    speed : array_like, shape (n_time,)
        Animal's running speed in cm/s.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Endpoint speed rule (``exclude_movement``); not part of the original.
        Default is 4.0.
    sharp_wave_band, ripple_band : tuple of (float, float), optional
        Difference-of-Gaussians pass-bands in Hz. The edges are the corner
        frequencies of Gaussian low-passes, not a brick wall: gain is well
        under 1 at the nominal edges and tails off gradually beyond them, as
        in the original. Defaults (2, 50) and
        (80, 250).
    sharp_wave_percentile, ripple_power_percentile : float, optional
        Cluster-derived thresholds: a candidate must exceed this percentile
        of the SWR cluster's sharp-wave feature and this percentile of the
        non-SWR cluster's ripple power. Defaults 10 and 50.
    window_size : float, optional
        Candidate block length in seconds. Default is 0.040, following
        neurocode ``DetectSWR`` (as does ``minimum_separation`` 0.050); buzcode
        ``bz_DetectSWR`` uses 0.200 and 0.100.
    local_window : float, optional
        Half-width in seconds of the window for local statistics. Candidates
        closer than this to either end of their block are not evaluated.
        Default is 5.0.
    sharp_wave_thresholds, ripple_thresholds : tuple of (float, float), optional
        ``(boundary, peak)`` in local standard deviations. Defaults (0.5, 2.5).
    minimum_separation : float, optional
        Minimum time from the previous candidate, in seconds, whether or not
        that candidate was kept. Two quirks of the original are kept: the
        last candidate is exempt from the test, and the first is measured
        from the start of the record. Default 0.050.
    minimum_sharp_wave_duration, maximum_sharp_wave_duration : float, optional
        Sharp-wave duration limits in seconds, applied as inclusive
        round-half-up sample counts (``sample_count_within``). A candidate is
        rejected when both its sharp wave and its ripple are below their
        minimum, or when the sharp wave exceeds the maximum. Defaults 0.020
        and 0.500.
    minimum_ripple_duration : float, optional
        Ripple duration minimum in seconds. Default 0.025. The ripple's sample
        count is the original's, one less than the inclusive count used for
        the sharp wave, so ``ripple_duration`` is one sample shorter than the
        span between its two boundary crossings.
    random_state : int or numpy.random.Generator, optional
        Seed, or a Generator, for the k-means initialization, as
        ``numpy.random.default_rng`` takes it. Default is 0, so two runs on
        the same data give the same events; the original's k-means is
        unseeded, which ``None`` reproduces.

    Returns
    -------
    ripple_times : pd.DataFrame
        One row per event, indexed by ``event_number``: ``start_time``,
        ``end_time``, ``peak_time`` (sharp-wave peak), ``duration``, the
        package's z-score statistics computed on the globally z-scored ripple
        power, the speed statistics, and ``sharp_wave_zscore``,
        ``sharp_wave_local_percentile``, ``ripple_power_zscore``,
        ``ripple_power_local_percentile`` (peak values against the local window),
        ``sharp_wave_duration`` and ``ripple_duration`` in seconds.

    References
    ----------
    .. [1] Long, J. D. II. ``bz_DetectSWR.m`` in buzcode, GPL-3. No
       accompanying paper.
       https://github.com/buzsakilab/buzcode/blob/0969ddf7f55ccaca8c71969bee4b21f310840047/analysis/SharpWaveRipples/bz_DetectSWR.m
    .. [2] AYA lab, neurocode, ``SharpWaveRipples/DetectSWR.m`` (no license
       file), doi:10.5281/zenodo.7819979
       https://github.com/ayalab1/neurocode/blob/d166a67ffb73096d8d11b14be6693d96ad63e4ed/SharpWaveRipples/DetectSWR.m

    """
    _validate_duration_limits(
        minimum_sharp_wave_duration,
        maximum_sharp_wave_duration,
        names=("minimum_sharp_wave_duration", "maximum_sharp_wave_duration"),
    )
    _validate_duration_limits(
        minimum_ripple_duration, None, names=("minimum_ripple_duration", "")
    )
    lfp = np.asarray(raw_lfps, dtype=float)
    if lfp.ndim != 2 or lfp.shape[1] != 2:
        msg = (
            "raw_lfps must have exactly two channels, shape (n_time, 2): the ripple "
            f"channel first and the sharp-wave channel second; got shape {lfp.shape}."
        )
        raise ValueError(msg)
    time, lfp, speed = _validate_detector_inputs(
        time, lfp, speed, sampling_frequency, speed_threshold
    )
    _check_band("sharp_wave_band", sharp_wave_band, sampling_frequency)
    _check_band("ripple_band", ripple_band, sampling_frequency)
    _check_thresholds(
        "sharp_wave_thresholds[0]",
        sharp_wave_thresholds[0],
        "sharp_wave_thresholds[1]",
        sharp_wave_thresholds[1],
    )
    _check_thresholds(
        "ripple_thresholds[0]",
        ripple_thresholds[0],
        "ripple_thresholds[1]",
        ripple_thresholds[1],
    )
    for name, percentile in (
        ("sharp_wave_percentile", sharp_wave_percentile),
        ("ripple_power_percentile", ripple_power_percentile),
    ):
        if not 0 < percentile < 100:
            msg = f"{name} must lie in (0, 100), got {percentile}."
            raise ValueError(msg)
    _check_positive(window_size=window_size, local_window=local_window)
    _check_gap(minimum_separation=minimum_separation)
    n_time = len(time)
    is_valid, blocks = _valid_blocks(time, lfp)
    _reject_flat_channels(lfp, blocks, "raw_lfps")
    slowest_kernel = len(_gaussian_lowpass_fir(sharp_wave_band[0], sampling_frequency))
    blocks = _drop_short_blocks(
        blocks,
        is_valid,
        slowest_kernel,
        f"the {sharp_wave_band[0]} Hz sharp-wave low-pass "
        f"({slowest_kernel / sampling_frequency:.2f} s)",
    )
    rng = np.random.default_rng(random_state)

    # features, within each block
    power_kernel = _gaussian_lowpass_fir(
        float(np.mean(ripple_band)) / np.pi, sampling_frequency
    )
    sharp_wave_diff = np.full(n_time, np.nan)
    ripple_power = np.full(n_time, np.nan)
    for start, stop in blocks:
        block_lfp = lfp[start:stop]
        band_lfp = _difference_of_gaussians_band(
            block_lfp, sharp_wave_band, sampling_frequency
        )
        sharp_wave_diff[start:stop] = band_lfp[:, 0] - band_lfp[:, 1]
        referenced = block_lfp - block_lfp.mean(axis=1, keepdims=True)
        ripple = np.abs(
            _difference_of_gaussians_band(referenced, ripple_band, sampling_frequency)
        )
        ripple_power[start:stop] = _firfilt(ripple, power_kernel).max(axis=1)

    # candidate feature pairs, one per non-overlapping window within a block
    window = int(np.floor(window_size * sampling_frequency))
    half_window = window // 2
    bound = int(local_window * sampling_frequency)
    feature_index_list: list[int] = []
    sharp_wave_list: list[float] = []
    ripple_list: list[float] = []
    in_range_list: list[bool] = []
    for block_start, block_stop in blocks:
        # arange to block_stop inclusive, so a block that is an exact multiple of
        # the window keeps its last complete window; a partial window is dropped
        for w0, w1 in pairwise(np.arange(block_start, block_stop + 1, window)):
            segment = sharp_wave_diff[w0:w1]
            local_arg = int(np.argmax(segment))
            peak = int(w0) + local_arg
            if local_arg in (0, window - 1):
                if peak in (block_start, block_stop - 1):
                    continue
                if int(np.argmax(sharp_wave_diff[peak - 1 : peak + 2])) != 1:
                    continue
            feature_index_list.append(peak)
            sharp_wave_list.append(float(segment[local_arg]))
            lo = max(peak - half_window, block_start)
            hi = min(peak + half_window, block_stop - 1)
            ripple_list.append(float(ripple_power[lo : hi + 1].max()))
            # the local statistics need the whole +/- local_window inside the block
            in_range_list.append(
                peak - bound >= block_start and peak + bound <= block_stop - 1
            )
    feature_index = np.asarray(feature_index_list, dtype=int)
    sharp_wave_feature = np.asarray(sharp_wave_list, dtype=float)
    ripple_feature = np.asarray(ripple_list, dtype=float)
    in_range = np.asarray(in_range_list, dtype=bool)
    if len(feature_index) < 2:
        msg = "Too few candidate windows to cluster; the recording is too short."
        raise ValueError(msg)

    features = np.column_stack([sharp_wave_feature, ripple_feature])
    _, labels = kmeans2(features, 2, iter=100, minit="++", seed=rng)
    is_swr_cluster = labels == (0 if np.sum(labels == 0) <= np.sum(labels == 1) else 1)
    if not np.any(is_swr_cluster) or np.all(is_swr_cluster):
        msg = "k-means did not separate the candidate features into two clusters."
        raise ValueError(msg)
    sharp_wave_cut = _matlab_percentile(
        sharp_wave_feature[is_swr_cluster], sharp_wave_percentile
    )
    ripple_cut = _matlab_percentile(ripple_feature[~is_swr_cluster], ripple_power_percentile)
    is_candidate = (
        is_swr_cluster
        & (sharp_wave_feature > sharp_wave_cut)
        & (ripple_feature > ripple_cut)
        & in_range
    )
    candidate_peaks = feature_index[is_candidate]
    candidate_sharp = sharp_wave_feature[is_candidate]
    candidate_ripple = ripple_feature[is_candidate]
    # time from the previous candidate; the first is measured from the record start
    separation = np.diff(time[candidate_peaks], prepend=time[0])

    sw_boundary, sw_peak = sharp_wave_thresholds
    rp_boundary, rp_peak = ripple_thresholds

    records = []
    n_candidates = len(candidate_peaks)
    # the duration limits as sample counts, the rule of sample_count_within,
    # measured once rather than per candidate
    min_ripple_samples = minimum_sample_count(time, minimum_ripple_duration)
    min_sharp_wave_samples = minimum_sample_count(time, minimum_sharp_wave_duration)
    max_sharp_wave_samples = minimum_sample_count(time, maximum_sharp_wave_duration)
    for ii, peak in enumerate(candidate_peaks):
        sw_window = sharp_wave_diff[peak - bound : peak + bound + 1]
        sw_median, sw_sd = np.median(sw_window), np.std(sw_window, ddof=1)
        if sw_window[bound] < sw_median + sw_peak * sw_sd:
            continue
        rp_window = ripple_power[peak - bound : peak + bound + 1]
        rp_median, rp_sd = np.median(rp_window), np.std(rp_window, ddof=1)
        if candidate_ripple[ii] < rp_median + rp_peak * rp_sd:
            continue
        if ii < n_candidates - 1 and separation[ii] < minimum_separation:
            continue
        below_before = np.flatnonzero(sw_window[: bound + 1] < sw_median + sw_boundary * sw_sd)
        below_after = np.flatnonzero(sw_window[bound:] < sw_median + sw_boundary * sw_sd)
        if len(below_before) == 0 or len(below_after) == 0:
            continue
        start_offset = bound - below_before[-1]  # samples before the peak
        stop_offset = below_after[0]  # samples after the peak
        sharp_wave_samples = start_offset + stop_offset + 1
        rp_peak_local = (bound - half_window) + int(
            np.argmax(rp_window[bound - half_window : bound + half_window + 1])
        )
        rp_before = np.flatnonzero(
            rp_window[: rp_peak_local + 1] < rp_median + rp_boundary * rp_sd
        )
        rp_after = np.flatnonzero(rp_window[rp_peak_local:] < rp_median + rp_boundary * rp_sd)
        if len(rp_before) == 0 or len(rp_after) == 0:
            continue
        ripple_samples = (rp_peak_local + rp_after[0]) - rp_before[-1]
        sharp_wave_long_enough = sharp_wave_samples >= min_sharp_wave_samples
        if ripple_samples < min_ripple_samples and not sharp_wave_long_enough:
            continue
        if sharp_wave_samples > max_sharp_wave_samples:
            # an over-long sharp wave is dropped, whatever the ripple
            continue
        records.append(
            {
                "peak": peak,
                "start": peak - start_offset,
                "end": peak + stop_offset,
                "sharp_wave_zscore": (candidate_sharp[ii] - sw_median) / sw_sd,
                "sharp_wave_local_percentile": np.mean(sw_window < candidate_sharp[ii]),
                "ripple_power_zscore": (candidate_ripple[ii] - rp_median) / rp_sd,
                "ripple_power_local_percentile": np.mean(rp_window < candidate_ripple[ii]),
                "sharp_wave_duration": sharp_wave_samples / sampling_frequency,
                "ripple_duration": ripple_samples / sampling_frequency,
            }
        )

    detected = pd.DataFrame(records)
    if records:
        event_times = np.column_stack([time[detected["start"]], time[detected["end"]]])
        keep = _is_immobile_at_endpoints(event_times, speed, time, speed_threshold)
        detected = detected[keep].reset_index(drop=True)
        event_times = event_times[keep]
    else:
        event_times = np.empty((0, 2))

    events = _get_event_stats(
        # the reported event spans the sharp wave, so the sustained-value window
        # is measured against the sharp-wave minimum
        event_times,
        time,
        normalize_signal(ripple_power),
        speed,
        minimum_duration=minimum_sharp_wave_duration,
        blocks=blocks,
    )
    events["peak_time"] = (
        time[detected["peak"].to_numpy(dtype=int)] if len(detected) else np.empty(0)
    )
    for column in (
        "sharp_wave_zscore",
        "sharp_wave_local_percentile",
        "ripple_power_zscore",
        "ripple_power_local_percentile",
        "sharp_wave_duration",
        "ripple_duration",
    ):
        events[column] = detected[column].to_numpy() if len(detected) else np.empty(0)
    return events
