"""The Carey, Tanaka & van der Meer joint ripple-power and multiunit candidate detector."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal import butter, oaconvolve, sosfiltfilt

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    BoolArray,
    FloatArray,
    IntArray,
    _boolean_run_bounds,
    _check_choice,
    _event_bounds,
    _is_immobile,
    _unit_area_gaussian,
    _warn_at_caller,
    get_envelope,
    nearest_sample_index,
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


CAREY_THRESHOLD_METHODS = ("zscore", "mean")
"""How ``Carey_candidate_detector`` scales the joint score before thresholding."""

CAREY_SMOOTHING_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
"""The narrow kernel ``SWRfreak`` smooths its spectra with."""

WEIGHTINGS = ("amplitude", "power")
"""How ``carey_spectral_ripple_score`` weights the Fourier magnitudes."""

_SPECTRUM_CHUNK = 65_536
"""Windows transformed at once, so the stacked windows stay near 100 MB."""


class _WindowedSpectrum:
    """The van der Meer lab's ``windowedFFT``: the magnitude spectrum of a
    ``window``-long stretch around a sample, tapered at each end over one
    period of ``high_pass_cutoff`` and folded so the taper overlaps itself."""

    def __init__(
        self, sampling_frequency: float, window: float, high_pass_cutoff: float, weight_by: str
    ) -> None:
        self.n_window = round(window * sampling_frequency)
        self.n_taper = round(sampling_frequency / high_pass_cutoff)
        self.n_core = self.n_window - self.n_taper
        if self.n_core < 1:
            msg = (
                f"window ({window} s) must be longer than one period of high_pass_cutoff "
                f"({1 / high_pass_cutoff} s)."
            )
            raise ValueError(msg)
        rise = 0.5 - 0.5 * np.cos(np.pi / self.n_taper * np.arange(self.n_taper))
        self.taper = np.concatenate([rise, np.ones(self.n_core), rise[::-1]])
        self.length = self.n_core + 2 * self.n_taper
        # a window's first sample, from the sample it is centered on
        self.before = self.n_core // 2 + self.n_taper
        self.after = self.length - self.before
        # MATLAB's round(length / 2) keeps the lower half, rounding half up
        self.n_coefficients = int(np.floor(self.n_window / 2 + 0.5))
        self.weights = (
            np.arange(1, self.n_coefficients + 1, dtype=float)
            if weight_by == "power"
            else np.ones(self.n_coefficients)
        )

    def of(self, windows: FloatArray) -> FloatArray:
        """Spectra of stacked windows, shape (n_windows, length) -> (n_windows, n_coefficients)."""
        tapered = windows * self.taper
        folded = tapered[:, : self.n_taper + self.n_core].copy()
        folded[:, : self.n_taper] += tapered[:, self.n_core + self.n_taper :]
        magnitude = np.abs(np.fft.fft(folded, axis=1))[:, : self.n_coefficients]
        spectra: FloatArray = magnitude * self.weights
        return spectra

    def at(self, data: FloatArray, centers: IntArray) -> FloatArray:
        """Spectra of the windows centered on ``centers``, which must fit in ``data``."""
        offsets = np.arange(self.length) - self.before
        return self.of(data[centers[:, np.newaxis] + offsets])


def _smooth_spectrum(spectrum: FloatArray) -> FloatArray:
    return np.convolve(spectrum, CAREY_SMOOTHING_KERNEL, mode="same")


@explain_call_errors
def carey_spectral_ripple_score(
    lfp: ArrayLike,
    sampling_frequency: float,
    ripple_times: ArrayLike | pd.DataFrame,
    *,
    window: float = 0.06,
    high_pass_cutoff: float = 100.0,
    noise_offset: float = 2.0,
    weight_by: str = "amplitude",
    step: int = 1,
) -> FloatArray:
    """The spectral ripple score behind the candidates of Carey et al. 2019.

    A reimplementation of the van der Meer lab's ``SWRfreak`` and ``amSWR``
    (with ``windowedFFT``) [1]_, the ripple score of the candidate events
    released with Carey, Tanaka & van der Meer 2019 [2]_. A template spectrum
    is built from example ripples: the mean magnitude spectrum of a
    ``window``-long stretch centered on each, minus that of the stretch
    ``noise_offset`` seconds later, each smoothed and normalized to unit sum.
    The score at each sample is the dot product of the template with the
    spectrum of the stretch centered there, frequencies below
    ``high_pass_cutoff`` left out, floored at zero and rescaled to mean 1.

    Pass it to ``Carey_candidate_detector(..., ripple_score=score)`` with
    ``threshold_method="mean"``, ``low_threshold=4`` and ``high_threshold=4``
    for the rule behind the paper's published candidates.

    Parameters
    ----------
    lfp : array_like, shape (n_time,) or (n_time, 1)
        Raw LFP from one pyramidal-layer channel. NaN marks a missing sample.
    sampling_frequency : float
        Sampling rate in Hz; the original ran at 2000.
    ripple_times : array_like, shape (n_ripples, 2), or pd.DataFrame
        Example ripples, ``[start_time, end_time]`` in samples' time, for the
        template: the paper's were picked by hand for each session. Detected
        ripples, such as a strict detector's largest, can stand in. Examples
        whose stretch, or whose noise stretch, is not all finite samples are
        left out with a warning.
    window : float, optional
        Length of each stretch in seconds. Default 0.06, the original's.
    high_pass_cutoff : float, optional
        Frequencies below this, in Hz, are left out of the score, and the
        taper at each end of a stretch lasts one period of it. Default 100.
    noise_offset : float, optional
        Seconds from each example to its noise stretch. Default 2, the
        original's.
    weight_by : {'amplitude', 'power'}, optional
        ``'amplitude'`` (default, as the published candidates) uses the
        Fourier magnitudes; ``'power'`` multiplies each by its coefficient's
        index, as ``SWRfreak``'s default did.
    step : int, optional
        Compute every ``step``-th sample and interpolate the rest with a
        shape-preserving cubic, as ``amSWR``'s ``stepSize`` does, within each
        run of finite samples. Default 1, every sample.

    Returns
    -------
    score : ndarray, shape (n_time,)
        Non-negative, mean 1 over the finite samples; NaN at missing samples,
        and 0 within one window's length of missing data or the recording
        edge, where the original is 0 too. With ``step=1`` it equals the
        original's score to rounding.

    Raises
    ------
    ValueError
        If the LFP is not one channel, a parameter is out of range, the
        window is no longer than one period of ``high_pass_cutoff``, or no
        example ripple can be used.

    References
    ----------
    .. [1] van der Meer lab, ``code-matlab/tasks/Alyssa_Tmaze/beta/SWRfreak.m``,
       ``amSWR.m`` and ``windowedFFT.m`` (Elyot Grant and A. Carey, 2015), at
       vandermeerlab commit ad0bbd4d01726a436b36671c0a8b2db81476e946.
    .. [2] Carey, A. A., Tanaka, Y., & van der Meer, M. A. A. (2019). Reward
       revaluation biases hippocampal replay content away from the preferred
       outcome. Nature Neuroscience, 22(9), 1450-1459.

    Examples
    --------
    >>> from ripple_detection.simulate import simulate_session, simulate_time
    >>> time = simulate_time(40_000, 2000)
    >>> ripples = [3.0, 6.0, 9.0, 12.0, 15.0]
    >>> session = simulate_session(time, ripples, n_channels=1, rng=0)
    >>> examples = np.array([(t - 0.02, t + 0.02) for t in ripples[:3]])
    >>> score = carey_spectral_ripple_score(session.lfps[:, 0], 2000, examples)
    >>> bool(score[round(12.0 * 2000)] > 4 * np.nanmedian(score))
    True

    """
    _check_positive(
        sampling_frequency=sampling_frequency,
        window=window,
        high_pass_cutoff=high_pass_cutoff,
        noise_offset=noise_offset,
    )
    _check_choice("weight_by", weight_by, WEIGHTINGS)
    if not (step >= 1 and step == int(step)):
        msg = f"step must be a whole number of at least 1, got {step}."
        raise ValueError(msg)
    if high_pass_cutoff >= sampling_frequency / 2:
        msg = (
            f"high_pass_cutoff ({high_pass_cutoff} Hz) must be below the Nyquist frequency "
            f"({sampling_frequency / 2} Hz)."
        )
        raise ValueError(msg)
    data = np.asarray(lfp, dtype=float)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 1:
        msg = f"lfp must be one channel, shape (n_time,); got shape {data.shape}."
        raise ValueError(msg)
    spectrum = _WindowedSpectrum(sampling_frequency, window, high_pass_cutoff, weight_by)
    n_time = data.size
    time = np.arange(n_time) / sampling_frequency

    # an example's stretch must lie in finite samples; the score is computed,
    # as amSWR computes it, from one window's length after the start of each
    # run of finite samples to one window's length before its end
    finite = np.isfinite(data)
    runs = _boolean_run_bounds(finite)
    fits = np.zeros(n_time, dtype=bool)
    scored = np.zeros(n_time, dtype=bool)
    for start, stop in runs:
        if stop - spectrum.after >= start + spectrum.before:
            fits[start + spectrum.before : stop - spectrum.after + 1] = True
        if stop - spectrum.n_window - 1 >= start + spectrum.n_window - 1:
            scored[start + spectrum.n_window - 1 : stop - spectrum.n_window] = True

    examples = _event_bounds(ripple_times)
    centers = nearest_sample_index(time, examples.mean(axis=1))
    noise = nearest_sample_index(time, examples.mean(axis=1) + noise_offset)
    in_range = (examples.mean(axis=1) + noise_offset) <= time[-1]
    usable = in_range & fits[centers] & fits[noise]
    if not np.any(usable):
        msg = (
            "No example ripple can build the template: each needs a stretch of finite "
            f"samples {window} s long around it and another {noise_offset} s later."
        )
        raise ValueError(msg)
    if not np.all(usable):
        _warn_at_caller(
            f"{int((~usable).sum())} of {len(usable)} example ripple(s) left out of the "
            "template: their stretch or noise stretch runs into missing data or the edge."
        )
    ripple_spectrum = _smooth_spectrum(spectrum.at(data, centers[usable]).sum(axis=0))
    noise_spectrum = _smooth_spectrum(spectrum.at(data, noise[usable]).sum(axis=0))
    template = _smooth_spectrum(
        ripple_spectrum / ripple_spectrum.sum() - noise_spectrum / noise_spectrum.sum()
    )
    # amSWR zeroes the coefficients below the cutoff, round(cutoff * window)
    template[: int(np.floor(high_pass_cutoff * window + 0.5))] = 0.0

    score = np.zeros(n_time)
    for run_start, run_stop in _boolean_run_bounds(scored):
        computed = np.arange(run_start, run_stop, step)
        values = np.empty(computed.size)
        for chunk in range(0, computed.size, _SPECTRUM_CHUNK):
            part = computed[chunk : chunk + _SPECTRUM_CHUNK]
            values[chunk : chunk + _SPECTRUM_CHUNK] = spectrum.at(data, part) @ template
        if step > 1 and computed.size > 1:
            if computed[-1] != run_stop - 1:
                computed = np.append(computed, run_stop - 1)
                values = np.append(
                    values, spectrum.at(data, np.array([run_stop - 1])) @ template
                )
            values = PchipInterpolator(computed, values)(np.arange(run_start, run_stop))
        score[run_start:run_stop] = values
    score = np.maximum(score, 0.0)
    score[~finite] = np.nan
    mean = np.nanmean(score)
    if not mean > 0:
        msg = "The spectral score is zero everywhere; the template matches nothing."
        raise ValueError(msg)
    return score / mean


@explain_call_errors
def Carey_candidate_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike | None,
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
    spike_smoothing_sigma: float = 0.020,
    spike_cap: float = 2.0,
    baseline_smoothing_sigma: float = 0.125,
    baseline_cap: float = 4.0,
    theta_lfp: ArrayLike | None = None,
    theta_band: tuple[float, float] = (6.0, 10.0),
    theta_threshold: float = 2.0,
    state_merge_gap: float = 0.050,
    minimum_state_duration: float = 0.050,
    maximum_duration: float | None = None,
    ripple_score: ArrayLike | None = None,
    threshold_method: str = "zscore",
) -> pd.DataFrame:
    """Detect candidate replay events from ripple power and multiunit activity jointly.

    The van der Meer lab's candidate-event code, written for the data of
    Carey, Tanaka & van der Meer 2019 [1]_: ``GenCandidateEvents`` with its
    Hilbert ripple score (``OldWizard``, the code's ``'HT'`` option) and its
    multiunit score ``amMUA`` by Elyot Grant and A. Carey [2]_. A ripple score
    and a multiunit score are combined as their **geometric mean**, so an
    event needs both a ripple and a population burst, then z-scored and
    segmented with two thresholds. Reimplemented from the code as read.

    This is not the configuration behind the paper's published candidates.
    The candidate files released with the paper [3]_ were made by the code's
    earlier ``precand`` step on its spectral ripple score, ``amSWR``: the dot
    product of a sliding 60 ms spectrum with a per-session SWR spectrum built
    from ripples picked by hand, corrected by a noise spectrum. There the joint
    score, rescaled to mean 0.5, is thresholded once at 4 (eight times its
    mean, not a z-score), and an event runs between the crossings of that same
    level. The multiunit score, the geometric mean, the low-speed and (with
    ``theta_lfp``) low-theta intervals, the 20 ms minimum and the five active
    units are the same as here. To run that configuration, compute the score
    with ``carey_spectral_ripple_score`` and pass ``filtered_lfps=None``,
    ``ripple_score=score``, ``threshold_method="mean"``, ``low_threshold=4``
    and ``high_threshold=4``.

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
      gaps under ``state_merge_gap`` and dropped unless longer than
      ``minimum_state_duration``) and, when ``theta_lfp`` is given, inside a
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
    filtered_lfps : array_like, shape (n_time, n_channels), or None
        Ripple-band-filtered LFP; the original uses 140-250 Hz on one channel.
        None when ``ripple_score`` is given instead.
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
        Boundary and peak thresholds on the joint score, z-scored or scaled
        as ``threshold_method`` says, the same two-threshold rule as
        ``Zugaro_ripple_detector``. Defaults 1 and 3 (the original's
        ``DetectorThreshold`` and ``DetectorThreshold2``).
    minimum_duration : float, optional
        Minimum candidate duration in seconds, applied as an inclusive
        round-half-up sample count (``sample_count_within``); the original's
        ``RemoveIV`` compared elapsed time strictly. Default 0.020.
    minimum_active_units : int, optional
        Minimum number of units with a spike inside the candidate. Default 5.
    ripple_smoothing_sigma : float, optional
        Gaussian standard deviation in seconds of the ripple-score smoothing.
        Default 0.010.
    spike_smoothing_sigma : float, optional
        Gaussian standard deviation in seconds of each unit's spike kernel.
        Default 0.020.
    spike_cap : float, optional
        Per-unit cap in coincident spikes. Default 2.
    baseline_smoothing_sigma : float, optional
        Gaussian standard deviation in seconds of the slow baseline. Default
        0.125.
    baseline_cap : float, optional
        Baseline cap in units' worth. Default 4.
    theta_lfp : array_like, shape (n_time,) or (n_time, 1), optional
        Raw LFP of a theta channel; when given, candidates during elevated
        theta are excluded, as the original's theta restriction does. It
        needs a channel the other inputs do not, so it is optional here.
        Default None (no theta exclusion).
    theta_band : tuple of (float, float), optional
        Theta pass-band in Hz, Butterworth of total order 4 (order 2 per edge, as MATLAB's `fdesign` 'N' counts it). Default (6, 10).
    theta_threshold : float, optional
        Theta-envelope z-score at or above which a period is excluded. Default 2.
    state_merge_gap, minimum_state_duration : float, optional
        Interval rules for the low-speed and low-theta periods, in seconds.
        Defaults 0.050 and 0.050 (vandermeerlab ``TSDtoIV`` defaults).
    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.
    ripple_score : array_like, shape (n_time,), optional
        A ripple score to use in place of the Hilbert score of
        ``filtered_lfps``, such as ``carey_spectral_ripple_score``'s:
        non-negative, NaN at missing samples. It is rescaled to mean 1, as the
        original rescales its ripple score, and not smoothed. Pass
        ``filtered_lfps=None`` with it. Default None.
    threshold_method : {'zscore', 'mean'}, optional
        How the joint score is scaled before the thresholds apply:
        ``'zscore'`` (default, ``GenCandidateEvents``) z-scores it;
        ``'mean'`` rescales it to mean 0.5, as the original's ``precand`` did,
        so a threshold of 4 is eight times the mean.

    Returns
    -------
    candidate_times : pd.DataFrame
        One row per candidate, indexed by ``event_number``, with the package's
        statistics computed on the joint score as scaled for thresholding, the
        speed statistics, and ``n_active_units``.

    References
    ----------
    .. [1] Carey, A. A., Tanaka, Y., & van der Meer, M. A. A. (2019). Reward
       revaluation biases hippocampal replay content away from the preferred
       outcome. Nature Neuroscience, 22(9), 1450-1459.
       doi:10.1038/s41593-019-0464-6
    .. [2] van der Meer lab, ``code-matlab/tasks/Alyssa_Tmaze/GenCandidateEvents.m``
       with ``beta/OldWizard.m``, ``beta/amMUA.m``, and ``beta/TSDtoIV2.m``.
       https://github.com/vandermeerlab/vandermeerlab/blob/82ba3fe29cc3912575b32a0fcdaaa1c4fe097231/code-matlab/tasks/Alyssa_Tmaze/GenCandidateEvents.m
    .. [3] Candidate events released with [1]_, ``Carey_etal_submitted/SWRcandidates``
       in https://github.com/vandermeerlab/papers, made with
       ``code-matlab/tasks/Alyssa_Tmaze/precand.m`` and ``beta/amSWR.m`` at
       vandermeerlab commit ad0bbd4d01726a436b36671c0a8b2db81476e946.

    Examples
    --------
    >>> from ripple_detection import filter_ripple_band
    >>> from ripple_detection.simulate import simulate_session, simulate_time
    >>> time = simulate_time(45_000, 1500)  # 30 s at 1500 Hz
    >>> session = simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], rng=0)
    >>> filtered_lfps = filter_ripple_band(session.lfps, sampling_frequency=1500)
    >>> events = Carey_candidate_detector(
    ...     time, filtered_lfps, session.multiunit, session.speed, 1500
    ... )
    >>> bool((events.n_active_units >= 5).all())
    True

    """
    _validate_duration_limits(minimum_duration, maximum_duration)
    _check_thresholds("low_threshold", low_threshold, "high_threshold", high_threshold)
    _check_smoothing_sigma(
        ripple_smoothing_sigma=ripple_smoothing_sigma,
        spike_smoothing_sigma=spike_smoothing_sigma,
        baseline_smoothing_sigma=baseline_smoothing_sigma,
    )
    _check_positive(spike_cap=spike_cap, baseline_cap=baseline_cap)
    _check_gap(state_merge_gap=state_merge_gap, minimum_state_duration=minimum_state_duration)
    if not np.isfinite(theta_threshold):
        msg = f"theta_threshold must be finite, got {theta_threshold}."
        raise ValueError(msg)
    _check_choice("threshold_method", threshold_method, CAREY_THRESHOLD_METHODS)
    one_of_the_two = (
        "Pass filtered_lfps, from which the Hilbert ripple score is formed, or "
        "ripple_score with filtered_lfps=None; exactly one of the two."
    )
    if filtered_lfps is not None and ripple_score is not None:
        raise ValueError(one_of_the_two)
    multiunit = np.asarray(multiunit, dtype=float)
    _validate_multiunit(multiunit)
    _check_minimum_active_units(minimum_active_units, multiunit.shape[1])
    signal: ArrayLike
    if ripple_score is None:
        if filtered_lfps is None:
            raise ValueError(one_of_the_two)
        signal = filtered_lfps
    else:
        score_signal = np.asarray(ripple_score, dtype=float)
        if score_signal.ndim == 1:
            score_signal = score_signal[:, np.newaxis]
        if score_signal.ndim != 2 or score_signal.shape[1] != 1:
            msg = f"ripple_score must have shape (n_time,), got {np.shape(ripple_score)}."
            raise ValueError(msg)
        if np.any(score_signal[np.isfinite(score_signal)] < 0):
            msg = "ripple_score must be non-negative: its geometric mean with the multiunit score is taken."
            raise ValueError(msg)
        signal = score_signal
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, signal, speed, sampling_frequency, speed_threshold
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
        if theta_signal.shape == (n_time, 1):
            theta_signal = theta_signal[:, 0]
        if theta_signal.shape != (n_time,):
            msg = (
                f"theta_lfp must have shape ({n_time},) or ({n_time}, 1), one channel, "
                f"got {theta_signal.shape}."
            )
            raise ValueError(msg)
        theta_envelope = _theta_envelope(theta_signal, time, sampling_frequency, theta_band)
        signals.append(theta_envelope)
    is_valid, blocks = _valid_blocks(time, *signals, minimum_duration=minimum_duration)
    _reject_flat_channels(
        filtered_lfps, blocks, "filtered_lfps" if ripple_score is None else "ripple_score"
    )

    # ripple score, rescaled to mean 1: given, or OldWizard ('amplitude',
    # 'wizard' kernel) from the ripple-band LFP
    ripple = np.full(n_time, np.nan)
    for start, stop in blocks:
        if ripple_score is not None:
            ripple[start:stop] = filtered_lfps[start:stop, 0]
            continue
        envelope = get_envelope(filtered_lfps[start:stop]).mean(axis=1)
        ripple[start:stop] = gaussian_filter1d(
            envelope,
            ripple_smoothing_sigma * sampling_frequency,
            truncate=3.0,
            mode="constant",
        )
    ripple = ripple / np.nanmean(ripple)

    # multiunit score (amMUA)
    sigma_samples = spike_smoothing_sigma * sampling_frequency
    spike_kernel = _unit_area_gaussian(sigma_samples, 5.0)
    cap = spike_cap / (sigma_samples * np.sqrt(2.0 * np.pi))
    baseline_kernel = _unit_area_gaussian(baseline_smoothing_sigma * sampling_frequency, 12.0)
    summed = np.full(n_time, np.nan)
    baseline = np.full(n_time, np.nan)
    for start, stop in blocks:
        # zero padding at the block edges, as np.convolve(..., "same") pads, for
        # these odd symmetric kernels; a block shorter than a kernel works too
        block_sum = np.zeros(stop - start)
        for unit in multiunit[start:stop].T:
            block_sum += np.minimum(_convolve_spikes(unit, spike_kernel), cap)
        summed[start:stop] = block_sum
        # FFT convolution: the baseline kernel (SD 125 ms, +/-12 SD) has thousands of taps,
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

    joint = np.sqrt(ripple * multiunit_score)
    if threshold_method == "zscore":
        zscored = normalize_signal(joint)
    else:
        # precand: rescmean(score, 0.5)
        zscored = joint * (0.5 / np.nanmean(joint))

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
                    minimum_state_duration,
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
