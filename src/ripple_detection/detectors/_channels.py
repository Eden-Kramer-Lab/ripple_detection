"""Choosing the channel to detect on."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, oaconvolve, sosfiltfilt

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import FloatArray, _check_sampling_interval
from ripple_detection.detectors._blocks import _contiguous_valid_blocks, _drop_short_blocks
from ripple_detection.detectors._validation import (
    MAXIMUM_PLAUSIBLE_MINIMUM,
    _check_band,
    _check_positive,
    _check_seconds,
)

NO_CONTENT_FRACTION = 1e-10
"""A channel's median RMS at or below this fraction of its largest absolute
value is no ripple-band content. Filtering a constant, or a constant with a
step or one glitch, leaves rounding noise in double precision: at most
1.4e-15 of the level, measured at 1000 Hz to 30 kHz. The quantization floor
of 16-bit samples, the coarsest recordings, is near 1e-6 of full scale."""


@explain_call_errors
def best_ripple_channel(
    lfps: ArrayLike,
    sampling_frequency: float,
    *,
    time: ArrayLike | None = None,
    band: tuple[float, float] = (140.0, 180.0),
    rms_window: float = 0.012,
) -> tuple[int, FloatArray]:
    """Pick the channel whose ripple-band RMS is most bursty, as buzcode does.

    buzcode's ``bz_GetBestRippleChan`` band-passes each channel to 140-180
    Hz (a Butterworth filter of order 4, applied forward and backward), takes
    a centred moving RMS over 15 samples (12 ms at 1250 Hz), and scores the
    channel by the RMS's mean over its median. A channel with large,
    occasional ripples over a quiet baseline scores high; noise alone, or a
    continuous rhythm in the band, scores near 1 whatever its amplitude. The
    highest score wins, the first channel on a tie, as MATLAB's ``max`` takes
    it. The original calls itself a placeholder for a channel's SNR.

    The candidates should be channels in the CA1 pyramidal layer: the score
    knows no anatomy, and a channel elsewhere with bursts in the band (a
    cortical or dentate site, or one picking up spike leakage) can win.
    Harvey et al. 2023 restricts the choice to CA1 shanks before choosing.

    Parameters
    ----------
    lfps : array_like, shape (n_time, n_channels)
        Raw LFP, any units (the filter is applied here). NaN or infinity in
        any channel marks that sample missing in every channel.
    sampling_frequency : float
        Sampling rate in Hz; the filter is designed for it.
    time : array_like, shape (n_time,), optional
        Increasing sample timestamps in seconds. When given, a step larger
        than 1.5 times the median step splits the recording as a missing
        sample does, and ``sampling_frequency`` is checked against the median
        step. Default None assumes a regular sample grid.
    band : tuple of (float, float), optional
        Pass-band in Hz. Default (140, 180), buzcode's.
    rms_window : float, optional
        Length in **seconds** of the moving RMS window, rounded half up to a
        whole number of samples. Default 0.012, buzcode's 15 samples at
        1250 Hz.

    Returns
    -------
    channel : int
        Column of ``lfps`` with the highest score.
    scores : ndarray, shape (n_channels,)
        Mean over median of each channel's ripple-band RMS, over the samples
        not missing. NaN for a channel with no ripple-band content: one whose
        median RMS is at most 1e-10 of its largest absolute value
        (``NO_CONTENT_FRACTION``). That is a dead or disconnected channel, or
        one railed (saturated) over most of the recording, constant but for
        rounding, whatever its level and whatever a glitch or a step in it
        adds to the mean.

    Raises
    ------
    ValueError
        If ``lfps`` is not 2-D with at least one channel; ``sampling_frequency``
        is not positive and finite; the band is not ``0 < low < high <
        Nyquist``; ``rms_window`` is not positive and finite, is 1 s or more
        (milliseconds given for seconds), or rounds to less than one sample;
        no sample is finite in every channel; no block of finite samples is
        longer than the filter's padding (24 samples); no channel can be
        scored. Also if ``time`` does not have one entry per row, holds a
        nonfinite or decreasing timestamp, has a median step of zero, or has a
        median step more than 10 percent from ``1 / sampling_frequency``.
    TypeError
        If ``sampling_frequency`` or ``rms_window`` is not a number.

    Warns
    -----
    UserWarning
        When blocks of finite samples too short to filter are treated as
        missing, or the median step of ``time`` is 2 to 10 percent from
        ``1 / sampling_frequency``.

    See Also
    --------
    ripple_detection.filter_ripple_band : The ripple-band filter the detectors take.

    Notes
    -----
    Transcribed from buzsakilab/buzcode
    `analysis/SharpWaveRipples/bz_GetBestRippleChan.m
    <https://github.com/buzsakilab/buzcode/blob/0969ddf7f55ccaca8c71969bee4b21f310840047/analysis/SharpWaveRipples/bz_GetBestRippleChan.m>`_
    and `utilities/fastrms.m
    <https://github.com/buzsakilab/buzcode/blob/0969ddf7f55ccaca8c71969bee4b21f310840047/utilities/fastrms.m>`_.
    The forward-backward pass pads each end with 24 samples, three times the
    filter order, as MATLAB's ``filtfilt`` does. The RMS window is centred as
    MATLAB's ``conv(..., 'same')`` centres it (for an even window, one more
    sample after than before) and zero-padded at each end of a block. At
    1250 Hz the scores agree with a direct transcription to 1e-9 (tested).

    This differs from the original in three ways:

    - **The filter is designed for the given rate.** The original
      normalizes the band by a fixed 625 Hz, so it is 140-180 Hz only at
      1250 Hz (at 30 kHz it would pass 3.4-4.3 kHz). The filter is the
      original's in second-order sections (``scipy.signal.sosfiltfilt``),
      which stay accurate where the transfer function loses precision, at
      high rates and narrow bands.
    - **No exclusion by units.** The original zeroes the score of a channel
      whose mean or median RMS is below 1, a level that means something only
      in ADC counts. Here a channel is left out (its score NaN) only when
      its median RMS is negligible relative to its own values, so
      microvolts, millivolts and counts choose alike.
    - **Missing samples and gaps.** The original assumes one unbroken
      recording. Here, as in every detector, the filter and the RMS run
      within each block of finite samples, split at a NaN in any channel or
      a gap in ``time``, and the mean and median are taken over the samples
      not missing. The original filters in single precision; this, in double.

    Other labs choose differently, and this does not implement them: Harvey
    et al. 2023's pipeline (ryanharvey1/ripple_heterogeneity) takes the CA1
    channel with the highest 100-250 Hz band power ratio, and neurocode's
    ``swrChannels`` (ayalab1/neurocode) combines a ripple signal-to-noise
    ratio with ripple-triggered wavelet power.

    Examples
    --------
    >>> import numpy as np
    >>> from ripple_detection import best_ripple_channel
    >>> fs = 1250
    >>> t = np.arange(20 * fs) / fs
    >>> rng = np.random.default_rng(0)
    >>> bursts = sum(
    ...     np.exp(-0.5 * ((t - c) / 0.015) ** 2) * np.sin(2 * np.pi * 160 * (t - c))
    ...     for c in range(1, 20, 2)
    ... )
    >>> continuous = 0.5 * np.sin(2 * np.pi * 160 * t)
    >>> lfps = np.column_stack([0 * t, 8 * bursts, continuous]) + rng.normal(size=(len(t), 3))
    >>> channel, scores = best_ripple_channel(lfps, fs)
    >>> channel  # the bursts, not the louder continuous rhythm
    1
    >>> scores.shape
    (3,)

    """
    _check_positive(sampling_frequency=sampling_frequency, rms_window=rms_window)
    _check_seconds(
        MAXIMUM_PLAUSIBLE_MINIMUM, "would average every ripple away", rms_window=rms_window
    )
    _check_band("band", band, sampling_frequency)
    # round half up, as minimum_sample_count rounds durations
    n_window = int(np.floor(rms_window * sampling_frequency + 0.5 + 1e-6))
    if n_window < 1:
        msg = (
            f"rms_window ({rms_window} s) is shorter than one sample at "
            f"{sampling_frequency:g} Hz."
        )
        raise ValueError(msg)

    values = np.asarray(lfps)
    if values.dtype.kind not in "iuf":
        values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        msg = (
            "lfps must be a 2-D array with shape (n_time, n_channels), the candidate "
            f"channels as columns; got shape {values.shape}."
        )
        raise ValueError(msg)
    if values.shape[1] == 0:
        msg = f"lfps must hold at least one channel; got shape {values.shape}."
        raise ValueError(msg)
    is_valid = np.isfinite(values).all(axis=1)
    if not np.any(is_valid):
        msg = "No sample of lfps is finite in every channel, so there is nothing to score."
        raise ValueError(msg)
    blocks = _contiguous_valid_blocks(is_valid, time)
    if time is not None and len(values) > 1:
        # the filter is designed for the stated rate, so the timestamps must agree
        _check_sampling_interval(
            float(np.median(np.diff(np.asarray(time, dtype=float)))), sampling_frequency
        )

    order = 4
    sos = butter(order, band, btype="bandpass", output="sos", fs=sampling_frequency)
    # MATLAB's filtfilt pads 3 x (nfilt - 1) samples, nfilt - 1 being the
    # band-pass's order of 2 x 4; sosfiltfilt needs more samples than that
    padlen = 3 * 2 * order
    blocks = _drop_short_blocks(blocks, is_valid, padlen + 1, "the band filter")

    window = np.ones(n_window)
    scores = np.full(values.shape[1], np.nan)
    # one channel at a time, as the original loops: a whole recording's filtered
    # copy, its square and its RMS in double for every channel at once would be
    # several times the input's size
    for channel in range(values.shape[1]):
        segments = [values[start:stop, channel].astype(float) for start, stop in blocks]
        rms = []
        for segment in segments:
            filtered = sosfiltfilt(sos, segment, padlen=padlen)
            # conv(x.^2, ones(n,1), 'same'): the full convolution from
            # floor(n / 2), so for an even n the window reaches one sample
            # further ahead than back
            summed = oaconvolve(filtered**2, window, mode="full")
            summed = summed[n_window // 2 : n_window // 2 + len(segment)]
            # the FFT convolution can round a sum of squares below zero
            rms.append(np.sqrt(np.maximum(summed, 0.0) / n_window))
        pooled = np.concatenate(rms)
        median = np.median(pooled)
        # rounding noise, not content: a constant (dead, disconnected or railed)
        # channel filters to about 1e-15 of its level, and one glitch or step
        # would then divide a finite mean by almost nothing
        largest = max(float(np.max(np.abs(segment))) for segment in segments)
        if median > NO_CONTENT_FRACTION * largest:
            scores[channel] = np.mean(pooled) / median

    if np.all(np.isnan(scores)):
        msg = (
            "No channel has ripple-band content to score: on every channel the median "
            f"RMS is at most {NO_CONTENT_FRACTION:g} of its largest value, as on a dead, "
            "disconnected or railed channel."
        )
        raise ValueError(msg)
    return int(np.nanargmax(scores)), scores
