"""The one missing-sample policy: valid samples, their contiguous blocks, and the
transforms and threshold tests that run within a block."""

import warnings
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike

from ripple_detection.core import (
    BoolArray,
    FloatArray,
    _get_normalization_mask,
    gaussian_smooth,
    get_envelope,
    minimum_sample_count,
    threshold_by_zscore,
)


def _valid_blocks(
    time: FloatArray, *signals: FloatArray, minimum_duration: float | None = None
) -> tuple[BoolArray, list[tuple[int, int]]]:
    """The samples every detector may use, and their contiguous blocks.

    A sample is valid when every channel of every signal is finite. Valid
    samples are split into blocks at every invalid sample and wherever the
    timestamp step exceeds 1.5 times the median step. Every step of every
    detector runs within a block, so nothing is computed across a gap and
    no event spans one.

    Speed is not one of the signals: it does not enter any trace, so a NaN
    in speed is an unknown speed, which the movement rules handle, not a
    missing sample that would split a block and cut a ripple in two.

    A block with fewer samples than ``minimum_duration`` spans cannot hold an
    event, so it is treated as missing with a warning that gives its sample
    ranges, and the detector raises if no block is left: a result emptied by
    missing data should say so, not look like a recording without ripples.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    *signals : ndarray, shape (n_time,) or (n_time, n_channels)
        The LFP and spikes the detector reads.
    minimum_duration : float, optional
        The detector's minimum event duration in seconds. Default None keeps
        every block.

    Returns
    -------
    is_valid : ndarray of bool, shape (n_time,)
    blocks : list of (start, stop)
        Half-open index ranges of the valid blocks, in order.

    Raises
    ------
    ValueError
        If no sample is valid, or no block is long enough for an event.

    """
    is_valid = np.ones(len(time), dtype=bool)
    for signal in signals:
        finite = np.isfinite(signal)
        is_valid &= finite.all(axis=1) if finite.ndim == 2 else finite
    if not np.any(is_valid):
        empty = [
            f"channel(s) {np.flatnonzero(~np.isfinite(signal).any(axis=0)).tolist()} "
            f"of signal {position + 1}"
            for position, signal in enumerate(signals)
            if signal.ndim == 2 and len(signal) and (~np.isfinite(signal).any(axis=0)).any()
        ]
        cause = (
            f"; {' and '.join(empty)} hold no finite sample. Drop them before detecting"
            if empty
            else ". Check the alignment of the inputs"
        )
        msg = f"Every sample has a NaN in at least one channel, so there is nothing to detect on{cause}."
        raise ValueError(msg)
    blocks = _contiguous_valid_blocks(is_valid, time)
    if minimum_duration is not None:
        blocks = _drop_short_blocks(
            blocks,
            is_valid,
            minimum_sample_count(time, minimum_duration),
            f"an event of minimum_duration ({minimum_duration} s)",
            stacklevel=4,
        )
    return is_valid, blocks


def _reject_flat_channels(signal: FloatArray, is_valid: BoolArray, name: str) -> None:
    """Raise for a channel that is constant over the valid samples.

    A dead or disconnected channel adds nothing to a sum or mean of
    envelopes, so a detector that combines channels would run on fewer than
    the caller passed, and dilute the rest, without a word. The per-channel
    detectors already raise on its zero normalization scale.
    """
    valid = signal[is_valid]
    if len(valid) < 2:
        return  # one sample has no spread; the normalization reports that
    flat = np.flatnonzero(np.all(valid == valid[0], axis=0))
    if flat.size:
        msg = (
            f"{name} channel(s) {flat.tolist()} are constant over the valid samples, "
            "as a dead or disconnected channel is. Drop them before detecting."
        )
        raise ValueError(msg)


def _drop_short_blocks(
    blocks: list[tuple[int, int]],
    is_valid: BoolArray,
    minimum_length: int,
    reason: str,
    stacklevel: int = 3,
) -> list[tuple[int, int]]:
    """Treat blocks shorter than a detector's transform needs as missing.

    Marks their samples invalid in place, warns with their sample ranges, and
    raises if no block remains.
    """
    short = [(start, stop) for start, stop in blocks if stop - start < minimum_length]
    if not short:
        return blocks
    for start, stop in short:
        is_valid[start:stop] = False
    kept = [(start, stop) for start, stop in blocks if stop - start >= minimum_length]
    if not kept:
        msg = (
            f"No block of finite samples is as long as the {minimum_length} samples that "
            f"{reason} needs."
        )
        raise ValueError(msg)
    warnings.warn(
        f"{len(short)} block(s) of finite samples shorter than the {minimum_length} samples "
        f"that {reason} needs are treated as missing (sample ranges "
        f"{short[:5]}{', ...' if len(short) > 5 else ''}).",
        UserWarning,
        stacklevel=stacklevel,
    )
    return kept


def _mask_invalid(signal: FloatArray, is_valid: BoolArray) -> FloatArray:
    """A copy of ``signal`` with NaN at every invalid sample, so a helper that
    splits blocks on its own splits them where the detector does."""
    masked = signal.copy()
    masked[~is_valid] = np.nan
    return masked


def _normalization_mask_over_valid(
    n_time: int, is_valid: BoolArray, normalization_mask: ArrayLike | None
) -> BoolArray:
    """The samples the normalization statistics come from: the caller's mask,
    if any, restricted to valid samples."""
    mask = _get_normalization_mask((n_time,), normalization_mask)
    mask = is_valid if mask is None else mask & is_valid
    if not np.any(mask):
        msg = (
            "The normalization mask selects no sample that is finite in every signal; "
            "cannot compute normalization statistics."
        )
        raise ValueError(msg)
    return mask


def _threshold_blocks(
    normalized: FloatArray,
    time: FloatArray,
    blocks: list[tuple[int, int]],
    minimum_duration: float,
    zscore_threshold: float,
) -> list[tuple[float, float]]:
    """``threshold_by_zscore`` within each block; events never span a gap.

    The blocks come from ``_valid_blocks`` with ``minimum_duration``, so each
    can hold an event.
    """
    events: list[tuple[float, float]] = []
    for start, stop in blocks:
        events.extend(
            threshold_by_zscore(
                normalized[start:stop], time[start:stop], minimum_duration, zscore_threshold
            )
        )
    return events


def _contiguous_valid_blocks(
    is_valid: BoolArray, time: FloatArray | None
) -> list[tuple[int, int]]:
    """Split rows into maximal contiguous valid blocks.

    A block ends at an invalid row or, when ``time`` is given, wherever the
    timestamp step exceeds 1.5 times the median step (a recording gap or the
    join between disjoint intervals). The median step is measured from
    ``time`` rather than taken from the nominal sampling rate, so an
    overstated rate cannot turn every sample into its own block.

    Parameters
    ----------
    is_valid : ndarray of bool, shape (n_time,)
        True for rows with finite data in every channel.
    time : ndarray, shape (n_time,), optional
        Sample timestamps in seconds. None declares a regular sample grid.

    Returns
    -------
    blocks : list of (start, stop)
        Half-open row ranges, in order.

    """
    n_time = len(is_valid)
    boundary = np.zeros(n_time + 1, dtype=bool)
    boundary[0] = boundary[-1] = True
    # a block boundary sits between rows i-1 and i where validity changes
    boundary[1:-1] |= is_valid[1:] != is_valid[:-1]
    if time is not None and n_time > 1:
        steps = np.diff(time)
        boundary[1:-1] |= steps > 1.5 * np.median(steps)
    edges = np.flatnonzero(boundary)
    return [(int(start), int(stop)) for start, stop in pairwise(edges) if is_valid[start]]


def _smoothed_envelope(
    filtered_lfps: FloatArray,
    blocks: list[tuple[int, int]],
    sampling_frequency: float,
    smoothing_sigma: float,
    square: bool = False,
) -> FloatArray:
    """Per-channel envelope, squared if asked, smoothed within each block.

    Neither the Hilbert transform nor the Gaussian kernel spans a gap.
    Samples outside every block are NaN.

    Parameters
    ----------
    filtered_lfps : ndarray, shape (n_time, n_channels)
        Ripple-band LFP.
    blocks : list of (start, stop)
        Half-open index ranges of the valid blocks (``_valid_blocks``).
    sampling_frequency : float
    smoothing_sigma : float
        Gaussian standard deviation in seconds.
    square : bool, optional
        Square the envelope before smoothing. Default is False.

    Returns
    -------
    smoothed : ndarray, shape (n_time, n_channels)

    """
    smoothed = np.full_like(filtered_lfps, np.nan)
    for start, stop in blocks:
        envelope = get_envelope(filtered_lfps[start:stop])
        if square:
            envelope = envelope**2
        smoothed[start:stop] = gaussian_smooth(envelope, smoothing_sigma, sampling_frequency)
    return smoothed
