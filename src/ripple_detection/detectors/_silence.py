"""Population events bounded by silences in the spike trains."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    BoolArray,
    FloatArray,
    IntArray,
    _check_choice,
    _gap_tolerance,
    sample_count_within,
)
from ripple_detection.detectors._blocks import _valid_blocks
from ripple_detection.detectors._units import _selected_units
from ripple_detection.detectors._validation import (
    _check_gap,
    _check_positive,
    _check_whole_number,
    _validate_duration_limits,
    _validate_multiunit,
    _validate_time_units,
)

COLUMNS = (
    "start_time",
    "end_time",
    "duration",
    "n_samples",
    "n_spikes",
    "n_active_units",
    "clipped_start",
    "clipped_end",
)
"""The columns :func:`detect_silence_bounded_events` returns, in order."""


def _time_scale(time: FloatArray) -> float:
    """Largest magnitude among the sorted times, which sets how far a
    difference of them can round."""
    return float(max(abs(time[0]), abs(time[-1])))


def _at_least(values: FloatArray, minimum: float, scale: float) -> BoolArray:
    """Whether each time difference reaches ``minimum``: one equal to it
    counts, which binary floating point would otherwise decide for it, within
    the tolerance the close-event rule uses for timestamps of magnitude
    ``scale``."""
    return np.asarray(values >= minimum - _gap_tolerance(minimum, scale), dtype=bool)


def _burst_onsets(spikes: FloatArray, time: FloatArray, maximum_isi: float | None) -> IntArray:
    """Sample indices at which any unit fires, keeping only the first spike of
    each unit's bursts when ``maximum_isi`` is given: a spike that follows the
    same unit's previous spike by less than it is dropped."""
    if maximum_isi is None:
        return np.flatnonzero(spikes.sum(axis=1) > 0)
    onsets = []
    scale = _time_scale(time)
    for unit in range(spikes.shape[1]):
        fired = np.flatnonzero(spikes[:, unit] > 0)
        if fired.size:
            gaps = np.diff(time[fired])
            is_first = np.concatenate([[True], _at_least(gaps, maximum_isi, scale)])
            onsets.append(fired[is_first])
    if not onsets:
        return np.empty(0, dtype=int)
    return np.unique(np.concatenate(onsets))


def _gap_events(
    spike_samples: IntArray, time: FloatArray, minimum_silence: float
) -> tuple[IntArray, BoolArray]:
    """Groups of spikes separated by silences of at least ``minimum_silence``.

    Returns ``[first, last]`` sample indices per event, both inclusive, and
    whether the silence before and after each event reaches the block's edge
    before its minimum, so it was not observed in full."""
    if spike_samples.size == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=bool)
    spike_times = time[spike_samples]
    scale = _time_scale(time)
    breaks = np.flatnonzero(_at_least(np.diff(spike_times), minimum_silence, scale))
    firsts = spike_samples[np.concatenate([[0], breaks + 1])]
    lasts = spike_samples[np.concatenate([breaks, [spike_samples.size - 1]])]
    clipped = np.column_stack(
        [
            ~_at_least(time[firsts] - time[0], minimum_silence, scale),
            ~_at_least(time[-1] - time[lasts], minimum_silence, scale),
        ]
    )
    return np.column_stack([firsts, lasts]), clipped


def _window_events(
    spike_samples: IntArray,
    time: FloatArray,
    minimum_silence: float,
    window: float,
    window_end_rule: str,
) -> tuple[IntArray, BoolArray]:
    """The window after each silence of at least ``minimum_silence``: from
    the spike that ends the silence to either the last spike or the fixed
    window endpoint, according to ``window_end_rule``.
    The first spike's silence is measured back to the block's start, so a
    spike too close to missing data or the recording edge is no onset: its
    silence was not observed in full. A window running past the block's end
    is clipped."""
    spike_times = time[spike_samples]
    silences = np.diff(np.concatenate([[time[0]], spike_times]))
    scale = _time_scale(time)
    ends_silence = _at_least(silences, minimum_silence, scale)
    # a spike exactly a window after the onset is in it, a sample later is not
    tolerance = _gap_tolerance(window, scale)
    bounds, clipped = [], []
    index = 0
    while index < spike_samples.size:
        if not ends_silence[index]:
            index += 1
            continue
        window_end = spike_times[index] + window
        last = int(np.searchsorted(spike_times, window_end + tolerance, side="right")) - 1
        last_sample = int(spike_samples[last])
        if window_end_rule == "fixed":
            last_sample = int(np.searchsorted(time, window_end + tolerance, side="right")) - 1
        bounds.append((spike_samples[index], last_sample))
        clipped.append((False, bool(window_end > time[-1] + tolerance)))
        index = last + 1
    if not bounds:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=bool)
    return np.asarray(bounds, dtype=int), np.asarray(clipped, dtype=bool)


@explain_call_errors
def detect_silence_bounded_events(
    time: ArrayLike,
    multiunit: ArrayLike,
    sampling_frequency: float,
    *,
    minimum_silence: float,
    window: float | None = None,
    window_end_rule: str = "last_spike",
    maximum_isi: float | None = None,
    units: ArrayLike | None = None,
    minimum_active_units: int = 1,
    minimum_active_fraction: float | None = None,
    minimum_duration: float = 0.0,
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect population events as spiking bounded by silences.

    Some published detectors use no rate threshold at all: an event is a
    burst of spikes from a chosen set of cells set off by silence. Two forms
    appear:

    - **Groups** (``window=None``): the pooled spike train is split wherever
      no selected unit fires for ``minimum_silence``, and each group of spikes
      is an event, from its first spike to its last (Foster & Wilson 2006:
      split at gaps over 50 ms; Liu et al. 2019: 100 ms of silence).
    - **Windows** (``window`` in seconds): an event starts at a spike that
      ends a silence of at least ``minimum_silence``. By default its end is
      the last spike within ``window``. Set ``window_end_rule='fixed'`` to
      retain the entire window (Diba & Buzsáki 2007: 60 ms of silence,
      then at least 5 cells in the next 300 ms).

    ``maximum_isi`` first collapses each unit's bursts to their first spike,
    so a unit firing a burst counts once, as the "letters" of Lee & Wilson
    2002 do. The participation and duration criteria then select events.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Sample timestamps in **seconds**, increasing.
    multiunit : array_like, shape (n_time, n_units)
        Spike counts or indicators per sample for each unit, usually sorted
        cells such as a template's place cells. A NaN in a selected unit
        marks the sample missing: nothing spans it, and a silence is not
        measured across it.
    sampling_frequency : float
        Sampling rate in Hz.
    minimum_silence : float
        Seconds without a spike from any selected unit that separates events
        (groups) or must precede one (windows). A silence equal to it counts.
    window : float, optional
        Length in seconds of the window after each silence. Default None,
        groups.
    window_end_rule : {'last_spike', 'fixed'}, optional
        With ``window``, end at the last spike (default) or at the last sample
        at or before onset + window. Fixed windows stop at missing data or
        the recording edge and are flagged as clipped. Duration and spike
        counts refer to the returned bounds. ``'fixed'`` requires ``window``.
    maximum_isi : float, optional
        Seconds: a spike following the same unit's previous spike by less
        than this is dropped before segmenting, so each burst is its first
        spike. Default None, every spike kept.
    units : array_like, optional
        The units whose spikes count: a boolean mask over the columns of
        ``multiunit`` or their indices. Default None, all.
    minimum_active_units : int, optional
        Least number of selected units firing in an event. Default 1.
    minimum_active_fraction : float, optional
        Least fraction, 0 to 1, of the selected units firing in an event.
        Default None.
    minimum_duration, maximum_duration : float, optional
        Duration limits in seconds over the returned bounds, as inclusive sample
        counts (``sample_count_within``). Defaults 0.0 and None.

    Returns
    -------
    events : pd.DataFrame
        One row per event, indexed by ``event_number`` from 1: ``start_time``
        and ``end_time`` (last spike or fixed-window end), ``duration``, ``n_samples``,
        ``n_spikes`` (every spike of the selected units inside the event,
        bursts included), ``n_active_units``, and ``clipped_start`` and
        ``clipped_end``, set when the silence before or after a group, or a
        window's full length, runs into missing data or the recording edge.

    Raises
    ------
    ValueError
        If ``multiunit`` is not spike counts, the lengths differ, time is not
        in seconds, a duration is not a plausible number of seconds, or a
        participation criterion asks for more units than are selected.

    Notes
    -----
    Speed is not an input: restrict the events afterwards with
    ``exclude_movement``, or to intervals with ``require_overlap``.

    Examples
    --------
    >>> fs = 1000
    >>> time = np.arange(2000) / fs
    >>> multiunit = np.zeros((2000, 3))
    >>> multiunit[[500, 520, 540], [0, 1, 2]] = 1
    >>> multiunit[[1200, 1210], [0, 1]] = 1
    >>> events = detect_silence_bounded_events(
    ...     time, multiunit, fs, minimum_silence=0.1, minimum_active_units=3
    ... )
    >>> events.start_time.tolist(), events.end_time.tolist(), events.n_active_units.tolist()
    ([0.5], [0.54], [3])

    """
    _check_positive(sampling_frequency=sampling_frequency)
    _check_positive(minimum_silence=minimum_silence)
    _check_gap(minimum_silence=minimum_silence)
    _check_choice("window_end_rule", window_end_rule, ("last_spike", "fixed"))
    if window is None and window_end_rule == "fixed":
        msg = "window_end_rule='fixed' requires window."
        raise ValueError(msg)
    if window is not None:
        _check_positive(window=window)
        _check_gap(window=window)
    if maximum_isi is not None:
        _check_positive(maximum_isi=maximum_isi)
        _check_gap(maximum_isi=maximum_isi)
    _validate_duration_limits(minimum_duration, maximum_duration)
    spikes = np.asarray(multiunit, dtype=float)
    _validate_multiunit(spikes)
    time = np.asarray(time, dtype=float)
    if len(spikes) != len(time):
        msg = f"multiunit has {len(spikes)} samples and time {len(time)}; they must match."
        raise ValueError(msg)
    _validate_time_units(time, sampling_frequency)
    selected = _selected_units(units, spikes.shape[1])
    _check_whole_number("minimum_active_units", minimum_active_units, 0)
    if minimum_active_units > len(selected):
        msg = (
            f"minimum_active_units is {minimum_active_units} but {len(selected)} unit(s) "
            "are selected, so no event could be kept."
        )
        raise ValueError(msg)
    if minimum_active_fraction is not None and not 0 <= minimum_active_fraction <= 1:
        msg = (
            f"minimum_active_fraction must be between 0 and 1, got {minimum_active_fraction}."
        )
        raise ValueError(msg)

    chosen = spikes[:, selected]
    _, blocks = _valid_blocks(time, chosen)
    rows = []
    for start, stop in blocks:
        block_spikes, block_time = chosen[start:stop], time[start:stop]
        onsets = _burst_onsets(block_spikes, block_time, maximum_isi)
        if window is None:
            bounds, clipped = _gap_events(onsets, block_time, minimum_silence)
        else:
            bounds, clipped = _window_events(
                onsets, block_time, minimum_silence, window, window_end_rule
            )
        for (first, last), (clip_start, clip_end) in zip(bounds, clipped, strict=True):
            inside = block_spikes[first : last + 1]
            rows.append(
                (
                    start + first,
                    start + last,
                    int(np.nansum(inside)),
                    int(np.sum(np.nansum(inside, axis=0) > 0)),
                    bool(clip_start),
                    bool(clip_end),
                )
            )

    table = np.asarray(rows, dtype=float).reshape(-1, 6)
    first_sample, last_sample = table[:, 0].astype(int), table[:, 1].astype(int)
    n_samples = last_sample - first_sample + 1
    n_active = table[:, 3].astype(int)
    keep = n_active >= minimum_active_units
    if minimum_active_fraction is not None:
        fraction = n_active / max(len(selected), 1)
        keep &= (fraction >= minimum_active_fraction) | np.isclose(
            fraction, minimum_active_fraction
        )
    if len(table):
        keep &= np.asarray(
            sample_count_within(n_samples, time, minimum_duration, maximum_duration)
        )
    start_time, end_time = time[first_sample[keep]], time[last_sample[keep]]
    events = pd.DataFrame(
        {
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "n_samples": n_samples[keep],
            "n_spikes": table[keep, 2].astype(int),
            "n_active_units": n_active[keep],
            "clipped_start": table[keep, 4].astype(bool),
            "clipped_end": table[keep, 5].astype(bool),
        },
        index=pd.Index(np.arange(int(keep.sum())) + 1, name="event_number"),
    )
    return events[list(COLUMNS)]
