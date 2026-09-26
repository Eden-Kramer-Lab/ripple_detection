"""Executable interpretations of the surveyed candidate-detection methods.

Use :meth:`Recording.from_arrays` with measured signals, selected cells and
curated state intervals, then :func:`run_method`. The returned DataFrame records
its method, DOI, output role and interpretation in ``attrs``. These are candidate
and ripple methods, not replay decoding. Unresolved choices are documented in
individual functions and in docs/literature/papers; simulation conventions are
confined to examples/literature_recipes.py and explicitly marked fallbacks.

The packaged literature CSV owns reported values. This module owns executable
interpretations, including choices absent from that table. Method coverage and
synthetic tests do not establish parity with historical event inventories.
"""

from __future__ import annotations

import contextvars
import dataclasses
import difflib
import functools
import hashlib
import inspect
import json
import os
import unicodedata
from collections.abc import Callable, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import fftconvolve, filtfilt, find_peaks, firwin

import ripple_detection as rd
from ripple_detection.core import BoolArray, FloatArray, IntArray, MergeMeasure, _matlab_smooth
from ripple_detection.detectors._blocks import _drop_short_blocks, _valid_blocks
from ripple_detection.detectors._long import (
    _difference_of_gaussians_band,
    _firfilt,
    _gaussian_lowpass_fir,
)
from ripple_detection.detectors._silence import WindowEndRule
from ripple_detection.detectors._state import RatioMeasure


@dataclass
class RecordedSignals:
    """Measured recording arrays on one time grid; no simulated ground truth.

    Build it with ``Recording.from_arrays``, which validates and copies the
    inputs; constructing it directly checks only that the arrays share the grid.

    Attributes
    ----------
    time : ndarray, shape (n_time,)
        Timestamps in seconds.
    sampling_frequency : float
        Nominal sampling rate in Hz.
    lfps : ndarray, shape (n_time, n_channels)
        Selected raw LFP channels; no columns when none were supplied.
    raw_lfp : ndarray, shape (n_time,)
        A copy of the first selected channel for single-channel rules; NaN when
        no channel was supplied.
    sharp_wave_lfp : ndarray, shape (n_time,)
        Stratum radiatum LFP; NaN when not supplied.
    multiunit : ndarray, shape (n_time, n_units)
        Spike counts of each unit on the grid.
    speed : ndarray, shape (n_time,), or None
        Speed in cm/s, NaN where unknown; None when not supplied.

    Raises
    ------
    ValueError
        An array does not have one row per timestamp.
    """

    time: FloatArray
    sampling_frequency: float
    lfps: FloatArray
    raw_lfp: FloatArray
    sharp_wave_lfp: FloatArray
    multiunit: FloatArray
    speed: FloatArray | None

    def __post_init__(self) -> None:
        n_time = len(self.time)
        for name, values, ndim in [
            ("lfps", self.lfps, 2),
            ("raw_lfp", self.raw_lfp, 1),
            ("sharp_wave_lfp", self.sharp_wave_lfp, 1),
            ("multiunit", self.multiunit, 2),
            ("speed", self.speed, 1),
        ]:
            if values is not None and (values.ndim != ndim or values.shape[0] != n_time):
                msg = (
                    f"{name} must have one row per timestamp ({n_time}) and {ndim} "
                    f"dimension(s); got shape {values.shape}."
                )
                raise ValueError(msg)


_NO_SPEED = (
    "This method uses the animal's speed, which was not supplied; pass speed "
    "(cm/s) to Recording.from_arrays, with NaN where it is unknown."
)


# The detection steps of the run_method call in progress: each appends its
# step name and event count, for attrs["diagnostics"].
_DETECTIONS: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "_DETECTIONS", default=None
)
_Events = TypeVar("_Events", bound=Sized)


def _counted(step: str, events: _Events) -> _Events:
    """Record how many events a detection step found, and return them."""
    record = _DETECTIONS.get()
    if record is not None:
        record.append({"step": step, "events": len(events)})
    return events


def _detected(
    function: Callable[..., pd.DataFrame], step: str | None = None
) -> Callable[..., pd.DataFrame]:
    """``function``, recording its event count as a detection step."""

    def detect(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return _counted(step or function.__name__, function(*args, **kwargs))

    return detect


def _known_speed(speed: FloatArray | None) -> FloatArray:
    """Speed a method's result depends on; absent speed is an error, not NaN."""
    if speed is None:
        raise ValueError(_NO_SPEED)
    return speed


def _speed_or_unknown(rec: Recording) -> FloatArray:
    """Speed for a call whose result ignores it (no speed rule): NaN if absent."""
    speed = rec.session.speed
    return np.full(len(rec.time), np.nan) if speed is None else speed


@dataclass(eq=False)
class Recording:
    """Recording inputs and transforms for the literature methods.

    Use ``from_arrays`` for real data. Select the intended LFP channels before
    passing them; single-channel methods use the first selected channel. Cell
    masks identify the caller's sorted populations, not automatically classified
    cells. Methods needing sleep require curated intervals for measured data;
    only explicit SimulatedSession inputs permit state proxies. Behavioral
    epochs (reward zones, rest, track ends) differ between methods, so they
    are not part of a recording: pass ``behavior_intervals`` to each call.
    """

    session: rd.SimulatedSession | RecordedSignals
    place_cells: BoolArray
    pyramidal: BoolArray
    sleep_intervals: FloatArray | None = None
    baseline_intervals: FloatArray | None = None
    reference_lfp: FloatArray | None = None
    templates: tuple[BoolArray, ...] = ()
    example_ripples: FloatArray | None = None
    external_ripples: FloatArray | None = None

    def __post_init__(self) -> None:
        n_time, n_units = len(self.time), self.multiunit.shape[1]
        self.place_cells = _cell_mask(self.place_cells, n_units)
        self.pyramidal = _cell_mask(self.pyramidal, n_units)
        self.templates = tuple(_cell_mask(template, n_units) for template in self.templates)
        self.sleep_intervals = _interval_array(self.sleep_intervals)
        self.baseline_intervals = _interval_array(self.baseline_intervals)
        self.example_ripples = _interval_array(self.example_ripples)
        self.external_ripples = _external_ripple_array(self.external_ripples)
        if self.reference_lfp is not None and np.shape(self.reference_lfp) != (n_time,):
            msg = f"reference_lfp must have one value per timestamp ({n_time})."
            raise ValueError(msg)

    @classmethod
    def from_arrays(
        cls,
        time: ArrayLike,
        sampling_frequency: float,
        *,
        lfps: ArrayLike | None = None,
        multiunit: ArrayLike | None = None,
        speed: ArrayLike | None = None,
        sharp_wave_lfp: ArrayLike | None = None,
        place_cells: ArrayLike | None = None,
        pyramidal: ArrayLike | None = None,
        sleep_intervals: ArrayLike | None = None,
        baseline_intervals: ArrayLike | None = None,
        artifact_intervals: ArrayLike | None = None,
        reference_lfp: ArrayLike | None = None,
        templates: Sequence[ArrayLike] = (),
        example_ripples: ArrayLike | None = None,
        external_ripples: ArrayLike | None = None,
    ) -> Recording:
        """Construct a recording without simulated data or inferred cell classes.

        Parameters
        ----------
        time : array_like, shape (n_time,)
            Strictly increasing timestamps in seconds. Gaps remain missing.
        sampling_frequency : float
            Nominal input rate in Hz; spikes and signals share this time grid.
        lfps, multiunit : array_like, optional
            Selected raw LFP channels and spike counts, shape (n_time, n_signals).
            Spike timestamps must be binned onto this grid by the caller; no
            information finer than the supplied counts can be recovered.
        speed, sharp_wave_lfp, reference_lfp : array_like, optional
            Speed in cm/s, radiatum LFP and reference LFP, each shape (n_time,).
            NaN speed is unknown speed. Without speed, a method whose result
            depends on speed raises; the others run.
        place_cells, pyramidal : array_like, optional
            Boolean masks, or distinct integer column indices into multiunit;
            a 0/1 integer array is read as indices and so rejects repeats.
        sleep_intervals, baseline_intervals, artifact_intervals : array_like, optional
            Sorted, disjoint inclusive [start, end] intervals in seconds.
            Artifacts mark all signal arrays missing. Baselines are consumed
            only by methods that explicitly request a caller-selected baseline.
        templates : sequence of array_like, optional
            Per-template cell masks or indices, with no fixed ensemble size.
        example_ripples : array_like, optional
            Manually selected start/end intervals for Carey's spectral template.
        external_ripples : array_like, optional
            Independently detected ripple intervals. For peak containment use
            three columns: start, end, peak; two columns imply midpoint peaks.

        Returns
        -------
        recording : Recording
            Copies of the supplied arrays. Missing inputs are never synthesized.

        Raises
        ------
        ValueError
            Timestamps, sampling rate, signal shapes, counts, selections or
            intervals are invalid or inconsistent.

        Notes
        -----
        The recording holds one float64 copy of every signal, so NaN can mark
        missing samples: 8 bytes per sample for each LFP channel and each
        unit, whatever the input's type. One hour at 1500 Hz with 100 units is
        4.32 GB of spike counts (5.4 million samples x 100 x 8 bytes); int16
        LFP and uint8 counts grow 4- and 8-fold. Construction peaks at about
        that retained size; select channels and units, or split long sessions,
        before building a recording from them.
        """
        timestamps = np.array(time, dtype=float)
        if timestamps.ndim != 1 or len(timestamps) < 2:
            msg = "time must be a one-dimensional array with at least two samples."
            raise ValueError(msg)
        if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
            msg = "sampling_frequency must be positive and finite."
            raise ValueError(msg)
        if np.any(np.diff(timestamps) <= 0):
            msg = "time must contain strictly increasing timestamps."
            raise ValueError(msg)
        _valid_blocks(timestamps, np.ones(len(timestamps)))
        if not np.isclose(np.median(np.diff(timestamps)), 1 / sampling_frequency, rtol=0.1):
            msg = "sampling_frequency must agree with the timestamp spacing."
            raise ValueError(msg)
        n = len(timestamps)

        def signal(value: ArrayLike | None, channels: bool = False) -> FloatArray:
            if value is None:
                return np.empty((n, 0)) if channels else np.full(n, np.nan)
            # The one copy, as float64 so NaN can mark missing samples.
            data = np.array(value, dtype=float)
            if channels and data.ndim == 1:
                data = data[:, None]
            if data.ndim != (2 if channels else 1) or data.shape[0] != n:
                msg = "Signals must have one row per timestamp."
                raise ValueError(msg)
            return data

        if multiunit is not None:
            _check_counts(np.asarray(multiunit))
        lfp_array, spikes = signal(lfps, True), signal(multiunit, True)
        sharp, reference = signal(sharp_wave_lfp), signal(reference_lfp)
        artifacts = _interval_array(artifact_intervals)
        if artifacts is not None:
            mask = _intervals_to_mask(timestamps, artifacts)
            for data in (lfp_array, spikes, sharp, reference):
                data[mask] = np.nan

        def cells(selection: ArrayLike | None) -> BoolArray:
            mask = np.zeros(spikes.shape[1], dtype=bool)
            if selection is None:
                return mask
            indices = np.asarray(selection)
            if indices.dtype == bool:
                if indices.shape != mask.shape:
                    msg = "Cell masks must have one entry per unit."
                    raise ValueError(msg)
                return indices.copy()
            if indices.ndim != 1 or (
                indices.size
                and (
                    not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0)
                    or np.any(indices >= len(mask))
                )
            ):
                msg = "Cell indices must be integers identifying existing units."
                raise ValueError(msg)
            if len(np.unique(indices)) != len(indices):
                msg = (
                    "Cell indices repeat a unit. Integers are read as unit indices; "
                    "pass a boolean mask (dtype=bool) to select units by position."
                )
                raise ValueError(msg)
            mask[indices.astype(int)] = True
            return mask

        external = (
            None if external_ripples is None else np.asarray(external_ripples, float).copy()
        )
        session = RecordedSignals(
            timestamps,
            sampling_frequency,
            lfp_array,
            lfp_array[:, 0].copy() if lfp_array.shape[1] else np.full(n, np.nan),
            sharp,
            spikes,
            None if speed is None else signal(speed),
        )
        return cls(
            session,
            cells(place_cells),
            cells(pyramidal),
            _interval_array(sleep_intervals),
            _interval_array(baseline_intervals),
            reference if reference_lfp is not None else None,
            tuple(cells(x) for x in templates),
            _interval_array(example_ripples),
            external,
        )

    @property
    def allows_simulation_proxies(self) -> bool:
        """Whether simulation-only fallbacks may stand in for missing inputs.

        Returns
        -------
        allowed : bool
            True only for a ``SimulatedSession``. A measured recording must
            supply curated states, templates, example ripples and unreported
            settings itself.
        """
        return isinstance(self.session, rd.SimulatedSession)

    @property
    def time(self) -> FloatArray:
        """Input timestamps.

        Returns
        -------
        time : ndarray, shape (n_time,)
            Seconds on the shared signal grid.
        """
        return self.session.time

    @property
    def fs(self) -> float:
        """Nominal sampling frequency.

        Returns
        -------
        frequency : float
            Samples per second.
        """
        return self.session.sampling_frequency

    @property
    def speed(self) -> FloatArray:
        """Observed animal speed.

        Returns
        -------
        speed : ndarray, shape (n_time,)
            Speed in cm/s; NaN values remain unknown.

        Raises
        ------
        ValueError
            No speed was supplied. A method whose result depends on speed
            refuses to run rather than treating every sample as unknown speed.
        """
        return _known_speed(self.session.speed)

    @property
    def multiunit(self) -> FloatArray:
        """Observed spike counts.

        Returns
        -------
        counts : ndarray, shape (n_time, n_units)
            Nonnegative counts on the shared grid, with missing observations retained.
        """
        return self.session.multiunit

    def filtered(self, band: tuple[float, float]) -> FloatArray:
        """Filter selected LFP channels within valid blocks.

        Parameters
        ----------
        band : pair of float
            Lower and upper passband edges in Hz.

        Returns
        -------
        filtered : ndarray, shape (n_time, n_channels)
            Newly computed bandpass signal. Mutating input arrays cannot leave a stale cache.
        """
        if self.session.lfps.shape[1] == 0:
            msg = "This method requires selected raw LFP channels."
            raise ValueError(msg)
        return rd.filter_ripple_band(self.session.lfps, self.fs, band=band, time=self.time)

    def envelope(self, band: tuple[float, float]) -> FloatArray:
        """Calculate Hilbert amplitudes of selected LFP channels.

        Parameters
        ----------
        band : pair of float
            Lower and upper passband edges in Hz.

        Returns
        -------
        amplitude : ndarray, shape (n_time, n_channels)
            Newly computed amplitudes, with missing samples and timestamp gaps preserved.
        """
        return rd.get_envelope(self.filtered(band), time=self.time)

    def transform(
        self, values: FloatArray, operation: Callable[[FloatArray], FloatArray]
    ) -> FloatArray:
        """Apply a shape-preserving operation within each valid block.

        Parameters
        ----------
        values : ndarray
            Input trace or channels with time on axis zero.
        operation : callable
            Function returning an array of the same shape for one valid block.

        Returns
        -------
        transformed : ndarray
            Result with the original shape and missing rows preserved.
        """
        return _transform(self.time, values, operation)

    def boxcar(self, values: FloatArray, width: float) -> FloatArray:
        """Smooth a trace with a centered uniform window.

        Parameters
        ----------
        values : ndarray
            Trace or channels with time on axis zero.
        width : float
            Window width in seconds. It becomes a sample count as durations
            do (``minimum_sample_count``: half up from the median timestamp
            step, at least one), then up to an odd count so the window is
            centered: 7 ms at 1500 Hz is 11 samples.

        Returns
        -------
        smoothed : ndarray
            Blockwise uniform-filter output; reflected block boundaries.
        """
        samples = rd.minimum_sample_count(self.time, width)
        samples += 1 - samples % 2
        return self.transform(values, lambda block: uniform_filter1d(block, samples, axis=0))

    def smooth(self, values: FloatArray, sigma: float) -> FloatArray:
        """Apply Gaussian smoothing within valid blocks.

        Parameters
        ----------
        values : ndarray
            Trace or channels with time on axis zero.
        sigma : float
            Gaussian standard deviation in seconds.

        Returns
        -------
        smoothed : ndarray
            Smoothed values with missing observations retained.
        """
        return self.transform(values, lambda block: rd.gaussian_smooth(block, sigma, self.fs))

    def merge(
        self,
        events: pd.DataFrame | FloatArray,
        gap: float,
        trace: FloatArray,
        *,
        inclusive: bool = False,
        measure: MergeMeasure = "gap",
    ) -> FloatArray:
        """Merge events only within the same valid detection block.

        Parameters
        ----------
        events : pandas.DataFrame or ndarray
            Event start/end times in seconds.
        gap : float
            Maximum separation in seconds for merging.
        trace : ndarray
            Detection signal whose missing samples define block boundaries.
        inclusive : bool, optional
            Merge events exactly at the specified separation.
        measure : str, optional
            Separation measure accepted by merge_close_events.

        Returns
        -------
        bounds : ndarray, shape (n_events, 2)
            Merged start/end times.

        Raises
        ------
        ValueError
            An input event crosses or lies outside a valid block.
        """
        event_bounds = bounds(events)
        if not len(event_bounds):
            return event_bounds
        groups = _event_block_groups(self.time, trace, event_bounds)
        merged = []
        for group in np.unique(groups):
            keep = groups == group
            selected = (
                events.loc[keep] if isinstance(events, pd.DataFrame) else event_bounds[keep]
            )
            merged.append(
                bounds(
                    rd.merge_close_events(selected, gap, inclusive=inclusive, measure=measure)
                )
            )
        return np.concatenate(merged)

    def mean_envelope(
        self, band: tuple[float, float], channels: int | None = None
    ) -> FloatArray:
        """Average Hilbert amplitudes across selected channels.

        Parameters
        ----------
        band : pair of float
            Passband edges in Hz.
        channels : int, optional
            Use the first this many selected channels; default uses all channels.

        Returns
        -------
        amplitude : ndarray, shape (n_time,)
            Mean amplitude on the shared grid.

        Raises
        ------
        ValueError
            Fewer than ``channels`` LFP channels were selected.
        """
        available = self.session.lfps.shape[1]
        if channels is not None and available < channels:
            msg = (
                f"This method averages {channels} selected LFP channels, but "
                f"{available} were supplied."
            )
            raise ValueError(msg)
        envelope = self.envelope(band)
        return envelope[:, :channels].mean(axis=1) if channels else envelope.mean(axis=1)

    def rate(self, units: BoolArray | None, sigma: float) -> FloatArray:
        """Estimate smoothed population firing rate on the input grid.

        Parameters
        ----------
        units : ndarray of bool or None
            Unit selection; None pools all supplied units.
        sigma : float
            Gaussian standard deviation in seconds.

        Returns
        -------
        rate : ndarray, shape (n_time,)
            Summed spikes per second after smoothing.
        """
        return self.smooth(self.counts(units) * self.fs, sigma)

    def counts(self, units: BoolArray | None = None) -> FloatArray:
        """Sum spike counts over a selected population.

        Parameters
        ----------
        units : ndarray of bool, optional
            Unit mask; default pools all supplied units.

        Returns
        -------
        counts : ndarray, shape (n_time,)
            Summed counts, preserving missing observations.

        Raises
        ------
        ValueError
            No supplied unit is selected.
        """
        spikes = self.multiunit if units is None else self.multiunit[:, units]
        if spikes.shape[1] == 0:
            msg = "This method requires spikes and a nonempty cell selection."
            raise ValueError(msg)
        return spikes.sum(axis=1)

    def ratio(
        self,
        theta: tuple[float, float] = (6.0, 12.0),
        delta: tuple[float, float] = (1.0, 4.0),
        smoothing_sigma: float = 1.0,
        measure: RatioMeasure = "amplitude",
    ) -> FloatArray:
        """Compute a theta/delta ratio from the raw LFP.

        Parameters
        ----------
        theta, delta : pair of float, optional
            Passband edges in Hz.
        smoothing_sigma : float, optional
            Gaussian standard deviation in seconds.
        measure : {"amplitude", "power"}, optional
            Quantity used for the ratio.

        Returns
        -------
        ratio : ndarray, shape (n_time,)
            Newly computed ratio; missing samples and timestamp gaps are preserved.
        """
        return rd.theta_delta_ratio(
            self.session.raw_lfp, self.fs, theta_band=theta, delta_band=delta,
            smoothing_sigma=smoothing_sigma, measure=measure,
            time=self.time,
        )  # fmt: skip

    def intervals_to_mask(self, intervals: FloatArray) -> BoolArray:
        """Select samples inside inclusive intervals.

        Parameters
        ----------
        intervals : ndarray, shape (n_intervals, 2)
            Start/end times in seconds.

        Returns
        -------
        mask : ndarray of bool, shape (n_time,)
            True for timestamps lying in any supplied interval.
        """
        return _intervals_to_mask(self.time, intervals)

    def mask_to_intervals(self, mask: BoolArray) -> FloatArray:
        """Convert a sample selection to contiguous intervals.

        Parameters
        ----------
        mask : ndarray of bool, shape (n_time,)
            Selected samples.

        Returns
        -------
        intervals : ndarray, shape (n_intervals, 2)
            Start/end timestamps of selected runs; timestamp gaps split runs.
        """
        return rd.state_intervals(mask.astype(float), self.time, 0.5, comparison=">")

    def sleep(
        self,
        speed_below: float,
        ratio_below: float,
        stillness: float = 0.0,
        theta: tuple[float, float] = (6.0, 12.0),
        delta: tuple[float, float] = (1.0, 4.0),
        smoothing_sigma: float = 1.0,
        measure: RatioMeasure = "amplitude",
    ) -> FloatArray:
        """Return curated sleep intervals or an explicit simulation proxy.

        Parameters
        ----------
        speed_below, ratio_below : float
            Simulation-only speed (cm/s) and theta/delta thresholds.
        stillness : float, optional
            Simulation-only minimum stillness duration in seconds.
        theta, delta : pair of float, optional
            Simulation-only passband edges in Hz.
        smoothing_sigma : float, optional
            Simulation-only Gaussian standard deviation in seconds.
        measure : {"amplitude", "power"}, optional
            Simulation-only ratio measure.

        Returns
        -------
        intervals : ndarray, shape (n_intervals, 2)
            Supplied sleep intervals, or the simulated stillness/low-theta overlap.

        Raises
        ------
        ValueError
            Sleep intervals are absent and the session is not a SimulatedSession.
        """
        if self.sleep_intervals is not None:
            return self.sleep_intervals
        if not self.allows_simulation_proxies:
            msg = "Supply sleep_intervals for a method requiring sleep/state scoring."
            raise ValueError(msg)
        still = rd.state_intervals(
            self.speed, self.time, speed_below, minimum_duration=stillness
        )
        ratio = self.ratio(theta, delta, smoothing_sigma, measure)
        low_theta = rd.state_intervals(ratio, self.time, ratio_below)
        return self.mask_to_intervals(
            self.intervals_to_mask(still) & self.intervals_to_mask(low_theta)
        )

    def awake(self, sleep: FloatArray) -> FloatArray:
        """Return the complement of supplied sleep intervals.

        Parameters
        ----------
        sleep : ndarray, shape (n_intervals, 2)
            Sleep start/end times in seconds.

        Returns
        -------
        intervals : ndarray, shape (n_intervals, 2)
            Selected non-sleep runs on the recording grid.
        """
        return self.mask_to_intervals(~self.intervals_to_mask(sleep))


def bounds(events: pd.DataFrame | FloatArray) -> FloatArray:
    """Extract event start/end times without changing their order.

    Parameters
    ----------
    events : pandas.DataFrame or ndarray
        Table with start_time/end_time columns or an array of time pairs.

    Returns
    -------
    bounds : ndarray, shape (n_events, 2)
        Floating-point start/end times in seconds, including shape (0, 2)
        for empty input. Other table columns are discarded.
    """
    if isinstance(events, pd.DataFrame):
        return np.asarray(
            events[["start_time", "end_time"]].to_numpy(dtype=float), dtype=float
        ).reshape(-1, 2)
    return np.asarray(events, dtype=float).reshape(-1, 2)


def within_duration(
    events: pd.DataFrame | FloatArray,
    low: float = 0.0,
    high: float = np.inf,
    *,
    sampling_frequency: float | None = None,
) -> FloatArray:
    """Keep events whose duration is from ``low`` to ``high`` seconds.

    Parameters
    ----------
    events : pandas.DataFrame or ndarray
        Table with start_time/end_time columns or an array of time pairs.
    low, high : float, optional
        Inclusive duration limits in seconds; defaults keep every duration.
        Durations are compared with a tolerance scaled to the timestamps'
        magnitude, so an event exactly at a limit is kept at any clock origin.
    sampling_frequency : float, optional
        When given, the bounds are an event's first and last samples at this
        rate and its duration counts one more sample period, the package's
        inclusive sample count (n samples last n / sampling_frequency). By
        default the duration is the elapsed time from start to end.

    Returns
    -------
    bounds : ndarray, shape (n_kept, 2)
        Start/end times of the kept events, in input order.
    """
    events = bounds(events)
    return events[_within_duration_mask(events, low, high, sampling_frequency)]


def _within_duration_mask(
    events: pd.DataFrame | FloatArray,
    low: float = 0.0,
    high: float = np.inf,
    sampling_frequency: float | None = None,
) -> BoolArray:
    """Which events last from ``low`` to ``high`` seconds, inclusive."""
    events = bounds(events)
    duration = events[:, 1] - events[:, 0]
    if sampling_frequency is not None:
        duration = duration + 1 / sampling_frequency
    tolerance = _time_tolerance(events)
    return (duration >= low - tolerance) & (duration <= high + tolerance)


def _time_tolerance(time: FloatArray) -> float:
    """Allow timestamp subtraction error at the recording's clock magnitude."""
    return max(1e-9, 4 * float(np.spacing(np.max(np.abs(time), initial=0.0))))


def _event_block_groups(
    time: FloatArray, trace: FloatArray, events: FloatArray, margin: float = 0.0
) -> IntArray:
    """Validate complete block containment and return each event's block index.

    ``margin`` widens each block on both sides: half a bin covers bounds at a
    bin's recorded samples, which lie up to half a bin from its center.
    """
    _, blocks = _valid_blocks(time, trace)
    intervals = np.asarray(
        [(time[a] - margin, time[b - 1] + margin) for a, b in blocks]
    ).reshape(-1, 2)
    groups = np.searchsorted(intervals[:, 0], events[:, 0], side="right") - 1
    if (
        not np.isfinite(events).all()
        or np.any(events[:, 1] < events[:, 0])
        or np.any(groups < 0)
        or np.any(events[:, 1] > intervals[groups, 1])
    ):
        msg = "Every event must lie within one valid detection block."
        raise ValueError(msg)
    return groups


def _event_slice(time: FloatArray, start: float, end: float, tolerance: float) -> slice:
    """Locate candidate samples without scanning the entire recording."""
    return slice(
        int(np.searchsorted(time, start - tolerance, side="left")),
        int(np.searchsorted(time, end + tolerance, side="right")),
    )


def _within_intervals_mask(events: FloatArray, intervals: ArrayLike) -> BoolArray:
    """Which ``(start, end)`` pairs lie inside one interval (``require_inside``'s
    rule, with the clock's rounding error allowed at either bound)."""
    return rd.core._inside_mask(
        np.asarray(events, dtype=float).reshape(-1, 2),
        rd.core._checked_intervals(intervals, "intervals"),
    )


def _only_in(rec: Recording, values: FloatArray, intervals: FloatArray) -> FloatArray:
    """``values`` (a trace or the spikes) missing outside the intervals, so
    detection runs inside them only."""
    mask = rec.intervals_to_mask(intervals)
    return np.where(mask if values.ndim == 1 else mask[:, None], values, np.nan)


def _zugaro_ripple_peaks(rec: Recording, band: tuple[float, float]) -> pd.DataFrame:
    """bz_FindRipples-like ripples on one pyramidal channel, for the
    Buzsaki-lineage papers that require a ripple peak but do not describe
    their ripple detector (assumed: Huszar et al. 2022's 5 SD peak and 2 SD
    bounds, 20-200 ms; its noise-channel veto is not reproduced)."""
    return _detected(rd.Zugaro_ripple_detector)(
        rec.time, rec.filtered(band)[:, :1], _speed_or_unknown(rec), rec.fs,
        low_threshold=2.0, high_threshold=5.0, maximum_duration=0.2,
        speed_threshold=np.inf,
    )  # fmt: skip


# Float validation reads this many elements at a time (8 MB of float64), so
# checking counts never copies the whole array.
_CHUNK_ELEMENTS = 2**20


def _check_counts(counts: np.ndarray[Any, Any]) -> None:
    """Raise unless every finite value is a nonnegative whole number."""
    msg = "multiunit must contain nonnegative integer spike counts."
    if counts.dtype == bool or counts.ndim == 0:
        return
    if np.issubdtype(counts.dtype, np.integer):
        if counts.size and counts.min() < 0:
            raise ValueError(msg)
        return
    rows = max(1, _CHUNK_ELEMENTS // max(1, int(np.prod(counts.shape[1:]))))
    for start in range(0, len(counts), rows):
        chunk = np.asarray(counts[start : start + rows], dtype=float)
        finite = chunk[np.isfinite(chunk)]
        if np.any(finite < 0) or np.any(finite != np.floor(finite)):
            raise ValueError(msg)


def _cell_mask(mask: ArrayLike, n_units: int) -> BoolArray:
    """A boolean selection with one entry per unit, as Recording holds them."""
    selection = np.asarray(mask)
    if selection.dtype != bool or selection.shape != (n_units,):
        msg = (
            f"Cell masks must be boolean arrays with one entry per unit ({n_units}); "
            "Recording.from_arrays also accepts unit indices."
        )
        raise ValueError(msg)
    return selection


def _external_ripple_array(value: ArrayLike | None) -> FloatArray | None:
    """Validated start/end(/peak) rows of an external ripple inventory."""
    if value is None:
        return None
    external = np.asarray(value, dtype=float).copy()
    if external.ndim != 2 or external.shape[1] not in (2, 3):
        msg = "external_ripples needs start/end and optional peak columns."
        raise ValueError(msg)
    _interval_array(external[:, :2])
    if external.shape[1] == 3 and (
        not np.isfinite(external).all()
        or np.any(external[:, 2] < external[:, 0])
        or np.any(external[:, 2] > external[:, 1])
    ):
        msg = "External ripple peaks must lie inside their intervals."
        raise ValueError(msg)
    return external


def _interval_array(value: ArrayLike | None) -> FloatArray | None:
    if value is None:
        return None
    intervals = np.asarray(value, dtype=float).copy()
    if intervals.size == 0:
        return intervals.reshape(0, 2)
    if (
        intervals.ndim != 2
        or intervals.shape[1] != 2
        or not np.isfinite(intervals).all()
        or np.any(intervals[:, 1] < intervals[:, 0])
        or np.any(intervals[1:, 0] <= intervals[:-1, 1])
    ):
        msg = "Intervals must be finite, sorted, disjoint start/end pairs."
        raise ValueError(msg)
    return intervals


def _baseline(rec: Recording, *, required: bool = False) -> BoolArray:
    if rec.baseline_intervals is not None:
        return rec.intervals_to_mask(rec.baseline_intervals)
    if required:
        msg = "Supply baseline_intervals for this method's normalization epoch."
        raise ValueError(msg)
    return np.ones(len(rec.time), dtype=bool)


def _baseline_samples(values: FloatArray, baseline: BoolArray) -> FloatArray:
    """Rows of ``values`` inside the baseline that are finite in every channel.

    Raises when there are none or when their spread is zero or not finite in
    every channel, since a statistic taken from them would be meaningless.
    """
    selected = values[baseline]
    finite = selected[
        np.isfinite(selected).all(axis=1) if selected.ndim == 2 else np.isfinite(selected)
    ]
    spread = np.std(finite, axis=0) if len(finite) else np.array(np.nan)
    if not len(finite) or not np.all(np.isfinite(spread)) or np.all(spread == 0):
        msg = (
            "The baseline (baseline_intervals, or the whole recording without them) "
            "needs finite samples with a nonzero spread."
        )
        raise ValueError(msg)
    return finite


def _zscore(values: FloatArray, mask: BoolArray | None = None, ddof: int = 0) -> FloatArray:
    baseline = values if mask is None else values[mask]
    finite = baseline[np.isfinite(baseline)]
    if finite.size <= ddof or float(np.std(finite, ddof=ddof)) == 0:
        msg = "Normalization needs a finite, nonconstant baseline."
        raise ValueError(msg)
    return (values - np.mean(finite)) / np.std(finite, ddof=ddof)


def _transform(
    time: FloatArray,
    values: FloatArray,
    operation: Callable[[FloatArray], FloatArray],
    minimum_length: int = 1,
    reason: str = "",
) -> FloatArray:
    """Apply an operation independently within each valid block on the given grid.

    Blocks shorter than ``minimum_length`` samples, which ``reason`` (the
    transform) cannot process, are left missing with a warning; if none is
    long enough, raise.
    """
    is_valid, blocks = _valid_blocks(time, values)
    if minimum_length > 1:
        blocks = _drop_short_blocks(blocks, is_valid, minimum_length, reason)
    result = np.full(values.shape, np.nan)
    for start, stop in blocks:
        result[start:stop] = operation(values[start:stop])
    return result


def _intervals_to_mask(time: FloatArray, intervals: FloatArray) -> BoolArray:
    """Select the union of inclusive intervals on the increasing time grid.

    Intervals may be in any order and overlap: each adds one over its samples
    (start <= time <= end), and a sample is selected where the sum is positive.
    """
    pairs = np.asarray(intervals, dtype=float).reshape(-1, 2)
    first = np.searchsorted(time, pairs[:, 0], side="left")
    stop = np.searchsorted(time, pairs[:, 1], side="right")
    nonempty = stop > first
    coverage = np.zeros(time.size + 1, dtype=int)
    np.add.at(coverage, first[nonempty], 1)
    np.add.at(coverage, stop[nonempty], -1)
    return np.asarray(np.cumsum(coverage[:-1]) > 0, dtype=bool)


@dataclass(frozen=True)
class PopulationTrace:
    """Population counts or rate on a native nonoverlapping bin grid.

    Build it with ``population_trace``; methods derive new traces with
    ``dataclasses.replace`` rather than changing one.

    Attributes
    ----------
    time, data : ndarray
        Bin centers (seconds) and counts/rate. Bins intersecting missing
        input are NaN, including partial edge bins.
    speed : ndarray or None
        Nearest observed speed (cm/s); None when the recording has no speed.
    sampling_frequency : float
        Reciprocal bin width in Hz.
    first_sample, last_sample : ndarray or None
        Timestamps (seconds) of the first and last recorded samples counted in
        each bin, NaN for a bin holding none. Events are reported at these
        samples. None for a trace built by hand, whose events are then
        reported at bin centers.

    Raises
    ------
    ValueError
        ``data``, ``speed``, ``first_sample`` or ``last_sample`` does not have
        one value per bin.
    """

    time: FloatArray
    data: FloatArray
    speed: FloatArray | None
    sampling_frequency: float
    first_sample: FloatArray | None = None
    last_sample: FloatArray | None = None

    def __post_init__(self) -> None:
        n_bins = len(self.time)
        per_bin = (self.data, self.speed, self.first_sample, self.last_sample)
        if any(values is not None and len(values) != n_bins for values in per_bin):
            msg = (
                f"PopulationTrace needs one value per bin ({n_bins}) in data, speed, "
                "first_sample and last_sample."
            )
            raise ValueError(msg)

    def sample_bounds(self, centers: FloatArray) -> FloatArray:
        """Convert bounds at bin centers to the bins' recorded samples.

        Parameters
        ----------
        centers : ndarray, shape (n_events, 2)
            Start/end bin centers, as ``detect_events_from_trace`` reports
            them on this grid.

        Returns
        -------
        bounds : ndarray, shape (n_events, 2)
            The first sample counted in the event's bins and the last: closed
            bounds holding exactly those samples. A bin holding no sample (bins
            narrower than the sample spacing, or uneven timestamps) is skipped,
            so bounds are always recorded timestamps. Bin centers for a trace
            built by hand, or for an event whose bins hold no sample.
        """
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        if self.first_sample is None or self.last_sample is None or not len(centers):
            return centers.copy()
        start_bin = rd.core.nearest_sample_index(self.time, centers[:, 0])
        end_bin = rd.core.nearest_sample_index(self.time, centers[:, 1])
        counted = np.flatnonzero(np.isfinite(self.first_sample))
        first = np.searchsorted(counted, start_bin, side="left")
        last = np.searchsorted(counted, end_bin, side="right") - 1
        has_samples = (first < len(counted)) & (last >= 0) & (first <= last)
        first = counted[np.clip(first, 0, len(counted) - 1)]
        last = counted[np.clip(last, 0, len(counted) - 1)]
        return np.where(
            has_samples[:, None],
            np.column_stack([self.first_sample[first], self.last_sample[last]]),
            centers,
        )

    def bins_inside(self, intervals: ArrayLike) -> BoolArray:
        """Which bins lie wholly inside one of the intervals.

        Parameters
        ----------
        intervals : array_like, shape (n_intervals, 2)
            Sorted, disjoint, inclusive [start, end] intervals in seconds.

        Returns
        -------
        inside : ndarray of bool, shape (n_bins,)
            True for a bin whose counted samples all lie inside one interval,
            so events restricted to these bins are reported inside it. A bin
            holding no sample, or a trace built by hand, is judged by its
            center.
        """
        first = self.time if self.first_sample is None else self.first_sample
        last = self.time if self.last_sample is None else self.last_sample
        first = np.where(np.isfinite(first), first, self.time)
        last = np.where(np.isfinite(last), last, self.time)
        return _within_intervals_mask(np.column_stack([first, last]), intervals)

    def smooth(self, sigma: float) -> FloatArray:
        """Gaussian-smooth the trace without crossing missing bins.

        Parameters
        ----------
        sigma : float
            Gaussian standard deviation in seconds.

        Returns
        -------
        smoothed : ndarray, shape (n_bins,)
            Smoothed values within each valid block; missing bins stay NaN.
        """
        return _transform(
            self.time,
            self.data,
            lambda x: rd.gaussian_smooth(x, sigma, self.sampling_frequency),
        )

    def detect(self, **kwargs: Any) -> pd.DataFrame:
        """Threshold this trace with ``detect_events_from_trace``.

        Parameters
        ----------
        **kwargs
            Keyword arguments of ``detect_events_from_trace``, such as
            ``threshold``, ``bound_threshold`` and ``minimum_event_duration``.
            Durations and ``close_event_threshold`` are in seconds between
            bin edges.

        Returns
        -------
        events : pandas.DataFrame
            ``detect_events_from_trace`` output. Bounds are the first and last
            recorded samples of the event's first and last bins (see
            ``sample_bounds``), so they hold exactly the samples the bins
            counted and a participation count sees no spike from a neighboring
            bin. ``duration`` is end minus start; the duration limits count
            bins (n bins last n bin widths). ``peak_time`` stays a bin center.

        Raises
        ------
        ValueError
            A speed rule is requested but the recording has no speed, or
            ``detect_events_from_trace`` rejects the options.
        """
        default = inspect.signature(rd.detect_events_from_trace).parameters["speed_threshold"]
        if self.speed is None and np.isfinite(kwargs.get("speed_threshold", default.default)):
            raise ValueError(_NO_SPEED)
        speed = np.full(len(self.time), np.nan) if self.speed is None else self.speed
        width = 1 / self.sampling_frequency
        if kwargs.get("close_event_threshold", 0.0) > 0:
            # Between centers, the gap between edges is one bin width longer.
            kwargs["close_event_threshold"] = kwargs["close_event_threshold"] + width
        events = _detected(
            rd.detect_events_from_trace, f"detect_events_from_trace ({width:g} s bins)"
        )(self.time, self.data, speed, self.sampling_frequency, **kwargs)
        events[["start_time", "end_time"]] = self.sample_bounds(bounds(events))
        events["duration"] = events.end_time - events.start_time
        return events

    def merge(
        self, events: pd.DataFrame | FloatArray, gap: float, *, inclusive: bool = False
    ) -> FloatArray:
        """Merge close events inside valid native-grid blocks only.

        Parameters
        ----------
        events : pandas.DataFrame or ndarray
            Event bounds in seconds on this grid: the recorded samples
            ``detect`` reports, or bin centers.
        gap : float
            Separation in seconds between the given bounds, from one event's
            end to the next's start, below which events are merged. Between
            recorded samples it is one sample period longer than edge to edge.
        inclusive : bool, optional
            Also merge events exactly ``gap`` apart.

        Returns
        -------
        bounds : ndarray, shape (n_events, 2)
            Merged start/end times; nothing is merged across a missing bin.

        Raises
        ------
        ValueError
            An input event crosses or lies outside a valid block.
        """
        event_bounds = bounds(events)
        if len(event_bounds) == 0:
            return event_bounds
        groups = _event_block_groups(
            self.time, self.data, event_bounds, margin=0.5 / self.sampling_frequency
        )
        merged = []
        for group in np.unique(groups):
            selected = event_bounds[groups == group]
            if len(selected):
                merged.append(rd.merge_close_events(selected, gap, inclusive=inclusive))
        return np.concatenate(merged) if merged else np.empty((0, 2))


def population_trace(
    rec: Recording,
    *,
    bin_width: float,
    units: BoolArray | None = None,
    smoothing_sigma: float = 0.0,
) -> PopulationTrace:
    """Bin observed spike counts before smoothing a population rate.

    Parameters
    ----------
    rec : Recording
        Counts on the input time grid. A count belongs to its supplied timestamp.
    bin_width : float
        Nonoverlapping bin width in seconds, anchored at the first timestamp.
    units : ndarray of bool, optional
        Population mask; default pools all supplied units.
    smoothing_sigma : float, optional
        Gaussian standard deviation in seconds, applied after binning.

    Returns
    -------
    trace : PopulationTrace
        Counts per second at bin centers. Only complete bins are formed, and
        bins with incomplete observed support are NaN. Binning cannot restore
        the precision of original spike times.
    """
    if not np.isfinite(bin_width) or bin_width <= 0:
        msg = "bin_width must be positive and finite."
        raise ValueError(msg)
    count = rec.counts(units)
    relative = rec.time - rec.time[0]
    tolerance = _time_tolerance(rec.time)
    nearest_edge = np.rint(relative / bin_width) * bin_width
    relative = np.where(np.abs(relative - nearest_edge) <= tolerance, nearest_edge, relative)
    n = int(np.floor((relative[-1] + 1 / rec.fs) / bin_width + 1e-7))
    if n < 2:
        msg = "The recording must span at least two complete bins."
        raise ValueError(msg)
    edges = np.arange(n + 1, dtype=float) * bin_width
    # np.histogram closes its last bin, so a sample on the final edge, which
    # starts an incomplete bin, would otherwise be counted in the last one.
    inside = relative < edges[-1]
    values = np.histogram(
        relative[inside],
        edges,
        weights=np.nan_to_num(count[inside], nan=0.0, posinf=0.0, neginf=0.0),
    )[0]
    centers = edges[:-1] + bin_width / 2
    observed = np.zeros(n, dtype=bool)
    _, blocks = _valid_blocks(rec.time, count)
    for start, stop in blocks:
        observed |= (edges[:-1] >= relative[start] - tolerance) & (
            edges[1:] <= relative[stop - 1] + 1 / rec.fs + tolerance
        )
    observed &= centers <= relative[-1] + tolerance
    values = np.where(observed, values / bin_width, np.nan)
    # Each bin's first and last recorded samples, by the histogram's own
    # half-open assignment, so reported bounds hold exactly the counted samples.
    counted_time = rec.time[inside]
    which = np.searchsorted(edges, relative[inside], side="right") - 1
    bins, first_index = np.unique(which, return_index=True)
    last_index = np.r_[first_index[1:] - 1, len(which) - 1]
    first_sample = np.full(n, np.nan)
    last_sample = np.full(n, np.nan)
    first_sample[bins] = counted_time[first_index]
    last_sample[bins] = counted_time[last_index]
    time = centers + rec.time[0]
    speed = (
        None
        if rec.session.speed is None
        else rec.session.speed[rd.core.nearest_sample_index(rec.time, time)]
    )
    trace = PopulationTrace(time, values, speed, 1 / bin_width, first_sample, last_sample)
    if smoothing_sigma:
        trace = dataclasses.replace(trace, data=trace.smooth(smoothing_sigma))
    return trace


def _detect_population(
    rec: Recording,
    units: BoolArray | None,
    sigma: float,
    *,
    bin_width: float = 0.001,
    **kwargs: Any,
) -> pd.DataFrame:
    trace = population_trace(rec, bin_width=bin_width, units=units, smoothing_sigma=sigma)
    if "normalization_mask" in kwargs:
        indices = rd.core.nearest_sample_index(rec.time, trace.time)
        kwargs["normalization_mask"] = np.asarray(kwargs["normalization_mask"])[indices]
    return trace.detect(**kwargs)


def _ripple_trace_events(
    rec: Recording,
    trace: FloatArray,
    *,
    threshold: float,
    bound_threshold: float = 0.0,
    **kwargs: Any,
) -> pd.DataFrame:
    kwargs.setdefault("minimum_duration", 0.0)
    kwargs.setdefault("speed_threshold", np.inf)
    return _detected(rd.detect_events_from_trace)(
        rec.time,
        trace,
        rec.speed if np.isfinite(kwargs["speed_threshold"]) else _speed_or_unknown(rec),
        rec.fs,
        threshold=threshold,
        bound_threshold=bound_threshold,
        **kwargs,
    )


def _local_peaks(
    rec: Recording, trace: FloatArray, level: float, *, before: float = 0.0, after: float = 0.0
) -> pd.DataFrame:
    """Every local peak above ``level``, each with its own window.

    Windows reach ``before`` and ``after`` seconds from the peak, cut at the
    edges of its valid block; ``clipped_start`` and ``clipped_end`` flag a cut.
    """
    # Keep every local peak, even if two occupy the same above-mean excursion.
    rows = []
    tolerance = _time_tolerance(rec.time)
    _, blocks = _valid_blocks(rec.time, trace)
    for start, stop in blocks:
        peaks, _ = find_peaks(trace[start:stop], height=np.nextafter(level, np.inf))
        for peak in peaks + start:
            first, last = rec.time[peak] - before, rec.time[peak] + after
            rows.append(
                (
                    max(rec.time[start], first),
                    min(rec.time[stop - 1], last),
                    rec.time[peak],
                    trace[peak],
                    bool(first < rec.time[start] - tolerance),
                    bool(last > rec.time[stop - 1] + tolerance),
                )
            )
    return _counted(
        "local peaks",
        pd.DataFrame(
            rows,
            columns=[
                "start_time",
                "end_time",
                "peak_time",
                "peak_value",
                "clipped_start",
                "clipped_end",
            ],
        ),
    )


def _mallory_candidates(time: FloatArray, z: FloatArray) -> pd.DataFrame:
    """Peaks >=3 bounded by inclusive mean crossings, merged by retained peak.

    A bound that reaches a valid block's edge before the trace falls to the
    mean is flagged in ``clipped_start`` or ``clipped_end``.
    """
    # [start, end, peak_time, peak_value, clipped_start, clipped_end]
    rows: list[list[float]] = []
    tolerance = _time_tolerance(time)
    _, blocks = _valid_blocks(time, z)
    for block_start, block_stop in blocks:
        peaks, _ = find_peaks(z[block_start:block_stop], height=3.0)
        block_events: list[list[float]] = []
        for peak in peaks + block_start:
            start = end = int(peak)
            while start > block_start and z[start] > 0:
                start -= 1
            while end < block_stop - 1 and z[end] > 0:
                end += 1
            event = [
                float(time[start]),
                float(time[end]),
                float(time[peak]),
                float(z[peak]),
                float(z[start] > 0),
                float(z[end] > 0),
            ]
            if block_events and event[0] == block_events[-1][0]:
                previous = block_events.pop()
                if previous[3] > event[3]:
                    event[2:4] = previous[2:4]
            block_events.append(event)
        merged: list[list[float]] = []
        for event in block_events:
            if merged and event[2] - merged[-1][2] <= 0.07 + tolerance:
                previous = merged.pop()
                event[0], event[4] = previous[0], previous[4]
                if previous[3] > event[3]:
                    event[2:4] = previous[2:4]
            merged.append(event)
        rows.extend(merged)
    result = pd.DataFrame(
        rows,
        columns=[
            "start_time",
            "end_time",
            "peak_time",
            "peak_value",
            "clipped_start",
            "clipped_end",
        ],
    )
    return _counted(
        "peaks bounded by mean crossings",
        result.astype({"clipped_start": bool, "clipped_end": bool}),
    )


# --------------------------------------------------------------------------- recipes


Role = Literal["candidate_detection", "secondary", "candidate_gate"]
"""A method's scientific use; ``list_methods`` defines each value."""

Stage = Literal["detection", "decoding_candidates"]
"""The initial inventory, or the candidates a paper's decoding analysis kept."""

Inventory = Literal["default", "additional"]
"""Whether the demonstration runs a method by default or it is an addition."""

RequirementKind = Literal["signal", "cells", "intervals", "external", "option"]
"""What a requirement is: a recorded signal, a cell selection, curated
intervals, an external inventory, or a method option."""

# Recording inputs and ``behavior_intervals``, the call's epochs, by kind; any
# other requirement names a method option.
_INPUT_KINDS: dict[str, RequirementKind] = {
    "lfps": "signal",
    "sharp_wave_lfp": "signal",
    "reference_lfp": "signal",
    "multiunit": "signal",
    "speed": "signal",
    "place_cells": "cells",
    "pyramidal": "cells",
    "templates": "cells",
    "sleep_intervals": "intervals",
    "baseline_intervals": "intervals",
    "behavior_intervals": "intervals",
    "example_ripples": "external",
    "external_ripples": "external",
}

_INPUT_DESCRIPTIONS = {
    "lfps": "raw LFP channels",
    "sharp_wave_lfp": "the stratum radiatum LFP",
    "reference_lfp": "a reference LFP",
    "multiunit": "spike counts per unit",
    "speed": "speed in cm/s",
    "place_cells": "a place-cell selection",
    "pyramidal": "a pyramidal-cell selection",
    "templates": "template cell selections",
    "sleep_intervals": "curated sleep intervals",
    "baseline_intervals": "baseline intervals",
    "behavior_intervals": "eligible behavioral epochs",
    "example_ripples": "example ripple intervals",
    "external_ripples": "an external ripple inventory",
}


@dataclass(frozen=True)
class Requirement:
    """An input a literature method needs, declared with its registration.

    ``check_method`` and ``run_method`` test these before running, and
    ``list_methods`` reports them, so the catalog and the checks share one
    declaration.

    Attributes
    ----------
    input : str
        A ``Recording.from_arrays`` input (``lfps``, ``sharp_wave_lfp``,
        ``reference_lfp``, ``multiunit``, ``speed``, ``place_cells``,
        ``pyramidal``, ``templates``, ``sleep_intervals``,
        ``baseline_intervals``, ``example_ripples``, ``external_ripples``),
        the call's ``behavior_intervals``, or a method option.
    meaning : str
        What it must hold for this method, such as "the first selected
        channel" or "track-end reward areas". Empty for the input's plain
        meaning.
    minimum : int
        Least number of selected LFP channels (``lfps`` only).
    measured_only : bool
        Whether only measured data need it: a ``SimulatedSession`` stands in
        with the documented simulation proxy.
    when : tuple of (str, object) pairs
        Needed only when every named option has that value, such as
        ``(("stage", "decoding_candidates"),)``; always when empty.
    unless : str or None
        Not needed when this other input is supplied (Krause's SWRs come
        from ``external_ripples`` or are detected from ``lfps``).
    """

    input: str
    meaning: str = ""
    minimum: int = 1
    measured_only: bool = False
    when: tuple[tuple[str, Any], ...] = ()
    unless: str | None = None

    @property
    def kind(self) -> RequirementKind:
        """What the input is.

        Returns
        -------
        kind : {"signal", "cells", "intervals", "external", "option"}
            The input's kind; any name that is not a recording input or
            ``behavior_intervals`` is a method option.
        """
        return _INPUT_KINDS.get(self.input, "option")

    def describe(self) -> str:
        """One line naming the input, its meaning and when it is needed.

        Returns
        -------
        description : str
            Such as ``"lfps: the first two selected channels (at least 2
            channels)"`` or ``"place_cells (if stage='decoding_candidates')"``.
        """
        text = self.input
        if self.meaning:
            text += f": {self.meaning}"
        notes = []
        if self.minimum > 1:
            notes.append(f"at least {self.minimum} channels")
        notes += [f"if {option}={value!r}" for option, value in self.when]
        if self.unless:
            notes.append(f"unless {self.unless} is supplied")
        if self.measured_only:
            notes.append("measured data")
        return text + (f" ({'; '.join(notes)})" if notes else "")


def _lfps(meaning: str, minimum: int = 1) -> Requirement:
    """Raw LFP channels, saying which of them the method uses."""
    return Requirement("lfps", meaning, minimum=minimum)


def _measured(input: str, meaning: str = "") -> Requirement:
    """An input a SimulatedSession replaces with its documented proxy."""
    return Requirement(input, meaning, measured_only=True)


def _when(input: str, meaning: str = "", **options: Any) -> Requirement:
    """An input needed only for these option values."""
    return Requirement(input, meaning, when=tuple(options.items()))


def _unmet(
    requirement: Requirement,
    rec: Recording,
    behavior_intervals: FloatArray | None,
    options: dict[str, Any],
) -> str | None:
    """Why ``requirement`` is unmet by this call, or None when it is met or
    does not apply."""
    if any(options.get(option) != value for option, value in requirement.when):
        return None
    if requirement.measured_only and rec.allows_simulation_proxies:
        return None
    if (
        requirement.unless
        and _unmet(Requirement(requirement.unless), rec, None, options) is None
    ):
        return None
    name = requirement.input
    session = rec.session
    if name == "lfps":
        lfps = getattr(session, "lfps", None)
        n_channels = 0 if lfps is None else int(np.shape(lfps)[1])
        if n_channels >= requirement.minimum:
            return None
        if requirement.minimum > 1:
            return (
                f"needs at least {requirement.minimum} selected LFP channels, got "
                f"{n_channels}; pass them to Recording.from_arrays"
            )
    elif name == "sharp_wave_lfp":
        values = getattr(session, "sharp_wave_lfp", None)
        if values is not None and np.isfinite(values).any():
            return None
    elif name in {"speed", "multiunit"}:
        values = getattr(session, name, None)
        if values is not None and (name == "speed" or np.shape(values)[1] > 0):
            return None
    elif name in {"place_cells", "pyramidal"}:
        if np.any(getattr(rec, name)):
            return None
    elif name == "templates":
        if rec.templates:
            return None
    elif name == "behavior_intervals":
        if behavior_intervals is not None:
            return None
        return "pass behavior_intervals to run_method or the named method"
    elif name in _INPUT_KINDS:
        if getattr(rec, name) is not None:
            return None
    else:
        if options.get(name) is not None:
            return None
        return f"pass {name}= explicitly (no published value is assumed for measured data)"
    return f"pass {name} to Recording.from_arrays"


def _requirement_problems(
    entry: Recipe,
    rec: Recording,
    behavior_intervals: FloatArray | None,
    options: dict[str, Any],
) -> list[str]:
    """Every declared requirement this call does not meet, one line each."""
    problems = []
    for requirement in entry.requirements:
        reason = _unmet(requirement, rec, behavior_intervals, options)
        if reason is not None:
            problems.append(f"{requirement.describe()} - {reason}")
    if entry.sampling_frequency is not None and not np.isclose(
        rec.fs, entry.sampling_frequency
    ):
        problems.append(
            f"input sampled at {entry.sampling_frequency:g} Hz - this recording is "
            f"{rec.fs:g} Hz; resample before building the Recording"
        )
    return problems


@dataclass(frozen=True)
class Recipe:
    """One registered literature method; ``list_methods`` shows every entry.

    Attributes
    ----------
    row : int
        The paper's row in ``load_literature_parameters()``.
    paper : str
        ``list_methods``' ``paper``.
    trigger : str
        ``list_methods``' ``output``.
    run : callable
        The public method, ``run(rec, **options)``; it dispatches through
        ``run_method`` and returns a DataFrame.
    note : str
        ``list_methods``' ``interpretation``.
    role : {"candidate_detection", "secondary", "candidate_gate"}
        ``list_methods``' ``role``.
    inventory : {"default", "additional"}
        ``list_methods``' ``inventory``: "default" entries are in ``RECIPES``,
        "additional" ones in ``VARIANTS``.
    requirements : tuple of Requirement
        ``list_methods``' ``requirements``.
    sampling_frequency : float or None
        ``list_methods``' ``sampling_frequency``.
    bin_width : float or None
        ``list_methods``' ``bin_width``.
    """

    row: int
    paper: str
    trigger: str
    run: Callable[..., pd.DataFrame]
    note: str
    role: Role = "candidate_detection"
    inventory: Inventory = "default"
    requirements: tuple[Requirement, ...] = ()
    sampling_frequency: float | None = None
    bin_width: float | None = None


RECIPES: list[Recipe] = []


P = ParamSpec("P")


# Raw functions compose intermediate inventories. Public calls all pass through
# run_method once, after the composition is complete.
_IMPLEMENTATIONS: dict[str, Callable[..., pd.DataFrame | FloatArray]] = {}
_ENTRIES: dict[str, Recipe] = {}


def _register(
    registry: list[Recipe],
    row: int,
    paper: str,
    trigger: str,
    *,
    role: Role = "candidate_detection",
    needs: Sequence[str | Requirement] = (),
    sampling_frequency: float | None = None,
    bin_width: float | None = None,
) -> Callable[[Callable[P, pd.DataFrame | FloatArray]], Callable[..., pd.DataFrame]]:
    inventory: Inventory = "default" if registry is RECIPES else "additional"
    requirements = tuple(
        need if isinstance(need, Requirement) else Requirement(need) for need in needs
    )
    behavior = next(
        (need.meaning for need in requirements if need.input == "behavior_intervals"), None
    )

    def register(
        function: Callable[P, pd.DataFrame | FloatArray],
    ) -> Callable[..., pd.DataFrame]:
        name = function.__name__
        if name in _IMPLEMENTATIONS:
            msg = f"A literature method named {name!r} is already registered."
            raise ValueError(msg)
        _IMPLEMENTATIONS[name] = function
        signature = _public_signature(function)

        @functools.wraps(function)
        def public_method(*args: Any, **options: Any) -> pd.DataFrame:
            # Options are checked by run_method, which names every problem.
            if not args and "rec" in options:
                args = (options.pop("rec"),)
            if len(args) != 1:
                msg = f"{name}() takes the recording positionally and options by keyword."
                raise TypeError(msg)
            return run_method(name, args[0], **options)

        public_method.__name__ = name
        public_method.__qualname__ = name
        public_method.__annotations__ = {
            **function.__annotations__,
            "behavior_intervals": "ArrayLike | None",
            "return": pd.DataFrame,
        }
        public_method.__signature__ = signature  # type: ignore[attr-defined]
        public_method.__doc__ = (function.__doc__ or "").rstrip() + _parameter_section(
            signature, behavior
        )
        entry = Recipe(
            row,
            paper,
            trigger,
            public_method,
            (function.__doc__ or "").strip(),
            role,
            inventory,
            requirements,
            sampling_frequency,
            bin_width,
        )
        registry.append(entry)
        _ENTRIES[name] = entry
        return public_method

    return register


def _public_signature(function: Callable[..., Any]) -> inspect.Signature:
    """The implementation's signature with the per-call ``behavior_intervals``
    last, keyword-only, returning a DataFrame."""
    parameters = [
        parameter
        for name, parameter in inspect.signature(function).parameters.items()
        if name != "behavior_intervals"
    ]
    parameters.append(
        inspect.Parameter(
            "behavior_intervals",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="ArrayLike | None",
        )
    )
    return inspect.Signature(parameters, return_annotation=pd.DataFrame)


def _parameter_section(signature: inspect.Signature, behavior: str | None) -> str:
    """The numpy-style Parameters and Returns of a registered method's wrapper."""
    lines = [
        "",
        "",
        "    Parameters",
        "    ----------",
        "    rec : Recording",
        "        Selected signals, cells and curated intervals.",
    ]
    for name, parameter in signature.parameters.items():
        if name in {"rec", "behavior_intervals"}:
            continue
        if parameter.default is inspect.Parameter.empty:
            lines += [f"    {name} : {parameter.annotation}", "        Required; see above."]
        else:
            lines += [
                f"    {name} : {parameter.annotation}, default {parameter.default!r}",
                "        See above.",
            ]
    lines += [
        "    behavior_intervals : array_like, shape (n_intervals, 2), optional",
        (
            f"        Required: the eligible {behavior}. Events not wholly inside one"
            if behavior
            else "        Eligible epochs; events not wholly inside one"
        ),
        "        interval are dropped (see run_method).",
        "",
        "    Returns",
        "    -------",
        "    events : pandas.DataFrame",
        "        Candidate bounds and available diagnostics, with method metadata in",
        "        attrs (see run_method).",
        "    ",
    ]
    return "\n".join(lines)


def _recipe(
    row: int, paper: str, trigger: str, **metadata: Any
) -> Callable[[Callable[P, pd.DataFrame | FloatArray]], Callable[..., pd.DataFrame]]:
    """Register a default inventory; ``metadata`` are ``_register``'s keywords."""
    return _register(RECIPES, row, paper, trigger, **metadata)


@_recipe(0, "Mallory 2025", "MUA", needs=("multiunit", "pyramidal", "speed"))
def mallory_2025(rec: Recording) -> pd.DataFrame:
    """Linear-track MUA candidates using the released peak-merging rule.

    Excitatory spikes, 12.5 ms Gaussian, stopped (<=5 cm/s) statistics,
    peak >=3 SD and inclusive mean-crossing samples. Merges within 70 ms
    compare the retained largest peak; ties retain the later peak. Supply
    one recording per normalization segment; baseline_intervals do not override
    its stopped samples. The ten-cell replay criterion
    and decoding are deliberately downstream of this candidate inventory.
    """
    trace = np.where(rec.speed <= 5.0, rec.rate(rec.pyramidal, 0.0125), np.nan)
    return _mallory_candidates(rec.time, _zscore(trace, ddof=1))


@_recipe(
    1,
    "Widloski 2025",
    "ripple label",
    role="secondary",
    needs=(_lfps("one channel per tetrode, envelopes averaged"), "speed"),
)
def widloski_2025(rec: Recording) -> pd.DataFrame | FloatArray:
    """Replays are defined by decoding (not reproduced). This is the ripple
    label: 100-220 Hz, one channel per tetrode, envelope smoothed with an
    80 ms Gaussian and averaged, z-scored over stopping (speed < 5), peak > 2 SD
    for >= 15 ms, bounds at the mean, merged < 50 ms (the text; the code merges
    none)."""
    return _detected(rd.detect_events_from_trace)(
        rec.time, rec.mean_envelope((100.0, 220.0)), rec.speed, rec.fs,
        threshold=2.0, smoothing_sigma=0.08, normalization_mask=rec.speed < 5,
        minimum_duration=0.015, close_event_threshold=0.05, close_event_rule="merge",
        speed_threshold=np.inf,
    )  # fmt: skip


def _population_with_ripple_peak(
    rec: Recording, sleep: FloatArray, behavior_intervals: FloatArray | None
) -> FloatArray:
    """Population candidates normalized over supplied NREM, with a ripple peak,
    inside the eligible quiet-waking/NREM ``behavior_intervals``.

    The historical ripple detector is unresolved. Real recordings require an
    external inventory; the simulation alone uses the documented Zugaro proxy.
    """
    events = _detect_population(
        rec,
        rec.pyramidal,
        0.015,
        threshold=3.0,
        normalization_mask=rec.intervals_to_mask(sleep),
        minimum_duration=0.0,
        minimum_event_duration=0.05,
        maximum_duration=0.5,
        speed_threshold=np.inf,
    )
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.pyramidal
    )
    if rec.external_ripples is None:
        if not rec.allows_simulation_proxies:
            msg = "Supply external_ripples; the historical LFP detector is unspecified."
            raise ValueError(msg)
        peaks = _zugaro_ripple_peaks(rec, (130.0, 200.0)).peak_time
    else:
        peaks = (
            rec.external_ripples[:, 2]
            if rec.external_ripples.shape[1] == 3
            else rec.external_ripples.mean(axis=1)
        )
    events = rd.require_times_inside(events, peaks)
    if behavior_intervals is not None:
        eligible = behavior_intervals
    elif not rec.allows_simulation_proxies:
        msg = "Supply eligible quiet-waking/NREM behavior_intervals separately from the NREM baseline."
        raise ValueError(msg)
    else:
        eligible = sleep
    return bounds(rd.require_inside(events, eligible))


@_recipe(
    2,
    "Yang 2024",
    "SWR+MUA",
    needs=(
        "multiunit",
        "pyramidal",
        _measured("sleep_intervals", "curated NREM, the normalization epoch"),
        _measured("behavior_intervals", "quiet-waking/NREM epochs"),
        _measured("external_ripples", "ripple intervals, peaks as a third column"),
    ),
    bin_width=0.001,
)
def yang_2024(
    rec: Recording, *, behavior_intervals: FloatArray | None = None
) -> pd.DataFrame | FloatArray:
    """Population candidates with a coincident externally supplied ripple peak.

    Supply curated NREM normalization and eligible quiet-waking/NREM intervals;
    the paper used SleepScoreMaster with manual curation. Only simulation uses
    a speed <4 and theta/delta <1 proxy and an assumed ripple detector. The
    15 ms Gaussian width is interpreted as SD as in the shared Grosmark path.
    """
    return _population_with_ripple_peak(rec, rec.sleep(4.0, 1.0), behavior_intervals)


def _tirole(rec: Recording) -> pd.DataFrame:
    """Tirole 2022's candidate rule, shared with Huelin Gorriz 2023."""
    trace = population_trace(rec, bin_width=0.001)
    kernel = np.exp(-0.5 * (np.arange(-20, 21) / 10) ** 2)
    kernel /= kernel.sum()

    def smooth_block(x: FloatArray) -> FloatArray:
        return np.asarray(filtfilt(kernel, [1.0], x, padlen=120), float)

    smoothed = _transform(
        trace.time,
        trace.data,
        smooth_block,
        minimum_length=121,  # filtfilt needs more samples than its padding
        reason="Tirole's 41-point forward/backward kernel",
    )
    trace = dataclasses.replace(trace, data=_zscore(smoothed, ddof=1))
    found = _tirole_bounds(trace.time, trace.data)
    events = trace.merge(within_duration(found, 0.1), 0.05)
    keep = []
    for start, end in events:
        n_steps = int(np.floor((end - start + _time_tolerance(rec.time)) / 0.01))
        grid = start + np.arange(n_steps + 1) * 0.01
        speed = rec.speed[rd.core.nearest_sample_index(rec.time, grid)]
        keep.append(bool(np.isfinite(speed).all() and np.median(speed) <= 5))
    events = events[np.asarray(keep, dtype=bool)]
    events = bounds(
        rd.require_active_units(
            events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
        )
    )
    ripple_time, amplitude = _tirole_ripple_amplitude(rec)
    z = _zscore(amplitude, ddof=1)
    selected = np.asarray(
        [np.any(z[(ripple_time > a) & (ripple_time < b)] >= 3) for a, b in events], dtype=bool
    )
    events = events[selected]
    # A merged event starts at its first part's start and ends at its last's end.
    reported = trace.sample_bounds(events)
    return pd.DataFrame(
        {
            "start_time": reported[:, 0],
            "end_time": reported[:, 1],
            "clipped_start": found.groupby("start_time")
            .clipped_start.any()[events[:, 0]]
            .to_numpy(dtype=bool)
            .reshape(-1),
            "clipped_end": found.groupby("end_time")
            .clipped_end.any()[events[:, 1]]
            .to_numpy(dtype=bool)
            .reshape(-1),
        }
    )


@_recipe(
    3,
    "Huelin Gorriz 2023",
    "SWR+MUA",
    needs=(
        "multiunit",
        "place_cells",
        "speed",
        _lfps("the first selected channel, for the ripple gate"),
    ),
    bin_width=0.001,
)
def huelin_gorriz_2023(
    rec: Recording, *, interpretation: str = "published_cap"
) -> pd.DataFrame | FloatArray:
    """Published duration cap or explicitly selected related Tirole code.

    The original extractor is missing from the Huelin Gorriz release. Both
    choices use Tirole's finite-kernel reconstruction, not a verified original
    pipeline. ``published_cap`` applies the reported 750 ms cap; ``related_code``
    retains the related extractor's uncapped durations.
    """
    if interpretation not in {"published_cap", "related_code"}:
        msg = "interpretation must be 'published_cap' or 'related_code'."
        raise ValueError(msg)
    events = _tirole(rec)
    if interpretation == "related_code":
        return events
    return events[_within_duration_mask(events, high=0.75)]


def _spiking_filter(rec: Recording, ripples: pd.DataFrame) -> pd.DataFrame:
    """eventSpikingTreshold: keep an event when the z-scored pyramidal count
    (10 ms windows), averaged over +/- 10 ms of its peak, exceeds 0.5."""
    count = rec.boxcar(rec.counts(rec.pyramidal), 0.01)
    near_peak = rec.boxcar(rd.normalize_signal(count), 0.02)
    if len(ripples) == 0:
        return ripples
    keep = near_peak[rd.core.nearest_sample_index(rec.time, ripples.peak_time)] > 0.5
    return ripples[keep]


def _check_stage(stage: str) -> None:
    if stage not in {"detection", "decoding_candidates"}:
        msg = "stage must be 'detection' or 'decoding_candidates'."
        raise ValueError(msg)


def _harvey_stage(
    rec: Recording, events: pd.DataFrame | FloatArray, stage: Stage
) -> FloatArray:
    """Replay selection on complete 20 ms bins, before scoring and shuffling."""
    _check_stage(stage)
    if stage == "detection":
        return bounds(events)
    candidates = within_duration(events, low=0.08)
    candidates = rec.merge(candidates, 0.0, rec.session.raw_lfp, inclusive=True)
    candidates = rd.require_active_units(
        candidates, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
    )
    counts = rec.counts(rec.place_cells)
    tolerance = _time_tolerance(rec.time)
    keep = []
    for start, end in candidates:
        n_bins = int(np.floor((end - start + tolerance) / 0.02))
        selected = _event_slice(rec.time, start, start + n_bins * 0.02, tolerance)
        relative = rec.time[selected] - start
        nearest = np.rint(relative / 0.02) * 0.02
        relative = np.where(np.abs(relative - nearest) <= tolerance, nearest, relative)
        inside = (relative >= 0) & (relative < n_bins * 0.02)
        if n_bins == 0 or not np.isfinite(counts[selected][inside]).all():
            keep.append(False)
            continue
        binned = np.histogram(
            relative[inside], np.arange(n_bins + 1) * 0.02, weights=counts[selected][inside]
        )[0]
        active = np.count_nonzero(
            np.any(rec.multiunit[selected][inside][:, rec.place_cells] > 0, axis=0)
        )
        keep.append(bool(active >= 5 and np.mean(binned == 0) < 0.5))
    return np.asarray(candidates[np.asarray(keep, dtype=bool)], dtype=float)


@_recipe(
    4,
    "Harvey 2023 (code)",
    "SWR",
    needs=(
        _lfps("the first selected channel: CA1 pyramidal layer"),
        Requirement(
            "sharp_wave_lfp",
            "the stratum radiatum channel; without one, harvey_2023_no_radiatum is the "
            "released branch",
        ),
        "multiunit",
        "pyramidal",
        _when("place_cells", stage="decoding_candidates"),
    ),
)
def harvey_2023_code(
    rec: Recording, *, stage: Stage = "detection"
) -> pd.DataFrame | FloatArray:
    """Released DetectSWR path on pyramidal and radiatum LFP, then spiking veto.

    Uses neurocode defaults: 2-50 Hz sharp waves, 80-250 Hz ripples,
    local thresholds 0.5/2.5 SD, sharp waves 20-500 ms, ripples >=25 ms.
    Supply selected pyramidal and radiatum channels and CA1 pyramidal spikes.
    Manual curation and EMG vetoes are external. Replay filtering is selectable
    with stage='decoding_candidates'; detection is the default.
    """
    ripples = _detected(rd.Long_sharp_wave_ripple_detector)(
        rec.time, rec.session.raw_lfp, _speed_or_unknown(rec), rec.fs,
        sharp_wave_lfp=rec.session.sharp_wave_lfp, speed_threshold=np.inf,
    )  # fmt: skip
    return _harvey_stage(rec, _spiking_filter(rec, ripples), stage)


@_recipe(
    4,
    "Harvey 2023 (text)",
    "SWR (needs radiatum)",
    needs=(
        _lfps("the first selected channel: CA1 pyramidal layer"),
        Requirement("sharp_wave_lfp", "the stratum radiatum channel"),
        _measured("baseline_intervals", "the normalization epoch (unstated in the paper)"),
        _when("multiunit", stage="decoding_candidates"),
        _when("place_cells", stage="decoding_candidates"),
    ),
)
def harvey_2023_text(
    rec: Recording, *, sharp_wave_polarity: float = -1.0, stage: Stage = "detection"
) -> FloatArray:
    """Published difference-of-Gaussians path, distinct from the code variant.

    Ripple 80-250 Hz, rectification and Gaussian low-pass at 55 Hz. A trace
    clipped at four SD estimates the power mean/SD; detection uses the
    unclipped trace at 4 SD, bounds 1 SD, >=15 ms. Coincident radiatum 5-40 Hz
    waves exceed 2.5 SD for 20-400 ms. Polarity and normalization epoch are
    unstated: negative polarity is an explicit default, baseline_intervals
    are required for measured data. Clipping before rectification follows
    the cited lab convention; this is not the released DetectSWR path.
    stage='decoding_candidates' optionally applies the released replay gates
    (>=80 ms, >=5 selected place cells, <50% empty 20 ms bins) to these SWRs.
    """
    if sharp_wave_polarity not in (-1.0, 1.0):
        msg = "sharp_wave_polarity must be -1 or 1."
        raise ValueError(msg)
    baseline = _baseline(rec, required=not rec.allows_simulation_proxies)
    band = rec.transform(
        rec.session.raw_lfp, lambda x: _difference_of_gaussians_band(x, (80.0, 250.0), rec.fs)
    )
    scale = float(np.std(_baseline_samples(band, baseline)))
    kernel = _gaussian_lowpass_fir(55.0, rec.fs)
    clipped = rec.transform(
        np.abs(np.clip(band, -4 * scale, 4 * scale)), lambda x: _firfilt(x, kernel)
    )
    power = rec.transform(np.abs(band), lambda x: _firfilt(x, kernel))
    clipped_baseline = _baseline_samples(clipped, baseline)
    mean, sd = float(np.mean(clipped_baseline)), float(np.std(clipped_baseline))
    ripples = _ripple_trace_events(
        rec,
        (power - mean) / sd,
        threshold=4.0,
        bound_threshold=1.0,
        normalization_method="none",
        minimum_event_duration=0.015,
    )
    sharp = rec.transform(
        rec.session.sharp_wave_lfp,
        lambda x: _difference_of_gaussians_band(x, (5.0, 40.0), rec.fs),
    )
    waves = _ripple_trace_events(
        rec,
        sharp_wave_polarity * sharp,
        threshold=2.5,
        bound_threshold=2.5,
        normalization_mask=baseline,
        minimum_event_duration=0.02,
        maximum_duration=0.4,
    )
    return _harvey_stage(rec, rd.require_overlap(ripples, waves), stage)


@_recipe(
    5,
    "Liu 2023",
    "SWR+MUA (needs radiatum)",
    needs=(
        _lfps("the first selected channel: CA1 pyramidal layer"),
        Requirement("sharp_wave_lfp", "the stratum radiatum channel"),
        "multiunit",
        "pyramidal",
    ),
    bin_width=0.001,
)
def liu_2023(rec: Recording) -> pd.DataFrame | FloatArray:
    """DetectSWR at neurocode defaults on the pyramidal and radiatum channels
    (the text's 1 SD bounds and 15-400 ms limits are not applied; manual
    curation is not reproduced); the candidates are pyramidal-cell bursts
    (10 ms Gaussian, assumed to be its SD; > 2 SD, bounds at the mean,
    100-500 ms) overlapping an SWR."""
    swrs = _detected(rd.Long_sharp_wave_ripple_detector)(
        rec.time, rec.session.raw_lfp, _speed_or_unknown(rec), rec.fs,
        sharp_wave_lfp=rec.session.sharp_wave_lfp, speed_threshold=np.inf,
    )  # fmt: skip
    bursts = _detect_population(
        rec, rec.pyramidal, 0.010,
        threshold=2.0, minimum_duration=0.0, minimum_event_duration=0.1,
        maximum_duration=0.5, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_overlap(bursts, swrs)


@_recipe(
    6,
    "Tirole 2022",
    "SWR+MUA",
    needs=(
        "multiunit",
        "place_cells",
        "speed",
        _lfps("the first selected channel, for the ripple gate"),
    ),
    bin_width=0.001,
)
def tirole_2022(rec: Recording) -> pd.DataFrame | FloatArray:
    """Released Tirole finite kernels and candidate order, with supplied cells.

    Native 1 ms counts of every supplied unit, 41-point gausswin(alpha=2),
    forward/backward filtering, sample-SD z scores, 10 ms-separated threshold
    anchors (z >= 3), inclusive below-zero crossings with 0.25/0.5 fallbacks
    within 300 ms; >=100 ms before <50 ms merging. Speed is sampled every
    10 ms and its median must be <=5 cm/s; >=5 active place cells follow.
    The ripple gate needs a z-scored ripple amplitude >=3 SD inside the event:
    the first selected LFP channel resampled to 1000 Hz, a 35-tap 125-300 Hz
    Hamming FIR applied forward and backward, Hilbert amplitude and a 15 ms
    moving average. Bin origin is the recording start; counts retain the input
    timestamp precision. The duration, merge, speed, cell and ripple rules use
    bin centers, as the release's onset-offset differences do; the reported
    bounds are the first and last recorded samples of the outer bins, as for
    other native grids. LFP resampling uses scipy's polyphase anti-alias
    filter, whose edge behavior can differ from the original
    acquisition/downsampling pipeline.
    """
    return _tirole(rec)


@_recipe(7, "Bush 2022", "MUA", needs=("multiunit", "pyramidal", "speed"))
def bush_2022(rec: Recording) -> pd.DataFrame | FloatArray:
    """Pyramidal cells, 5 ms Gaussian, peak z >= 3, bounds at z >= 0; merged
    when <= 40 ms apart, events <= 40 ms dropped, then >= 5 or 15% of
    pyramidal cells (whichever is larger), median speed <= 10, <= 0.5 s."""
    trace = rec.rate(rec.pyramidal, 0.005)
    events = _detected(rd.detect_events_from_trace)(
        rec.time, trace, rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    merged = rec.merge(events, 0.04, trace, inclusive=True)
    merged = within_duration(merged, low=0.04 + 1 / rec.fs)
    merged = rd.require_active_units(
        merged, rec.multiunit, rec.time,
        minimum_active_units=5, minimum_active_fraction=0.15, units=rec.pyramidal,
    )  # fmt: skip
    merged = rd.exclude_movement(merged, rec.speed, rec.time, 10.0, rule="median")
    return within_duration(merged, high=0.5)


@_recipe(8, "Berners-Lee 2022", "MUA", needs=("multiunit", "speed"), bin_width=0.001)
def berners_lee_2022(rec: Recording) -> pd.DataFrame:
    """Released finite-kernel SDEs on the caller-selected spike population.

    1 ms bins, 100-point Gaussian (SD 10 samples), zero-padded filter2-style
    convolution. Sample-SD statistics over |speed|<5; mean crossings clipped
    at movement, peak >=3 SD, strictly >100 and <500 native bins. The release
    pools its input spikedata; cell-type inclusion is a caller selection,
    not an inferred exclusion of interneurons. Bin origin is recording start.
    """
    trace = population_trace(rec, bin_width=0.001)
    kernel = np.exp(-0.5 * ((np.arange(100) - 49.5) / 10) ** 2)
    kernel /= kernel.sum()

    def smooth(x: FloatArray) -> FloatArray:
        # MATLAB conv2(...,'same') crops at floor(kernel_length/2).
        return np.convolve(x, kernel, mode="full")[50 : 50 + len(x)]

    stopped = np.abs(_known_speed(trace.speed)) < 5
    data = _zscore(_transform(trace.time, trace.data, smooth), stopped, ddof=1)
    data[~stopped] = np.nan
    return dataclasses.replace(trace, data=data).detect(
        threshold=3.0,
        bound_threshold=np.nextafter(0.0, np.inf),
        normalization_method="none",
        minimum_duration=0.0,
        minimum_event_duration=0.101,
        maximum_duration=0.499,
        speed_threshold=np.inf,
    )


def _pfeiffer_2015_swrs(
    rec: Recording,
    threshold: float = 3.0,
    channels: int | None = None,
    maximum_duration: float = 2.0,
) -> pd.DataFrame:
    """Pfeiffer & Foster 2015's SWR rule at ``threshold`` SD, over the first
    ``channels`` selected channels (all by default), 50 ms to ``maximum_duration``."""
    return _detected(rd.detect_events_from_trace)(
        rec.time, rec.mean_envelope((150.0, 250.0), channels), rec.speed, rec.fs,
        threshold=threshold, smoothing_sigma=0.0125, normalization_mask=rec.speed < 5,
        minimum_duration=0.0, minimum_event_duration=0.05, maximum_duration=maximum_duration,
        speed_threshold=5.0,
    )  # fmt: skip


@_recipe(
    10,
    "Krause 2022",
    "SWR",
    needs=(
        "multiunit",
        "place_cells",
        Requirement(
            "lfps",
            "every selected channel, averaged, for Pfeiffer 2015 SWRs",
            unless="external_ripples",
        ),
        Requirement("speed", unless="external_ripples"),
    ),
)
def krause_2022(rec: Recording) -> FloatArray:
    """Supplied or Pfeiffer-style SWRs trimmed using per-SWR 3 ms place-cell bins.

    Zero-padded four-bin convolution, divided by cell count, >2 spikes/s;
    first to last crossing separated by >=10 bins. Supply external_ripples
    to reuse the original inventory. The released get_spikemat keeps bins
    whose right edge is strictly before the SWR end. Its start bound uses
    the first high bin's start; its end subtracts trailing bin widths from
    the SWR end, retaining the unbinned remainder. Thus moving the SWR end
    can move the trimmed end with identical spikes. This released-code
    convention is preserved; floating-point bin-edge drift is not.
    Missing spike support rejects that SWR instead of smoothing across it.
    """
    swrs = bounds(
        _pfeiffer_2015_swrs(rec)
        if rec.external_ripples is None
        else rec.external_ripples[:, :2]
    )
    count = rec.counts(rec.place_cells)
    _, blocks = _valid_blocks(rec.time, count)
    intervals = np.asarray([(rec.time[a], rec.time[b - 1]) for a, b in blocks])
    tolerance = _time_tolerance(rec.time)
    found = []
    n_missing = n_short = 0
    for start, end in swrs:
        group = np.searchsorted(intervals[:, 0], start, side="right") - 1
        if group < 0 or end > intervals[group, 1]:
            n_missing += 1
            continue
        n_bins = int(np.ceil((end - start - tolerance) / 0.003)) - 1
        if n_bins < 11:
            n_short += 1
            continue
        samples = _event_slice(rec.time, start, start + n_bins * 0.003, tolerance)
        relative = rec.time[samples] - start
        nearest_edge = np.rint(relative / 0.003) * 0.003
        relative = np.where(
            np.abs(relative - nearest_edge) <= tolerance, nearest_edge, relative
        )
        selected = (relative >= 0) & (relative < n_bins * 0.003)
        counts = np.histogram(
            relative[selected], np.arange(n_bins + 1) * 0.003, weights=count[samples][selected]
        )[0]
        rate = np.convolve(
            counts / (rec.place_cells.sum() * 0.003), np.ones(4) / 4, mode="same"
        )
        high = np.flatnonzero(rate > 2)
        if len(high) > 1 and high[-1] - high[0] >= 10:
            found.append((start + high[0] * 0.003, end - (n_bins - high[-1]) * 0.003))
    if n_missing or n_short:
        rd.core._warn_at_caller(
            f"{n_missing + n_short} of {len(swrs)} SWR(s) skipped: {n_missing} crossing "
            f"missing place-cell spikes or a timestamp gap, {n_short} with fewer than the "
            "11 complete 3 ms bins the trimming rule needs."
        )
    return np.asarray(found, float).reshape(-1, 2)


@_recipe(
    11,
    "Mou 2022",
    "MUA",
    needs=(
        "multiunit",
        _when("place_cells", "one template's cells", stage="decoding_candidates"),
    ),
    bin_width=0.01,
)
def mou_2022(
    rec: Recording, *, normalization: str = "minmax", stage: Stage = "detection"
) -> FloatArray:
    """10 ms all-spike bins, 20 ms Gaussian; explicit minmax or maximum scaling.

    The original normalization callback is missing. ``normalization='minmax'``
    names the reconstruction explicitly; 'maximum' divides by the maximum.
    Peaks >0.35, bounds 0.15, gaps <30 ms. Detection returns all PBEs.
    stage='decoding_candidates' adds >=4 active cells from one template
    supplied via place_cells.
    """
    _check_stage(stage)
    if normalization not in {"minmax", "maximum"}:
        msg = "normalization must be minmax or maximum."
        raise ValueError(msg)
    trace = population_trace(rec, bin_width=0.01, smoothing_sigma=0.02)
    offset = float(np.nanmin(trace.data)) if normalization == "minmax" else 0.0
    span = np.nanmax(trace.data) - offset
    if not np.isfinite(span) or span == 0:
        msg = "Min-max normalization needs a nonconstant population trace."
        raise ValueError(msg)
    trace = dataclasses.replace(trace, data=(trace.data - offset) / span)
    events = trace.detect(
        threshold=0.35,
        bound_threshold=0.15,
        normalization_method="none",
        minimum_duration=0.0,
        close_event_threshold=0.03,
        close_event_rule="merge",
        speed_threshold=np.inf,
    )
    if stage == "detection":
        return bounds(events)
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=4, units=rec.place_cells
    )


@_recipe(
    12,
    "Berners-Lee 2021",
    "SWR",
    needs=(
        _lfps(
            "the first three selected channels (the tetrodes with the most pyramidal cells), averaged",
            minimum=3,
        ),
        "speed",
    ),
)
def berners_lee_2021(rec: Recording) -> pd.DataFrame | FloatArray:
    """Pfeiffer & Foster 2015's rule at 2 SD on three selected tetrodes.

    Supply the channels from the three tetrodes with the most pyramidal cells
    first; the function uses the first three channels without ranking them.
    """
    return _pfeiffer_2015_swrs(rec, threshold=2.0, channels=3)


@_recipe(
    13,
    "Denovellis 2021",
    "SWR",
    needs=(_lfps("every selected CA1 channel, squared and summed"), "speed"),
    sampling_frequency=1500.0,
)
def denovellis_2021(rec: Recording) -> pd.DataFrame:
    """Historical Kay trace: square filtered LFP, sum, 4 ms Gaussian, then root.

    Uses the 101-tap equiripple filter at 1500 Hz and the paper/0.1.8.dev0
    signal definition, before the package
    switched to squared Hilbert envelopes. Z over the session, >=2 SD for
    >=15 ms, mean bounds, endpoint speed <=4. The modern package's missing
    data policy and duration rounding remain in force. Select CA1 channels
    before calling; the two-spiking-tetrode analysis filter is downstream.
    """
    from scipy.signal import remez

    if not np.isclose(rec.fs, 1500):
        msg = "The historical Denovellis filter requires input sampled at 1500 Hz."
        raise ValueError(msg)
    kernel = remez(101, [0, 125, 150, 250, 275, 750], [0, 1, 0], fs=1500)

    def historical_filter(x: FloatArray) -> FloatArray:
        return np.asarray(filtfilt(kernel, [1.0], x, axis=0), float)

    filtered = _transform(
        rec.time,
        rec.session.lfps,
        historical_filter,
        minimum_length=3 * len(kernel) + 1,  # more than filtfilt's default padding
        reason="the historical 101-tap ripple filter",
    )
    trace = np.sqrt(np.maximum(0, rec.smooth(np.sum(filtered**2, axis=1), 0.004)))
    return _ripple_trace_events(
        rec, trace, threshold=2.0, minimum_duration=0.015, speed_threshold=4.0
    )


@_recipe(
    14,
    "Gillespie 2021",
    "SWR",
    needs=(_lfps("every selected channel, combined by the Kay consensus"), "speed"),
)
def gillespie_2021(rec: Recording) -> pd.DataFrame | FloatArray:
    """Kay consensus trace over every selected channel (square root of the
    summed squared 150-250 Hz envelopes, 4 ms Gaussian), 2 SD for >= 15 ms,
    bounds at the mean, speed < 4 cm/s at both ends."""
    return _detected(rd.Kay_ripple_detector)(
        rec.time,
        rec.filtered((150.0, 250.0)),
        rec.speed,
        rec.fs,
        speed_threshold=np.nextafter(4.0, -np.inf),
    )


def _michon(rec: Recording, *, order: str = "text") -> FloatArray:
    """5 ms population bins; smoothing/detrending order is explicitly selectable.

    The ripple envelope averages the first three selected channels, or all of
    them when fewer are selected (the papers used one to three tetrodes).
    """
    if order not in {"text", "code"}:
        msg = "order must be 'text' or 'code'."
        raise ValueError(msg)

    def detrended(time: FloatArray, values: FloatArray, fs: float) -> FloatArray:
        window = round(3 * fs) | 1

        def detrend(x: FloatArray) -> FloatArray:
            return np.asarray(x - median_filter(x, size=window, mode="nearest"), dtype=float)

        def smooth(x: FloatArray) -> FloatArray:
            return rd.gaussian_smooth(x, 0.015, fs)

        first, second = (smooth, detrend) if order == "text" else (detrend, smooth)
        return _transform(time, _transform(time, values, first), second)

    common: dict[str, Any] = {
        "bound_threshold": 0.5,
        "minimum_duration": 0.0,
        "close_event_threshold": 0.02,
        "close_event_rule": "merge",
        "speed_threshold": np.inf,
    }
    ripples = _detected(rd.detect_events_from_trace)(
        rec.time,
        detrended(
            rec.time,
            rec.mean_envelope((140.0, 225.0), channels=min(3, rec.session.lfps.shape[1])),
            rec.fs,
        ),
        rec.speed,
        rec.fs,
        threshold=8.0,
        minimum_event_duration=0.04,
        **common,
    )
    population = population_trace(rec, bin_width=0.005)
    population = dataclasses.replace(
        population,
        data=detrended(population.time, population.data, population.sampling_frequency),
    )
    bursts = population.detect(threshold=4.0, minimum_event_duration=0.08, **common)
    return rd.exclude_movement(rd.require_overlap(bursts, ripples), rec.speed, rec.time, 5.0)


@_recipe(
    15,
    "Michon 2021",
    "SWR+MUA",
    needs=(
        _lfps("the first three selected channels (all, if fewer), averaged"),
        "multiunit",
        "speed",
    ),
    bin_width=0.005,
)
def michon_2021(rec: Recording, *, order: str = "text") -> FloatArray:
    """5 ms MUA bins; 15 ms smoothing, 3 s detrending; text/code order selectable."""
    return _michon(rec, order=order)


@_recipe(
    16, "Igata 2021 (candidates)", "SWR+MUA", needs=("multiunit", "speed"), bin_width=0.001
)
def igata_2021(rec: Recording) -> pd.DataFrame | FloatArray:
    """The candidate stage only: the rate of all recorded neurons (15 ms
    Gaussian) z-scored over stopping (< 5 cm/s), > 2 SD, bounds at the mean,
    > 4 active neurons, 50 ms-2 s; no speed rule on events (none stated). The
    GMM split on rate and ripple power that follows is under-specified."""
    events = _detect_population(
        rec, None, 0.015,
        threshold=2.0, normalization_mask=rec.speed < 5, minimum_duration=0.0,
        minimum_event_duration=0.05, maximum_duration=2.0, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(events, rec.multiunit, rec.time, minimum_active_units=5)


@_recipe(
    17,
    "Gridchyn 2020",
    "adaptive MUA triggers",
    needs=("multiunit", Requirement("baseline_intervals", "the pre-rest epoch")),
)
def gridchyn_2020(
    rec: Recording,
    *,
    initial_multiplier: float = 3.5,
    target_rate: float = 1.0,
    update_interval: float = 60.0,
    gain: float = 0.5,
    refractory: float = 0.15,
) -> pd.DataFrame:
    """Causal replay of trailing 20 ms counts and adaptive HSE triggers.

    Counts use half-open windows (t - 20 ms, t], evaluated on the supplied
    sample grid. Supply a pre-rest baseline. The initial multiplier is 3.5; every minute
    it changes by 0.5*(observed trigger rate - 1 Hz). Trigger refractory is
    150 ms. Offline bounds extend to baseline crossings, onset to first spike.
    Outputs retain each trigger's time and multiplier as columns, and the
    update history and expected count in attrs. The multiplier has no floor;
    a warning reports the first update that takes it to zero or below.

    This implements the published feedback rule, not hardware integration.
    The C++ release's apparent +1 rate-count offset is not reproduced; rate
    here is actual triggers since the previous update divided by observed
    time. The feedback clock pauses across missing data; this offline gap
    policy is separate from the original hardware's wall-clock behavior.
    No trigger or counting window crosses missing spikes or timestamp gaps.
    """
    if (
        initial_multiplier <= 0
        or target_rate < 0
        or update_interval <= 0
        or gain < 0
        or refractory < 0
        or not np.isfinite(
            [initial_multiplier, target_rate, update_interval, gain, refractory]
        ).all()
    ):
        msg = "Invalid adaptive detector parameters."
        raise ValueError(msg)
    count = rec.counts()
    mask = _baseline(rec, required=True) & np.isfinite(count)
    if not mask.any():
        msg = "The supplied pre-rest baseline contains no valid spikes."
        raise ValueError(msg)
    expected = float(np.mean(count[mask]) * rec.fs * 0.02)
    if expected <= 0:
        msg = "Pre-rest must have a positive population firing rate."
        raise ValueError(msg)
    trailing = np.full(len(count), np.nan)
    _, blocks = _valid_blocks(rec.time, count)
    for start, stop in blocks:
        relative = rec.time[start:stop] - rec.time[start]
        # Half-open (now - 20 ms, now]: exactly 20 samples at 1 kHz.
        left = np.searchsorted(
            relative, relative - 0.02 + _time_tolerance(rec.time), side="right"
        )
        cumulative = np.r_[0.0, np.cumsum(count[start:stop])]
        trailing[start:stop] = cumulative[1:] - cumulative[left]
    factor = initial_multiplier
    last_update = 0.0
    observed_time = 0.0
    last_trigger = -np.inf
    recent = 0
    history: list[tuple[float, float, float]] = []
    rows = []
    for block_start, block_stop in blocks:
        block_clock = observed_time + (1 / rec.fs if block_start != blocks[0][0] else 0)
        for index in range(block_start, block_stop):
            now = float(rec.time[index])
            observed_time = block_clock + float(rec.time[index] - rec.time[block_start])
            if observed_time - last_update > update_interval:
                rate = recent / (observed_time - last_update)
                factor += gain * (rate - target_rate)
                # The published feedback rule has no documented floor.
                if factor <= 0 and not any(f <= 0 for _, _, f in history):
                    rd.core._warn_at_caller(
                        f"The Gridchyn threshold multiplier reached {factor:.3g} at "
                        f"{now:.6g} s; the published rule has no floor, so every "
                        "sample with a spike can trigger until it recovers."
                    )
                history.append((now, rate, factor))
                last_update, recent = observed_time, 0
            if (
                trailing[index] >= expected * factor
                and count[index] > 0
                and now - last_trigger >= refractory
            ):
                start = end = index
                while start > block_start and trailing[start - 1] > expected:
                    start -= 1
                while end < block_stop - 1 and trailing[end + 1] > expected:
                    end += 1
                spikes = np.flatnonzero(count[start : end + 1] > 0)
                if len(spikes):
                    start += int(spikes[0])
                rows.append(
                    (
                        rec.time[start],
                        rec.time[end],
                        now,
                        factor,
                        start == block_start,
                        end == block_stop - 1,
                    )
                )
                last_trigger = now
                recent += 1
    result = pd.DataFrame(
        rows,
        columns=[
            "start_time",
            "end_time",
            "trigger_time",
            "threshold_multiplier",
            "clipped_start",
            "clipped_end",
        ],
    )
    result.attrs["threshold_updates"] = [
        {"time": t, "observed_rate": r, "multiplier": f} for t, r, f in history
    ]
    result.attrs["expected_count"] = expected
    return result


@_recipe(
    18,
    "Kaefer 2020",
    "SWR label",
    role="secondary",
    needs=(
        _lfps("every selected channel, band RMS averaged"),
        "reference_lfp",
        Requirement(
            "baseline_intervals", "the normalization epoch (unspecified in the paper)"
        ),
    ),
)
def kaefer_2020(rec: Recording) -> pd.DataFrame:
    """Secondary SWR label: reference-subtracted 240 ms FFT chunks every 20 ms.

    Average each channel's 150-250 Hz RMS, peak 5 SD, bounds 1.5 SD. Measured
    inputs require reference_lfp and baseline_intervals. The FFT uses a
    rectangular frequency mask and chunk centers (taper/anchor unspecified).
    The paper sampled at 5 kHz; the same chunk durations are supported at
    other rates. Replay's adaptive spike-window decoding is separate.
    """
    if rec.reference_lfp is None:
        msg = "Supply reference_lfp for the Kaefer ripple inventory."
        raise ValueError(msg)
    raw = rec.session.lfps - rec.reference_lfp[:, None]
    width, step = round(0.24 * rec.fs), round(0.02 * rec.fs)
    if width < 2 or step < 1:
        msg = "Sampling frequency is too low for FFT windows."
        raise ValueError(msg)
    centers = np.arange(width // 2, len(rec.time) - (width - width // 2) + 1, step)
    power = np.full(len(centers), np.nan)
    is_valid, blocks = _valid_blocks(rec.time, raw)
    blocks = _drop_short_blocks(blocks, is_valid, width, "Kaefer's 240 ms FFT chunk")
    frequencies = np.fft.rfftfreq(width, 1 / rec.fs)
    band = (frequencies >= 150) & (frequencies <= 250)
    # A chunk is used only when it lies inside one valid block.
    block_starts, block_stops = np.asarray(blocks).reshape(-1, 2).T
    starts = centers - width // 2
    block = np.searchsorted(block_starts, starts, side="right") - 1
    fits = (block >= 0) & (starts + width <= block_stops[np.maximum(block, 0)])
    for j in np.flatnonzero(fits):
        start = starts[j]
        spectrum = np.fft.rfft(raw[start : start + width], axis=0)
        spectrum[~band] = 0
        filtered = np.fft.irfft(spectrum, n=width, axis=0)
        power[j] = np.sqrt(np.mean(filtered**2, axis=0)).mean()
    time = rec.time[centers]
    baseline = _baseline(rec, required=True)[centers]
    return _detected(rd.detect_events_from_trace)(
        time,
        power,
        _speed_or_unknown(rec)[centers],
        rec.fs / step,
        threshold=5.0,
        bound_threshold=1.5,
        normalization_mask=baseline,
        minimum_duration=0.0,
        speed_threshold=np.inf,
    )


@_recipe(
    19,
    "Bhattarai 2020",
    "SWR+MUA",
    needs=(
        _lfps("the first two selected channels", minimum=2),
        "multiunit",
        Requirement("place_cells", "the block-specific place cells"),
    ),
)
def bhattarai_2020(
    rec: Recording,
    *,
    power_measure: str = "squared_signal",
    window_end_rule: WindowEndRule = "last_spike",
) -> FloatArray:
    """Post-silence population candidates coinciding with a Bhattarai SWR.

    Power is explicitly selectable; event-end policy is unresolved in the
    supplement (last-spike is the default interpretation, fixed is available).
    Supply block-specific place cells; silence is measured in that population.
    """
    swrs = _IMPLEMENTATIONS["bhattarai_2020_ripples"](rec, power_measure=power_measure)
    replays = _detected(rd.detect_silence_bounded_events)(
        rec.time, rec.multiunit, rec.fs,
        minimum_silence=0.06 + 1 / rec.fs, window=0.3, window_end_rule=window_end_rule, units=rec.place_cells,
        minimum_active_units=5,
    )  # fmt: skip
    return rd.require_overlap(replays, swrs)


@_recipe(
    20,
    "Stella 2019",
    "SWR",
    needs=(
        _lfps("each selected electrode; the maximum z-scored RMS"),
        _measured("sleep_intervals", "curated non-REM"),
        _measured("frequencies", "wavelet frequencies in Hz"),
        _measured("cycles", "wavelet cycles"),
    ),
)
def stella_2019(
    rec: Recording, *, frequencies: ArrayLike | None = None, cycles: float | None = None
) -> pd.DataFrame:
    """Per-electrode Morlet RMS, z over selected non-REM periods, maximum, 5/2 SD.

    Measured data require explicit wavelet frequencies/cycles and curated
    non-REM sleep_intervals, since these are not specified by the paper.
    Simulation alone uses six frequencies, seven cycles and a theta/delta
    proxy. Wavelets use symmetric odd-length support reaching four Gaussian
    SD (rounded up to whole samples) and unit L1 normalization.
    """
    if frequencies is None or cycles is None:
        if not rec.allows_simulation_proxies:
            msg = "Supply unreported wavelet frequencies and cycles explicitly."
            raise ValueError(msg)
        frequencies, cycles = np.linspace(150, 250, 6), 7.0
    grid = np.asarray(frequencies, float)
    if (
        grid.ndim != 1
        or not len(grid)
        or not np.isfinite(grid).all()
        or np.any(grid <= 0)
        or np.any(grid >= rec.fs / 2)
        or not np.isfinite(cycles)
        or cycles <= 0
    ):
        msg = "Wavelet frequencies must be between zero and Nyquist; cycles must be positive."
        raise ValueError(msg)
    if rec.sleep_intervals is not None or not rec.allows_simulation_proxies:
        allowed = rec.intervals_to_mask(rec.sleep(4.0, 1.0))
    else:
        allowed = rec.ratio(measure="power") <= 2.0
    raw = np.where(allowed[:, None], rec.session.lfps, np.nan)
    power = np.zeros(raw.shape)
    for frequency in grid:
        sigma = cycles / (2 * np.pi * frequency)
        half_width = int(np.ceil(4 * sigma * rec.fs))
        time = np.arange(-half_width, half_width + 1) / rec.fs
        wavelet = np.exp(2j * np.pi * frequency * time) * np.exp(-(time**2) / (2 * sigma**2))
        wavelet /= np.abs(wavelet).sum()

        def wavelet_power(x: FloatArray, kernel: Any = wavelet) -> FloatArray:
            return np.asarray(
                np.abs(fftconvolve(x, kernel[:, None], mode="same", axes=0)) ** 2, float
            )

        power += rec.transform(raw, wavelet_power)
    rms = np.sqrt(power / len(grid))
    return _ripple_trace_events(
        rec,
        rd.normalize_signal(rms).max(axis=1),
        threshold=5.0,
        bound_threshold=2.0,
        normalization_method="none",
    )


@_recipe(21, "Xu 2019", "MUA", needs=("multiunit", "pyramidal"), bin_width=0.001)
def xu_2019(rec: Recording) -> pd.DataFrame | FloatArray:
    """Pyramidal cells, 15 ms Gaussian, peak > 3 SD, bounds at the mean,
    75-750 ms, >= 4 cells, >= 5 spikes, >= 10% of cells, onset at the first
    spike. Normalization period not stated (assumed the whole session)."""
    events = _detect_population(
        rec, rec.pyramidal, 0.015,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.075,
        maximum_duration=0.75, speed_threshold=np.inf,
    )  # fmt: skip
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=4,
        minimum_active_fraction=0.1, minimum_spikes=5, units=rec.pyramidal,
    )  # fmt: skip
    return rd.trim_events_to_trace(
        events, rec.counts(rec.pyramidal), rec.time, 1.0, sides="start"
    )


def _farooq(rec: Recording, sleep: FloatArray, units: BoolArray) -> FloatArray:
    """Population frames inside supplied state periods on native 1 ms bins.

    The reported 15 ms Gaussian width is interpreted as SD; its width
    convention is not independently established by the paper.
    """
    events = _detect_population_in(
        rec,
        sleep,
        rec.pyramidal,
        0.015,
        threshold=2.0,
        bound_threshold=2.0,
        minimum_duration=0.0,
        minimum_event_duration=0.1,
        maximum_duration=0.8,
        speed_threshold=np.inf,
    )
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=units
    )


@_recipe(
    22,
    "Farooq 2019 (Neuron)",
    "MUA",
    needs=("multiunit", "pyramidal", _measured("sleep_intervals", "curated SWS")),
    bin_width=0.001,
)
def farooq_2019_neuron(rec: Recording) -> pd.DataFrame | FloatArray:
    """Population frames in caller-supplied SWS, with 15 ms interpreted as Gaussian SD.

    The width convention is unresolved. Only simulation uses the proxy:
    speed < 1 cm/s for >= 5 s (scaled from 5 min) and theta/delta
    (6-12 / 1-4 Hz Hilbert amplitude, 5 s Gaussian) < 2; >= 5 neurons (the
    Methods; the Results say place cells). This paper's frames are defined
    during slow-wave sleep."""
    return _farooq(rec, rec.sleep(1.0, 2.0, stillness=5.0, smoothing_sigma=5.0), rec.pyramidal)


@_recipe(
    23,
    "Farooq 2019 (Science)",
    "MUA",
    needs=(
        "multiunit",
        "pyramidal",
        Requirement("place_cells", "the place-responsive cells"),
        _measured("sleep_intervals", "curated SWS"),
    ),
    bin_width=0.001,
)
def farooq_2019_science(rec: Recording) -> pd.DataFrame | FloatArray:
    """The reported 15 ms Gaussian width is interpreted as SD (unresolved).

    SWS: speed < 2 cm/s and theta/delta (4-10 / 1-3 Hz, 10 s Gaussian) below
    its mean; >= 5 place-responsive cells. Measured recordings require curated
    sleep intervals. Awake frames are available in farooq_2019_science_awake."""
    if rec.sleep_intervals is not None or not rec.allows_simulation_proxies:
        return _farooq(rec, rec.sleep(2.0, 1.0), rec.place_cells)
    ratio = rec.ratio((4.0, 10.0), (1.0, 3.0), smoothing_sigma=10.0)
    still = rd.state_intervals(rec.speed, rec.time, 2.0)
    low = rd.state_intervals(ratio, rec.time, float(np.nanmean(ratio)))
    sleep = rec.mask_to_intervals(rec.intervals_to_mask(still) & rec.intervals_to_mask(low))
    return _farooq(rec, sleep, rec.place_cells)


@_recipe(
    24,
    "Chenani 2019",
    "MUA",
    needs=("multiunit", "place_cells", _measured("behavior_intervals", "reward zones")),
    bin_width=0.001,
)
def chenani_2019(rec: Recording) -> pd.DataFrame | FloatArray:
    """Place-cell rate, 30 ms Gaussian, peak >= 3 SD, bounds >= 1 SD, >= 5
    active cells. Supply reward-zone behavior_intervals; zones chosen by eye
    are not inferred automatically; measured recordings without them raise."""
    events = _detect_population(
        rec, rec.place_cells, 0.030,
        threshold=3.0, bound_threshold=1.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
    )


@_recipe(
    25,
    "Michon 2019",
    "SWR+MUA",
    needs=(
        _lfps("the first three selected channels (all, if fewer), averaged"),
        "multiunit",
        "speed",
    ),
    bin_width=0.005,
)
def michon_2019(rec: Recording, *, order: str = "text") -> FloatArray:
    """Same offline conjunction as Michon 2021; text/code preprocessing order selectable."""
    return _michon(rec, order=order)


@_recipe(
    26,
    "Liu 2019",
    "MUA",
    needs=("multiunit", "pyramidal", _measured("sleep_intervals", "curated SWS")),
)
def liu_2019(rec: Recording) -> pd.DataFrame | FloatArray:
    """Pyramidal spikes inside SWS (speed < 1 cm/s and theta/delta < 2, 5 s
    Gaussian), split at >= 100 ms of silence, >= 4 cells, 80 ms-1.2 s. The
    awake-rest frames are available separately in liu_2019_awake."""
    sleep = rec.sleep(1.0, 2.0, smoothing_sigma=5.0)
    return _detected(rd.detect_silence_bounded_events)(
        rec.time, _only_in(rec, rec.multiunit, sleep), rec.fs,
        minimum_silence=0.1, units=rec.pyramidal, minimum_active_units=4,
        minimum_duration=0.08, maximum_duration=1.2,
    )  # fmt: skip


def _karlsson_rule(rec: Recording, speed_threshold: float) -> pd.DataFrame:
    """Karlsson & Frank 2009: each tetrode's 4 ms-smoothed envelope, 3 SD for
    >= 15 ms on any tetrode, bounds at the mean, overlapping events combined.

    Speed at both ends at or below ``speed_threshold``; a paper's strict
    "less than" passes the next float below its limit."""
    return _detected(rd.Karlsson_ripple_detector)(
        rec.time,
        rec.filtered((150.0, 250.0)),
        rec.speed,
        rec.fs,
        speed_threshold=speed_threshold,
    )


@_recipe(
    27,
    "Shin 2019",
    "SWR",
    needs=(
        _lfps("each selected channel"),
        "speed",
        _when("multiunit", stage="decoding_candidates"),
        _when("place_cells", stage="decoding_candidates"),
    ),
)
def shin_2019(rec: Recording, *, stage: Stage = "detection") -> pd.DataFrame | FloatArray:
    """The Karlsson rule at <= 4 cm/s; for the analyses, whole events >= 50 ms
    with >= 5 place cells are selected by stage='decoding_candidates'.
    The default returns the initial SWR inventory."""
    _check_stage(stage)
    events = _karlsson_rule(rec, 4.0)
    if stage == "detection":
        return events
    events = within_duration(events, low=0.05)
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
    )


@_recipe(
    28,
    "Carey 2019",
    "SWR+MUA",
    needs=(
        _lfps("the first selected channel, for the spectral template and theta"),
        "multiunit",
        "speed",
        _measured("example_ripples", "manually selected ripple intervals"),
    ),
)
def carey_2019(rec: Recording) -> pd.DataFrame:
    """Published amSWR spectral score and joint MUA candidates.

    Measured recordings require manually selected example_ripples. Simulation
    alone uses the five largest Kay events. The joint score is rescaled to
    mean 0.5, thresholded at 4 inside low-speed/low-theta intervals, >=20 ms
    and >=5 units. The spectral template uses the first selected raw channel.
    """
    examples: pd.DataFrame | FloatArray
    if rec.example_ripples is None:
        if not rec.allows_simulation_proxies:
            msg = "Supply example_ripples for Carey's spectral template."
            raise ValueError(msg)
        kay = _detected(rd.Kay_ripple_detector)(
            rec.time, rec.filtered((150.0, 250.0)), rec.speed, rec.fs
        )
        examples = kay.nlargest(5, "max_zscore")
    else:
        examples = rec.example_ripples
    score = rd.carey_spectral_ripple_score(rec.time, rec.session.raw_lfp, rec.fs, examples)
    return _detected(rd.Carey_candidate_detector)(
        rec.time,
        None,
        rec.multiunit,
        rec.speed,
        rec.fs,
        ripple_score=score,
        threshold_method="mean",
        low_threshold=4.0,
        high_threshold=4.0,
        theta_lfp=rec.session.raw_lfp,
    )


@_recipe(
    29,
    "Muessig 2019",
    "SWR+MUA",
    needs=(
        "multiunit",
        "pyramidal",
        _lfps("each selected channel; the most variable one is used"),
        _measured("sleep_intervals", "the curated rest or RUN state of the trial"),
        _when("speed", sample_speed_veto=True),
    ),
    bin_width=0.001,
)
def muessig_2019(
    rec: Recording, *, trial: str = "rest", sample_speed_veto: bool = False
) -> FloatArray:
    """Native 1 ms pyramidal bursts overlapping separate RMS ripple windows.

    10 ms Gaussian, above 3 SD with 3 SD bounds, 100-750 ms. Pass one trial
    per recording and supply curated eligible intervals via sleep_intervals.
    The paper defines rest using mean speed and theta/delta power in 1.6 s
    windows, stepped by 0.8 s. Mean speed is <2.5 cm/s for rest trials and
    <1 cm/s for RUN. State estimation is external for measured data: supplied
    intervals must already implement the selected trial's criteria.

    By default, events must lie wholly in those intervals, without a second
    speed test. sample_speed_veto=True adds the previous stricter requirement
    that all native-grid speed samples be known and below the trial's limit. This
    additional veto is not specified by the paper. Simulation without supplied
    intervals retains the explicitly approximate speed/theta-delta state proxy.
    """
    if trial not in {"rest", "run"}:
        msg = "trial must be 'rest' or 'run'."
        raise ValueError(msg)
    limit = 1.0 if trial == "run" else 2.5
    bursts = _detect_population(
        rec,
        rec.pyramidal,
        0.010,
        threshold=3.0,
        bound_threshold=3.0,
        minimum_duration=0.0,
        minimum_event_duration=0.1,
        maximum_duration=0.75,
        speed_rule="all",
        speed_threshold=np.nextafter(limit, -np.inf) if sample_speed_veto else np.inf,
    )
    events = rd.require_overlap(bursts, _IMPLEMENTATIONS["muessig_2019_ripples"](rec))
    return bounds(rd.require_inside(events, rec.sleep(limit, 2.0, measure="power")))


@_recipe(
    30,
    "Drieu 2018",
    "MUA",
    needs=("multiunit", "place_cells", _measured("sleep_intervals", "curated SWS")),
    bin_width=0.001,
)
def drieu_2018(rec: Recording, *, stage: Stage = "detection") -> pd.DataFrame | FloatArray:
    """Place-cell bursts in supplied SWS: 10 ms Gaussian, 3 SD/mean, <=500 ms.

    Detection returns all bursts. stage='decoding_candidates' adds >=3 active
    place cells and elapsed duration >60 ms for trajectory analysis. Only a
    SimulatedSession may use the demonstration's shortened theta/delta proxy.
    """
    _check_stage(stage)
    events = _drieu_events(rec)
    if stage == "detection":
        return events
    candidates = bounds(events)
    candidates = candidates[
        (candidates[:, 1] - candidates[:, 0]) > 0.06 + _time_tolerance(candidates)
    ]
    return rd.require_active_units(
        candidates, rec.multiunit, rec.time, minimum_active_units=3, units=rec.place_cells
    )


def _drieu_events(rec: Recording) -> pd.DataFrame | FloatArray:
    """Place-cell rate, 10 ms Gaussian, peak > 3 SD, bounds at the mean,
    <= 500 ms, inside SWS found by k-means (two clusters, assumed) on a
    theta/delta power ratio (6-10 / 1-4 Hz; Hilbert power for the paper's
    spectrogram, clustered over the whole session rather than sleep sessions),
    epochs longer than 2 s (scaled from 120 s) with gaps < 1 s bridged."""
    if rec.sleep_intervals is not None or not rec.allows_simulation_proxies:
        sleep = rec.sleep(2.0, 1.0)
        return _detect_population_in(
            rec,
            sleep,
            rec.place_cells,
            0.010,
            threshold=3.0,
            minimum_duration=0.0,
            maximum_duration=0.5,
            speed_threshold=np.inf,
        )
    ratio = rec.ratio((6.0, 10.0), (1.0, 4.0), measure="power")
    sleep = rd.state_intervals(
        ratio, rec.time, rd.two_cluster_threshold(ratio), comparison="<=",
        merge_gap=1.0, minimum_duration=2.0,
    )  # fmt: skip
    return _detected(rd.detect_events_from_trace)(
        rec.time, _only_in(rec, rec.rate(rec.place_cells, 0.010), sleep), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, maximum_duration=0.5, speed_threshold=np.inf,
    )  # fmt: skip


@_recipe(31, "Maboudi 2018", "MUA", needs=("multiunit", "pyramidal", "speed"), bin_width=0.001)
def maboudi_2018(rec: Recording) -> FloatArray:
    """Linear-track PBEs: pooled 1 ms bins, finite 20 ms SD/60 ms half-width Gaussian.

    Session-normalized peak >=3 SD, mean bounds, mean speed <=5, >=80 ms and
    >=4 supplied pyramidal cells. The Gaussian uses zero padding at valid
    block edges. These are continuous PBE bounds, not the later 20 ms binned
    analysis support; the released archive distinguishes those inventories.
    """
    trace = population_trace(rec, bin_width=0.001)
    kernel = np.exp(-0.5 * (np.arange(-60, 61) / 20) ** 2)
    kernel /= kernel.sum()
    trace = dataclasses.replace(
        trace,
        data=_transform(
            trace.time,
            trace.data,
            lambda x: np.asarray(fftconvolve(x, kernel, mode="same"), float),
        ),
    )
    events = trace.detect(
        threshold=3.0,
        minimum_duration=0.0,
        minimum_event_duration=0.08,
        speed_rule="mean",
        speed_threshold=5.0,
    )
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=4, units=rec.pyramidal
    )


@_recipe(
    32,
    "Ólafsdóttir 2017",
    "MUA",
    needs=(
        "multiunit",
        "place_cells",
        "speed",
        _measured("behavior_intervals", "corner epochs"),
    ),
    bin_width=0.001,
)
def olafsdottir_2017(rec: Recording, *, analysis: str = "arm") -> pd.DataFrame | FloatArray:
    """Native place-cell MUA candidates, with separate arm/trajectory participation.

    5 ms Gaussian, 3 SD/mean, >=40 ms, all event speeds <=3. Supply corner
    behavior_intervals; measured recordings without them raise.
    Arm reactivation adds no cell-count criterion; analysis='trajectory'
    requires >=15% and >5 place cells.
    """
    if analysis not in {"arm", "trajectory"}:
        msg = "analysis must be 'arm' or 'trajectory'."
        raise ValueError(msg)
    events = _detect_population(
        rec,
        rec.place_cells,
        0.005,
        threshold=3.0,
        minimum_duration=0.0,
        minimum_event_duration=0.04,
        speed_rule="all",
        speed_threshold=3.0,
    )
    if analysis == "arm":
        return events
    return rd.require_active_units(
        events,
        rec.multiunit,
        rec.time,
        minimum_active_units=6,
        minimum_active_fraction=0.15,
        units=rec.place_cells,
    )


@_recipe(
    33,
    "Wu 2017",
    "MUA",
    needs=(
        "multiunit",
        _when("place_cells", "one template's cells", stage="decoding_candidates"),
    ),
    bin_width=0.01,
)
def wu_2017(rec: Recording, *, stage: Stage = "detection") -> pd.DataFrame | FloatArray:
    """Nonoverlapping 10 ms all-spike bins, no smoothing; 4 SD, mean bounds, 50-400 ms.

    stage='decoding_candidates' adds >=4 active template cells; supply one
    template via place_cells. Detection and session normalization are defaults."""
    _check_stage(stage)
    events = _detect_population(
        rec,
        None,
        0.0,
        bin_width=0.01,
        threshold=4.0,
        minimum_duration=0.0,
        minimum_event_duration=0.05,
        maximum_duration=0.4,
        speed_threshold=np.inf,
    )

    if stage == "detection":
        return events
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=4, units=rec.place_cells
    )


@_recipe(
    34,
    "Yamamoto 2017 (one reading)",
    "SWR+MUA",
    needs=(_lfps("the first selected channel"), "multiunit"),
    bin_width=0.01,
)
def yamamoto_2017(rec: Recording) -> pd.DataFrame | FloatArray:
    """One reading of an ambiguous rule: summed spikes in nonoverlapping 10 ms bins, peak > 3 SD, bounds at 1 SD, kept when
    overlapping a period of 140-200 Hz power above 3 SD on one channel. The
    paper does not say how the two combine or which trace sets the bounds."""
    ripples = _detected(rd.detect_events_from_trace)(
        rec.time, rec.envelope((140.0, 200.0))[:, 0] ** 2, _speed_or_unknown(rec), rec.fs,
        threshold=3.0, bound_threshold=3.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    bursts = _detect_population(
        rec, None, 0.0, bin_width=0.01,
        threshold=3.0, bound_threshold=1.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_overlap(bursts, ripples)


@_recipe(35, "Tang 2017", "SWR", needs=(_lfps("each selected channel"), "speed"))
def tang_2017(rec: Recording) -> pd.DataFrame | FloatArray:
    """The Karlsson rule at < 4 cm/s (smoothing and minimum inherited)."""
    return _karlsson_rule(rec, np.nextafter(4.0, -np.inf))


@_recipe(
    36,
    "Grosmark 2016",
    "SWR+MUA",
    needs=(
        "multiunit",
        "pyramidal",
        _measured("sleep_intervals", "curated NREM, the normalization epoch"),
        _measured("behavior_intervals", "quiet-waking/NREM epochs"),
        _measured("external_ripples", "ripple intervals, peaks as a third column"),
        _when("place_cells", stage="decoding_candidates"),
    ),
    bin_width=0.001,
)
def grosmark_2016(
    rec: Recording,
    *,
    stage: Stage = "detection",
    behavior_intervals: FloatArray | None = None,
) -> FloatArray:
    """Population/ripple conjunction, with a separate decoding-candidate stage.

    The reported 15 ms Gaussian width is interpreted as its SD (unresolved).
    Detection retains 50-500 ms events with >=5 pyramidal cells. Selecting
    stage='decoding_candidates' additionally requires >=100 ms and >=5 or 10%
    of supplied place cells. Measured data require NREM for normalization,
    eligible quiet-waking/NREM behavior_intervals, and external ripple peaks.
    """
    _check_stage(stage)
    events = _population_with_ripple_peak(rec, rec.sleep(4.0, 1.0), behavior_intervals)
    if stage == "detection":
        return events
    return rd.require_active_units(
        within_duration(events, low=0.1, sampling_frequency=rec.fs),
        rec.multiunit,
        rec.time,
        minimum_active_units=5,
        minimum_active_fraction=0.1,
        units=rec.place_cells,
    )


@_recipe(
    37,
    "Ambrose 2016",
    "SWR",
    needs=(_lfps("every selected channel (one per tetrode), averaged"), "speed"),
)
def ambrose_2016(rec: Recording) -> pd.DataFrame | FloatArray:
    """Pfeiffer & Foster 2015's trace, the mean envelope over every selected
    channel (the paper used one channel from each of four to seven tetrodes;
    select them before calling), > 3 SD, detected only
    while stopped (< 5 cm/s, stated in the paper). Statistics also come from
    stopping periods (the lab's convention, inferred); no duration limits are
    reported in the main Methods or supplement. The proximity to the well is
    not reproduced."""
    return _detected(rd.detect_events_from_trace)(
        rec.time, rec.mean_envelope((150.0, 250.0)), rec.speed, rec.fs,
        threshold=3.0, smoothing_sigma=0.0125, minimum_duration=0.0,
        speed_rule="restrict", speed_threshold=np.nextafter(5.0, -np.inf),
    )  # fmt: skip


@_recipe(
    38,
    "Jadhav 2016",
    "SWR",
    needs=(
        _lfps("each selected channel"),
        "speed",
        _when("multiunit", stage="decoding_candidates"),
    ),
)
def jadhav_2016(rec: Recording, *, stage: Stage = "detection") -> pd.DataFrame | FloatArray:
    """The Karlsson rule at < 4 cm/s; SWRs within 1 s after the previous
    one's start dropped; stage='decoding_candidates' adds >=4 active CA1 cells (all supplied units).
    The default returns the initial SWR inventory."""
    _check_stage(stage)
    events = rd.exclude_close_events(
        _karlsson_rule(rec, np.nextafter(4.0, -np.inf)), 1.0, measure_from="start"
    )
    if stage == "detection":
        return events
    return rd.require_active_units(events, rec.multiunit, rec.time, minimum_active_units=4)


@_recipe(
    39,
    "Ólafsdóttir 2016",
    "MUA",
    needs=("multiunit", "place_cells"),
    bin_width=0.001,
)
def olafsdottir_2016(rec: Recording) -> pd.DataFrame | FloatArray:
    """Place cells, 5 ms Gaussian, > 3 SD, bounds at the mean, >= 40 ms,
    >=15% of the place cells; no speed rule. The paper detects "from the rest
    session" (an hour and a half in a rest enclosure) and states no speed,
    immobility or sleep criterion, so pass the rest-session recording alone:
    detection and statistics span the whole recording passed."""
    events = _detect_population(
        rec, rec.place_cells, 0.005,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.04,
        speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_fraction=0.15, units=rec.place_cells
    )


@_recipe(40, "Silva 2015", "MUA", needs=("multiunit", "pyramidal", "speed"), bin_width=0.001)
def silva_2015(rec: Recording) -> pd.DataFrame | FloatArray:
    """Sorted units without interneurons (pyramidal; the Results say all
    recorded units, the Fig. 1c legend place cells), 10 ms Gaussian, > 3 SD,
    bounds at the mean, only while < 5 cm/s, 100-500 ms."""
    return _detect_population(
        rec, rec.pyramidal, 0.010,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.1,
        maximum_duration=0.5, speed_rule="restrict",
        speed_threshold=np.nextafter(5.0, -np.inf),
    )  # fmt: skip


@_recipe(
    41,
    "Ólafsdóttir 2015",
    "MUA",
    needs=(
        "multiunit",
        Requirement("templates", "one cell selection per directional template"),
        _measured("behavior_intervals", "rest epochs"),
    ),
)
def olafsdottir_2015(rec: Recording, *, minimum_active_units: int = 0) -> FloatArray:
    """Per-template silence-bounded candidates before optional decoding filters.

    Supply templates as cell masks; >=15% of a template in <=300 ms bounded
    by >=50 ms of silence. The optional minimum_active_units can impose the
    decoding-stage seven-cell criterion. Rest epochs are caller-supplied
    behavior_intervals; measured recordings without them raise.
    No ensemble size is assumed.
    """
    if not rec.templates:
        msg = "Supply templates: one cell selection per directional template."
        raise ValueError(msg)
    if any(not template.any() for template in rec.templates):
        msg = "A template selects no cells; supply each directional template's cells."
        raise ValueError(msg)
    found = []
    small = 0
    for template in rec.templates:
        if template.sum() < minimum_active_units:
            small += 1
            continue
        events = _detected(rd.detect_silence_bounded_events)(
            rec.time,
            rec.multiunit,
            rec.fs,
            minimum_silence=0.05,
            units=template,
            minimum_active_fraction=0.15,
            maximum_duration=0.3,
            minimum_active_units=minimum_active_units,
        )
        found.append(bounds(events))
    if small:
        rd.core._warn_at_caller(
            f"{small} of {len(rec.templates)} template(s) skipped: fewer than "
            f"{minimum_active_units} cells, so no event could meet minimum_active_units."
        )
    events = np.concatenate(found) if found else np.empty((0, 2))
    return np.asarray(events[np.argsort(events[:, 0], kind="stable")], float)


@_recipe(
    42,
    "Pfeiffer 2015",
    "SWR",
    needs=(_lfps("every selected channel (one per tetrode), averaged"), "speed"),
)
def pfeiffer_2015(rec: Recording) -> pd.DataFrame | FloatArray:
    """Mean 150-250 Hz Hilbert envelope over every selected channel (one per
    tetrode; select them before calling), 12.5 ms Gaussian, above 3 SD with
    statistics over speed < 5 cm/s (one reading of "excluding periods of
    movement"), speed <= 5 cm/s at both ends, bounds at the mean, 50 ms-2 s."""
    return _pfeiffer_2015_swrs(rec)


@_recipe(43, "Wu 2014", "MUA", needs=("multiunit", "place_cells", "speed"), bin_width=0.01)
def wu_2014(rec: Recording) -> pd.DataFrame | FloatArray:
    """Place-cell density in nonoverlapping 10 ms bins, 15 ms Gaussian, > 2 SD
    over the session, bounds at the mean, speed < 5 at both ends (assumed; the
    paper does not say which samples). The reward-area restriction is not
    reproduced."""
    return _detect_population(
        rec, rec.place_cells, 0.015, bin_width=0.01,
        threshold=2.0, minimum_duration=0.0, speed_threshold=np.nextafter(5.0, -np.inf),
    )  # fmt: skip


@_recipe(
    44,
    "Wikenheiser 2013",
    "SWR",
    needs=(
        _lfps("every selected channel, averaged"),
        "multiunit",
        _measured("window_anchor", "'samples', 'peaks' or 'onsets'"),
        Requirement(
            "sleep_intervals",
            "curated rest, including the stillness rule",
            measured_only=True,
            when=(("branch", "rest"),),
        ),
        _when("theta_delta", "the caller's z-scored theta/delta trace", branch="run_lia"),
        _when("speed", branch="run_lia"),
        _when("baseline_intervals", "the normalization epoch", normalization="baseline"),
    ),
)
def wikenheiser_2013(
    rec: Recording,
    *,
    branch: str = "rest",
    window_anchor: str | None = None,
    theta_delta: ArrayLike | None = None,
    normalization: str = "session",
) -> FloatArray:
    """150 ms ripple windows, >=3 cells and >=5 spikes, rest or run-LIA branch.

    Power is squared mean 140-220 Hz envelope (channel/measure interpretation).
    Anchors are where z-scored power reaches 1 SD (z >= 1; the paper says
    "exceeded"). The window anchor is unspecified: measured data must choose
    'samples', 'peaks' or 'onsets'. Each anchor gets a 150 ms window, and
    overlapping (or touching) windows within a valid LFP block are joined, as
    the paper's "Overlapping events were concatenated"; windows are clipped to
    their block, so no event spans missing data.
    The paper does not define its baseline epoch. normalization='session'
    uses all valid LFP samples; 'baseline' explicitly selects supplied
    baseline_intervals. Unrelated baseline intervals do not affect the default.
    Measured rest requires curated sleep_intervals including the surrounding
    stillness rule; run-LIA requires the caller's z-scored theta/delta trace,
    with event-mean ratio <0 and mean speed <2. Simulation rest uses its
    explicitly shortened state proxy.
    """
    if branch not in {"rest", "run_lia"}:
        msg = "branch must be 'rest' or 'run_lia'."
        raise ValueError(msg)
    if window_anchor is None:
        if not rec.allows_simulation_proxies:
            msg = "Supply the unreported window_anchor explicitly."
            raise ValueError(msg)
        window_anchor = "samples"
    if window_anchor not in {"samples", "peaks", "onsets"}:
        msg = "window_anchor must be 'samples', 'peaks' or 'onsets'."
        raise ValueError(msg)
    if normalization not in {"session", "baseline"}:
        msg = "normalization must be 'session' or 'baseline'."
        raise ValueError(msg)
    mask = _baseline(rec, required=True) if normalization == "baseline" else None
    power = _zscore(rec.mean_envelope((140.0, 220.0)) ** 2, mask)
    found = []
    _, blocks = _valid_blocks(rec.time, power)
    for start, stop in blocks:
        z = power[start:stop]
        if window_anchor == "samples":
            anchors = np.flatnonzero(z >= 1)
        elif window_anchor == "peaks":
            anchors = find_peaks(z, height=1)[0]
        else:
            anchors = np.flatnonzero(np.diff(np.r_[False, z >= 1].astype(int)) == 1)
        windows = rd.windows_around_times(rec.time[start + anchors], 0.075)
        found.append(np.clip(windows, rec.time[start], rec.time[stop - 1]))
    events = _counted(
        "joined ripple-power windows", np.concatenate(found) if found else np.empty((0, 2))
    )
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=3, minimum_spikes=5
    )
    if branch == "rest":
        if rec.sleep_intervals is not None or not rec.allows_simulation_proxies:
            return bounds(rd.require_inside(events, rec.sleep(2.0, 0.0)))
        ratio = _zscore(rec.ratio((6.0, 10.0), (2.0, 4.0), measure="power"))
        still = rd.state_intervals(rec.speed, rec.time, 2.0, minimum_duration=2.0)
        low = rd.state_intervals(ratio, rec.time, 0.0)
        rest = rec.mask_to_intervals(rec.intervals_to_mask(still) & rec.intervals_to_mask(low))
        return bounds(rd.require_inside(events, rest))
    if theta_delta is None:
        msg = "Supply the z-scored theta_delta trace for the run-LIA branch."
        raise ValueError(msg)
    ratio = np.asarray(theta_delta, float)
    if ratio.shape != rec.time.shape:
        msg = "theta_delta must have one value per input timestamp."
        raise ValueError(msg)
    events = bounds(
        rd.exclude_movement(
            events, rec.speed, rec.time, np.nextafter(2.0, -np.inf), rule="mean"
        )
    )
    keep = []
    for start, end in events:
        values = ratio[(rec.time >= start) & (rec.time <= end)]
        keep.append(bool(len(values) and np.isfinite(values).all() and np.mean(values) < 0))
    return np.asarray(events[np.asarray(keep, dtype=bool)], float)


@_recipe(
    45, "Pfeiffer 2013", "MUA", needs=("multiunit", "pyramidal", "speed"), bin_width=0.001
)
def pfeiffer_2013(rec: Recording) -> pd.DataFrame | FloatArray:
    """Clustered pyramidal units' histogram (interneurons excluded, inferred)
    only while < 5 cm/s, 10 ms Gaussian, > 3 SD, bounds at the mean; bounds
    moved inward until the first and last 20 ms windows (5 ms steps) hold 2
    spikes; then (order assumed) >= 10% of units and 50 ms-2 s."""
    events = _detect_population(
        rec, rec.pyramidal, 0.010,
        threshold=3.0, minimum_duration=0.0, speed_rule="restrict",
        speed_threshold=np.nextafter(5.0, -np.inf),
    )  # fmt: skip
    events = rd.trim_events_to_spike_windows(
        events, rec.multiunit, rec.time, units=rec.pyramidal
    )
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_fraction=0.1, units=rec.pyramidal
    )
    return within_duration(events, 0.05, 2.0, sampling_frequency=rec.fs)


@_recipe(
    46,
    "Carr 2012",
    "SWR",
    needs=(
        _lfps("each selected CA1 channel"),
        "speed",
        _when("multiunit", stage="decoding_candidates"),
        _when("place_cells", stage="decoding_candidates"),
    ),
)
def carr_2012(rec: Recording, *, stage: Stage = "detection") -> pd.DataFrame | FloatArray:
    """The Karlsson rule on CA1 at < 4 cm/s; stage='decoding_candidates'
    adds >=5 active place cells; the default returns the initial SWR inventory."""
    _check_stage(stage)
    events = _karlsson_rule(rec, np.nextafter(4.0, -np.inf))
    if stage == "detection":
        return events
    return rd.require_active_units(
        events, rec.multiunit, rec.time,
        minimum_active_units=5, units=rec.place_cells,
    )  # fmt: skip


@_recipe(47, "Bendor 2012", "MUA", needs=("multiunit",), bin_width=0.001)
def bendor_2012(rec: Recording) -> pd.DataFrame | FloatArray:
    """Davidson's multiunit signal (all spikes, 15 ms Gaussian), peak z >= 4,
    bounds z >= 2, merged < 50 ms, >= 50 ms; z over the whole session
    (assumed). The NREM, REM and awake labelling of events is not reproduced."""
    return _detect_population(
        rec, None, 0.015,
        threshold=4.0, bound_threshold=2.0, minimum_duration=0.0,
        close_event_threshold=0.05, close_event_rule="merge", minimum_event_duration=0.05,
        speed_threshold=np.inf,
    )  # fmt: skip


@_recipe(
    48,
    "Gupta 2010",
    "SWR gate only",
    role="candidate_gate",
    needs=(_lfps("every selected channel (one per tetrode), averaged"),),
)
def gupta_2010(rec: Recording, *, log_amplitude: bool = True) -> pd.DataFrame | FloatArray:
    """Events are windows grown by a spike-order score (not reproduced). This
    is the SWR gate: 180-220 Hz Hilbert amplitude averaged over tetrodes,
    log-transformed as in Jackson 2006, above 2 SD over the whole session.
    Retaining the log transform is an explicit inference; log_amplitude=False
    selects the untransformed interpretation. The >=3 active cells and reward
    pause apply to sequence windows, which this gate does not construct."""
    amplitude = rec.mean_envelope((180.0, 220.0))
    trace = np.log(np.maximum(amplitude, np.finfo(float).tiny)) if log_amplitude else amplitude
    return _detected(rd.detect_events_from_trace)(
        rec.time, trace, _speed_or_unknown(rec), rec.fs,
        threshold=2.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip


@_recipe(49, "Karlsson 2009", "SWR", needs=(_lfps("each selected channel"), "speed"))
def karlsson_2009(rec: Recording) -> pd.DataFrame | FloatArray:
    """The Karlsson rule: each selected channel's 150-250 Hz envelope, 4 ms
    Gaussian, 3 SD for >= 15 ms on any channel, bounds at the mean,
    overlapping events combined, speed < 2 cm/s at both ends. The paper used
    CA1 and CA3 tetrodes; select one channel per tetrode before calling."""
    return _karlsson_rule(rec, np.nextafter(2.0, -np.inf))


@_recipe(50, "Davidson 2009", "MUA", needs=("multiunit", "speed"), bin_width=0.001)
def davidson_2009(rec: Recording) -> pd.DataFrame | FloatArray:
    """All spikes, 15 ms Gaussian, peak >= 3 SD over stopping (< 5 cm/s),
    bounds at the mean, speed < 5 at both ends; within 30 s of running (RUN:
    speed >15 cm/s). Supply the full relevant behavioral recording."""
    events = _detect_population(
        rec, None, 0.015,
        threshold=3.0, normalization_mask=rec.speed < 5, minimum_duration=0.0,
        speed_threshold=np.nextafter(5.0, -np.inf),
    )  # fmt: skip
    running = rd.state_intervals(rec.speed, rec.time, 15.0, comparison=">")
    return rd.require_overlap(events, running + np.array([-30.0, 30.0]))


@_recipe(
    51,
    "Diba 2007",
    "MUA",
    needs=(
        "multiunit",
        Requirement("place_cells", "one directional template's cells"),
        "speed",
        _measured("behavior_intervals", "track-end reward areas"),
    ),
)
def diba_2007(rec: Recording) -> pd.DataFrame | FloatArray:
    """>= 60 ms of silence (of the template's cells, assumed), then >= 5 and
    >= 30% of the template's cells (whichever is greater) in the next 300 ms,
    speed <=10 at both ends (assumed). Supply place_cells for one directional
    template and behavior_intervals for the eligible track-end reward areas;
    measured recordings without them raise."""
    events = _detected(rd.detect_silence_bounded_events)(
        rec.time, rec.multiunit, rec.fs,
        minimum_silence=0.06, window=0.3, window_end_rule="fixed", units=rec.place_cells,
        minimum_active_units=5, minimum_active_fraction=0.3,
    )  # fmt: skip
    return rd.exclude_movement(events, rec.speed, rec.time, 10.0)


@_recipe(
    52,
    "Ji 2007",
    "MUA",
    needs=(
        "multiunit",
        _measured("sleep_intervals", "curated SWS"),
        _when("place_cells", "one template's cells", stage="decoding_candidates"),
    ),
    bin_width=0.01,
)
def ji_2007(
    rec: Recording,
    *,
    stage: Stage = "detection",
    histogram_bins: int = 100,
    histogram_smoothing: int = 3,
    merge_gap: float = 0.08,
) -> pd.DataFrame:
    """10 ms pooled counts, 30 ms Gaussian, SWS histogram-minimum threshold.

    Histogram settings are reconstruction choices; the reported per-animal
    merge gaps range from 70 to 90 ms. Supply sleep_intervals for measured data.
    stage='decoding_candidates' adds >=4 active cells from one template
    supplied via place_cells; detection returns all frames.
    """
    _check_stage(stage)
    sleep = rec.sleep(4.0, 1.0)
    trace = population_trace(rec, bin_width=0.01, smoothing_sigma=0.03)
    counts = trace.data * 0.01
    mask = trace.bins_inside(sleep)
    level = rd.histogram_minimum_threshold(
        counts[mask], bins=histogram_bins, smoothing_window=histogram_smoothing
    )
    counts[~mask] = np.nan
    events = dataclasses.replace(trace, data=counts).detect(
        threshold=level,
        bound_threshold=level,
        normalization_method="none",
        minimum_duration=0.0,
        close_event_threshold=merge_gap,
        close_event_rule="merge",
        speed_threshold=np.inf,
    )

    if stage == "detection":
        return events
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=4, units=rec.place_cells
    )


@_recipe(
    53,
    "Foster 2006",
    "MUA",
    needs=(
        "multiunit",
        Requirement("place_cells", "one probe sequence's cells"),
        "speed",
        _measured("behavior_intervals", "facing-direction epochs"),
    ),
)
def foster_2006(rec: Recording) -> pd.DataFrame | FloatArray:
    """Probe cells' spikes during stopping (< 5 cm/s, assumed) pooled and split
    at gaps of more than 50 ms, >=1/3 of the cells, <=500 ms. Supply
    place_cells for one probe sequence and behavior_intervals for the
    eligible facing-direction epochs; measured recordings without them raise."""
    stopped = rec.mask_to_intervals(rec.speed < 5)
    return _detected(rd.detect_silence_bounded_events)(
        rec.time, _only_in(rec, rec.multiunit, stopped), rec.fs,
        minimum_silence=0.05 + 1 / rec.fs, units=rec.place_cells,
        minimum_active_fraction=1 / 3, maximum_duration=0.5,
    )  # fmt: skip


@_recipe(
    54,
    "Lee 2002",
    "MUA",
    needs=(
        "multiunit",
        Requirement("place_cells", "one directional template's cells"),
        _measured("sleep_intervals", "curated SWS"),
    ),
)
def lee_2002(rec: Recording) -> pd.DataFrame | FloatArray:
    """Template cells' spikes in supplied SWS, with within-cell bursts collapsed.

    Supply place_cells for one directional template and curated sleep_intervals
    (the paper used a theta/total power ratio and video). Each cell's spikes
    with ISI <50 ms collapse to their first spike; the resulting letters split
    at gaps >100 ms. Only simulation uses speed <4 and theta/delta <1 as SWS."""
    sleep = rec.sleep(4.0, 1.0)
    return _detected(rd.detect_silence_bounded_events)(
        rec.time, _only_in(rec, rec.multiunit, sleep), rec.fs,
        minimum_silence=0.1 + 1 / rec.fs, maximum_isi=0.05, units=rec.place_cells,
    )  # fmt: skip


@_recipe(
    55,
    "Nádasdy 1999",
    "SWR",
    needs=(
        _lfps("each selected channel, RMS summed"),
        Requirement("baseline_intervals", "the normalization epoch (unreported)"),
        _measured("sleep_intervals", "curated sleep"),
        _measured("rms_window", "RMS window in seconds"),
        _measured("bound_threshold", "boundary threshold in SD"),
    ),
)
def nadasdy_1999(
    rec: Recording, *, rms_window: float | None = None, bound_threshold: float | None = None
) -> FloatArray:
    """150-250 Hz per-channel RMS sum, 7 SD, during supplied sleep.

    RMS window, baseline and bounds are not reported. Measured data require
    rms_window, bound_threshold and baseline_intervals. The demonstration
    explicitly uses 4 ms RMS and mean bounds; these are not source values.
    """
    if rms_window is None or bound_threshold is None:
        if not rec.allows_simulation_proxies:
            msg = "Supply unreported rms_window and bound_threshold explicitly."
            raise ValueError(msg)
        rms_window, bound_threshold = 0.004, 0.0
    events = _rms_ripples(
        rec,
        band=(150.0, 250.0),
        threshold=7.0,
        rms_window=rms_window,
        bound_threshold=bound_threshold,
        reference_subtract=False,
    )
    return bounds(
        rd.require_inside(events, rec.sleep(4.0, 1.0, theta=(5.0, 10.0), delta=(2.0, 4.0)))
    )


@_recipe(
    56,
    "Kudrimoti 1999",
    "SWR",
    needs=(
        _lfps("the first selected channel"),
        _measured("sleep_intervals", "curated SWS, also the normalization epoch"),
        _measured("threshold_sd", "threshold in SD"),
    ),
)
def kudrimoti_1999(rec: Recording, *, threshold_sd: float | None = None) -> pd.DataFrame:
    """100-300 Hz amplitude above a caller-selected threshold for >=25 ms in SWS.

    The threshold is unreported. Measured inputs require threshold_sd; only
    the simulation uses a 3 SD assumption. Supplied sleep defines the baseline.
    """
    if threshold_sd is None:
        if not rec.allows_simulation_proxies:
            msg = "Supply the unreported threshold_sd explicitly."
            raise ValueError(msg)
        threshold_sd = 3.0
    sleep = rec.sleep(4.0, 1.0)
    amplitude = _only_in(rec, rec.envelope((100.0, 300.0))[:, 0], sleep)
    return _ripple_trace_events(
        rec,
        amplitude,
        threshold=threshold_sd,
        bound_threshold=threshold_sd,
        minimum_duration=0.025,
        normalization_mask=rec.intervals_to_mask(sleep),
    )


NOT_REPRODUCED: dict[int, tuple[str, str, tuple[str, ...]]] = {
    1: (
        "Widloski 2025",
        (
            "Decoded replay definition is not implemented; the available methods return "
            "secondary ripple and population-burst labels."
        ),
        ("widloski_2025", "widloski_2025_bursts"),
    ),
    18: (
        "Kaefer 2020",
        (
            "Adaptive-window trajectory decoding is not implemented; the available method "
            "returns secondary SWR labels."
        ),
        ("kaefer_2020",),
    ),
    48: (
        "Gupta 2010",
        (
            "Flexible spike-sequence windows are not implemented; the available method "
            "returns only the SWR gate."
        ),
        ("gupta_2010",),
    ),
    9: (
        "Widloski 2022",
        (
            "Events are defined by decoding; ripple amplitude and spike density are "
            "reference traces only."
        ),
        (),
    ),
}
"""Papers whose own event definition is not implemented, keyed by survey row:
``(paper, reason, methods)``, where ``methods`` names the inventories that
remain available for that paper (secondary labels or a gate), if any."""


# --------------------------------------------------------------------------- run


def _tirole_bounds(time: FloatArray, z: FloatArray) -> pd.DataFrame:
    """Released Tirole bounds around 10 ms-separated threshold anchors.

    A side without a crossing at any fallback level ends at the 300 ms search
    limit or the valid block's edge, flagged in ``clipped_start`` or
    ``clipped_end``. Identical bounds from different anchors appear once.
    """
    found = []
    tolerance = _time_tolerance(time)
    _, blocks = _valid_blocks(time, z)
    for start, stop in blocks:
        high = np.flatnonzero(z[start:stop] >= 3) + start
        if not len(high):
            continue
        anchors = high[np.r_[True, np.diff(time[high]) >= 0.01 - tolerance]]
        for anchor in anchors:
            left = max(start, int(np.searchsorted(time, time[anchor] - 0.3 - tolerance)))
            right = min(
                stop - 1,
                int(np.searchsorted(time, time[anchor] + 0.3 + tolerance, side="right")) - 1,
            )
            onset, offset = left, right
            clipped_start = clipped_end = True
            for level in (0.0, 0.25, 0.5):
                eligible = (
                    z[left : anchor + 1] < level
                    if level == 0
                    else z[left : anchor + 1] <= level
                )
                crossing = np.flatnonzero(eligible)
                if len(crossing):
                    onset = left + int(crossing[-1])
                    clipped_start = False
                    break
            for level in (0.0, 0.25, 0.5):
                eligible = (
                    z[anchor : right + 1] < level
                    if level == 0
                    else z[anchor : right + 1] <= level
                )
                crossing = np.flatnonzero(eligible)
                if len(crossing):
                    offset = anchor + int(crossing[0])
                    clipped_end = False
                    break
            found.append((time[onset], time[offset], clipped_start, clipped_end))
    result = pd.DataFrame(
        found, columns=["start_time", "end_time", "clipped_start", "clipped_end"]
    ).astype(
        {"start_time": float, "end_time": float, "clipped_start": bool, "clipped_end": bool}
    )
    return _counted(
        "threshold anchors' bounds",
        result.groupby(["start_time", "end_time"], as_index=False)
        .any()
        .sort_values(["start_time", "end_time"], ignore_index=True),
    )


def _tirole_ripple_amplitude(rec: Recording) -> tuple[FloatArray, FloatArray]:
    from fractions import Fraction

    from scipy.signal import resample_poly

    if rec.session.lfps.shape[1] == 0:
        msg = "Tirole requires a selected raw LFP channel."
        raise ValueError(msg)
    ratio = Fraction(1000 / rec.fs).limit_denominator(10000)
    if not np.isclose(float(ratio), 1000 / rec.fs, rtol=1e-8):
        msg = "Resample LFP to 1000 Hz before using this method."
        raise ValueError(msg)
    raw = rec.session.lfps[:, 0]
    _, blocks = _valid_blocks(rec.time, raw)
    times, amplitudes = [], []
    kernel = firwin(35, [125, 300], pass_zero=False, fs=1000, window="hamming")
    short = []
    for start, stop in blocks:
        signal = np.asarray(
            resample_poly(raw[start:stop], ratio.numerator, ratio.denominator), float
        )
        time = rec.time[start] + np.arange(len(signal)) / 1000
        keep = time <= rec.time[stop - 1]
        signal, time = signal[keep], time[keep]
        if len(signal) <= 102:  # filtfilt needs more samples than its padding
            short.append((start, stop))
            continue
        filtered = filtfilt(kernel, [1.0], signal, padlen=102)
        amplitude = _matlab_smooth(rd.get_envelope(filtered), 15)
        times.append(time)
        amplitudes.append(amplitude)
    if not times:
        msg = (
            "No block of finite samples is as long as the 103 samples at 1000 Hz that "
            "Tirole's ripple filter needs."
        )
        raise ValueError(msg)
    if short:
        rd.core._warn_at_caller(
            f"{len(short)} block(s) of finite samples shorter than the 103 samples at "
            f"1000 Hz that Tirole's ripple filter needs are treated as missing (sample "
            f"ranges {short[:5]}{', ...' if len(short) > 5 else ''})."
        )
    return np.concatenate(times), np.concatenate(amplitudes)


def _detect_population_in(
    rec: Recording, intervals: FloatArray, units: BoolArray | None, sigma: float, **kwargs: Any
) -> pd.DataFrame:
    trace = population_trace(rec, bin_width=0.001, units=units, smoothing_sigma=sigma)
    mask = trace.bins_inside(intervals)
    return dataclasses.replace(trace, data=np.where(mask, trace.data, np.nan)).detect(**kwargs)


# Additional inventories use a separate registry from the demonstration's
# default inventories. A variant that reuses another paper's rule says so in
# its docstring and inherits that rule's unknowns (maboudi_2018_open_field is
# pfeiffer_2013; foster_2006_ripples is lee_2002_ripples).
VARIANTS: list[Recipe] = []


def _variant(
    row: int, paper: str, trigger: str, **metadata: Any
) -> Callable[[Callable[P, pd.DataFrame | FloatArray]], Callable[..., pd.DataFrame]]:
    """Register an additional inventory; ``metadata`` are ``_register``'s keywords."""
    return _register(VARIANTS, row, paper, trigger, **metadata)


@_variant(
    4,
    "Harvey 2023 (no radiatum)",
    "SWR",
    needs=(
        _lfps("the first selected channel: the highest ripple power"),
        "multiunit",
        "pyramidal",
        _when("place_cells", stage="decoding_candidates"),
    ),
)
def harvey_2023_no_radiatum(rec: Recording, *, stage: Stage = "detection") -> FloatArray:
    """Released FindRipples branch for sessions without a radiatum channel.

    One selected high-ripple-power channel, 100-250 Hz, thresholds 1/3 SD,
    20-300 ms, <50 ms merging, then the pyramidal-spiking veto. EMG curation
    remains external. stage='decoding_candidates' applies the released replay
    gates: >=80 ms before overlap merging, >=5 place cells and <50% empty
    nonoverlapping 20 ms bins. Detection is the default.
    """
    ripples = _detected(rd.Zugaro_ripple_detector)(
        rec.time,
        rec.filtered((100.0, 250.0))[:, :1],
        _speed_or_unknown(rec),
        rec.fs,
        low_threshold=1.0,
        high_threshold=3.0,
        minimum_inter_ripple_interval=0.05,
        minimum_duration=0.02,
        maximum_duration=0.3,
        speed_threshold=np.inf,
    )
    return _harvey_stage(rec, _spiking_filter(rec, ripples), stage)


@_variant(
    0,
    "Mallory 2025",
    "ripple candidates",
    role="secondary",
    needs=(_lfps("the first selected channel"), "speed"),
)
def mallory_2025_ripples(rec: Recording) -> pd.DataFrame:
    """Single-channel 150-250 Hz Hilbert amplitude, 12.5 ms smoothing.

    Stopped (<=5 cm/s) normalization and the released 70 ms retained-peak
    merging rule. Supply one recording per normalization segment;
    baseline_intervals do not override the segment's stopped samples.
    Artifact intervals and selected channels are caller inputs.
    """
    amplitude = rec.smooth(rec.envelope((150.0, 250.0))[:, 0], 0.0125)
    amplitude = np.where(rec.speed <= 5, amplitude, np.nan)
    return _mallory_candidates(rec.time, _zscore(amplitude, ddof=1))


@_variant(
    7,
    "Bush 2022",
    "ripple candidates",
    role="secondary",
    needs=(
        _lfps("the first selected channel: the highest theta SNR"),
        "multiunit",
        "pyramidal",
        "speed",
    ),
    sampling_frequency=4800.0,
)
def bush_2022_ripples(rec: Recording, *, fir_window: str = "hamming") -> FloatArray:
    """400th-order 150-250 Hz FIR, Hilbert amplitude, 5 ms Gaussian.

    Supply the highest-theta-SNR channel first. The paper's 4800 Hz input
    rate is required; its FIR window is unstated (Hamming is explicit here).
    Forward/backward application is an implementation choice: the effective
    filter has order 800 and the squared single-pass magnitude response.
    The population detector's duration, cell and median-speed rules follow.
    """
    if not np.isclose(rec.fs, 4800):
        msg = "Bush's 400th-order filter requires LFP sampled at 4800 Hz."
        raise ValueError(msg)
    kernel = firwin(401, [150, 250], fs=rec.fs, pass_zero=False, window=fir_window)

    def filtered(x: FloatArray) -> FloatArray:
        return np.asarray(filtfilt(kernel, [1.0], x, padlen=1200), float)

    band = _transform(
        rec.time,
        rec.session.raw_lfp,
        filtered,
        minimum_length=1201,  # more than filtfilt's padding
        reason="Bush's 400th-order FIR",
    )
    trace = rec.smooth(rd.get_envelope(band, time=rec.time), 0.005)
    events = _ripple_trace_events(rec, trace, threshold=3.0)
    events = rec.merge(events, 0.04, trace, inclusive=True)
    events = within_duration(events, 0.04 + 1 / rec.fs)
    events = rd.require_active_units(
        events,
        rec.multiunit,
        rec.time,
        minimum_active_units=5,
        minimum_active_fraction=0.15,
        units=rec.pyramidal,
    )
    return within_duration(
        rd.exclude_movement(events, rec.speed, rec.time, 10.0, rule="median"), high=0.5
    )


@_variant(
    16,
    "Igata 2021",
    "per-channel ripple candidates",
    role="secondary",
    needs=(_lfps("each selected channel"), "speed"),
)
def igata_2021_ripples(rec: Recording) -> pd.DataFrame:
    """150-250 Hz envelope, 4 ms Gaussian, stopped baseline, 3 SD/mean, 50-500 ms.

    Returns one inventory per selected channel, with a channel column. The
    paper does not specify a channel-combination rule. These are separate
    offline ripples, not the under-specified GMM-selected population events.
    """
    rows = []
    amplitude = rec.envelope((150.0, 250.0))
    for channel in range(amplitude.shape[1]):
        events = _ripple_trace_events(
            rec,
            amplitude[:, channel],
            threshold=3.0,
            smoothing_sigma=0.004,
            normalization_mask=rec.speed < 5,
            minimum_event_duration=0.05,
            maximum_duration=0.5,
        )
        events["channel"] = channel
        rows.append(events)
    return pd.concat(rows, ignore_index=True)


def _rms_ripples(
    rec: Recording,
    *,
    band: tuple[float, float],
    threshold: float,
    rms_window: float,
    bound_threshold: float,
    reference_subtract: bool,
    aggregation: str = "sum",
) -> pd.DataFrame:
    if not np.isfinite(rms_window) or rms_window <= 0:
        msg = "rms_window must be positive and finite."
        raise ValueError(msg)
    lfp = rec.session.lfps
    if lfp.shape[1] == 0:
        msg = "Select at least one LFP channel."
        raise ValueError(msg)
    if reference_subtract:
        if rec.reference_lfp is None:
            msg = "Supply reference_lfp for reference subtraction."
            raise ValueError(msg)
        lfp = lfp - rec.reference_lfp[:, None]
    filtered = rd.filter_ripple_band(lfp, rec.fs, band=band, time=rec.time)
    rms = np.sqrt(np.maximum(0, rec.boxcar(filtered**2, rms_window)))
    if aggregation not in {"sum", "mean", "first"}:
        msg = "aggregation must be sum, mean or first."
        raise ValueError(msg)
    trace = (
        rms.sum(axis=1)
        if aggregation == "sum"
        else rms.mean(axis=1)
        if aggregation == "mean"
        else rms[:, 0]
    )
    return _ripple_trace_events(
        rec,
        trace,
        threshold=threshold,
        bound_threshold=bound_threshold,
        normalization_mask=_baseline(rec, required=True),
    )


@_variant(
    17,
    "Gridchyn 2020",
    "ripple candidates",
    role="secondary",
    needs=(
        _lfps("each selected channel, RMS summed"),
        "reference_lfp",
        Requirement("baseline_intervals", "the pre-rest epoch"),
    ),
)
def gridchyn_2020_ripples(
    rec: Recording, *, rms_window: float, bound_threshold: float
) -> pd.DataFrame:
    """Reference-subtracted 150-250 Hz RMS sum, 6 SD over supplied pre-rest.

    RMS window and boundary threshold are not reported and must be supplied.
    The shared equiripple filter is a reconstruction choice.
    """
    return _rms_ripples(
        rec,
        band=(150.0, 250.0),
        threshold=6.0,
        rms_window=rms_window,
        bound_threshold=bound_threshold,
        reference_subtract=True,
    )


@_variant(
    21,
    "Xu 2019",
    "ripple candidates",
    role="secondary",
    needs=(
        _lfps("each selected channel, RMS summed"),
        "reference_lfp",
        Requirement("baseline_intervals", "the first-sleep epoch"),
    ),
)
def xu_2019_ripples(
    rec: Recording, *, rms_window: float, bound_threshold: float
) -> pd.DataFrame:
    """Reference-subtracted 150-250 Hz RMS sum, 7 SD over first-sleep baseline.

    Supply baseline_intervals, reference_lfp, and the unreported RMS window
    and boundary threshold. The shared equiripple filter is a choice.
    """
    return _rms_ripples(
        rec,
        band=(150.0, 250.0),
        threshold=7.0,
        rms_window=rms_window,
        bound_threshold=bound_threshold,
        reference_subtract=True,
    )


@_variant(
    22,
    "Farooq 2019 (Neuron)",
    "ripple candidates",
    role="secondary",
    needs=(
        _lfps("the first selected channel"),
        Requirement("baseline_intervals", "the normalization epoch"),
    ),
)
def farooq_2019_neuron_ripples(
    rec: Recording, *, threshold: float, bound_threshold: float, smoothing_sigma: float
) -> pd.DataFrame:
    """175-225 Hz Hilbert amplitude on the first selected channel.

    The paper does not provide the peak/bound threshold or smoothing; all
    are required inputs, along with a caller-selected normalization baseline.
    """
    return _ripple_trace_events(
        rec,
        rec.envelope((175.0, 225.0))[:, 0],
        threshold=threshold,
        bound_threshold=bound_threshold,
        smoothing_sigma=smoothing_sigma,
        normalization_mask=_baseline(rec, required=True),
    )


@_variant(
    23,
    "Farooq 2019 (Science)",
    "ripple candidates",
    role="secondary",
    needs=(
        _lfps("the first selected channel"),
        "speed",
        _measured("sleep_intervals", "curated sleep"),
    ),
)
def farooq_2019_science_ripples(
    rec: Recording, *, power_measure: str, bound_threshold: float
) -> pd.DataFrame:
    """140-250 Hz power >3 SD during supplied sleep and speed <2 cm/s.

    The power definition and event bounds are unreported; power_measure must
    be 'hilbert' (squared envelope) or 'squared_signal'. The selected sleep
    samples at speed <2 set the statistics. Supplied baseline_intervals
    further restrict this normalization subset; they do not replace sleep.
    """
    power = _power(rec, (140.0, 250.0), power_measure)
    mask = rec.intervals_to_mask(rec.sleep(2.0, 1.0)) & (rec.speed < 2)
    return _ripple_trace_events(
        rec,
        np.where(mask, power, np.nan),
        threshold=3.0,
        bound_threshold=bound_threshold,
        normalization_mask=mask & _baseline(rec),
    )


def _power(rec: Recording, band: tuple[float, float], measure: str) -> FloatArray:
    if measure == "hilbert":
        return rec.envelope(band)[:, 0] ** 2
    if measure == "squared_signal":
        return rec.filtered(band)[:, 0] ** 2
    msg = "power_measure must be 'hilbert' or 'squared_signal'."
    raise ValueError(msg)


@_variant(
    24, "Chenani 2019", "unclassified HFE candidates", needs=(_lfps("each selected channel"),)
)
def chenani_2019_hfe(rec: Recording, *, ar_coefficients: ArrayLike) -> pd.DataFrame:
    """AR(2)-whitened 100-250 Hz Hilbert amplitude, 12 ms Gaussian, 3/1 SD.

    Supply the two fitted AR coefficients (one pair per selected channel);
    whitening computes x[t]-a1*x[t-1]-a2*x[t-2]. The fit convention is not
    specified by the paper. Outputs are per-channel HFE candidates, not SWRs:
    multitaper/PCA clustering and stable-partition selection remain external.
    """
    from scipy.signal import lfilter

    coefficients = np.asarray(ar_coefficients, dtype=float)
    n_channels = rec.session.lfps.shape[1]
    if coefficients.shape != (n_channels, 2) or not np.isfinite(coefficients).all():
        msg = "ar_coefficients must have two finite values per LFP channel."
        raise ValueError(msg)
    rows = []
    for channel, pair in enumerate(coefficients):

        def whiten(x: FloatArray, coefficients: FloatArray = pair) -> FloatArray:
            return np.asarray(lfilter(np.r_[1.0, -coefficients], [1.0], x), float)

        white = rec.transform(rec.session.lfps[:, channel], whiten)

        band = rd.filter_ripple_band(white, rec.fs, band=(100.0, 250.0), time=rec.time)
        amplitude = rd.get_envelope(band, time=rec.time)
        events = _ripple_trace_events(
            rec, amplitude, threshold=3.0, bound_threshold=1.0, smoothing_sigma=0.012
        )
        events["channel"] = channel
        rows.append(events)
    if not rows:
        msg = "Select at least one LFP channel."
        raise ValueError(msg)
    return pd.concat(rows, ignore_index=True)


@_variant(
    26,
    "Liu 2019",
    "ripple peaks and centered controls",
    role="secondary",
    needs=(
        _lfps("the first selected channel"),
        Requirement("baseline_intervals", "the normalization epoch"),
    ),
)
def liu_2019_ripples(
    rec: Recording, *, smoothing_sigma: float, window: float = 0.24
) -> pd.DataFrame:
    """150-250 Hz squared Hilbert amplitude peaks >3 SD; centered 240 ms controls.

    Channel selection, smoothing and baseline are unspecified in the paper.
    Supply the first channel, smoothing_sigma and baseline_intervals. Each
    local peak remains separate, including overlapping control windows.
    """
    if not np.isfinite(window) or window < 0:
        msg = "window must be nonnegative and finite."
        raise ValueError(msg)
    power = rec.smooth(rec.envelope((150.0, 250.0))[:, 0] ** 2, smoothing_sigma)
    return _local_peaks(
        rec,
        _zscore(power, _baseline(rec, required=True)),
        3.0,
        before=window / 2,
        after=window / 2,
    )


@_variant(
    30,
    "Drieu 2018",
    "ripple candidates",
    role="secondary",
    needs=(_lfps("every selected channel, averaged per band"),),
)
def drieu_2018_ripples(rec: Recording, *, signal_measure: str) -> pd.DataFrame:
    """Detrended 100-250 Hz minus 300-500 Hz signal, 3/1 SD, >20 and <110 ms.

    signal_measure explicitly chooses 'amplitude' or 'power', an unresolved
    interpretation of the subtraction in the paper. Average selected channels
    in each band, discard negative differences, normalize positive values.
    """
    from scipy.signal import detrend

    raw = rec.transform(rec.session.lfps, lambda x: np.asarray(detrend(x, axis=0), float))
    if signal_measure not in {"amplitude", "power"}:
        msg = "signal_measure must be 'amplitude' or 'power'."
        raise ValueError(msg)
    traces = []
    for band in ((100.0, 250.0), (300.0, 500.0)):
        filtered = rd.filter_ripple_band(raw, rec.fs, band=band, time=rec.time)
        amplitude = rd.get_envelope(filtered, time=rec.time)
        traces.append(
            (amplitude if signal_measure == "amplitude" else amplitude**2).mean(axis=1)
        )
    difference = traces[0] - traces[1]
    positive = np.where(difference > 0, difference, np.nan)
    return _ripple_trace_events(
        rec,
        positive,
        threshold=3.0,
        bound_threshold=1.0,
        minimum_event_duration=0.02 + 1 / rec.fs,
        maximum_duration=0.11 - 1 / rec.fs,
    )


@_variant(
    32,
    "Ólafsdóttir 2017",
    "ripple candidates",
    role="secondary",
    needs=(_lfps("the first selected channel"),),
    sampling_frequency=1200.0,
)
def olafsdottir_2017_ripples(rec: Recording) -> FloatArray:
    """150-250 Hz squared Hilbert modulus, 2.5 SD/mean, 40-500 ms, then <40 ms merge.

    Supply LFP already sampled at the reported 1200 Hz. One selected channel
    is used; the source does not establish a channel-combination rule.
    """
    if not np.isclose(rec.fs, 1200):
        msg = "Supply the selected LFP sampled at 1200 Hz."
        raise ValueError(msg)
    power = rec.envelope((150.0, 250.0))[:, 0] ** 2
    events = _ripple_trace_events(
        rec, power, threshold=2.5, minimum_event_duration=0.04, maximum_duration=0.5
    )
    return rec.merge(events, 0.04, power)


@_variant(
    43,
    "Wu 2014",
    "ripple peaks",
    role="secondary",
    needs=(_lfps("every selected channel, averaged"), "speed"),
)
def wu_2014_ripples(rec: Recording) -> pd.DataFrame:
    """150-250 Hz mean envelope, 8 ms Gaussian, local peaks >2.5 stopped-baseline SD."""
    trace = rec.smooth(rec.mean_envelope((150.0, 250.0)), 0.008)
    return _local_peaks(rec, _zscore(trace, rec.speed < 5), 2.5)


@_variant(
    45,
    "Pfeiffer 2013",
    "ripple candidates",
    role="secondary",
    needs=(_lfps("every selected channel, averaged"), "speed"),
)
def pfeiffer_2013_ripples(rec: Recording) -> pd.DataFrame:
    """150-250 Hz mean envelope, 12.5 ms Gaussian, 3 SD/mean over stopping.

    Detection and statistics use only speed <5 cm/s. No duration limits
    are borrowed from the later 2015 paper.
    """
    return _ripple_trace_events(
        rec,
        rec.mean_envelope((150.0, 250.0)),
        threshold=3.0,
        smoothing_sigma=0.0125,
        normalization_mask=rec.speed < 5,
        speed_rule="restrict",
        speed_threshold=np.nextafter(5.0, -np.inf),
    )


@_variant(
    50,
    "Davidson 2009",
    "ripple peaks",
    role="secondary",
    needs=(_lfps("every selected channel, averaged"), "speed"),
)
def davidson_2009_ripples(rec: Recording) -> pd.DataFrame:
    """150-250 Hz mean envelope, 12.5 ms Gaussian, local peaks >2.5 stopped-baseline SD."""
    trace = rec.smooth(rec.mean_envelope((150.0, 250.0)), 0.0125)
    return _local_peaks(rec, _zscore(trace, rec.speed < 5), 2.5)


@_variant(
    51,
    "Diba 2007",
    "ripple candidates",
    role="secondary",
    needs=(
        _lfps("the first selected channel: CA1"),
        Requirement("baseline_intervals", "the normalization epoch"),
    ),
)
def diba_2007_ripples(rec: Recording, *, rms_window: float) -> pd.DataFrame:
    """Single CA1 channel, 100-300 Hz RMS, 2 SD peak and 1.5 SD bounds.

    RMS window and baseline are not reported; supply rms_window and
    baseline_intervals explicitly. The shared equiripple filter is a choice.
    """
    return _rms_ripples(
        rec,
        band=(100.0, 300.0),
        threshold=2.0,
        rms_window=rms_window,
        bound_threshold=1.5,
        reference_subtract=False,
        aggregation="first",
    )


@_variant(
    52,
    "Ji 2007",
    "ripple candidates",
    role="secondary",
    needs=(
        _lfps("the first selected channel"),
        Requirement(
            "baseline_intervals", "the epoch whose filtered-LFP SD sets the thresholds"
        ),
    ),
)
def ji_2007_ripples(rec: Recording) -> FloatArray:
    """Rectified 80-250 Hz LFP, low 3*S/high 7*S where S is filtered-LFP SD.

    Merge low-threshold intervals with gaps <50 ms before requiring a high
    peak, so weak neighbors can extend a strong event. No mean is subtracted
    from the rectified signal. Supply baseline_intervals for S.
    """
    filtered = rec.filtered((80.0, 250.0))[:, 0]
    scale = float(np.nanstd(filtered[_baseline(rec, required=True)]))
    if not np.isfinite(scale) or scale <= 0:
        msg = "Filtered-LFP baseline must have positive finite SD."
        raise ValueError(msg)
    trace = np.abs(filtered)
    low = _ripple_trace_events(
        rec, trace, threshold=3 * scale, bound_threshold=3 * scale, normalization_method="none"
    )
    merged = rec.merge(low, 0.05, trace)
    return rd.require_trace_peak(merged, trace, rec.time, 7 * scale)


@_variant(
    54,
    "Lee 2002",
    "ripple candidates",
    role="secondary",
    needs=(_lfps("the first selected channel"), _measured("sleep_intervals", "curated SWS")),
)
def lee_2002_ripples(rec: Recording) -> FloatArray:
    """Rectified 100-400 Hz, SWS mean+5 SD, crossings <=20 ms apart joined, >=20 ms.

    Channel and filter design are unspecified; use the first selected channel
    and the shared equiripple filter. Supplied sleep_intervals set statistics.
    """
    trace = np.abs(rec.filtered((100.0, 400.0))[:, 0])
    baseline = rec.intervals_to_mask(rec.sleep(4.0, 1.0))
    z = _zscore(trace, baseline)
    events = _ripple_trace_events(
        rec, z, threshold=5.0, bound_threshold=5.0, normalization_method="none"
    )
    return within_duration(rec.merge(events, 0.02, trace, inclusive=True), low=0.02)


@_variant(
    53,
    "Foster 2006",
    "ripple candidates",
    role="secondary",
    needs=(_lfps("the first selected channel"), _measured("sleep_intervals", "curated SWS")),
)
def foster_2006_ripples(rec: Recording) -> pd.DataFrame:
    """Inherited Lee ripple rule; reported event time is the interval midpoint."""
    events = _IMPLEMENTATIONS["lee_2002_ripples"](rec)
    result = pd.DataFrame(events, columns=["start_time", "end_time"])
    result["event_time"] = events.mean(axis=1)
    return result


@_variant(
    1,
    "Widloski 2025",
    "population burst labels",
    role="secondary",
    needs=("multiunit", "speed"),
    bin_width=0.001,
)
def widloski_2025_bursts(rec: Recording) -> pd.DataFrame:
    """All good clusters in 1 ms bins, 80 ms Gaussian, stopped baseline, 3 SD/mean, >=50 ms."""
    return _detect_population(
        rec,
        None,
        0.08,
        threshold=3.0,
        normalization_mask=rec.speed < 5,
        minimum_duration=0.0,
        minimum_event_duration=0.05,
        speed_threshold=np.inf,
    )


@_variant(
    10,
    "Krause 2022",
    "HSE candidates",
    role="secondary",
    needs=("multiunit", "speed"),
    bin_width=0.001,
)
def krause_2022_hse(rec: Recording, *, interpretation: str = "text") -> pd.DataFrame:
    """Pooled 1 ms spike bins, 3 SD/mean, explicitly distinct text/code branches.

    Text: 20 ms Gaussian, stopped events; baseline period unspecified (uses
    supplied baseline or whole epoch). Code: 10 ms Gaussian, whole-epoch
    statistics, events cut at moving samples, >50 ms after cutting.
    """
    if interpretation not in {"text", "code"}:
        msg = "interpretation must be 'text' or 'code'."
        raise ValueError(msg)
    trace = population_trace(
        rec, bin_width=0.001, smoothing_sigma=0.02 if interpretation == "text" else 0.01
    )
    if interpretation == "text":
        mask = np.ones(len(trace.time), dtype=bool)
        if rec.baseline_intervals is not None:
            mask = _intervals_to_mask(trace.time, rec.baseline_intervals)
        return trace.detect(
            threshold=3.0,
            minimum_duration=0.0,
            speed_rule="all",
            speed_threshold=5.0,
            normalization_mask=mask,
        )
    speed = _known_speed(trace.speed)
    data = _zscore(trace.data)
    data[~np.isfinite(speed) | (speed > 5)] = np.nan
    return dataclasses.replace(trace, data=data).detect(
        threshold=3.0,
        normalization_method="none",
        minimum_duration=0.0,
        minimum_event_duration=0.051,
        speed_threshold=np.inf,
    )


@_variant(
    13,
    "Denovellis 2021",
    "MUA candidates",
    role="secondary",
    needs=("multiunit", "speed"),
    bin_width=0.002,
)
def denovellis_2021_mua(rec: Recording) -> pd.DataFrame:
    """Historical 2 ms MUA grid, 15 ms Gaussian, 2 SD for >=15 ms, speed <=4 cm/s.

    Uses supplied multiunit columns; tetrode identities and the primary
    ripple analysis's two-tetrode participation filter remain caller inputs.
    """
    return _detect_population(
        rec,
        None,
        0.015,
        bin_width=0.002,
        threshold=2.0,
        minimum_duration=0.015,
        speed_threshold=4.0,
    )


@_variant(
    14,
    "Gillespie 2021",
    "MUA candidates",
    role="secondary",
    needs=("multiunit", "speed"),
    bin_width=0.001,
)
def gillespie_2021_mua(rec: Recording) -> pd.DataFrame:
    """Published 1 ms MUA bins, 15 ms Gaussian, stopped (<4) baseline, 3 SD/mean,
    speed <4 at both ends (which samples is not stated).

    The released helper's different 5 ms kernel is not silently substituted.
    """
    return _detect_population(
        rec,
        None,
        0.015,
        threshold=3.0,
        normalization_mask=rec.speed < 4,
        minimum_duration=0.0,
        speed_threshold=np.nextafter(4.0, -np.inf),
    )


@_variant(
    31,
    "Maboudi 2018",
    "open-field population candidates",
    needs=("multiunit", "pyramidal", "speed"),
    bin_width=0.001,
)
def maboudi_2018_open_field(rec: Recording) -> FloatArray:
    """Open-field Pfeiffer 2013 criteria; separate from linear-track PBEs."""
    return _IMPLEMENTATIONS["pfeiffer_2013"](rec)


@_variant(
    29,
    "Muessig 2019",
    "ripple windows",
    role="secondary",
    needs=(_lfps("each selected channel; the most variable one is used"),),
)
def muessig_2019_ripples(rec: Recording) -> pd.DataFrame:
    """7 ms RMS, 100-250 Hz, most-variable channel, >99th percentile, +/-50 ms.

    Pass one trial per recording, or baseline_intervals for the trial used
    for channel selection and percentile estimation. Local peaks stay separate.
    """
    rms = np.sqrt(np.maximum(0, rec.boxcar(rec.filtered((100.0, 250.0)) ** 2, 0.007)))
    baseline = _baseline_samples(rms, _baseline(rec))
    channel = int(np.argmax(np.std(baseline, axis=0)))
    rms = rms[:, channel]
    level = float(np.percentile(baseline[:, channel], 99))
    return _local_peaks(rec, rms, level, before=0.05, after=0.05)


@_variant(
    19,
    "Bhattarai 2020",
    "ripple candidates",
    role="secondary",
    needs=(_lfps("the first two selected channels", minimum=2), "multiunit", "place_cells"),
)
def bhattarai_2020_ripples(
    rec: Recording, *, power_measure: str = "squared_signal"
) -> FloatArray:
    """Two-channel 100-250 Hz power, 50 ms boxcar, 3/1 SD, >20 ms then <=100 ms merge.

    Squared signal or squared Hilbert amplitude explicitly resolves the
    unreported instantaneous-power definition. The merge is interpreted as
    end-to-start; >=5 supplied place cells follow merging.
    """
    if rec.session.lfps.shape[1] < 2:
        msg = "Select the two reported LFP channels."
        raise ValueError(msg)
    if power_measure == "squared_signal":
        power = rec.filtered((100.0, 250.0))[:, :2] ** 2
    elif power_measure == "hilbert":
        power = rec.envelope((100.0, 250.0))[:, :2] ** 2
    else:
        msg = "power_measure must be 'squared_signal' or 'hilbert'."
        raise ValueError(msg)
    trace = rec.boxcar(power, 0.05).mean(axis=1)
    events = _ripple_trace_events(rec, trace, threshold=3.0, bound_threshold=1.0)
    events = rec.merge(
        within_duration(events, low=0.02 + 1 / rec.fs), 0.1, trace, inclusive=True
    )
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
    )


@_variant(
    23,
    "Farooq 2019 (Science)",
    "awake-rest population frames",
    needs=(
        "multiunit",
        "pyramidal",
        "place_cells",
        "speed",
        Requirement("behavior_intervals", "awake rest on the track"),
    ),
    bin_width=0.001,
)
def farooq_2019_science_awake(
    rec: Recording, *, behavior_intervals: FloatArray | None = None
) -> FloatArray:
    """Same frame criteria within supplied awake-rest epochs and speed <1 cm/s.

    Supply behavior_intervals selecting awake track rest; their curation must
    settle whether to apply the ambiguous theta/delta restriction.
    """
    if behavior_intervals is None:
        msg = "Supply behavior_intervals for awake rest on the track."
        raise ValueError(msg)
    mask = rec.intervals_to_mask(behavior_intervals) & (rec.speed < 1)
    return _farooq(rec, rec.mask_to_intervals(mask), rec.place_cells)


@_variant(
    26,
    "Liu 2019",
    "awake-rest silence-bounded frames",
    needs=(
        "multiunit",
        "pyramidal",
        "speed",
        Requirement("behavior_intervals", "track-end rest epochs"),
    ),
)
def liu_2019_awake(
    rec: Recording, *, behavior_intervals: FloatArray | None = None
) -> pd.DataFrame:
    """Silence-bounded frames at supplied track-end rest epochs, speed <2 cm/s:
    pyramidal spikes split at >=100 ms of silence, >=4 cells, 80 ms-1.2 s."""
    if behavior_intervals is None:
        msg = "Supply track-end behavior_intervals for awake frames."
        raise ValueError(msg)
    mask = rec.intervals_to_mask(behavior_intervals) & (rec.speed < 2)
    return _detected(rd.detect_silence_bounded_events)(
        rec.time,
        np.where(mask[:, None], rec.multiunit, np.nan),
        rec.fs,
        minimum_silence=0.1,
        units=rec.pyramidal,
        minimum_active_units=4,
        minimum_duration=0.08,
        maximum_duration=1.2,
    )


@_variant(
    26,
    "Liu 2019",
    "ripple-associated sleep frames",
    needs=(
        _lfps("the first selected channel"),
        Requirement("baseline_intervals", "the ripple power's normalization epoch"),
        "multiunit",
        "pyramidal",
        _measured("sleep_intervals", "curated SWS"),
    ),
)
def liu_2019_ripple_frames(rec: Recording, *, smoothing_sigma: float) -> FloatArray:
    """liu_2019's silence-bounded SWS frames that contain a >3 SD local peak
    of liu_2019_ripples' power (first selected channel, caller-selected
    smoothing_sigma, baseline_intervals for its statistics)."""
    peaks: pd.DataFrame = _IMPLEMENTATIONS["liu_2019_ripples"](
        rec, smoothing_sigma=smoothing_sigma, window=0.0
    )
    return rd.require_times_inside(_IMPLEMENTATIONS["liu_2019"](rec), peaks.peak_time)


def list_methods() -> pd.DataFrame:
    """List executable inventories and the options each caller must supply.

    Returns
    -------
    methods : pandas.DataFrame
        One row per method. The columns, defined here for every place that
        reports them (``Recipe``, result ``attrs``, the demonstration CSV):

        name
            The function name, accepted by ``run_method``. Names distinguish
            a paper's protocols and inventories.
        doi
            The paper's DOI URL, as in ``load_literature_parameters()``.
        paper
            First author and year as the survey spells them (Ólafsdóttir,
            Nádasdy), naming the variant when a paper has several.
        output
            What the events are, such as "MUA", "SWR+MUA" or "ripple peaks".
        role
            The events' scientific use: "candidate_detection" for the paper's
            candidate events; "secondary" for an inventory the paper uses
            alongside them (ripple labels, controls, a separate ripple or MUA
            inventory); "candidate_gate" for a gate that is not itself an
            event definition (Gupta 2010).
        inventory
            The demonstration's grouping, independent of role: "default" for
            the inventory examples/literature_recipes.py runs per paper,
            "additional" for the others.
        required_options
            Keyword options the signature requires.
        signals, cells, intervals, external_inputs, measured_options
            The ``requirements`` of each kind (signal; cell selection; sleep,
            baseline or behavior intervals; example or external ripples;
            options without a published value that measured data must set),
            each described as ``Requirement.describe`` does: what it must
            hold and when it is needed. ``behavior_intervals`` are passed per
            call; everything else except options goes to
            ``Recording.from_arrays``.
        requirements
            Every ``Requirement`` as a dict of its fields (``input``, ``kind``,
            ``meaning``, ``minimum``, ``measured_only``, ``when``, ``unless``),
            the declaration ``check_method`` and ``run_method`` test.
        sampling_frequency
            The input rate in Hz the method's filter requires, or NaN for any.
        stages
            The ``stage`` values accepted: ``("detection",)``, or also
            ``"decoding_candidates"`` for methods with a decoding-candidate stage.
        bin_width
            The bin width in seconds of the population grid the events are
            found on (``population_trace``), or NaN when they come from the
            input samples or a grid the docstring describes (Kaefer's 20 ms
            FFT stride, Krause's per-SWR 3 ms bins). On a population grid the
            detection's duration limits count bins while ``duration`` is the
            elapsed time between the closed bounds (see ``run_method``).
        interpretation
            The method's docstring: its rule, interpretation and assumptions.
    """
    survey = rd.load_literature_parameters()
    rows = []
    for entry in (*RECIPES, *VARIANTS):
        parameters = inspect.signature(_IMPLEMENTATIONS[entry.run.__name__]).parameters

        def described(*kinds: RequirementKind, entry: Recipe = entry) -> tuple[str, ...]:
            return tuple(need.describe() for need in entry.requirements if need.kind in kinds)

        rows.append(
            {
                "name": entry.run.__name__,
                "doi": survey.loc[entry.row, "DOI"],
                "paper": entry.paper,
                "output": entry.trigger,
                "role": entry.role,
                "inventory": entry.inventory,
                "required_options": tuple(
                    name
                    for name, parameter in parameters.items()
                    if name != "rec" and parameter.default is inspect.Parameter.empty
                ),
                "signals": described("signal"),
                "cells": described("cells"),
                "intervals": described("intervals"),
                "external_inputs": described("external"),
                "measured_options": described("option"),
                "requirements": tuple(
                    {**dataclasses.asdict(need), "kind": need.kind}
                    for need in entry.requirements
                ),
                "sampling_frequency": entry.sampling_frequency,
                "stages": ("detection", "decoding_candidates")
                if "stage" in parameters
                else ("detection",),
                "bin_width": entry.bin_width,
                "interpretation": entry.note,
            }
        )
    return pd.DataFrame(rows)


def _folded(text: str) -> str:
    """Case- and accent-insensitive form, so "Olafsdottir" finds "Ólafsdóttir"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def _unknown_method(name: str) -> str:
    """Why ``name`` names no method: the methods of a paper or DOI it names,
    or close method names. One paper can have several methods, so none is
    chosen for the caller."""
    query = _folded(name)
    doi = query.removeprefix("https://doi.org/").removeprefix("doi:").strip()
    survey = rd.load_literature_parameters()
    matches = [
        method
        for method, entry in _ENTRIES.items()
        if query
        and (
            _folded(entry.paper).startswith(query)
            or _folded(str(survey.loc[entry.row, "DOI"])).removeprefix("https://doi.org/")
            == doi
        )
    ]
    if matches:
        return (
            f"{name!r} is not a method name; methods for that paper: {', '.join(matches)}. "
            "Choose one: they are different inventories (see list_methods())."
        )
    close = difflib.get_close_matches(name, list(_ENTRIES), n=3, cutoff=0.6)
    hint = f" Did you mean {' or '.join(close)}?" if close else ""
    return f"Unknown literature method {name!r}.{hint} See list_methods() for every name."


def _option_problems(name: str, options: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Options the method does not take or requires, and every option resolved
    to its given or default value."""
    parameters = {
        key: parameter
        for key, parameter in inspect.signature(_IMPLEMENTATIONS[name]).parameters.items()
        if key not in {"rec", "behavior_intervals"}
    }
    accepted = ", ".join(parameters) or "none"
    problems = [
        f"{key} - {name} takes no option {key!r}: literature methods run fixed "
        f"published rules, not tunable detectors; its options are: {accepted}. "
        "To tune a threshold, use the package's detectors or detect_events_from_trace."
        for key in options
        if key not in parameters
    ]
    resolved = {}
    for key, parameter in parameters.items():
        if key in options:
            resolved[key] = options[key]
        elif parameter.default is inspect.Parameter.empty:
            problems.append(f"{key} - a required option of {name}; pass {key}=")
        else:
            resolved[key] = parameter.default
    return problems, resolved


def _check(
    name: str,
    recording: Recording,
    behavior_intervals: ArrayLike | None,
    options: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, Any], FloatArray | None]:
    """The call's option problems and input problems, the resolved options and
    the validated behavior intervals."""
    if name not in _ENTRIES:
        raise KeyError(_unknown_method(name))
    entry = _ENTRIES[name]
    eligible = _interval_array(behavior_intervals)
    option_problems, resolved = _option_problems(name, options)
    input_problems = _requirement_problems(entry, recording, eligible, resolved)
    stage = resolved.get("stage", "detection")
    if stage not in {"detection", "decoding_candidates"}:
        input_problems.append(
            f"stage={stage!r} - expected 'detection' or 'decoding_candidates'"
        )
    return option_problems, input_problems, resolved, eligible


def check_method(
    name: str,
    recording: Recording,
    *,
    behavior_intervals: ArrayLike | None = None,
    **options: Any,
) -> list[str]:
    """List everything a call of a literature method would lack, without running it.

    Parameters
    ----------
    name : str
        Exact function name from ``list_methods()``.
    recording : Recording
        The inputs the call would use.
    behavior_intervals : array_like, shape (n_intervals, 2), optional
        The call's eligible epochs, as ``run_method`` takes them.
    **options
        The call's method options.

    Returns
    -------
    problems : list of str
        One line per problem, each starting with the input or option it
        names: a declared requirement the call does not meet (see
        ``list_methods``: signals, cell selections, curated intervals,
        external inventories, options measured data must set), a required
        rate the recording does not have, a keyword the method does not take,
        a required option not given, or an unknown ``stage``. Empty when the
        call can run; it can still fail on the data themselves (a constant
        baseline, blocks too short for a filter).

    Raises
    ------
    KeyError
        Unknown method name.
    ValueError
        ``behavior_intervals`` are not sorted, disjoint start/end pairs.

    Examples
    --------
    >>> import numpy as np
    >>> from ripple_detection.literature_methods import Recording, check_method
    >>> time = np.arange(3000) / 1500
    >>> recording = Recording.from_arrays(time, 1500, multiunit=np.zeros((3000, 4)))
    >>> for problem in check_method("yang_2024", recording):
    ...     print(problem.split(" - ")[0])
    pyramidal
    sleep_intervals: curated NREM, the normalization epoch (measured data)
    behavior_intervals: quiet-waking/NREM epochs (measured data)
    external_ripples: ripple intervals, peaks as a third column (measured data)
    """
    option_problems, input_problems, _, _ = _check(
        name, recording, behavior_intervals, options
    )
    return option_problems + input_problems


def run_method(
    name: str,
    recording: Recording,
    *,
    behavior_intervals: ArrayLike | None = None,
    **options: Any,
) -> pd.DataFrame:
    """Run a named paper/protocol inventory and attach its scientific context.

    Parameters
    ----------
    name : str
        Exact function name from list_methods(). DOI-only dispatch is avoided
        because one paper can describe several distinct inventories.
    recording : Recording
        Measured or simulated inputs. Channel/cell selection belongs to callers.
    behavior_intervals : array_like, shape (n_intervals, 2), optional
        Sorted, disjoint, inclusive [start, end] eligible epochs in seconds,
        for this call only: their meaning differs between methods (reward
        zones, rest, corners, track ends, facing direction). Events not
        wholly inside one interval are dropped; normalization is unchanged.
        Methods whose ``requirements`` name the epochs they need raise on
        measured data without them; awake-frame methods also restrict their
        detection trace to them, as their docstrings say.
    **options
        Named method options, including required settings absent from sources.

    Returns
    -------
    events : pandas.DataFrame
        Indexed by ``event_number`` from 1. Every method's result starts with
        ``start_time`` and ``end_time`` (seconds; closed bounds),
        ``duration`` (``end_time - start_time``), ``peak_time`` (NaN where
        the method defines no peak) and ``clipped_start``/``clipped_end``
        (an edge cut by missing data or the recording's edge; False where the
        method does not track clipping, see ``clipping_tracked``), then the
        method's own columns (channel, trigger, statistics).

        ``attrs`` holds, in JSON types (lists for arrays, so results
        concatenate and ``save_events`` can write them), ``method``,
        ``ripple_detection_version`` (the version that detected the events,
        which ``save_events`` and ``load_events`` keep), ``doi``,
        ``output``, ``role``, ``inventory`` and ``interpretation`` (defined
        in ``list_methods``); ``options`` (every keyword option, defaults
        resolved; a per-sample trace such as Wikenheiser's ``theta_delta`` by
        its ``shape``, ``dtype`` and ``sha256``); ``behavior_intervals`` (the
        intervals supplied, or None); ``inputs`` (the sample count and time
        range, LFP channel and unit counts, whether the radiatum, reference
        and speed were supplied, the place-cell, pyramidal and template unit
        indices, and the sleep, baseline, example and external intervals);
        ``grid``;
        ``clipping_tracked`` (whether the method reports clipping);
        ``diagnostics``; plus any the method adds (Gridchyn:
        ``threshold_updates`` and ``expected_count``).

        ``diagnostics`` says what the recording held and where events were
        lost, to read before changing anything when a result is empty or
        surprising: ``recording_seconds`` (samples / rate); ``signals``, the
        ``valid_fraction`` and ``valid_seconds`` of each supplied signal
        (finite in every channel or unit; known speed); ``interval_seconds``,
        the time the sleep, baseline and behavior intervals cover (samples
        inside / rate; None when not supplied); ``detections``, each
        detection step the method ran, in call order, with the ``events`` it
        found before the method's later filters (speed, duration and
        participation rules inside a shared detector are applied before its
        count; cell, overlap, state and stage filters after); and the counts
        ``events_before_behavior_intervals`` and ``events``.

        ``grid`` holds ``input_sampling_frequency`` (Hz), ``bin_width``
        (seconds of the population grid, or None for the input samples),
        ``native_sampling_frequency`` (its reciprocal, or the input rate),
        ``duration`` (the convention above) and, on a population grid,
        ``bin_limits``. On a grid, duration limits count bins while
        ``duration`` is elapsed time between the first and last samples the
        bins counted: spikes filling 10.000-10.049 s at 1000 Hz make five
        10 ms bins, reported as 10.000-10.049 s with ``duration`` 0.049 s,
        and a 50 ms minimum (five bins) keeps the event.

    Raises
    ------
    KeyError
        Unknown method name.
    TypeError
        A keyword the method does not take, or a required option missing.
        The message lists every other problem ``check_method`` finds too.
    ValueError
        Missing or invalid inputs for the selected method: the message lists
        every problem ``check_method`` finds, at once.
    """
    option_problems, input_problems, resolved, eligible = _check(
        name, recording, behavior_intervals, options
    )
    if option_problems or input_problems:
        msg = f"{name} cannot run on this call:\n- " + "\n- ".join(
            option_problems + input_problems
        )
        raise TypeError(msg) if option_problems else ValueError(msg)
    entry = _ENTRIES[name]
    implementation = _IMPLEMENTATIONS[name]
    call = inspect.signature(implementation).bind(recording, **resolved)
    if "behavior_intervals" in inspect.signature(implementation).parameters:
        call.arguments["behavior_intervals"] = eligible
    detections: list[dict[str, Any]] = []
    token = _DETECTIONS.set(detections)
    try:
        raw = implementation(*call.args, **call.kwargs)
    finally:
        _DETECTIONS.reset(token)
    result = (
        raw.copy()
        if isinstance(raw, pd.DataFrame)
        else pd.DataFrame(bounds(raw), columns=["start_time", "end_time"])
    )
    n_found = len(result)
    if eligible is not None:
        result = result.loc[_within_intervals_mask(bounds(result), eligible)].copy()
    clipping_tracked = {"clipped_start", "clipped_end"} <= set(result.columns)
    result = _output_core(result)
    result.attrs.update(
        {
            "method": name,
            "ripple_detection_version": rd.__version__,
            "doi": rd.load_literature_parameters().loc[entry.row, "DOI"],
            "output": entry.trigger,
            "role": entry.role,
            "inventory": entry.inventory,
            "interpretation": entry.note,
            "options": {
                key: value
                for key, value in call.arguments.items()
                if key not in {"rec", "behavior_intervals"}
            },
            "behavior_intervals": eligible,
            "inputs": _input_summary(recording),
            "grid": _grid(entry, recording.fs),
            "clipping_tracked": clipping_tracked,
            "diagnostics": _diagnostics(recording, eligible, detections, n_found, len(result)),
        }
    )
    # JSON-ready, so results concatenate (pandas compares attrs) and export.
    result.attrs = _jsonable(result.attrs, len(recording.time))
    return result


def _input_summary(rec: Recording) -> dict[str, Any]:
    """Which inputs and selections a call ran on, for provenance."""
    session = rec.session
    lfps = getattr(session, "lfps", None)
    sharp = getattr(session, "sharp_wave_lfp", None)
    return {
        "n_samples": len(rec.time),
        "time_range": [float(rec.time[0]), float(rec.time[-1])],
        "n_lfp_channels": 0 if lfps is None else int(np.shape(lfps)[1]),
        "sharp_wave_lfp": sharp is not None and bool(np.isfinite(sharp).any()),
        "reference_lfp": rec.reference_lfp is not None,
        "speed": getattr(session, "speed", None) is not None,
        "n_units": int(rec.multiunit.shape[1]),
        "place_cells": np.flatnonzero(rec.place_cells),
        "pyramidal": np.flatnonzero(rec.pyramidal),
        "templates": [np.flatnonzero(template) for template in rec.templates],
        "sleep_intervals": rec.sleep_intervals,
        "baseline_intervals": rec.baseline_intervals,
        "example_ripples": rec.example_ripples,
        "external_ripples": rec.external_ripples,
    }


def _jsonable(value: Any, n_time: int) -> Any:
    """``value`` in JSON's types: arrays become lists, except per-sample
    traces (one row per timestamp), recorded by shape, dtype and SHA-256;
    non-finite numbers become None."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item, n_time) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.ndim and len(value) == n_time and n_time > 1:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            }
        return _jsonable(value.tolist(), n_time)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, n_time) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item(), n_time)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_events(events: pd.DataFrame, path: str | os.PathLike[str]) -> Path:
    """Write a literature method's events as CSV with a JSON provenance sidecar.

    Parameters
    ----------
    events : pandas.DataFrame
        A ``run_method`` (or named method) result, whose ``attrs`` hold its
        provenance.
    path : str or path-like
        The CSV file to write. The sidecar goes beside it with the suffix
        ``.json`` (``events.csv`` and ``events.json``).

    Returns
    -------
    sidecar : pathlib.Path
        The JSON file written: ``saved_with_ripple_detection_version`` (the
        version saving the file), the table's ``columns`` and their dtypes, and
        ``attrs`` (method, the ``ripple_detection_version`` that detected the
        events, DOI, output, role, inventory, interpretation, resolved options,
        behavior intervals, the input selections, grid, clipping and
        diagnostics; see ``run_method``). Loading and saving again keeps the
        detecting version.
        Strict JSON: no NaN or Infinity.

    Raises
    ------
    ValueError
        ``events`` carries no method provenance, or ``path`` ends in
        ``.json`` (the sidecar would overwrite it).
    """
    table = Path(path)
    if table.suffix == ".json":
        msg = "Name the CSV file, not its .json sidecar."
        raise ValueError(msg)
    if "method" not in events.attrs:
        msg = "These events carry no method provenance; save a run_method result."
        raise ValueError(msg)
    sidecar = table.with_suffix(".json")
    provenance = {
        # The version that detected the events is in attrs, set by run_method
        # and kept through load_events; this is only the version that saved them.
        "saved_with_ripple_detection_version": rd.__version__,
        "columns": {column: str(dtype) for column, dtype in events.dtypes.items()},
        "attrs": _jsonable(dict(events.attrs), -1),
    }
    events.to_csv(table, index_label="event_number")
    sidecar.write_text(json.dumps(provenance, indent=2, allow_nan=False, ensure_ascii=False))
    return sidecar


def load_events(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read events written by ``save_events``, restoring dtypes and ``attrs``.

    Parameters
    ----------
    path : str or path-like
        The CSV file; its ``.json`` sidecar must sit beside it.

    Returns
    -------
    events : pandas.DataFrame
        Indexed by ``event_number``, with each column's saved dtype and the
        saved provenance in ``attrs`` (JSON types: lists for arrays).
    """
    table = Path(path)
    provenance = json.loads(table.with_suffix(".json").read_text())
    # The C parser's default float conversion can be off by an ulp; a bound off
    # by one can drop a spike on an event's last sample.
    events = pd.read_csv(table, index_col="event_number", float_precision="round_trip").astype(
        provenance["columns"]
    )
    events.index = events.index.astype("int64")
    events.attrs = provenance["attrs"]
    return events


def _finite_rows(values: FloatArray) -> BoolArray:
    """Rows finite in every column. A row sum is NaN or infinite exactly when a
    value is (signals and counts are far from overflowing), so this needs one
    value per row rather than a boolean copy of the whole array."""
    with np.errstate(invalid="ignore", over="ignore"):  # inf - inf is NaN: not finite
        return np.asarray(np.isfinite(values.sum(axis=1)), dtype=bool)


def _diagnostics(
    rec: Recording,
    behavior_intervals: FloatArray | None,
    detections: list[dict[str, Any]],
    n_found: int,
    n_events: int,
) -> dict[str, Any]:
    """What the recording held and where events were lost, for attrs."""
    session = rec.session
    n_time = len(rec.time)
    signals: dict[str, Any] = {}
    lfps = getattr(session, "lfps", None)
    if lfps is not None and np.shape(lfps)[1]:
        signals["lfps"] = _finite_rows(lfps)
    sharp = getattr(session, "sharp_wave_lfp", None)
    if sharp is not None and np.isfinite(sharp).any():
        signals["sharp_wave_lfp"] = np.isfinite(sharp)
    if rec.reference_lfp is not None:
        signals["reference_lfp"] = np.isfinite(rec.reference_lfp)
    if rec.multiunit.shape[1]:
        signals["multiunit"] = _finite_rows(rec.multiunit)
    speed = getattr(session, "speed", None)
    if speed is not None:
        signals["speed"] = np.isfinite(speed)

    def seconds(samples: int) -> float:
        return float(samples / rec.fs)

    intervals = {
        "sleep_intervals": rec.sleep_intervals,
        "baseline_intervals": rec.baseline_intervals,
        "behavior_intervals": behavior_intervals,
    }
    return {
        "recording_seconds": seconds(n_time),
        "signals": {
            name: {
                "valid_fraction": float(valid.mean()),
                "valid_seconds": seconds(int(valid.sum())),
            }
            for name, valid in signals.items()
        },
        "interval_seconds": {
            name: None
            if values is None
            else seconds(int(_intervals_to_mask(rec.time, values).sum()))
            for name, values in intervals.items()
        },
        "detections": detections,
        "events_before_behavior_intervals": n_found,
        "events": n_events,
    }


_CORE_COLUMNS = [
    "start_time",
    "end_time",
    "duration",
    "peak_time",
    "clipped_start",
    "clipped_end",
]


def _output_core(events: pd.DataFrame) -> pd.DataFrame:
    """The shared columns first, then the method's own, numbered from 1."""
    core = pd.DataFrame(
        {
            "start_time": events.start_time.to_numpy(dtype=float),
            "end_time": events.end_time.to_numpy(dtype=float),
            "duration": (events.end_time - events.start_time).to_numpy(dtype=float),
            "peak_time": (
                events.peak_time.to_numpy(dtype=float)
                if "peak_time" in events
                else np.full(len(events), np.nan)
            ),
            **{
                flag: (
                    events[flag].to_numpy(dtype=bool)
                    if flag in events
                    else np.zeros(len(events), dtype=bool)
                )
                for flag in ("clipped_start", "clipped_end")
            },
        }
    )
    extra = events.drop(columns=[c for c in _CORE_COLUMNS if c in events]).reset_index(
        drop=True
    )
    result = pd.concat([core, extra], axis=1)
    result.index = pd.RangeIndex(1, len(result) + 1, name="event_number")
    result.attrs = dict(events.attrs)  # what the method adds (Gridchyn's updates)
    return result


def _grid(entry: Recipe, sampling_frequency: float) -> dict[str, Any]:
    """The sample or bin grid of a method's events and its duration convention."""
    grid: dict[str, Any] = {
        "input_sampling_frequency": sampling_frequency,
        "bin_width": entry.bin_width,
        "native_sampling_frequency": (
            sampling_frequency if entry.bin_width is None else 1 / entry.bin_width
        ),
        "duration": (
            "end_time - start_time: elapsed seconds between the closed bounds, one "
            "sample period less than the samples they hold span"
        ),
    }
    if entry.bin_width is not None:
        grid["bin_limits"] = (
            "the population detection's duration limits count bins: n bins last n * "
            "bin_width, so an event of n whole bins has a duration near n * bin_width "
            "- 1 / input_sampling_frequency"
        )
    return grid


__all__ = [
    "NOT_REPRODUCED",
    "RECIPES",
    "VARIANTS",
    "Inventory",
    "PopulationTrace",
    "Recipe",
    "RecordedSignals",
    "Recording",
    "Requirement",
    "RequirementKind",
    "Role",
    "Stage",
    "bounds",
    "check_method",
    "list_methods",
    "load_events",
    "population_trace",
    "run_method",
    "save_events",
    "within_duration",
] + [entry.run.__name__ for entry in (*RECIPES, *VARIANTS)]
