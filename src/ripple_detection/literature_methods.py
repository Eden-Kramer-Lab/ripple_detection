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

import functools
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ParamSpec

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import fftconvolve, filtfilt, find_peaks, firwin

import ripple_detection as rd
from ripple_detection.core import BoolArray, FloatArray, IntArray, _matlab_smooth
from ripple_detection.detectors._blocks import _valid_blocks
from ripple_detection.detectors._long import (
    _difference_of_gaussians_band,
    _firfilt,
    _gaussian_lowpass_fir,
)


@dataclass
class RecordedSignals:
    """Real recording arrays; no simulated ground truth is required."""

    time: FloatArray
    sampling_frequency: float
    lfps: FloatArray
    raw_lfp: FloatArray
    sharp_wave_lfp: FloatArray
    multiunit: FloatArray
    speed: FloatArray | None


_NO_SPEED = (
    "This method uses the animal's speed, which was not supplied; pass speed "
    "(cm/s) to Recording.from_arrays, with NaN where it is unknown."
)


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
    only explicit SimulatedSession inputs permit state proxies.
    """

    session: rd.SimulatedSession | RecordedSignals
    place_cells: BoolArray
    pyramidal: BoolArray
    sleep_intervals: FloatArray | None = None
    baseline_intervals: FloatArray | None = None
    reference_lfp: FloatArray | None = None
    templates: tuple[BoolArray, ...] = ()
    behavior_intervals: FloatArray | None = None
    example_ripples: FloatArray | None = None
    external_ripples: FloatArray | None = None

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
        behavior_intervals: ArrayLike | None = None,
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
            Boolean masks or integer column indices into multiunit.
        sleep_intervals, baseline_intervals, artifact_intervals : array_like, optional
            Sorted, disjoint inclusive [start, end] intervals in seconds.
            Artifacts mark all signal arrays missing. Baselines are consumed
            only by methods that explicitly request a caller-selected baseline.
        templates : sequence of array_like, optional
            Per-template cell masks or indices, with no fixed ensemble size.
        behavior_intervals : array_like, optional
            Allowed intervals applied by run_method as whole-event containment.
            Explicit awake-frame methods also select their detection trace
            using these intervals, as described in their docstrings.
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
        """
        timestamps = np.asarray(time, dtype=float).copy()
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
            data = np.asarray(value, dtype=float).copy()
            if channels and data.ndim == 1:
                data = data[:, None]
            if data.ndim != (2 if channels else 1) or data.shape[0] != n:
                msg = "Signals must have one row per timestamp."
                raise ValueError(msg)
            return data

        lfp_array, spikes = signal(lfps, True), signal(multiunit, True)
        finite_spikes = spikes[np.isfinite(spikes)]
        if np.any(finite_spikes < 0) or np.any(finite_spikes != np.floor(finite_spikes)):
            msg = "multiunit must contain nonnegative integer spike counts."
            raise ValueError(msg)
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
            mask[indices.astype(int)] = True
            return mask

        external = (
            None if external_ripples is None else np.asarray(external_ripples, float).copy()
        )
        if external is not None:
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
        session = RecordedSignals(
            timestamps,
            sampling_frequency,
            lfp_array,
            lfp_array[:, 0] if lfp_array.shape[1] else np.full(n, np.nan),
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
            _interval_array(behavior_intervals),
            _interval_array(example_ripples),
            external,
        )

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
            Window width in seconds, rounded to at least one sample.

        Returns
        -------
        smoothed : ndarray
            Blockwise uniform-filter output; reflected block boundaries.
        """
        return self.transform(
            values,
            lambda block: uniform_filter1d(block, max(1, round(width * self.fs)), axis=0),
        )

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
        measure: str = "gap",
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
                rd.merge_close_events(selected, gap, inclusive=inclusive, measure=measure)
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
        """
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
        measure: str = "amplitude",
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
        measure: str = "amplitude",
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
        if not isinstance(self.session, rd.SimulatedSession):
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
    events: pd.DataFrame | FloatArray, low: float = 0.0, high: float = np.inf
) -> FloatArray:
    """Events whose elapsed duration is from ``low`` to ``high`` seconds."""
    events = bounds(events)
    duration = events[:, 1] - events[:, 0]
    tolerance = _time_tolerance(events)
    return events[(duration >= low - tolerance) & (duration <= high + tolerance)]


def _time_tolerance(time: FloatArray) -> float:
    """Allow timestamp subtraction error at the recording's clock magnitude."""
    return max(1e-9, 4 * float(np.spacing(np.max(np.abs(time), initial=0.0))))


def _event_block_groups(
    time: FloatArray, trace: FloatArray, events: FloatArray, margin: float = 0.0
) -> IntArray:
    """Validate complete block containment and return each event's block index.

    ``margin`` widens each block on both sides, half a bin for bin-edge bounds.
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


def within_intervals(events: pd.DataFrame | FloatArray, intervals: FloatArray) -> FloatArray:
    """Events lying entirely inside one of the intervals."""
    events, intervals = bounds(events), np.asarray(intervals, dtype=float).reshape(-1, 2)
    if len(intervals) == 0:
        return events[:0]
    which = np.searchsorted(intervals[:, 0], events[:, 0], side="right") - 1
    inside = (which >= 0) & (events[:, 1] <= intervals[np.clip(which, 0, None), 1])
    return np.asarray(events[inside], dtype=float)


def only_in(rec: Recording, values: FloatArray, intervals: FloatArray) -> FloatArray:
    """``values`` (a trace or the spikes) missing outside the intervals, so
    detection runs inside them only."""
    mask = rec.intervals_to_mask(intervals)
    return np.where(mask if values.ndim == 1 else mask[:, None], values, np.nan)


def zugaro_ripple_peaks(rec: Recording, band: tuple[float, float]) -> pd.DataFrame:
    """bz_FindRipples-like ripples on one pyramidal channel, for the
    Buzsaki-lineage papers that require a ripple peak but do not describe
    their ripple detector (assumed: Huszar et al. 2022's 5 SD peak and 2 SD
    bounds, 20-200 ms; its noise-channel veto is not reproduced)."""
    return rd.Zugaro_ripple_detector(
        rec.time, rec.filtered(band)[:, :1], _speed_or_unknown(rec), rec.fs,
        low_threshold=2.0, high_threshold=5.0, maximum_duration=0.2,
        speed_threshold=np.inf,
    )  # fmt: skip


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


def _zscore(values: FloatArray, mask: BoolArray | None = None, ddof: int = 0) -> FloatArray:
    baseline = values if mask is None else values[mask]
    finite = baseline[np.isfinite(baseline)]
    if finite.size <= ddof or float(np.std(finite, ddof=ddof)) == 0:
        msg = "Normalization needs a finite, nonconstant baseline."
        raise ValueError(msg)
    return (values - np.mean(finite)) / np.std(finite, ddof=ddof)


def _transform(
    time: FloatArray, values: FloatArray, operation: Callable[[FloatArray], FloatArray]
) -> FloatArray:
    """Apply an operation independently within each valid block on the given grid."""
    _, blocks = _valid_blocks(time, values)
    result = np.full(values.shape, np.nan)
    for start, stop in blocks:
        result[start:stop] = operation(values[start:stop])
    return result


def _intervals_to_mask(time: FloatArray, intervals: FloatArray) -> BoolArray:
    """Select the union of inclusive intervals on the supplied time grid."""
    mask = np.zeros(time.size, dtype=bool)
    for start, end in np.asarray(intervals).reshape(-1, 2):
        mask |= (time >= start) & (time <= end)
    return mask


@dataclass
class PopulationTrace:
    """Population counts or rate on a native nonoverlapping bin grid.

    Attributes
    ----------
    time, data : ndarray
        Bin centers (seconds) and counts/rate. Bins intersecting missing
        input are NaN, including partial edge bins.
    speed : ndarray or None
        Nearest observed speed (cm/s); None when the recording has no speed.
    sampling_frequency : float
        Reciprocal bin width in Hz.
    """

    time: FloatArray
    data: FloatArray
    speed: FloatArray | None
    sampling_frequency: float

    def smooth(self, sigma: float) -> FloatArray:
        """Return a Gaussian-smoothed trace without crossing missing bins."""
        return _transform(
            self.time,
            self.data,
            lambda x: rd.gaussian_smooth(x, sigma, self.sampling_frequency),
        )

    def detect(self, **kwargs: Any) -> pd.DataFrame:
        """Threshold this trace using detect_events_from_trace options.

        Bounds are the outer edges of an event's first and last bins, so an
        event of n bins lasts n bin widths, as the duration limits count it.
        ``close_event_threshold`` is likewise the gap between edges.
        """
        default = inspect.signature(rd.detect_events_from_trace).parameters["speed_threshold"]
        if self.speed is None and np.isfinite(kwargs.get("speed_threshold", default.default)):
            raise ValueError(_NO_SPEED)
        speed = np.full(len(self.time), np.nan) if self.speed is None else self.speed
        width = 1 / self.sampling_frequency
        if kwargs.get("close_event_threshold", 0.0) > 0:
            # Between centers, the gap between edges is one bin width longer.
            kwargs["close_event_threshold"] = kwargs["close_event_threshold"] + width
        events = rd.detect_events_from_trace(
            self.time, self.data, speed, self.sampling_frequency, **kwargs
        )
        events["start_time"] -= width / 2
        events["end_time"] += width / 2
        events["duration"] = events.end_time - events.start_time
        return events

    def merge(
        self, events: pd.DataFrame | FloatArray, gap: float, *, inclusive: bool = False
    ) -> FloatArray:
        """Merge inside valid native-grid blocks only.

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
    time = centers + rec.time[0]
    speed = (
        None
        if rec.session.speed is None
        else rec.session.speed[rd.core.nearest_sample_index(rec.time, time)]
    )
    trace = PopulationTrace(time, values, speed, 1 / bin_width)
    if smoothing_sigma:
        trace.data = trace.smooth(smoothing_sigma)
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
    return rd.detect_events_from_trace(
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
    # Keep every local peak, even if two occupy the same above-mean excursion.
    rows = []
    _, blocks = _valid_blocks(rec.time, trace)
    for start, stop in blocks:
        peaks, _ = find_peaks(trace[start:stop], height=np.nextafter(level, np.inf))
        for peak in peaks + start:
            rows.append(  # noqa: PERF401 - explicit per-peak boundaries
                (
                    max(rec.time[start], rec.time[peak] - before),
                    min(rec.time[stop - 1], rec.time[peak] + after),
                    rec.time[peak],
                    trace[peak],
                )
            )
    return pd.DataFrame(rows, columns=["start_time", "end_time", "peak_time", "peak_value"])


def _mallory_candidates(time: FloatArray, z: FloatArray) -> pd.DataFrame:
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
            event = [float(time[start]), float(time[end]), float(time[peak]), float(z[peak])]
            if block_events and event[0] == block_events[-1][0]:
                previous = block_events.pop()
                if previous[3] > event[3]:
                    event[2:] = previous[2:]
            block_events.append(event)
        merged: list[list[float]] = []
        for event in block_events:
            if merged and event[2] - merged[-1][2] <= 0.07 + tolerance:
                previous = merged.pop()
                event[0] = previous[0]
                if previous[3] > event[3]:
                    event[2:] = previous[2:]
            merged.append(event)
        rows.extend(merged)
    return pd.DataFrame(rows, columns=["start_time", "end_time", "peak_time", "peak_value"])


# --------------------------------------------------------------------------- recipes


@dataclass
class Recipe:
    row: int
    paper: str
    trigger: str
    run: Callable[..., pd.DataFrame | FloatArray]
    note: str
    role: str = "candidate_detection"


RECIPES: list[Recipe] = []


P = ParamSpec("P")


# Raw functions compose intermediate inventories. Public calls all pass through
# run_method once, after the composition is complete.
_IMPLEMENTATIONS: dict[str, Callable[..., pd.DataFrame | FloatArray]] = {}


def _register(
    registry: list[Recipe], row: int, paper: str, trigger: str, role: str
) -> Callable[[Callable[P, pd.DataFrame | FloatArray]], Callable[P, pd.DataFrame]]:
    def register(
        function: Callable[P, pd.DataFrame | FloatArray],
    ) -> Callable[P, pd.DataFrame]:
        name = function.__name__
        _IMPLEMENTATIONS[name] = function

        @functools.wraps(function)
        def public_method(*args: P.args, **options: P.kwargs) -> pd.DataFrame:
            call = inspect.signature(function).bind(*args, **options)
            recording = call.arguments.pop("rec")
            return run_method(name, recording, **call.arguments)

        public_method.__name__ = name
        public_method.__qualname__ = name
        public_method.__annotations__ = {**function.__annotations__, "return": pd.DataFrame}
        public_method.__signature__ = inspect.signature(function).replace(  # type: ignore[attr-defined]
            return_annotation=pd.DataFrame
        )
        public_method.__doc__ = (
            (function.__doc__ or "").rstrip()
            + """

    Parameters
    ----------
    rec : Recording
        Selected signals, cells and curated intervals.
    **options
        Method-specific keyword arguments described above and in the signature.

    Returns
    -------
    events : pandas.DataFrame
        Candidate bounds and available diagnostics, with method metadata in
        attrs. Supplied behavior_intervals retain wholly contained events.
    """
        )
        registry.append(
            Recipe(row, paper, trigger, public_method, (function.__doc__ or "").strip(), role)
        )
        return public_method

    return register


def recipe(
    row: int, paper: str, trigger: str, *, role: str = "candidate_detection"
) -> Callable[[Callable[P, pd.DataFrame | FloatArray]], Callable[P, pd.DataFrame]]:
    return _register(RECIPES, row, paper, trigger, role)


@recipe(0, "Mallory 2025", "MUA")
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


@recipe(1, "Widloski 2025", "secondary ripple label", role="secondary_label")
def widloski_2025(rec: Recording) -> pd.DataFrame | FloatArray:
    """Replays are defined by decoding (not reproduced). This is the ripple
    label: 100-220 Hz, one channel per tetrode, envelope smoothed with an
    80 ms Gaussian and averaged, z-scored over stopping (speed < 5), peak > 2 SD
    for >= 15 ms, bounds at the mean, merged < 50 ms (the text; the code merges
    none)."""
    return rd.detect_events_from_trace(
        rec.time, rec.mean_envelope((100.0, 220.0)), rec.speed, rec.fs,
        threshold=2.0, smoothing_sigma=0.08, normalization_mask=rec.speed < 5,
        minimum_duration=0.015, close_event_threshold=0.05, close_event_rule="merge",
        speed_threshold=np.inf,
    )  # fmt: skip


def _population_with_ripple_peak(rec: Recording, sleep: FloatArray) -> FloatArray:
    """Population candidates normalized over supplied NREM, with a ripple peak.

    The historical ripple detector is unresolved. Real recordings require an
    external inventory; the simulation alone uses the documented Zugaro proxy.
    Supply behavior_intervals to run_method for quiet-waking/NREM eligibility.
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
        if not isinstance(rec.session, rd.SimulatedSession):
            msg = "Supply external_ripples; the historical LFP detector is unspecified."
            raise ValueError(msg)
        peaks = zugaro_ripple_peaks(rec, (130.0, 200.0)).peak_time
    else:
        peaks = (
            rec.external_ripples[:, 2]
            if rec.external_ripples.shape[1] == 3
            else rec.external_ripples.mean(axis=1)
        )
    events = rd.require_times_inside(events, peaks)
    if rec.behavior_intervals is not None:
        eligible = rec.behavior_intervals
    elif not isinstance(rec.session, rd.SimulatedSession):
        msg = "Supply eligible quiet-waking/NREM behavior_intervals separately from the NREM baseline."
        raise ValueError(msg)
    else:
        eligible = sleep
    return within_intervals(events, eligible)


@recipe(2, "Yang 2024", "SWR+MUA")
def yang_2024(rec: Recording) -> pd.DataFrame | FloatArray:
    """Population candidates with a coincident externally supplied ripple peak.

    Supply curated NREM normalization and eligible quiet-waking/NREM intervals;
    the paper used SleepScoreMaster with manual curation. Only simulation uses
    a speed <4 and theta/delta <1 proxy and an assumed ripple detector. The
    15 ms Gaussian width is interpreted as SD as in the shared Grosmark path.
    """
    return _population_with_ripple_peak(rec, rec.sleep(4.0, 1.0))


def _tirole(rec: Recording) -> FloatArray:
    """Released Tirole finite kernels and candidate order, with supplied cells.

    Native 1 ms counts, 41-point gausswin(alpha=2), forward/backward filtering,
    sample-SD z scores, 10 ms-separated threshold anchors, inclusive below-zero
    crossings with 0.25/0.5 fallbacks within 300 ms; >=100 ms before <50 ms
    merging. Speed is sampled every 10 ms; place-cell and ripple gates follow.
    Bin origin is the recording start; counts retain the input timestamp precision.
    The duration, merge, speed, cell and ripple rules use bin centers, as the
    release's onset-offset differences do; the reported bounds are the outer
    bin edges, half a bin wider on each side, as for other native grids.
    LFP resampling uses scipy's polyphase anti-alias filter, whose edge behavior
    can differ from the original acquisition/downsampling pipeline.
    """
    trace = population_trace(rec, bin_width=0.001)
    kernel = np.exp(-0.5 * (np.arange(-20, 21) / 10) ** 2)
    kernel /= kernel.sum()

    def smooth_block(x: FloatArray) -> FloatArray:
        if len(x) <= 120:
            return np.full_like(x, np.nan)
        return np.asarray(filtfilt(kernel, [1.0], x, padlen=120), float)

    trace.data = _zscore(_transform(trace.time, trace.data, smooth_block), ddof=1)
    events = _tirole_bounds(trace.time, trace.data)
    events = trace.merge(within_duration(events, 0.1), 0.05)
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
    half_bin = 0.5 / trace.sampling_frequency
    return np.asarray(events[selected] + np.array([-half_bin, half_bin]), dtype=float)


@recipe(3, "Huelin Gorriz 2023", "SWR+MUA")
def huelin_gorriz_2023(rec: Recording, *, interpretation: str = "published_cap") -> FloatArray:
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
    return within_duration(events, high=0.75) if interpretation == "published_cap" else events


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


def _harvey_stage(rec: Recording, events: pd.DataFrame | FloatArray, stage: str) -> FloatArray:
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


@recipe(4, "Harvey 2023 (code)", "SWR")
def harvey_2023_code(rec: Recording, *, stage: str = "detection") -> pd.DataFrame | FloatArray:
    """Released DetectSWR path on pyramidal and radiatum LFP, then spiking veto.

    Uses neurocode defaults: 2-50 Hz sharp waves, 80-250 Hz ripples,
    local thresholds 0.5/2.5 SD, sharp waves 20-500 ms, ripples >=25 ms.
    Supply selected pyramidal and radiatum channels and CA1 pyramidal spikes.
    Manual curation and EMG vetoes are external. Replay filtering is selectable
    with stage='decoding_candidates'; detection is the default.
    """
    ripples = rd.Long_sharp_wave_ripple_detector(
        rec.time, rec.session.raw_lfp, _speed_or_unknown(rec), rec.fs,
        sharp_wave_lfp=rec.session.sharp_wave_lfp, speed_threshold=np.inf,
    )  # fmt: skip
    return _harvey_stage(rec, _spiking_filter(rec, ripples), stage)


@recipe(4, "Harvey 2023 (text)", "SWR (needs radiatum)")
def harvey_2023_text(
    rec: Recording, *, sharp_wave_polarity: float = -1.0, stage: str = "detection"
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
    baseline = _baseline(rec, required=not isinstance(rec.session, rd.SimulatedSession))
    band = rec.transform(
        rec.session.raw_lfp, lambda x: _difference_of_gaussians_band(x, (80.0, 250.0), rec.fs)
    )
    scale = float(np.nanstd(band[baseline]))
    kernel = _gaussian_lowpass_fir(55.0, rec.fs)
    clipped = rec.transform(
        np.abs(np.clip(band, -4 * scale, 4 * scale)), lambda x: _firfilt(x, kernel)
    )
    power = rec.transform(np.abs(band), lambda x: _firfilt(x, kernel))
    mean, sd = float(np.nanmean(clipped[baseline])), float(np.nanstd(clipped[baseline]))
    if not np.isfinite(sd) or sd <= 0:
        msg = "Clipped power needs a finite nonconstant baseline."
        raise ValueError(msg)
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


@recipe(5, "Liu 2023", "SWR+MUA (needs radiatum)")
def liu_2023(rec: Recording) -> pd.DataFrame | FloatArray:
    """DetectSWR at neurocode defaults on the pyramidal and radiatum channels
    (the text's 1 SD bounds and 15-400 ms limits are not applied; manual
    curation is not reproduced); the candidates are pyramidal-cell bursts
    (10 ms Gaussian, assumed to be its SD; > 2 SD, bounds at the mean,
    100-500 ms) overlapping an SWR."""
    swrs = rd.Long_sharp_wave_ripple_detector(
        rec.time, rec.session.raw_lfp, _speed_or_unknown(rec), rec.fs,
        sharp_wave_lfp=rec.session.sharp_wave_lfp, speed_threshold=np.inf,
    )  # fmt: skip
    bursts = _detect_population(
        rec, rec.pyramidal, 0.010,
        threshold=2.0, minimum_duration=0.0, minimum_event_duration=0.1,
        maximum_duration=0.5, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_overlap(bursts, swrs)


@recipe(6, "Tirole 2022", "SWR+MUA")
def tirole_2022(rec: Recording) -> pd.DataFrame | FloatArray:
    """See _tirole."""
    return _tirole(rec)


@recipe(7, "Bush 2022", "MUA")
def bush_2022(rec: Recording) -> pd.DataFrame | FloatArray:
    """Pyramidal cells, 5 ms Gaussian, peak z >= 3, bounds at z >= 0; merged
    when <= 40 ms apart, events <= 40 ms dropped, then >= 5 or 15% of
    pyramidal cells (whichever is larger), median speed <= 10, <= 0.5 s."""
    trace = rec.rate(rec.pyramidal, 0.005)
    events = rd.detect_events_from_trace(
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


@recipe(8, "Berners-Lee 2022", "MUA")
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

    trace.data = _transform(trace.time, trace.data, smooth)
    stopped = np.abs(_known_speed(trace.speed)) < 5
    trace.data = _zscore(trace.data, stopped, ddof=1)
    trace.data[~stopped] = np.nan
    return trace.detect(
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
    """Mean over tetrodes of the 150-250 Hz envelope, 12.5 ms Gaussian, above
    ``threshold`` SD with statistics over speed < 5 (one reading of
    "excluding periods of movement"), speed <= 5 at both ends, bounds at the
    mean, 50 ms to ``maximum_duration``."""
    return rd.detect_events_from_trace(
        rec.time, rec.mean_envelope((150.0, 250.0), channels), rec.speed, rec.fs,
        threshold=threshold, smoothing_sigma=0.0125, normalization_mask=rec.speed < 5,
        minimum_duration=0.0, minimum_event_duration=0.05, maximum_duration=maximum_duration,
        speed_threshold=5.0,
    )  # fmt: skip


@recipe(10, "Krause 2022", "SWR")
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
    for start, end in swrs:
        group = np.searchsorted(intervals[:, 0], start, side="right") - 1
        if group < 0 or end > intervals[group, 1]:
            continue
        n_bins = int(np.ceil((end - start - tolerance) / 0.003)) - 1
        if n_bins < 11:
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
    return np.asarray(found, float).reshape(-1, 2)


@recipe(11, "Mou 2022", "MUA")
def mou_2022(
    rec: Recording, *, normalization: str = "minmax", stage: str = "detection"
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
    trace.data = (trace.data - offset) / span
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


@recipe(12, "Berners-Lee 2021", "SWR")
def berners_lee_2021(rec: Recording) -> pd.DataFrame | FloatArray:
    """Pfeiffer & Foster 2015's rule at 2 SD on three selected tetrodes.

    Supply the channels from the three tetrodes with the most pyramidal cells
    first; the function uses the first three channels without ranking them.
    """
    return _pfeiffer_2015_swrs(rec, threshold=2.0, channels=3)


@recipe(13, "Denovellis 2021", "SWR")
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
        if len(x) <= 303:
            return np.full_like(x, np.nan)
        return np.asarray(filtfilt(kernel, [1.0], x, axis=0), float)

    filtered = rec.transform(rec.session.lfps, historical_filter)
    trace = np.sqrt(np.maximum(0, rec.smooth(np.sum(filtered**2, axis=1), 0.004)))
    return _ripple_trace_events(
        rec, trace, threshold=2.0, minimum_duration=0.015, speed_threshold=4.0
    )


@recipe(14, "Gillespie 2021", "SWR")
def gillespie_2021(rec: Recording) -> pd.DataFrame | FloatArray:
    """Kay consensus trace, 2 SD for 15 ms, speed < 4."""
    return rd.Kay_ripple_detector(rec.time, rec.filtered((150.0, 250.0)), rec.speed, rec.fs)


def _michon(rec: Recording, *, order: str = "text") -> FloatArray:
    """5 ms population bins; smoothing/detrending order is explicitly selectable."""
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
    ripples = rd.detect_events_from_trace(
        rec.time,
        detrended(rec.time, rec.mean_envelope((140.0, 225.0), channels=3), rec.fs),
        rec.speed,
        rec.fs,
        threshold=8.0,
        minimum_event_duration=0.04,
        **common,
    )
    population = population_trace(rec, bin_width=0.005)
    population.data = detrended(
        population.time, population.data, population.sampling_frequency
    )
    bursts = population.detect(threshold=4.0, minimum_event_duration=0.08, **common)
    return rd.exclude_movement(rd.require_overlap(bursts, ripples), rec.speed, rec.time, 5.0)


@recipe(15, "Michon 2021", "SWR+MUA")
def michon_2021(rec: Recording, *, order: str = "text") -> FloatArray:
    """5 ms MUA bins; 15 ms smoothing, 3 s detrending; text/code order selectable."""
    return _michon(rec, order=order)


@recipe(16, "Igata 2021 (candidates)", "SWR+MUA")
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


@recipe(17, "Gridchyn 2020", "adaptive MUA triggers")
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
    Outputs retain trigger time, multiplier, and update history in attrs.

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
    history = []
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


@recipe(18, "Kaefer 2020", "secondary SWR label", role="secondary_label")
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
    _, blocks = _valid_blocks(rec.time, raw)
    frequencies = np.fft.rfftfreq(width, 1 / rec.fs)
    band = (frequencies >= 150) & (frequencies <= 250)
    for j, center in enumerate(centers):
        start, stop = center - width // 2, center - width // 2 + width
        if any(start >= a and stop <= b for a, b in blocks):
            spectrum = np.fft.rfft(raw[start:stop], axis=0)
            spectrum[~band] = 0
            filtered = np.fft.irfft(spectrum, n=width, axis=0)
            power[j] = np.sqrt(np.mean(filtered**2, axis=0)).mean()
    time = rec.time[centers]
    baseline = _baseline(rec, required=True)[centers]
    return rd.detect_events_from_trace(
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


@recipe(19, "Bhattarai 2020", "SWR+MUA")
def bhattarai_2020(
    rec: Recording,
    *,
    power_measure: str = "squared_signal",
    window_end_rule: str = "last_spike",
) -> FloatArray:
    """Post-silence population candidates coinciding with a Bhattarai SWR.

    Power is explicitly selectable; event-end policy is unresolved in the
    supplement (last-spike is the default interpretation, fixed is available).
    Supply block-specific place cells; silence is measured in that population.
    """
    swrs = _IMPLEMENTATIONS["bhattarai_2020_ripples"](rec, power_measure=power_measure)
    replays = rd.detect_silence_bounded_events(
        rec.time, rec.multiunit, rec.fs,
        minimum_silence=0.06 + 1 / rec.fs, window=0.3, window_end_rule=window_end_rule, units=rec.place_cells,
        minimum_active_units=5,
    )  # fmt: skip
    return rd.require_overlap(replays, swrs)


@recipe(20, "Stella 2019", "SWR")
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
        if not isinstance(rec.session, rd.SimulatedSession):
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
    if rec.sleep_intervals is not None or not isinstance(rec.session, rd.SimulatedSession):
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


@recipe(21, "Xu 2019", "MUA")
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


@recipe(22, "Farooq 2019 (Neuron)", "MUA")
def farooq_2019_neuron(rec: Recording) -> pd.DataFrame | FloatArray:
    """Population frames in caller-supplied SWS, with 15 ms interpreted as Gaussian SD.

    The width convention is unresolved. Only simulation uses the proxy:
    speed < 1 cm/s for >= 5 s (scaled from 5 min) and theta/delta
    (6-12 / 1-4 Hz Hilbert amplitude, 5 s Gaussian) < 2; >= 5 neurons (the
    Methods; the Results say place cells). This paper's frames are defined
    during slow-wave sleep."""
    return _farooq(rec, rec.sleep(1.0, 2.0, stillness=5.0, smoothing_sigma=5.0), rec.pyramidal)


@recipe(23, "Farooq 2019 (Science)", "MUA")
def farooq_2019_science(rec: Recording) -> pd.DataFrame | FloatArray:
    """The reported 15 ms Gaussian width is interpreted as SD (unresolved).

    SWS: speed < 2 cm/s and theta/delta (4-10 / 1-3 Hz, 10 s Gaussian) below
    its mean; >= 5 place-responsive cells. Measured recordings require curated
    sleep intervals. Awake frames are available in farooq_2019_science_awake."""
    if rec.sleep_intervals is not None or not isinstance(rec.session, rd.SimulatedSession):
        return _farooq(rec, rec.sleep(2.0, 1.0), rec.place_cells)
    ratio = rec.ratio((4.0, 10.0), (1.0, 3.0), smoothing_sigma=10.0)
    still = rd.state_intervals(rec.speed, rec.time, 2.0)
    low = rd.state_intervals(ratio, rec.time, float(np.nanmean(ratio)))
    sleep = rec.mask_to_intervals(rec.intervals_to_mask(still) & rec.intervals_to_mask(low))
    return _farooq(rec, sleep, rec.place_cells)


@recipe(24, "Chenani 2019", "MUA")
def chenani_2019(rec: Recording) -> pd.DataFrame | FloatArray:
    """Place-cell rate, 30 ms Gaussian, peak >= 3 SD, bounds >= 1 SD, >= 5
    active cells. Supply reward-zone behavior_intervals to run_method; zones
    chosen by eye are not inferred automatically."""
    events = _detect_population(
        rec, rec.place_cells, 0.030,
        threshold=3.0, bound_threshold=1.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
    )


@recipe(25, "Michon 2019", "SWR+MUA")
def michon_2019(rec: Recording, *, order: str = "text") -> FloatArray:
    """Same offline conjunction as Michon 2021; text/code preprocessing order selectable."""
    return _michon(rec, order=order)


@recipe(26, "Liu 2019", "MUA")
def liu_2019(rec: Recording) -> pd.DataFrame | FloatArray:
    """Pyramidal spikes inside SWS (speed < 1 cm/s and theta/delta < 2, 5 s
    Gaussian), split at >= 100 ms of silence, >= 4 cells, 80 ms-1.2 s. The
    awake-rest frames are available separately in liu_2019_awake."""
    sleep = rec.sleep(1.0, 2.0, smoothing_sigma=5.0)
    return rd.detect_silence_bounded_events(
        rec.time, only_in(rec, rec.multiunit, sleep), rec.fs,
        minimum_silence=0.1, units=rec.pyramidal, minimum_active_units=4,
        minimum_duration=0.08, maximum_duration=1.2,
    )  # fmt: skip


def _karlsson_rule(rec: Recording, speed_threshold: float) -> pd.DataFrame:
    """Karlsson & Frank 2009: each tetrode's 4 ms-smoothed envelope, 3 SD for
    >= 15 ms on any tetrode, bounds at the mean, overlapping events combined."""
    return rd.Karlsson_ripple_detector(
        rec.time,
        rec.filtered((150.0, 250.0)),
        rec.speed,
        rec.fs,
        speed_threshold=speed_threshold,
    )


@recipe(27, "Shin 2019", "SWR")
def shin_2019(rec: Recording, *, stage: str = "detection") -> pd.DataFrame | FloatArray:
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


@recipe(28, "Carey 2019", "SWR+MUA")
def carey_2019(rec: Recording) -> pd.DataFrame:
    """Published amSWR spectral score and joint MUA candidates.

    Measured recordings require manually selected example_ripples. Simulation
    alone uses the five largest Kay events. The joint score is rescaled to
    mean 0.5, thresholded at 4 inside low-speed/low-theta intervals, >=20 ms
    and >=5 units. The spectral template uses the first selected raw channel.
    """
    examples: pd.DataFrame | FloatArray
    if rec.example_ripples is None:
        if not isinstance(rec.session, rd.SimulatedSession):
            msg = "Supply example_ripples for Carey's spectral template."
            raise ValueError(msg)
        kay = rd.Kay_ripple_detector(rec.time, rec.filtered((150.0, 250.0)), rec.speed, rec.fs)
        examples = kay.nlargest(5, "max_zscore")
    else:
        examples = rec.example_ripples
    score = rd.carey_spectral_ripple_score(rec.time, rec.session.raw_lfp, rec.fs, examples)
    return rd.Carey_candidate_detector(
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


@recipe(29, "Muessig 2019", "SWR+MUA")
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
    return within_intervals(events, rec.sleep(limit, 2.0, measure="power"))


@recipe(30, "Drieu 2018", "MUA")
def drieu_2018(rec: Recording, *, stage: str = "detection") -> pd.DataFrame | FloatArray:
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
    if rec.sleep_intervals is not None or not isinstance(rec.session, rd.SimulatedSession):
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
    return rd.detect_events_from_trace(
        rec.time, only_in(rec, rec.rate(rec.place_cells, 0.010), sleep), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, maximum_duration=0.5, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(31, "Maboudi 2018", "MUA")
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
    trace.data = _transform(
        trace.time,
        trace.data,
        lambda x: np.asarray(fftconvolve(x, kernel, mode="same"), float),
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


@recipe(32, "Olafsdottir 2017", "MUA")
def olafsdottir_2017(rec: Recording, *, analysis: str = "arm") -> pd.DataFrame | FloatArray:
    """Native place-cell MUA candidates, with separate arm/trajectory participation.

    5 ms Gaussian, 3 SD/mean, >=40 ms, all event speeds <=3. Supply corner
    behavior_intervals to run_method. Arm reactivation adds no cell-count
    criterion; analysis='trajectory' requires >=15% and >5 place cells.
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


@recipe(33, "Wu 2017", "MUA")
def wu_2017(rec: Recording, *, stage: str = "detection") -> pd.DataFrame | FloatArray:
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


@recipe(34, "Yamamoto 2017 (one reading)", "SWR+MUA")
def yamamoto_2017(rec: Recording) -> pd.DataFrame | FloatArray:
    """One reading of an ambiguous rule: summed spikes in nonoverlapping 10 ms bins, peak > 3 SD, bounds at 1 SD, kept when
    overlapping a period of 140-200 Hz power above 3 SD on one channel. The
    paper does not say how the two combine or which trace sets the bounds."""
    ripples = rd.detect_events_from_trace(
        rec.time, rec.envelope((140.0, 200.0))[:, 0] ** 2, _speed_or_unknown(rec), rec.fs,
        threshold=3.0, bound_threshold=3.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    bursts = _detect_population(
        rec, None, 0.0, bin_width=0.01,
        threshold=3.0, bound_threshold=1.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_overlap(bursts, ripples)


@recipe(35, "Tang 2017", "SWR")
def tang_2017(rec: Recording) -> pd.DataFrame | FloatArray:
    """The Karlsson rule at < 4 cm/s (smoothing and minimum inherited)."""
    return _karlsson_rule(rec, 4.0)


@recipe(36, "Grosmark 2016", "SWR+MUA")
def grosmark_2016(rec: Recording, *, stage: str = "detection") -> FloatArray:
    """Population/ripple conjunction, with a separate decoding-candidate stage.

    The reported 15 ms Gaussian width is interpreted as its SD (unresolved).
    Detection retains 50-500 ms events with >=5 pyramidal cells. Selecting
    stage='decoding_candidates' additionally requires >=100 ms and >=5 or 10%
    of supplied place cells. Measured data require NREM for normalization,
    eligible quiet-waking/NREM behavior_intervals, and external ripple peaks.
    """
    if stage not in {"detection", "decoding_candidates"}:
        msg = "stage must be 'detection' or 'decoding_candidates'."
        raise ValueError(msg)
    events = _population_with_ripple_peak(rec, rec.sleep(4.0, 1.0))
    if stage == "detection":
        return events
    return rd.require_active_units(
        within_duration(events, low=0.1),
        rec.multiunit,
        rec.time,
        minimum_active_units=5,
        minimum_active_fraction=0.1,
        units=rec.place_cells,
    )


@recipe(37, "Ambrose 2016", "SWR")
def ambrose_2016(rec: Recording) -> pd.DataFrame | FloatArray:
    """Pfeiffer & Foster 2015's trace on 4 tetrodes, > 3 SD, detected only
    while stopped (< 5 cm/s, stated in the paper). Statistics also come from
    stopping periods (the lab's convention, inferred); no duration limits are
    reported in the main Methods or supplement. The proximity to the well is
    not reproduced."""
    return rd.detect_events_from_trace(
        rec.time, rec.mean_envelope((150.0, 250.0)), rec.speed, rec.fs,
        threshold=3.0, smoothing_sigma=0.0125, minimum_duration=0.0,
        speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip


@recipe(38, "Jadhav 2016", "SWR")
def jadhav_2016(rec: Recording, *, stage: str = "detection") -> pd.DataFrame | FloatArray:
    """The Karlsson rule at < 4 cm/s; SWRs within 1 s after the previous
    one's start dropped; stage='decoding_candidates' adds >=4 active CA1 cells (all supplied units).
    The default returns the initial SWR inventory."""
    _check_stage(stage)
    events = rd.exclude_close_events(_karlsson_rule(rec, 4.0), 1.0, measure_from="start")
    if stage == "detection":
        return events
    return rd.require_active_units(events, rec.multiunit, rec.time, minimum_active_units=4)


@recipe(39, "Olafsdottir 2016", "MUA")
def olafsdottir_2016(rec: Recording) -> pd.DataFrame | FloatArray:
    """Place cells, 5 ms Gaussian, > 3 SD, bounds at the mean, >= 40 ms,
    >=15% of the place cells; no speed rule. Supply a rest recording;
    detection and statistics span the full supplied recording."""
    events = _detect_population(
        rec, rec.place_cells, 0.005,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.04,
        speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_fraction=0.15, units=rec.place_cells
    )


@recipe(40, "Silva 2015", "MUA")
def silva_2015(rec: Recording) -> pd.DataFrame | FloatArray:
    """Sorted units without interneurons (pyramidal; the Results say all
    recorded units, the Fig. 1c legend place cells), 10 ms Gaussian, > 3 SD,
    bounds at the mean, only while < 5 cm/s, 100-500 ms."""
    return _detect_population(
        rec, rec.pyramidal, 0.010,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.1,
        maximum_duration=0.5, speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip


@recipe(41, "Olafsdottir 2015", "MUA")
def olafsdottir_2015(rec: Recording, *, minimum_active_units: int = 0) -> FloatArray:
    """Per-template silence-bounded candidates before optional decoding filters.

    Supply templates as cell masks; >=15% of a template in <=300 ms bounded
    by >=50 ms of silence. The optional minimum_active_units can impose the
    decoding-stage seven-cell criterion. Rest epochs are caller-supplied
    behavior_intervals to run_method. No ensemble size is assumed.
    """
    if not rec.templates:
        msg = "Supply templates: one cell selection per directional template."
        raise ValueError(msg)
    found = []
    for template in rec.templates:
        if template.sum() < max(1, minimum_active_units):
            continue
        events = rd.detect_silence_bounded_events(
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
    events = np.concatenate(found) if found else np.empty((0, 2))
    return np.asarray(events[np.argsort(events[:, 0], kind="stable")], float)


@recipe(42, "Pfeiffer 2015", "SWR")
def pfeiffer_2015(rec: Recording) -> pd.DataFrame | FloatArray:
    """See _pfeiffer_2015_swrs."""
    return _pfeiffer_2015_swrs(rec)


@recipe(43, "Wu 2014", "MUA")
def wu_2014(rec: Recording) -> pd.DataFrame | FloatArray:
    """Place-cell density in nonoverlapping 10 ms bins, 15 ms Gaussian, > 2 SD
    over the session, bounds at the mean, speed < 5 at both ends (assumed; the
    paper does not say which samples). The reward-area restriction is not
    reproduced."""
    return _detect_population(
        rec, rec.place_cells, 0.015, bin_width=0.01,
        threshold=2.0, minimum_duration=0.0, speed_threshold=5.0,
    )  # fmt: skip


@recipe(44, "Wikenheiser 2013", "SWR")
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
    The window anchor is unspecified: measured data must choose 'samples',
    'peaks' or 'onsets'. Each anchor retains its own window, clipped to its
    valid LFP block.
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
        if not isinstance(rec.session, rd.SimulatedSession):
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
    events = np.concatenate(found) if found else np.empty((0, 2))
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=3, minimum_spikes=5
    )
    if branch == "rest":
        if rec.sleep_intervals is not None or not isinstance(rec.session, rd.SimulatedSession):
            return within_intervals(events, rec.sleep(2.0, 0.0))
        ratio = _zscore(rec.ratio((6.0, 10.0), (2.0, 4.0), measure="power"))
        still = rd.state_intervals(rec.speed, rec.time, 2.0, minimum_duration=2.0)
        low = rd.state_intervals(ratio, rec.time, 0.0)
        rest = rec.mask_to_intervals(rec.intervals_to_mask(still) & rec.intervals_to_mask(low))
        return within_intervals(events, rest)
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


@recipe(45, "Pfeiffer 2013", "MUA")
def pfeiffer_2013(rec: Recording) -> pd.DataFrame | FloatArray:
    """Clustered pyramidal units' histogram (interneurons excluded, inferred)
    only while < 5 cm/s, 10 ms Gaussian, > 3 SD, bounds at the mean; bounds
    moved inward until the first and last 20 ms windows (5 ms steps) hold 2
    spikes; then (order assumed) >= 10% of units and 50 ms-2 s."""
    events = _detect_population(
        rec, rec.pyramidal, 0.010,
        threshold=3.0, minimum_duration=0.0, speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip
    events = rd.trim_events_to_spike_windows(
        events, rec.multiunit, rec.time, units=rec.pyramidal
    )
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_fraction=0.1, units=rec.pyramidal
    )
    return within_duration(events, 0.05, 2.0)


@recipe(46, "Carr 2012", "SWR")
def carr_2012(rec: Recording, *, stage: str = "detection") -> pd.DataFrame | FloatArray:
    """The Karlsson rule on CA1 at < 4 cm/s; stage='decoding_candidates'
    adds >=5 active place cells; the default returns the initial SWR inventory."""
    _check_stage(stage)
    events = _karlsson_rule(rec, 4.0)
    if stage == "detection":
        return events
    return rd.require_active_units(
        events, rec.multiunit, rec.time,
        minimum_active_units=5, units=rec.place_cells,
    )  # fmt: skip


@recipe(47, "Bendor 2012", "MUA")
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


@recipe(48, "Gupta 2010", "SWR gate only", role="candidate_gate")
def gupta_2010(rec: Recording, *, log_amplitude: bool = True) -> pd.DataFrame | FloatArray:
    """Events are windows grown by a spike-order score (not reproduced). This
    is the SWR gate: 180-220 Hz Hilbert amplitude averaged over tetrodes,
    log-transformed as in Jackson 2006, above 2 SD over the whole session.
    Retaining the log transform is an explicit inference; log_amplitude=False
    selects the untransformed interpretation. The >=3 active cells and reward
    pause apply to sequence windows, which this gate does not construct."""
    amplitude = rec.mean_envelope((180.0, 220.0))
    trace = np.log(np.maximum(amplitude, np.finfo(float).tiny)) if log_amplitude else amplitude
    return rd.detect_events_from_trace(
        rec.time, trace, _speed_or_unknown(rec), rec.fs,
        threshold=2.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(49, "Karlsson 2009", "SWR")
def karlsson_2009(rec: Recording) -> pd.DataFrame | FloatArray:
    """The Karlsson rule at < 2 cm/s (CA1 and CA3 tetrodes)."""
    return _karlsson_rule(rec, 2.0)


@recipe(50, "Davidson 2009", "MUA")
def davidson_2009(rec: Recording) -> pd.DataFrame | FloatArray:
    """All spikes, 15 ms Gaussian, peak >= 3 SD over stopping (< 5 cm/s),
    bounds at the mean, speed < 5 at both ends; within 30 s of running (RUN:
    speed >15 cm/s). Supply the full relevant behavioral recording."""
    events = _detect_population(
        rec, None, 0.015,
        threshold=3.0, normalization_mask=rec.speed < 5, minimum_duration=0.0,
        speed_threshold=5.0,
    )  # fmt: skip
    running = rd.state_intervals(rec.speed, rec.time, 15.0, comparison=">")
    return rd.require_overlap(events, running + np.array([-30.0, 30.0]))


@recipe(51, "Diba 2007", "MUA")
def diba_2007(rec: Recording) -> pd.DataFrame | FloatArray:
    """>= 60 ms of silence (of the template's cells, assumed), then >= 5 and
    >= 30% of the template's cells (whichever is greater) in the next 300 ms,
    speed <=10 at both ends (assumed). Supply place_cells for one directional
    template and behavior_intervals for the eligible track-end reward areas."""
    events = rd.detect_silence_bounded_events(
        rec.time, rec.multiunit, rec.fs,
        minimum_silence=0.06, window=0.3, window_end_rule="fixed", units=rec.place_cells,
        minimum_active_units=5, minimum_active_fraction=0.3,
    )  # fmt: skip
    return rd.exclude_movement(events, rec.speed, rec.time, 10.0)


@recipe(52, "Ji 2007", "MUA")
def ji_2007(
    rec: Recording,
    *,
    stage: str = "detection",
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
    trace.data *= 0.01
    mask = _intervals_to_mask(trace.time, sleep)
    level = rd.histogram_minimum_threshold(
        trace.data[mask], bins=histogram_bins, smoothing_window=histogram_smoothing
    )
    trace.data[~mask] = np.nan
    events = trace.detect(
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


@recipe(53, "Foster 2006", "MUA")
def foster_2006(rec: Recording) -> pd.DataFrame | FloatArray:
    """Probe cells' spikes during stopping (< 5 cm/s, assumed) pooled and split
    at gaps of more than 50 ms, >=1/3 of the cells, <=500 ms. Supply
    place_cells for one probe sequence and behavior_intervals for the
    eligible facing-direction epochs."""
    stopped = rec.mask_to_intervals(rec.speed < 5)
    return rd.detect_silence_bounded_events(
        rec.time, only_in(rec, rec.multiunit, stopped), rec.fs,
        minimum_silence=0.05 + 1 / rec.fs, units=rec.place_cells,
        minimum_active_fraction=1 / 3, maximum_duration=0.5,
    )  # fmt: skip


@recipe(54, "Lee 2002", "MUA")
def lee_2002(rec: Recording) -> pd.DataFrame | FloatArray:
    """Template cells' spikes in supplied SWS, with within-cell bursts collapsed.

    Supply place_cells for one directional template and curated sleep_intervals
    (the paper used a theta/total power ratio and video). Each cell's spikes
    with ISI <50 ms collapse to their first spike; the resulting letters split
    at gaps >100 ms. Only simulation uses speed <4 and theta/delta <1 as SWS."""
    sleep = rec.sleep(4.0, 1.0)
    return rd.detect_silence_bounded_events(
        rec.time, only_in(rec, rec.multiunit, sleep), rec.fs,
        minimum_silence=0.1 + 1 / rec.fs, maximum_isi=0.05, units=rec.place_cells,
    )  # fmt: skip


@recipe(55, "Nadasdy 1999", "SWR")
def nadasdy_1999(
    rec: Recording, *, rms_window: float | None = None, bound_threshold: float | None = None
) -> FloatArray:
    """150-250 Hz per-channel RMS sum, 7 SD, during supplied sleep.

    RMS window, baseline and bounds are not reported. Measured data require
    rms_window, bound_threshold and baseline_intervals. The demonstration
    explicitly uses 4 ms RMS and mean bounds; these are not source values.
    """
    if rms_window is None or bound_threshold is None:
        if not isinstance(rec.session, rd.SimulatedSession):
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
    return within_intervals(events, rec.sleep(4.0, 1.0, theta=(5.0, 10.0), delta=(2.0, 4.0)))


@recipe(56, "Kudrimoti 1999", "SWR")
def kudrimoti_1999(rec: Recording, *, threshold_sd: float | None = None) -> pd.DataFrame:
    """100-300 Hz amplitude above a caller-selected threshold for >=25 ms in SWS.

    The threshold is unreported. Measured inputs require threshold_sd; only
    the simulation uses a 3 SD assumption. Supplied sleep defines the baseline.
    """
    if threshold_sd is None:
        if not isinstance(rec.session, rd.SimulatedSession):
            msg = "Supply the unreported threshold_sd explicitly."
            raise ValueError(msg)
        threshold_sd = 3.0
    sleep = rec.sleep(4.0, 1.0)
    amplitude = only_in(rec, rec.envelope((100.0, 300.0))[:, 0], sleep)
    return _ripple_trace_events(
        rec,
        amplitude,
        threshold=threshold_sd,
        bound_threshold=threshold_sd,
        minimum_duration=0.025,
        normalization_mask=rec.intervals_to_mask(sleep),
    )


NOT_REPRODUCED = {
    1: (
        "Widloski 2025",
        "Decoded replay definition is not implemented; the available method returns secondary ripple labels.",
    ),
    18: (
        "Kaefer 2020",
        "Adaptive-window trajectory decoding is not implemented; the available method returns secondary SWR labels.",
    ),
    48: (
        "Gupta 2010",
        "Flexible spike-sequence windows are not implemented; the available method returns only the SWR gate.",
    ),
    9: (
        "Widloski 2022",
        (
            "events are defined by decoding; ripple amplitude and spike density are "
            "reference traces only"
        ),
    ),
}


# --------------------------------------------------------------------------- run


def _tirole_bounds(time: FloatArray, z: FloatArray) -> FloatArray:
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
            for level in (0.0, 0.25, 0.5):
                eligible = (
                    z[left : anchor + 1] < level
                    if level == 0
                    else z[left : anchor + 1] <= level
                )
                crossing = np.flatnonzero(eligible)
                if len(crossing):
                    onset = left + int(crossing[-1])
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
                    break
            found.append((time[onset], time[offset]))
    return np.unique(np.asarray(found, dtype=float).reshape(-1, 2), axis=0)


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
    for start, stop in blocks:
        signal = np.asarray(
            resample_poly(raw[start:stop], ratio.numerator, ratio.denominator), float
        )
        time = rec.time[start] + np.arange(len(signal)) / 1000
        keep = time <= rec.time[stop - 1]
        signal, time = signal[keep], time[keep]
        if len(signal) <= 102:
            continue
        filtered = filtfilt(kernel, [1.0], signal, padlen=102)
        amplitude = _matlab_smooth(rd.get_envelope(filtered), 15)
        times.append(time)
        amplitudes.append(amplitude)
    if not times:
        msg = "No LFP block is long enough for the Tirole filter."
        raise ValueError(msg)
    return np.concatenate(times), np.concatenate(amplitudes)


def _detect_population_in(
    rec: Recording, intervals: FloatArray, units: BoolArray | None, sigma: float, **kwargs: Any
) -> pd.DataFrame:
    trace = population_trace(rec, bin_width=0.001, units=units, smoothing_sigma=sigma)
    mask = _intervals_to_mask(trace.time, intervals)
    trace.data[~mask] = np.nan
    return trace.detect(**kwargs)


# Additional inventories use a separate registry from the demonstration's
# default inventories. No variant inherits a different paper's unknowns.
VARIANTS: list[Recipe] = []


def variant(
    row: int, paper: str, trigger: str, *, role: str = "candidate_detection"
) -> Callable[[Callable[P, pd.DataFrame | FloatArray]], Callable[P, pd.DataFrame]]:
    return _register(VARIANTS, row, paper, trigger, role)


@variant(4, "Harvey 2023 (no radiatum)", "SWR")
def harvey_2023_no_radiatum(rec: Recording, *, stage: str = "detection") -> FloatArray:
    """Released FindRipples branch for sessions without a radiatum channel.

    One selected high-ripple-power channel, 100-250 Hz, thresholds 1/3 SD,
    20-300 ms, <50 ms merging, then the pyramidal-spiking veto. EMG curation
    remains external. stage='decoding_candidates' applies the released replay
    gates: >=80 ms before overlap merging, >=5 place cells and <50% empty
    nonoverlapping 20 ms bins. Detection is the default.
    """
    ripples = rd.Zugaro_ripple_detector(
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


@variant(0, "Mallory 2025", "secondary ripple candidates")
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


@variant(7, "Bush 2022", "secondary ripple candidates")
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
        if len(x) <= 1200:
            return np.full_like(x, np.nan)
        return np.asarray(filtfilt(kernel, [1.0], x, padlen=1200), float)

    trace = rec.smooth(
        rd.get_envelope(rec.transform(rec.session.raw_lfp, filtered), time=rec.time), 0.005
    )
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


@variant(16, "Igata 2021", "secondary per-channel ripple candidates")
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


@variant(17, "Gridchyn 2020", "secondary ripple candidates")
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


@variant(21, "Xu 2019", "secondary ripple candidates")
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


@variant(22, "Farooq 2019 (Neuron)", "secondary ripple candidates")
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


@variant(23, "Farooq 2019 (Science)", "secondary ripple candidates")
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


@variant(24, "Chenani 2019", "unclassified HFE candidates")
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


@variant(26, "Liu 2019", "secondary ripple peaks and centered controls")
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


@variant(30, "Drieu 2018", "secondary ripple candidates")
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


@variant(32, "Olafsdottir 2017", "secondary ripple candidates")
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


@variant(43, "Wu 2014", "secondary ripple peaks")
def wu_2014_ripples(rec: Recording) -> pd.DataFrame:
    """150-250 Hz mean envelope, 8 ms Gaussian, local peaks >2.5 stopped-baseline SD."""
    trace = rec.smooth(rec.mean_envelope((150.0, 250.0)), 0.008)
    return _local_peaks(rec, _zscore(trace, rec.speed < 5), 2.5)


@variant(45, "Pfeiffer 2013", "secondary ripple candidates")
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


@variant(50, "Davidson 2009", "secondary ripple peaks")
def davidson_2009_ripples(rec: Recording) -> pd.DataFrame:
    """150-250 Hz mean envelope, 12.5 ms Gaussian, local peaks >2.5 stopped-baseline SD."""
    trace = rec.smooth(rec.mean_envelope((150.0, 250.0)), 0.0125)
    return _local_peaks(rec, _zscore(trace, rec.speed < 5), 2.5)


@variant(51, "Diba 2007", "secondary ripple candidates")
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


@variant(52, "Ji 2007", "secondary ripple candidates")
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


@variant(54, "Lee 2002", "secondary ripple candidates")
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


@variant(53, "Foster 2006", "secondary ripple candidates")
def foster_2006_ripples(rec: Recording) -> pd.DataFrame:
    """Inherited Lee ripple rule; reported event time is the interval midpoint."""
    events = _IMPLEMENTATIONS["lee_2002_ripples"](rec)
    result = pd.DataFrame(events, columns=["start_time", "end_time"])
    result["event_time"] = events.mean(axis=1)
    return result


@variant(1, "Widloski 2025", "population burst labels", role="secondary_label")
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


@variant(10, "Krause 2022", "secondary HSE candidates")
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
    trace.data = _zscore(trace.data)
    speed = _known_speed(trace.speed)
    trace.data[~np.isfinite(speed) | (speed > 5)] = np.nan
    return trace.detect(
        threshold=3.0,
        normalization_method="none",
        minimum_duration=0.0,
        minimum_event_duration=0.051,
        speed_threshold=np.inf,
    )


@variant(13, "Denovellis 2021", "secondary MUA candidates")
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


@variant(14, "Gillespie 2021", "secondary MUA candidates")
def gillespie_2021_mua(rec: Recording) -> pd.DataFrame:
    """Published 1 ms MUA bins, 15 ms Gaussian, stopped (<4) baseline, 3 SD/mean.

    The released helper's different 5 ms kernel is not silently substituted.
    """
    return _detect_population(
        rec,
        None,
        0.015,
        threshold=3.0,
        normalization_mask=rec.speed < 4,
        minimum_duration=0.0,
        speed_threshold=4.0,
    )


@variant(31, "Maboudi 2018", "open-field population candidates")
def maboudi_2018_open_field(rec: Recording) -> FloatArray:
    """Open-field Pfeiffer 2013 criteria; separate from linear-track PBEs."""
    return _IMPLEMENTATIONS["pfeiffer_2013"](rec)


@variant(29, "Muessig 2019", "secondary ripple windows")
def muessig_2019_ripples(rec: Recording) -> pd.DataFrame:
    """7 ms RMS, 100-250 Hz, most-variable channel, >99th percentile, +/-50 ms.

    Pass one trial per recording, or baseline_intervals for the trial used
    for channel selection and percentile estimation. Local peaks stay separate.
    """
    rms = np.sqrt(np.maximum(0, rec.boxcar(rec.filtered((100.0, 250.0)) ** 2, 0.007)))
    baseline = _baseline(rec)
    rms = rms[:, int(np.argmax(np.nanstd(rms[baseline], axis=0)))]
    level = float(np.nanpercentile(rms[baseline], 99))
    return _local_peaks(rec, rms, level, before=0.05, after=0.05)


@variant(19, "Bhattarai 2020", "separate ripple candidates")
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


@variant(23, "Farooq 2019 (Science)", "awake-rest population frames")
def farooq_2019_science_awake(rec: Recording) -> FloatArray:
    """Same frame criteria within supplied awake-rest epochs and speed <1 cm/s.

    Supply behavior_intervals selecting awake track rest; their curation must
    settle whether to apply the ambiguous theta/delta restriction.
    """
    if rec.behavior_intervals is None:
        msg = "Supply behavior_intervals for awake rest on the track."
        raise ValueError(msg)
    mask = rec.intervals_to_mask(rec.behavior_intervals) & (rec.speed < 1)
    return _farooq(rec, rec.mask_to_intervals(mask), rec.place_cells)


@variant(26, "Liu 2019", "awake-rest silence-bounded frames")
def liu_2019_awake(rec: Recording) -> pd.DataFrame:
    """Silence-bounded frames at supplied track-end rest epochs, speed <2 cm/s."""
    if rec.behavior_intervals is None:
        msg = "Supply track-end behavior_intervals for awake frames."
        raise ValueError(msg)
    mask = rec.intervals_to_mask(rec.behavior_intervals) & (rec.speed < 2)
    return rd.detect_silence_bounded_events(
        rec.time,
        np.where(mask[:, None], rec.multiunit, np.nan),
        rec.fs,
        minimum_silence=0.1,
        units=rec.pyramidal,
        minimum_active_units=4,
        minimum_duration=0.08,
        maximum_duration=1.2,
    )


@variant(26, "Liu 2019", "ripple-associated sleep frames")
def liu_2019_ripple_frames(rec: Recording, *, smoothing_sigma: float) -> FloatArray:
    """Sleep frames containing a >3 SD ripple-power local peak."""
    peaks: pd.DataFrame = _IMPLEMENTATIONS["liu_2019_ripples"](
        rec, smoothing_sigma=smoothing_sigma, window=0.0
    )
    return rd.require_times_inside(_IMPLEMENTATIONS["liu_2019"](rec), peaks.peak_time)


def list_methods() -> pd.DataFrame:
    """List executable inventories and the options each caller must supply.

    Returns
    -------
    methods : pandas.DataFrame
        Function name, survey DOI, paper, output role, demonstration grouping, required options and
        interpretation. Names distinguish protocols and secondary inventories.
    """
    survey = rd.load_literature_parameters()
    rows = []
    for entry in (*RECIPES, *VARIANTS):
        parameters = inspect.signature(entry.run).parameters
        rows.append(
            {
                "name": entry.run.__name__,
                "doi": survey.loc[entry.row, "DOI"],
                "paper": entry.paper,
                "output": entry.trigger,
                "role": entry.role,
                "inventory": "default" if entry in RECIPES else "additional",
                "required_options": tuple(
                    name
                    for name, parameter in parameters.items()
                    if name != "rec" and parameter.default is inspect.Parameter.empty
                ),
                "interpretation": entry.note,
            }
        )
    return pd.DataFrame(rows)


def run_method(name: str, recording: Recording, **options: Any) -> pd.DataFrame:
    """Run a named paper/protocol inventory and attach its scientific context.

    Parameters
    ----------
    name : str
        Exact function name from list_methods(). DOI-only dispatch is avoided
        because one paper can describe several distinct inventories.
    recording : Recording
        Measured or simulated inputs. Channel/cell selection belongs to callers.
    **options
        Named method options, including required settings absent from sources.

    Returns
    -------
    events : pandas.DataFrame
        At least start_time, end_time and duration (elapsed seconds), retaining
        any method-specific peak, channel or trigger columns. attrs records
        method, DOI, output role, interpretation and resolved method options.
        behavior_intervals, if supplied, retain only wholly contained events.
        The dispatcher does not change normalization. Explicit awake-frame
        methods also use these intervals to select their detection trace.

    Raises
    ------
    KeyError
        Unknown method name.
    TypeError
        Missing required method options or unknown keyword arguments.
    ValueError
        Missing or invalid recording inputs for the selected method.
    """
    entries = {entry.run.__name__: entry for entry in (*RECIPES, *VARIANTS)}
    if name not in entries:
        msg = f"Unknown literature method {name!r}; inspect list_methods()."
        raise KeyError(msg)
    entry = entries[name]
    call = inspect.signature(entry.run).bind(recording, **options)
    call.apply_defaults()
    raw = _IMPLEMENTATIONS[name](*call.args, **call.kwargs)
    result = (
        raw.copy()
        if isinstance(raw, pd.DataFrame)
        else pd.DataFrame(bounds(raw), columns=["start_time", "end_time"])
    )
    if recording.behavior_intervals is not None:
        keep = np.zeros(len(result), dtype=bool)
        for start, end in recording.behavior_intervals:
            keep |= (result.start_time.to_numpy() >= start) & (
                result.end_time.to_numpy() <= end
            )
        result = result.loc[keep].copy()
    if "duration" not in result:
        result["duration"] = result.end_time - result.start_time
    result.attrs.update(
        {
            "method": name,
            "doi": rd.load_literature_parameters().loc[entry.row, "DOI"],
            "output": entry.trigger,
            "role": entry.role,
            "inventory": "default" if entry in RECIPES else "additional",
            "interpretation": entry.note,
            "options": {key: value for key, value in call.arguments.items() if key != "rec"},
            "input_sampling_frequency": recording.fs,
            "behavior_intervals_applied": recording.behavior_intervals is not None,
        }
    )
    return result


__all__ = [
    "NOT_REPRODUCED",
    "RECIPES",
    "VARIANTS",
    "PopulationTrace",
    "Recipe",
    "RecordedSignals",
    "Recording",
    "bounds",
    "list_methods",
    "population_trace",
    "run_method",
    "within_duration",
    "within_intervals",
] + [entry.run.__name__ for entry in (*RECIPES, *VARIANTS)]
