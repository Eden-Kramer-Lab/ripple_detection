"""The multiunit high-synchrony-event detector."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    _is_immobile_at_endpoints,
    get_multiunit_population_firing_rate,
    nearest_sample_index,
)
from ripple_detection.detectors._blocks import (
    _valid_blocks,
)
from ripple_detection.detectors._events import (
    _count_active_units,
    _finish_events,
    _get_event_stats,
    _threshold_trace,
)
from ripple_detection.detectors._validation import (
    _check_finite_non_negative,
    _check_gap,
    _check_minimum_active_units,
    _check_smoothing_sigma,
    _validate_detector_inputs,
    _validate_duration_limits,
    _validate_multiunit,
)


@explain_call_errors
def multiunit_HSE_detector(
    time: ArrayLike,
    multiunit: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    *,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 2.0,
    smoothing_sigma: float = 0.015,
    close_event_threshold: float = 0.0,
    normalization_method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
    maximum_duration: float | None = None,
    minimum_active_units: int = 0,
) -> pd.DataFrame:
    """Detect High Synchrony Events from multiunit spiking activity.

    Identifies periods of elevated population spiking during immobility. The
    population firing rate, summed over units, is smoothed with a Gaussian
    kernel and z-scored. It is then thresholded with the same rules as the LFP
    detectors: at or above ``zscore_threshold`` for ``minimum_duration``, then
    extended to where the rate returns to the mean.

    The 15 ms smoothing kernel follows Davidson et al. 2009 [1]_. The
    selection rule does not. Davidson et al. define a candidate event as a
    period above the mean whose *peak* exceeds 3 s.d. They take the statistics
    from stopped periods only and impose no sustained-duration requirement.
    To approximate that convention, pass ``zscore_threshold=3.0``,
    ``minimum_duration=0.0`` and ``normalization_mask=speed < 5.0``, their
    stopped-period criterion. The defaults here, 2 s.d. held for 15 ms with
    statistics over all samples, are this package's own convention.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in **seconds**.
    multiunit : array_like, shape (n_time, n_units)
        Spikes per sample for each unit: indicators (0 or 1) or counts
        (0, 1, 2, ...). The two differ only in bins holding more than one
        spike, where counts weigh the bin by its spikes. Values must be
        non-negative whole numbers; a rate in Hz or a smoothed rate raises
        when it is not whole, and NaN marks a missing sample.
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point in **cm/s**.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (in cm/s) for event detection. An event is kept only if
        the speed at its first and last sample is at or below this value
        (``exclude_movement``); speed inside the event is not tested. Default
        is 4.0 cm/s, which corresponds to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, pass ``np.inf``, which also
        keeps events whose speed is unknown (NaN).
    minimum_duration : float, optional
        Minimum event duration in **seconds**. Default is 0.015 (15 milliseconds).
        The firing rate must stay at or above ``zscore_threshold`` for at least
        ``minimum_sample_count(time, minimum_duration)`` consecutive samples,
        rounded half up from the median timestamp step (23 at 1500 Hz and 15 ms);
        the event is then extended to the surrounding mean-crossings, so the reported
        ``duration`` is typically longer.
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    zscore_threshold : float, optional
        Detection sensitivity threshold in standard deviations above mean.
        Default is 2.0. Lower values (e.g., 1.5) detect more events but may
        include false positives. Higher values (e.g., 3.0) are more conservative.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in **seconds**.
        Default is 0.015 (15 ms, longer than ripple detectors for smoother
        population firing rate estimates).
    close_event_threshold : float, optional
        Minimum time in **seconds** between events. Events closer than this
        are excluded -- the later event is dropped, not merged. Default is 0.0
        (no exclusion). Set to 0.05-0.1 s to drop closely-spaced events.
    normalization_method : {'zscore', 'median_mad'}, optional
        Method for normalizing the firing rate. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains outliers.
        The median/MAD method is more resistant to extreme values.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask to specify which samples to use for computing normalization
        statistics. For example, use `speed <= speed_threshold` to compute
        statistics only during immobility.
        Default is None (use all data).

    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.
    minimum_active_units : int, optional
        Minimum number of units with at least one spike inside an event.
        Events with fewer are dropped, after the speed rule and before close
        events are excluded, so a dropped event suppresses no neighbour.
        Default is 0, which imposes no criterion. Published criteria are most often around five units.
        ``Carey_candidate_detector`` applies the same rule with its original's
        default of 5.
    Returns
    -------
    high_synchrony_events : pd.DataFrame
        DataFrame with detected events and comprehensive statistics (see
        Kay_ripple_detector for column descriptions), plus
        ``n_active_units``, the number of units with at least one spike
        inside the event.

        Returns empty DataFrame if no events detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 2.0 to 1.5)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your multiunit data shows synchronous spiking activity

    Notes
    -----
    Missing samples: a NaN or infinite value anywhere in ``multiunit`` marks
    that sample missing, as does a step in ``time`` larger than 1.5 times its
    median step. The population rate is smoothed within each contiguous block
    of valid samples, no event spans a gap, and an event cut off by one is
    flagged in ``clipped_start`` and ``clipped_end``. A block too short for an
    event of ``minimum_duration`` is treated as missing, with a warning, and
    no block left raises. A spike count that is absent rather than missing
    should be 0, not NaN. A NaN in ``speed`` is an
    unknown speed, not a missing sample: it splits no block, and an event
    whose first or last sample has unknown speed fails the speed criterion.

    The 15 ms smoothing is Davidson et al.'s; the other defaults (2 SD, a
    15 ms minimum, 4 cm/s) are this package's convention. Published
    multiunit-burst detectors in the same lineage use their own values
    (Davidson et al. 2009 among them), so set them explicitly when
    reproducing a paper.

    References
    ----------
    .. [1] Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009).
       Hippocampal replay of extended experience. Neuron, 63(4), 497-507.
       doi:10.1016/j.neuron.2009.07.027

    Examples
    --------
    >>> from ripple_detection import filter_ripple_band
    >>> from ripple_detection.simulate import simulate_session, simulate_time
    >>> time = simulate_time(45_000, 1500)  # 30 s at 1500 Hz
    >>> session = simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], random_state=0)
    >>> events = multiunit_HSE_detector(time, session.multiunit, session.speed, 1500)
    >>> "n_active_units" in events, bool(len(events))
    (True, True)

    """
    _validate_duration_limits(minimum_duration, maximum_duration)
    _check_finite_non_negative(zscore_threshold=zscore_threshold)
    _check_gap(close_event_threshold=close_event_threshold)
    _check_smoothing_sigma(smoothing_sigma=smoothing_sigma)
    multiunit = np.asarray(multiunit, dtype=float)
    _validate_multiunit(multiunit)
    _check_minimum_active_units(minimum_active_units, multiunit.shape[1])
    time, multiunit, speed = _validate_detector_inputs(
        time, multiunit, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, multiunit, minimum_duration=minimum_duration)

    firing_rate = np.full(len(time), np.nan)
    for start, stop in blocks:
        firing_rate[start:stop] = get_multiunit_population_firing_rate(
            multiunit[start:stop], sampling_frequency, smoothing_sigma
        )

    normalized, candidates = _threshold_trace(
        firing_rate,
        time,
        is_valid,
        blocks,
        minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold,
        normalization_method=normalization_method,
        normalization_mask=normalization_mask,
    )
    n_active = _count_active_units(
        multiunit,
        np.column_stack(
            [
                nearest_sample_index(time, candidates[:, 0]),
                nearest_sample_index(time, candidates[:, 1]),
            ]
        ),
    )
    keep = _is_immobile_at_endpoints(candidates, speed, time, speed_threshold) & (
        n_active >= minimum_active_units
    )
    event_times, kept = _finish_events(
        candidates, keep, time, close_event_threshold, maximum_duration
    )
    events = _get_event_stats(event_times, time, normalized, speed, minimum_duration, blocks)
    events["n_active_units"] = n_active[kept]
    return events
