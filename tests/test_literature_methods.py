"""Behavioral checks for packaged literature methods, beyond simulation coverage."""

import inspect
import re

import numpy as np
import pandas as pd
import pytest
from scipy.signal import filtfilt, remez

import ripple_detection as rd
from ripple_detection import literature_methods as lm

RIPPLE_TIMES = [3, 6, 9, 12, 15, 18]


def _measured_inputs(fs=1500, origin=0.0, ripple_duration=(0.08, 0.16)):
    """from_arrays inputs for 20 s of simulated signals, as measured data."""
    time = np.arange(int(20 * fs)) / fs
    session = rd.simulate_session(
        time,
        RIPPLE_TIMES,
        n_channels=3,
        n_units=20,
        baseline_rate=1,
        ripple_rate_gain=40,
        ripple_duration=ripple_duration,
        rng=21,
    )
    time = time + origin
    return {
        "time": time,
        "sampling_frequency": fs,
        "lfps": session.lfps,
        "multiunit": session.multiunit,
        "speed": np.zeros(len(time)),
        "sharp_wave_lfp": session.sharp_wave_lfp,
        "reference_lfp": np.zeros(len(time)),
        "place_cells": np.arange(15),
        "pyramidal": np.arange(20),
        "sleep_intervals": [[time[0], time[-1]]],
        "baseline_intervals": [[time[0], time[-1]]],
        "templates": [np.arange(10)],
    }


def _eligible_epochs(name, time):
    """behavior_intervals spanning the recording, for a method that needs them."""
    if "behavior_intervals" not in {need.input for need in lm._ENTRIES[name].requirements}:
        return {}
    return {"behavior_intervals": [[time[0], time[-1]]]}


@pytest.fixture(scope="module")
def measured():
    return lm.Recording.from_arrays(**_measured_inputs())


def test_measured_recording_copies_inputs_and_masks_artifacts():
    time = np.arange(100) / 1000
    lfp = np.ones((100, 2))
    spikes = np.zeros((100, 3))
    rec = lm.Recording.from_arrays(
        time,
        1000,
        lfps=lfp,
        multiunit=spikes,
        place_cells=[0, 2],
        artifact_intervals=[[0.02, 0.03]],
    )
    assert lfp.sum() == 200
    assert np.isnan(rec.session.lfps[20:31]).all()
    assert np.isnan(rec.multiunit[20:31]).all()
    np.testing.assert_array_equal(rec.place_cells, [True, False, True])
    with pytest.raises(ValueError, match="speed"):
        rec.speed  # noqa: B018 - the accessor raises
    with pytest.raises(ValueError, match="cell selection"):
        rec.counts(rec.pyramidal)
    with pytest.raises(ValueError, match="sleep_intervals"):
        rec.sleep(4, 1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"place_cells": [0.5]}, "Cell indices"),
        ({"place_cells": [-1]}, "Cell indices"),
        ({"place_cells": [3]}, "Cell indices"),
        ({"place_cells": [True]}, "Cell masks"),
        ({"sleep_intervals": [[0.03, 0.01]]}, "Intervals"),
        ({"sleep_intervals": [[0.01, 0.03], [0.02, 0.05]]}, "Intervals"),
        ({"speed": np.zeros(5)}, "one row"),
        ({"external_ripples": [[0.01, 0.02, 0.03]]}, "peaks"),
    ],
)
def test_recording_rejects_invalid_selections(overrides, message):
    with pytest.raises(ValueError, match=message):
        lm.Recording.from_arrays(
            np.arange(100) / 1000, 1000, multiunit=np.zeros((100, 3)), **overrides
        )


def test_native_bins_are_nonoverlapping_and_conserve_counts():
    time = np.arange(100) / 1000
    spikes = np.zeros((100, 2))
    spikes[[2, 9, 10, 18, 31], 0] = [1, 2, 3, 4, 5]
    rec = lm.Recording.from_arrays(time, 1000, multiunit=spikes)
    trace = lm.population_trace(rec, bin_width=0.01)
    np.testing.assert_allclose(trace.time, np.arange(10) * 0.01 + 0.005)
    np.testing.assert_allclose(trace.data * 0.01, [3, 7, 0, 5, 0, 0, 0, 0, 0, 0])
    assert trace.sampling_frequency == 100


@pytest.mark.parametrize("kind", ["nan", "inf", "timestamp"])
def test_native_bins_preserve_missing_support(kind):
    time = np.arange(1000) / 1000
    spikes = np.ones((1000, 1))
    if kind == "timestamp":
        time[500:] += 0.02
    else:
        spikes[500:520] = np.nan if kind == "nan" else np.inf
    rec = lm.Recording.from_arrays(time, 1000, multiunit=spikes)
    trace = lm.population_trace(rec, bin_width=0.01, smoothing_sigma=0.005)
    missing = (trace.time > 0.50) & (trace.time < 0.52)
    assert np.isnan(trace.data[missing]).all()
    assert np.isfinite(trace.data[trace.time < 0.49]).all()
    assert np.isfinite(trace.data[(trace.time > 0.53) & (trace.time < 0.99)]).all()


def test_mallory_merges_relative_to_retained_largest_peak():
    time = np.arange(400) / 1000
    z = np.full(400, -1.0)
    z[[100, 150, 210]] = [8, 4, 5]
    result = lm._mallory_candidates(time, z)
    # The second peak merges into the first. Its smaller peak must not pull
    # the third (60 ms later) into a cluster 110 ms from the retained peak.
    np.testing.assert_allclose(result.peak_time, [0.1, 0.21])
    np.testing.assert_allclose(lm.bounds(result), [[0.099, 0.151], [0.209, 0.211]])
    z[150] = 8
    tied = lm._mallory_candidates(time, z)
    assert len(tied) == 1
    assert tied.peak_time.iloc[0] == 0.15


def test_mallory_never_merges_across_a_missing_sample():
    time = np.arange(300) / 1000
    z = np.full(300, -1.0)
    z[[100, 150]] = 5
    z[125] = np.nan
    assert len(lm._mallory_candidates(time, z)) == 2


def test_tirole_boundaries_include_crossing_samples_and_fallbacks():
    time = np.arange(1000) / 1000
    z = np.full(1000, -1.0)
    z[400:601] = 0.1
    z[450:501] = 4
    np.testing.assert_allclose(lm.bounds(lm._tirole_bounds(time, z)), [[0.399, 0.601]])
    z[:] = 0.2
    z[450:501] = 4
    # No z<0 crossing exists; each side independently uses z<=0.25.
    np.testing.assert_allclose(lm.bounds(lm._tirole_bounds(time, z)), [[0.449, 0.501]])


def test_tirole_uses_finite_forward_backward_kernel(measured, monkeypatch):
    observed = []
    counts = np.zeros(1001)
    counts[500] = 1
    trace = lm.PopulationTrace(np.arange(1001) / 1000, counts, np.zeros(1001), 1000)
    monkeypatch.setattr(lm, "population_trace", lambda *args, **kwargs: trace)
    original_zscore = lm._zscore

    def capture(values, *args, **kwargs):
        observed.append(values.copy())
        return original_zscore(values, *args, **kwargs)

    monkeypatch.setattr(lm, "_zscore", capture)
    monkeypatch.setattr(
        lm,
        "_tirole_bounds",
        lambda *args: pd.DataFrame(
            columns=["start_time", "end_time", "clipped_start", "clipped_end"]
        ),
    )
    lm.tirole_2022(measured)
    # Released process_clusters.m: gausswin(41, 2), normalized, filtfilt.
    # Fixed interior impulse response computed independently with SciPy's
    # windows.gaussian(41, 10) and two direct convolutions. This is a
    # numerical reference to the stated rule, not an author-recording output.
    expected = [
        0.00003164857171567986,
        0.0018196887550036961,
        0.009720857095430696,
        0.02317874471117811,
        0.03051345412830434,
        0.02317874471117811,
        0.009720857095430696,
        0.0018196887550036961,
        0.00003164857171567986,
    ]
    np.testing.assert_allclose(observed[0][460:541:10], expected, rtol=1e-12)
    assert observed[0].sum() == pytest.approx(1)
    assert np.count_nonzero(observed[0]) == 81
    np.testing.assert_allclose(observed[0], observed[0][::-1], atol=1e-15)


def test_gridchyn_trailing_window_and_adaptation_are_observable():
    time = np.arange(4000) / 1000
    spikes = np.zeros((4000, 1))
    spikes[np.arange(0, 1000, 100)] = 1  # pre-rest expected 0.2 spikes per window
    spikes[[1100, 1700, 2300, 3100]] = 5
    rec = lm.Recording.from_arrays(
        time, 1000, multiunit=spikes, baseline_intervals=[[0, 0.999]]
    )
    events = lm.gridchyn_2020(rec, update_interval=1, target_rate=1)
    assert events.attrs["expected_count"] == pytest.approx(0.2)
    np.testing.assert_allclose(
        events.trigger_time, [0, 0.2, 0.4, 0.6, 0.8, 1.1, 1.7, 2.3, 3.1]
    )
    first = events.attrs["threshold_updates"][0]
    assert first["multiplier"] == pytest.approx(3.5 + 0.5 * (5 / 1.001 - 1))
    assert np.all(events.start_time <= events.trigger_time)
    assert np.all(events.end_time >= events.trigger_time)
    # A centered boxcar would trigger before a burst; trailing counts cannot.
    assert not np.any((events.trigger_time > 1.0) & (events.trigger_time < 1.1))


def test_peak_inventory_keeps_two_peaks_in_one_excursion(measured):
    trace = np.zeros(len(measured.time))
    trace[100:200] = 2
    trace[[125, 175]] = 4
    events = lm._local_peaks(measured, trace, 3)
    assert len(events) == 2
    np.testing.assert_allclose(events.peak_time, measured.time[[125, 175]])


def test_ji_ripple_merges_weak_neighbors_before_high_peak_gate(measured, monkeypatch):
    filtered = np.tile([-1.0, 1.0], len(measured.time) // 2)
    filtered[1000:1005] = 8
    filtered[1020:1025] = 4
    monkeypatch.setattr(measured, "filtered", lambda band: filtered[:, None])
    events = lm.ji_2007_ripples(measured)
    np.testing.assert_allclose(lm.bounds(events), measured.time[[1000, 1024]][None, :])


VARIANT_OPTIONS = {
    "gridchyn_2020_ripples": {"rms_window": 0.01, "bound_threshold": 1.5},
    "xu_2019_ripples": {"rms_window": 0.01, "bound_threshold": 1.5},
    "farooq_2019_neuron_ripples": {
        "threshold": 3.0,
        "bound_threshold": 0.0,
        "smoothing_sigma": 0.004,
    },
    "farooq_2019_science_ripples": {"power_measure": "hilbert", "bound_threshold": 0.0},
    "chenani_2019_hfe": {"ar_coefficients": np.zeros((3, 2))},
    "liu_2019_ripples": {"smoothing_sigma": 0.004},
    "liu_2019_ripple_frames": {"smoothing_sigma": 0.004},
    "drieu_2018_ripples": {"signal_measure": "amplitude"},
    "diba_2007_ripples": {"rms_window": 0.01},
}
RECIPE_OPTIONS = {
    "stella_2019": {"frequencies": [150, 200, 250], "cycles": 7},
    "nadasdy_1999": {"rms_window": 0.004, "bound_threshold": 0},
    "kudrimoti_1999": {"threshold_sd": 3},
    "wikenheiser_2013": {"window_anchor": "peaks"},
}
METHOD_RATES = {"bush_2022_ripples": 4800, "olafsdottir_2017_ripples": 1200}
ALL_METHODS = [entry.run.__name__ for entry in (*lm.RECIPES, *lm.VARIANTS)]


def _method_inputs(name, origin=0.0):
    """Every input a method needs on measured data, and its required options."""
    inputs = _measured_inputs(METHOD_RATES.get(name, 1500), origin)
    ripples = np.asarray(RIPPLE_TIMES, dtype=float) + origin
    # The last ripple's noise stretch would run past the recording's end.
    inputs["example_ripples"] = np.c_[ripples - 0.04, ripples + 0.04][:-1]
    inputs["external_ripples"] = np.c_[ripples - 0.04, ripples + 0.04, ripples]
    options = {**VARIANT_OPTIONS, **RECIPE_OPTIONS}.get(name, {})
    return inputs, options | _eligible_epochs(name, inputs["time"])


@pytest.mark.parametrize("name", ALL_METHODS)
def test_methods_without_speed_raise_or_do_not_use_it(name):
    """Absent speed must not read as unknown speed at every sample, which
    silently empties every speed-restricted inventory."""
    inputs, options = _method_inputs(name)
    n_time = len(inputs["time"])

    def run(rec):
        # Reported speed statistics describe the input, not the selection.
        try:
            events = lm.run_method(name, rec, **options)
        except ValueError as error:
            return str(error)
        return events.drop(columns=[c for c in events if "speed" in c])

    still, moving = (
        run(lm.Recording.from_arrays(**{**inputs, "speed": speed}))
        for speed in (np.zeros(n_time), np.full(n_time, 100.0))
    )
    assert isinstance(still, pd.DataFrame), still
    del inputs["speed"]
    without = run(lm.Recording.from_arrays(**inputs))
    if isinstance(without, str):
        assert "speed" in without, without
        # Only a method whose result depends on speed may refuse to run without
        # it; one finding nothing on this recording cannot show the dependence.
        assert isinstance(moving, str) or not still.equals(moving) or still.empty, name
        return
    pd.testing.assert_frame_equal(without, still)
    pd.testing.assert_frame_equal(without, moving)


# Rectified-LFP thresholds need a stronger ripple than the generic simulation.
RECTIFIED_LFP_METHODS = {"ji_2007_ripples", "lee_2002_ripples", "foster_2006_ripples"}
# Added inventories that find nothing on the measured fixture, and why.
EMPTY_ON_THE_FIXTURE = {
    # Its population bursts stay above 2 SD for at most 93 ms, short of 100 ms.
    "farooq_2019_science_awake",
}


def _with_strong_ripple(inputs, at=9.0):
    """A 180 Hz, 35 ms SD ripple of amplitude 50 added to the first channel."""
    lfps = inputs["lfps"].copy()
    relative = inputs["time"] - at
    lfps[:, 0] += (
        50 * np.cos(2 * np.pi * 180 * relative) * np.exp(-0.5 * (relative / 0.035) ** 2)
    )
    return {**inputs, "lfps": lfps}


@pytest.mark.parametrize("entry", lm.VARIANTS, ids=lambda x: x.run.__name__)
def test_every_added_inventory_runs_with_explicit_inputs(entry):
    name = entry.run.__name__
    inputs = _measured_inputs(METHOD_RATES.get(name, 1500))
    if name in RECTIFIED_LFP_METHODS:
        inputs = _with_strong_ripple(inputs)
    rec = lm.Recording.from_arrays(**inputs)
    options = VARIANT_OPTIONS.get(name, {}) | _eligible_epochs(name, rec.time)
    events = lm.run_method(name, rec, **options)
    assert isinstance(events, pd.DataFrame)
    assert bool(len(events)) is (name not in EMPTY_ON_THE_FIXTURE), name
    assert events.attrs["method"] == name
    assert events.attrs["doi"].startswith("https://doi.org/")
    assert (events.start_time <= events.end_time).all()
    assert (events.start_time >= rec.time[0]).all()
    assert (events.end_time <= rec.time[-1]).all()


def test_method_inventory_has_distinct_secondary_roles():
    inventory = lm.list_methods()
    assert inventory.name.is_unique
    expected = {0, 7, 16, 17, 21, 22, 23, 24, 26, 30, 32, 43, 45, 50, 51, 52, 53, 54}
    assert expected <= {entry.row for entry in lm.VARIANTS}
    assert inventory.loc[inventory.name == "xu_2019_ripples", "required_options"].item() == (
        "rms_window",
        "bound_threshold",
    )


def test_missing_settings_are_not_silently_fabricated(measured):
    with pytest.raises(TypeError, match="rms_window"):
        lm.run_method("xu_2019_ripples", measured)
    with pytest.raises(ValueError, match="threshold_sd"):
        lm.run_method("kudrimoti_1999", measured)
    with pytest.raises(ValueError, match="example_ripples"):
        lm.run_method("carey_2019", measured)
    with pytest.raises(ValueError, match="external_ripples"):
        lm.run_method("yang_2024", measured, behavior_intervals=[[0, 20]])
    with pytest.raises(KeyError, match="Unknown literature method"):
        lm.run_method("unrecognized", measured)


def test_run_method_applies_behavior_containment_and_preserves_context(measured):
    result = lm.run_method("pfeiffer_2013_ripples", measured, behavior_intervals=[[5, 10]])
    assert len(result)
    assert (result.start_time >= 5).all()
    assert (result.end_time <= 10).all()
    np.testing.assert_array_equal(result.attrs["behavior_intervals"], [[5, 10]])
    assert "behavior_intervals" not in result.attrs["options"]
    assert "duration limits" in result.attrs["interpretation"]
    unrestricted = lm.run_method("pfeiffer_2013_ripples", measured)
    assert unrestricted.attrs["behavior_intervals"] is None
    assert len(unrestricted) > len(result)


def test_behavior_intervals_belong_to_a_call_not_to_the_recording(measured):
    """Epochs mean different things per method (reward zones, rest, track
    ends), so one recording serves every method and each call says its own."""
    assert not hasattr(measured, "behavior_intervals")
    with pytest.raises(TypeError, match="behavior_intervals"):
        lm.Recording.from_arrays(**_measured_inputs(), behavior_intervals=[[0, 1]])
    zones = [[5.0, 10.0]]
    chenani = lm.chenani_2019(measured, behavior_intervals=zones)
    assert ((chenani.start_time >= 5) & (chenani.end_time <= 10)).all()
    # The next method on the same recording is not restricted by that call.
    karlsson = lm.karlsson_2009(measured)
    assert ((karlsson.start_time < 5) | (karlsson.end_time > 10)).any()
    parameter = inspect.signature(lm.karlsson_2009).parameters["behavior_intervals"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    with pytest.raises(ValueError, match="Intervals"):
        lm.run_method("karlsson_2009", measured, behavior_intervals=[0, 1])


@pytest.mark.parametrize(
    "name",
    [
        "mallory_2025_ripples",
        "igata_2021_ripples",
        "gridchyn_2020_ripples",
        "xu_2019_ripples",
        "farooq_2019_neuron_ripples",
        "farooq_2019_science_ripples",
        "chenani_2019_hfe",
        "liu_2019_ripples",
        "drieu_2018_ripples",
        "wu_2014_ripples",
        "pfeiffer_2013_ripples",
        "davidson_2009_ripples",
        "diba_2007_ripples",
        "ji_2007_ripples",
        "lee_2002_ripples",
        "foster_2006_ripples",
        "muessig_2019_ripples",
        "bhattarai_2020_ripples",
    ],
)
def test_secondary_ripples_do_not_bridge_artifact_intervals(measured, name):
    inputs = {
        "time": measured.time,
        "sampling_frequency": measured.fs,
        "lfps": measured.session.lfps,
        "multiunit": measured.multiunit,
        "speed": measured.speed,
        "reference_lfp": measured.reference_lfp,
        "place_cells": measured.place_cells,
        "pyramidal": measured.pyramidal,
        "sleep_intervals": measured.sleep_intervals,
        "baseline_intervals": measured.baseline_intervals,
    }
    if name in RECTIFIED_LFP_METHODS:
        inputs = _with_strong_ripple(inputs)
    clean = lm.Recording.from_arrays(**inputs)
    rec = lm.Recording.from_arrays(**inputs, artifact_intervals=[[8.99, 9.01]])
    before = lm.run_method(name, clean, **VARIANT_OPTIONS.get(name, {}))
    events = lm.run_method(name, rec, **VARIANT_OPTIONS.get(name, {}))
    assert ((before.start_time < 9.01) & (before.end_time > 8.99)).any()
    assert not ((events.start_time < 9.01) & (events.end_time > 8.99)).any()


def test_historical_denovellis_trace_squares_filtered_lfp(measured, monkeypatch):
    captured = []

    def capture(rec, trace, **kwargs):
        captured.append(trace)
        return pd.DataFrame(columns=["start_time", "end_time"])

    monkeypatch.setattr(lm, "_ripple_trace_events", capture)
    lm.denovellis_2021(measured)
    # Independent expected source expression; the Hilbert-envelope alternative
    # would remove the within-cycle oscillation before smoothing.
    from scipy.ndimage import gaussian_filter1d

    kernel = remez(101, [0, 125, 150, 250, 275, 750], [0, 1, 0], fs=1500)
    filtered = filtfilt(kernel, [1.0], measured.session.lfps, axis=0)
    power = np.sum(filtered**2, axis=1)
    kernel_sd = 0.004 * measured.fs
    expected = np.sqrt(
        gaussian_filter1d(power, kernel_sd, truncate=8, mode="constant")
        / gaussian_filter1d(np.ones_like(power), kernel_sd, truncate=8, mode="constant")
    )
    np.testing.assert_allclose(captured[0], expected)


def test_kaefer_uses_reference_subtraction_and_20ms_grid(measured, monkeypatch):
    captured = []

    def capture(time, trace, speed, fs, **kwargs):
        captured.append((time, trace, fs))
        return pd.DataFrame(columns=["start_time", "end_time"])

    monkeypatch.setattr(lm.rd, "detect_events_from_trace", capture)
    rec = lm.Recording.from_arrays(
        measured.time,
        measured.fs,
        lfps=measured.session.lfps[:, 0],
        reference_lfp=measured.session.lfps[:, 0],
        baseline_intervals=[[0, 19]],
    )
    lm.kaefer_2020(rec)
    time, rms, fs = captured[0]
    assert fs == 50
    np.testing.assert_allclose(np.diff(time), 0.02)
    np.testing.assert_allclose(rms, 0)


def test_template_inventory_uses_caller_ensemble_size():
    time = np.arange(2000) / 1000
    spikes = np.zeros((2000, 8))
    spikes[[500, 510, 520], [1, 3, 6]] = 1
    rec = lm.Recording.from_arrays(time, 1000, multiunit=spikes, templates=[[1, 3, 6]])
    rest = [[0, 2]]
    events = lm.run_method("olafsdottir_2015", rec, behavior_intervals=rest)
    np.testing.assert_allclose(lm.bounds(events), [[0.5, 0.52]])
    with pytest.warns(UserWarning, match="fewer than 7 cells"):
        assert not len(
            lm.run_method(
                "olafsdottir_2015", rec, minimum_active_units=7, behavior_intervals=rest
            )
        )


def test_native_bins_are_stable_with_unix_timestamps():
    relative = np.arange(3000) / 1500
    spikes = np.zeros((3000, 1))
    spikes[np.arange(0, 3000, 15)] = 1
    traces = []
    for origin in (0, 1_700_000_000):
        rec = lm.Recording.from_arrays(relative + origin, 1500, multiunit=spikes)
        traces.append(lm.population_trace(rec, bin_width=0.01))
    np.testing.assert_allclose(traces[0].data, traces[1].data)
    np.testing.assert_allclose(traces[0].time, traces[1].time - 1_700_000_000, atol=1e-6)


@pytest.mark.parametrize(
    ("method", "width"), [("maboudi_2018", 121), ("berners_lee_2022", 100)]
)
def test_finite_population_kernels_have_reported_support(monkeypatch, method, width):
    time = np.arange(4000) / 1000
    spikes = np.zeros((4000, 5))
    spikes[2000] = 1
    rec = lm.Recording.from_arrays(
        time, 1000, multiunit=spikes, speed=np.zeros(len(time)), pyramidal=np.arange(5)
    )
    observed = []

    def capture(trace, **kwargs):
        observed.append(trace.data.copy())
        return pd.DataFrame(columns=["start_time", "end_time"])

    monkeypatch.setattr(lm.PopulationTrace, "detect", capture)
    getattr(lm, method)(rec)
    data = observed[0]
    # For Berners-Lee, normalization moves zero to the background z value.
    baseline = np.nanmin(data)
    changed = np.flatnonzero(data > baseline + 1e-9)
    assert len(changed) == width
    assert np.all(np.diff(changed) == 1)


def test_gridchyn_trailing_window_excludes_left_boundary_but_keeps_interior_spikes():
    time = np.arange(2000) / 1000
    spikes = np.zeros((2000, 1))
    spikes[np.arange(0, 1000, 100)] = 1
    spikes[[1200, 1220]] = 1
    rec = lm.Recording.from_arrays(
        time, 1000, multiunit=spikes, baseline_intervals=[[0, 0.999]]
    )
    result = lm.gridchyn_2020(rec, initial_multiplier=8)
    # 0.2 expected * 8 = 1.6: neither single spike is enough.
    assert not len(result)
    rec.session.multiunit[1220] = 0
    rec.session.multiunit[1219] = 1
    result = lm.gridchyn_2020(rec, initial_multiplier=8)
    np.testing.assert_allclose(result.trigger_time, [1.219])


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("huelin_gorriz_2023", {"interpretation": "related_code"}),
        ("mou_2022", {"normalization": "maximum"}),
        ("michon_2019", {"order": "code"}),
        ("krause_2022_hse", {"interpretation": "code"}),
        ("stella_2019", {"frequencies": [150, 200, 250], "cycles": 7}),
        ("muessig_2019", {"trial": "run"}),
        ("wikenheiser_2013", {"branch": "run_lia", "window_anchor": "peaks"}),
        ("wikenheiser_2013", {"window_anchor": "onsets"}),
        ("nadasdy_1999", {"rms_window": 0.004, "bound_threshold": 0}),
        ("kudrimoti_1999", {"threshold_sd": 3}),
        ("bhattarai_2020", {"power_measure": "hilbert", "window_end_rule": "fixed"}),
        ("drieu_2018_ripples", {"signal_measure": "power"}),
        (
            "farooq_2019_science_ripples",
            {"power_measure": "squared_signal", "bound_threshold": 0},
        ),
    ],
)
def test_named_interpretations_accept_measured_inputs(measured, name, options):
    options = options.copy()
    if options.get("branch") == "run_lia":
        options["theta_delta"] = -np.ones_like(measured.time)
    events = lm.run_method(name, measured, **options)
    for key, value in options.items():
        np.testing.assert_equal(events.attrs["options"][key], value)
    assert (events.start_time <= events.end_time).all()


def test_analysis_participation_does_not_change_initial_detection(monkeypatch):
    time = np.arange(3000) / 1000
    spikes = np.zeros((3000, 20))
    spikes[550, :5] = 1
    spikes[1550, :6] = 1
    rec = lm.Recording.from_arrays(
        time,
        1000,
        multiunit=spikes,
        place_cells=np.arange(20),
        pyramidal=np.arange(20),
        sleep_intervals=[[0, 2.999]],
        external_ripples=[[0.5, 0.6]],  # the candidates are replaced below
        speed=np.zeros(len(time)),
    )
    eligible = {"behavior_intervals": [[0, 2.999]]}
    candidates = np.array([[0.5, 0.575], [1.5, 1.7]])
    monkeypatch.setattr(lm, "_population_with_ripple_peak", lambda *args: candidates.copy())
    np.testing.assert_allclose(lm.bounds(lm.grosmark_2016(rec, **eligible)), candidates)
    np.testing.assert_allclose(
        lm.bounds(lm.grosmark_2016(rec, stage="decoding_candidates", **eligible)),
        candidates[1:],
    )
    frame = pd.DataFrame(candidates, columns=["start_time", "end_time"])
    monkeypatch.setattr(lm, "_detect_population", lambda *args, **kwargs: frame.copy())
    np.testing.assert_allclose(lm.bounds(lm.olafsdottir_2017(rec, **eligible)), candidates)
    np.testing.assert_allclose(
        lm.bounds(lm.olafsdottir_2017(rec, analysis="trajectory", **eligible)),
        candidates[1:],
    )


def test_krause_trims_each_swr_without_spikes_from_its_neighbors():
    time = np.arange(2000) / 1000
    spikes = np.zeros((2000, 10))
    spikes[[497, 652]] = 100
    rec = lm.Recording.from_arrays(
        time, 1000, multiunit=spikes, place_cells=np.arange(10), external_ripples=[[0.5, 0.65]]
    )
    # Smoothing over the whole recording leaks both exterior bursts into the
    # SWR and can create a spurious first-to-last-crossing population burst.
    assert not len(lm.krause_2022(rec))


@pytest.mark.parametrize("origin", [0.0, 86400.0, 1_700_000_000.0])
def test_duration_limits_keep_boundary_events_at_any_clock_origin(origin):
    starts = np.arange(272) * 0.1 + origin
    exact = np.c_[starts, starts + 0.04]
    np.testing.assert_array_equal(lm.within_duration(exact, 0.04, 0.04), exact)
    short = exact.copy()
    short[:, 1] -= 0.001
    long = exact.copy()
    long[:, 1] += 0.001
    assert not len(lm.within_duration(short, 0.04))
    assert not len(lm.within_duration(long, high=0.04))


@pytest.mark.parametrize("origin", [0.0, 1_700_000_000.0])
def test_mallory_seventy_ms_boundary_is_clock_invariant(origin):
    time = np.arange(1000) / 1000 + origin
    trace = np.full(1000, -1.0)
    trace[[100, 170, 241]] = [4, 5, 6]
    events = lm._mallory_candidates(time, trace)
    np.testing.assert_allclose(events.peak_time - origin, [0.17, 0.241], atol=1e-6)
    np.testing.assert_allclose(
        lm.bounds(events) - origin, [[0.099, 0.171], [0.24, 0.242]], atol=1e-6
    )


@pytest.mark.parametrize("origin", [0.0, 1_700_000_000.0])
def test_tirole_ten_ms_anchors_and_search_bounds_are_clock_invariant(origin):
    time = np.arange(1200) / 1000 + origin
    z = np.full(1200, -1.0)
    z[400:721] = 0.6
    z[[400, 410]] = 4
    events = lm.bounds(lm._tirole_bounds(time, z))
    np.testing.assert_allclose(events - origin, [[0.399, 0.700], [0.399, 0.710]], atol=1e-6)


@pytest.mark.parametrize("origin", [0.0, 1_700_000_000.0])
@pytest.mark.parametrize(
    ("swr_end", "expected_end"), [(0.650, 0.629), (0.651, 0.627), (0.652, 0.628)]
)
def test_krause_bounds_match_released_bin_helper_and_trimming(origin, swr_end, expected_end):
    time = np.arange(4000) / 2000 + origin
    spikes = np.zeros((4000, 10))
    spikes[np.arange(1061, 1242, 6)] = 1
    rec = lm.Recording.from_arrays(
        time,
        2000,
        multiunit=spikes,
        place_cells=np.arange(10),
        external_ripples=np.array([[0.5, swr_end]]) + origin,
    )
    result = lm.krause_2022(rec)
    assert len(result) == 1
    # Reference outputs executed from get_spikemat and select_population_burst
    # at DrugowitschLab/HippocampalSWRDynamics cda23b7. The first case has
    # 49 bins (right edge strictly before SWR end); the others have 50.
    # Released bounds retain the unbinned remainder at the end, deliberately.
    np.testing.assert_allclose(lm.bounds(result) - origin, [[0.527, expected_end]], atol=1e-6)


@pytest.mark.parametrize("origin", [0.0, 1_700_000_000.0])
def test_gridchyn_uses_twenty_samples_for_a_twenty_ms_window(origin):
    time = np.arange(1000) / 1000 + origin
    rec = lm.Recording.from_arrays(
        time,
        1000,
        multiunit=np.ones((1000, 1)),
        baseline_intervals=[[time[0], time[-1]]],
    )
    # One count per millisecond is exactly the baseline, never 1.025x it.
    assert not len(lm.gridchyn_2020(rec, initial_multiplier=1.025, gain=0))
    at_baseline = lm.gridchyn_2020(rec, initial_multiplier=1, gain=0)
    assert len(at_baseline)
    assert at_baseline.trigger_time.iloc[0] - origin == pytest.approx(0.019, abs=1e-6)
    assert at_baseline.attrs["expected_count"] == 20


@pytest.mark.parametrize(
    "name",
    [
        "denovellis_2021",
        "xu_2019",
        "wu_2014",
        "bendor_2012",
        "chenani_2019",
        "olafsdottir_2016",
        "mallory_2025",
        "mallory_2025_ripples",
        "wikenheiser_2013",
    ],
)
def test_session_normalization_ignores_unrelated_baseline_intervals(measured, name):
    from dataclasses import replace

    full = replace(measured, baseline_intervals=None)
    restricted = replace(measured, baseline_intervals=np.array([[0, 1.5]]))
    options = {"window_anchor": "peaks"} if name == "wikenheiser_2013" else {}
    options |= _eligible_epochs(name, measured.time)
    expected = lm.run_method(name, full, **options)
    assert len(expected), name
    actual = lm.run_method(name, restricted, **options)
    pd.testing.assert_frame_equal(actual, expected)


def test_krause_text_baseline_changes_detection_but_code_baseline_does_not(measured):
    from dataclasses import replace

    full = replace(measured, baseline_intervals=None)
    restricted = replace(measured, baseline_intervals=np.array([[0, 1.5]]))
    text_full = lm.krause_2022_hse(full)
    text_restricted = lm.krause_2022_hse(restricted)
    assert len(text_full)
    assert len(text_restricted)
    assert not text_full.equals(text_restricted)
    pd.testing.assert_frame_equal(
        lm.krause_2022_hse(full, interpretation="code"),
        lm.krause_2022_hse(restricted, interpretation="code"),
    )


def test_recording_transforms_are_fresh_and_do_not_keep_recordings_alive():
    import gc
    import weakref

    time = np.arange(3000) / 1500
    rec = lm.Recording.from_arrays(
        time, 1500, lfps=np.random.default_rng(3).normal(size=(3000, 2))
    )
    before = rec.envelope((150, 250))
    rec.session.lfps *= 2
    np.testing.assert_allclose(rec.envelope((150, 250)), before * 2)
    rec.ratio()
    reference = weakref.ref(rec)
    del rec
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    ("name", "options", "message"),
    [
        ("stella_2019", {}, "frequencies"),
        ("nadasdy_1999", {}, "rms_window"),
        ("kudrimoti_1999", {}, "threshold_sd"),
        ("carey_2019", {}, "example_ripples"),
        ("yang_2024", {"behavior_intervals": [[0, 20]]}, "external_ripples"),
        ("chenani_2019", {}, "behavior_intervals"),
    ],
)
def test_arbitrary_session_objects_cannot_enable_simulation_fallbacks(
    measured, name, options, message
):
    from dataclasses import replace
    from types import SimpleNamespace

    rec = replace(measured, session=SimpleNamespace(**vars(measured.session)))
    with pytest.raises(ValueError, match=message):
        lm.run_method(name, rec, **options)
    rec.sleep_intervals = None
    with pytest.raises(ValueError, match="sleep_intervals"):
        rec.sleep(4, 1)


@pytest.mark.parametrize("name", ["pfeiffer_2015", "bendor_2012", "mallory_2025_ripples"])
def test_direct_methods_and_registry_apply_identical_behavior_filters(measured, name):
    unrestricted = lm.run_method(name, measured)
    assert len(unrestricted) > 1
    direct = getattr(lm, name)(measured, behavior_intervals=[[5, 10]])
    dispatched = lm.run_method(name, measured, behavior_intervals=[[5, 10]])
    pd.testing.assert_frame_equal(direct, dispatched)
    np.testing.assert_equal(direct.attrs, dispatched.attrs)
    assert 0 < len(direct) < len(unrestricted)
    assert (direct.start_time >= 5).all()
    assert (direct.end_time <= 10).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"time": [0]}, "at least two"),
        ({"time": [[0, 1]]}, "one-dimensional"),
        ({"time": [0, np.nan]}, "finite"),
        ({"time": [0.001, 0]}, "increasing"),
        ({"time": [0, 0]}, "increasing"),
        ({"sampling_frequency": 0}, "positive and finite"),
        ({"sampling_frequency": np.inf}, "positive and finite"),
        ({"sampling_frequency": np.nan}, "positive and finite"),
        ({"sampling_frequency": 20}, "timestamp spacing"),
        ({"lfps": np.zeros((100, 1, 1))}, "one row"),
        ({"multiunit": np.full((100, 1), -1)}, "nonnegative integer"),
        ({"multiunit": np.full((100, 1), 0.5)}, "nonnegative integer"),
        ({"place_cells": [[0]]}, "Cell indices"),
        ({"baseline_intervals": [[0, np.nan]]}, "Intervals"),
        ({"external_ripples": [0, 1]}, "start/end"),
        ({"external_ripples": [[0, 1, 0.5, 2]]}, "start/end"),
        ({"external_ripples": [[0, 1, np.nan]]}, "peaks"),
    ],
)
def test_recording_constructor_error_paths(overrides, message):
    options = {
        "time": np.arange(100) / 1000,
        "sampling_frequency": 1000,
        "multiunit": np.zeros((100, 3)),
    }
    options.update(overrides)
    with pytest.raises(ValueError, match=message):
        lm.Recording.from_arrays(**options)


@pytest.mark.parametrize(
    ("name", "selected"),
    [
        ("mou_2022", [1, 2]),
        ("wu_2017", [1, 2]),
        ("ji_2007", [1, 2]),
        ("shin_2019", [2]),
        ("carr_2012", [2]),
        ("jadhav_2016", [1, 2]),
        ("drieu_2018", [2]),
    ],
)
def test_detection_and_decoding_stages_have_distinct_participation_rules(
    monkeypatch, name, selected
):
    time = np.arange(4000) / 1000
    spikes = np.zeros((4000, 5))
    for sample, count in [(520, 3), (1520, 4), (2520, 5)]:
        spikes[sample, :count] = 1
    rec = lm.Recording.from_arrays(
        time,
        1000,
        lfps=np.zeros(len(time)),  # the detections are replaced below
        multiunit=spikes,
        place_cells=np.arange(5),
        speed=np.zeros(len(time)),
        sleep_intervals=[[0, time[-1]]],
    )
    candidates = np.array([[0.5, 0.55], [1.5, 1.56], [2.5, 2.6]])
    frame = pd.DataFrame(candidates, columns=["start_time", "end_time"])
    monkeypatch.setattr(lm, "_karlsson_rule", lambda *args: frame.copy())
    monkeypatch.setattr(lm, "_detect_population", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(lm, "_drieu_events", lambda *args: frame.copy())
    monkeypatch.setattr(lm.PopulationTrace, "detect", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(lm.rd, "histogram_minimum_threshold", lambda *args, **kwargs: 0.5)
    np.testing.assert_allclose(lm.bounds(lm.run_method(name, rec)), candidates)
    np.testing.assert_allclose(
        lm.bounds(lm.run_method(name, rec, stage="decoding_candidates")), candidates[selected]
    )
    with pytest.raises(ValueError, match="stage"):
        lm.run_method(name, rec, stage="invalid")


def test_harvey_code_uses_detectswr_and_replay_filters_are_separate(monkeypatch):
    time = np.arange(4000) / 1000
    spikes = np.zeros((4000, 5))
    spikes[510:550:20] = 1  # too short
    spikes[1010:1100:20, :4] = 1  # too few cells
    spikes[1510] = 1  # too many empty bins
    spikes[[2010, 2030, 2050]] = 1  # passes
    spikes[[2510, 2530, 2550]] = 1  # exactly half of six bins empty
    spikes[3010:3100:20, :4] = 1
    spikes[3105, 4] = 1  # fifth cell only in the discarded partial final bin
    rec = lm.Recording.from_arrays(
        time,
        1000,
        lfps=np.zeros(len(time)),
        sharp_wave_lfp=np.ones(len(time)),
        multiunit=spikes,
        speed=np.zeros(len(time)),
        place_cells=np.arange(5),
        pyramidal=np.arange(5),
    )
    candidates = np.array(
        [[0.5, 0.56], [1, 1.1], [1.5, 1.6], [2, 2.1], [2.5, 2.62], [3, 3.11]]
    )
    frame = pd.DataFrame(candidates, columns=["start_time", "end_time"])
    called = []

    def detectswr(time, raw, speed, fs, **kwargs):
        called.append(kwargs)
        np.testing.assert_array_equal(raw, rec.session.raw_lfp)
        np.testing.assert_array_equal(kwargs["sharp_wave_lfp"], rec.session.sharp_wave_lfp)
        return frame.copy()

    monkeypatch.setattr(lm.rd, "Long_sharp_wave_ripple_detector", detectswr)
    monkeypatch.setattr(lm, "_spiking_filter", lambda rec, events: events)
    np.testing.assert_allclose(lm.bounds(lm.harvey_2023_code(rec)), candidates)
    np.testing.assert_allclose(
        lm.bounds(lm.harvey_2023_code(rec, stage="decoding_candidates")), candidates[[3]]
    )
    assert len(called) == 2
    assert called[0]["speed_threshold"] == np.inf


def test_gupta_log_transform_is_explicit_and_changes_the_gate(measured, monkeypatch):
    amplitude = np.ones(len(measured.time))
    amplitude[1000:1200] = 3
    amplitude[15000:15020] = 100
    monkeypatch.setattr(measured, "mean_envelope", lambda *args, **kwargs: amplitude)
    logged = lm.gupta_2010(measured)
    linear = lm.gupta_2010(measured, log_amplitude=False)
    assert len(logged) == 2
    assert len(linear) == 1
    assert linear.start_time.iloc[0] == measured.time[15000]


def test_kaefer_detects_a_burst_on_the_reference_subtracted_fft_grid():
    time = np.arange(15000) / 1500
    rng = np.random.default_rng(11)
    reference = rng.normal(size=len(time))
    background = rng.normal(scale=0.1, size=(len(time), 2))
    ripple = 5 * np.cos(2 * np.pi * 180 * time) * ((time >= 4) & (time < 4.16))
    inputs = {
        "time": time,
        "sampling_frequency": 1500,
        "reference_lfp": reference,
        "baseline_intervals": [[0, 2]],
    }
    positive = lm.Recording.from_arrays(
        **inputs, lfps=reference[:, None] + background + ripple[:, None]
    )
    negative = lm.Recording.from_arrays(**inputs, lfps=reference[:, None] + background)
    events = lm.kaefer_2020(positive)
    assert ((events.start_time < 4.16) & (events.end_time > 4)).any()
    control = lm.kaefer_2020(negative)
    assert not ((control.start_time < 4.16) & (control.end_time > 4)).any()
    # Window centers are 120 ms from the origin, then every 20 ms.
    edges = lm.bounds(events)
    np.testing.assert_allclose(
        (edges - 0.12) / 0.02, np.rint((edges - 0.12) / 0.02), atol=1e-9
    )


def test_published_huelin_cap_changes_the_returned_inventory(measured, monkeypatch):
    candidates = np.array([[1, 1.6], [3, 3.9]])
    monkeypatch.setattr(lm, "_tirole", lambda *args: candidates.copy())
    np.testing.assert_allclose(lm.bounds(lm.huelin_gorriz_2023(measured)), candidates[:1])
    np.testing.assert_allclose(
        lm.bounds(lm.huelin_gorriz_2023(measured, interpretation="related_code")), candidates
    )


def test_mou_scaling_options_change_bounds_when_background_is_nonzero(monkeypatch):
    time = np.arange(1000) / 1000
    rec = lm.Recording.from_arrays(time, 1000, multiunit=np.ones((1000, 5)))
    rate = np.full(100, 2.0)
    rate[40:60] = 10

    def trace(*args, **kwargs):
        return lm.PopulationTrace(np.arange(100) / 100, rate.copy(), np.zeros(100), 100)

    monkeypatch.setattr(lm, "population_trace", trace)
    # A trace built by hand has no recorded samples per bin, so its events
    # are reported at the centers of the bins centered on 0.40-0.59 s.
    np.testing.assert_allclose(lm.bounds(lm.mou_2022(rec)), [[0.40, 0.59]])
    np.testing.assert_allclose(
        lm.bounds(lm.mou_2022(rec, normalization="maximum")), [[0.0, 0.99]]
    )


@pytest.mark.parametrize(
    ("name", "first", "second"),
    [
        (
            "bhattarai_2020_ripples",
            {"power_measure": "squared_signal"},
            {"power_measure": "hilbert"},
        ),
        ("drieu_2018_ripples", {"signal_measure": "amplitude"}, {"signal_measure": "power"}),
        (
            "farooq_2019_science_ripples",
            {"power_measure": "hilbert", "bound_threshold": 0},
            {"power_measure": "squared_signal", "bound_threshold": 0},
        ),
        ("wikenheiser_2013", {"window_anchor": "peaks"}, {"window_anchor": "onsets"}),
        (
            "nadasdy_1999",
            {"rms_window": 0.004, "bound_threshold": 0},
            {"rms_window": 0.02, "bound_threshold": 0},
        ),
    ],
)
def test_interpretation_options_change_observable_boundaries(measured, name, first, second):
    a = lm.run_method(name, measured, **first)
    b = lm.run_method(name, measured, **second)
    assert len(a), name
    # Compare bounds rather than attrs, which necessarily contain different options.
    assert not np.array_equal(lm.bounds(a), lm.bounds(b)), name


def test_muessig_optional_sample_veto_uses_trial_speed_limits(monkeypatch):
    time = np.arange(4000) / 1000
    spikes = np.zeros((4000, 10))
    spikes[1000:1200] = 1
    rec = lm.Recording.from_arrays(
        time,
        1000,
        multiunit=spikes,
        speed=np.full(len(time), 1.5),
        pyramidal=np.arange(10),
        lfps=np.zeros(len(time)),  # the ripple windows are replaced below
        sleep_intervals=[[0, time[-1]]],
    )
    monkeypatch.setitem(
        lm._IMPLEMENTATIONS,
        "muessig_2019_ripples",
        lambda *args: np.array([[0, rec.time[-1]]]),
    )
    rest = lm.muessig_2019(rec, trial="rest", sample_speed_veto=True)
    assert len(rest) == 1
    assert rest.start_time.iloc[0] < 1.05
    assert rest.end_time.iloc[0] > 1.15
    assert not len(lm.muessig_2019(rec, trial="run", sample_speed_veto=True))


@pytest.mark.parametrize("trial", ["rest", "run"])
@pytest.mark.parametrize("brief_speed", [10.0, np.nan])
def test_muessig_curated_state_accepts_brief_speed_excursions(monkeypatch, trial, brief_speed):
    from dataclasses import replace

    time = np.arange(8000) / 1000
    spikes = np.zeros((len(time), 10))
    spikes[1000:1200] = spikes[5000:5200] = 1
    speed = np.full(len(time), 0.5)
    speed[1070:1074] = brief_speed
    # A brief excursion does not disqualify this 1.6 s state window.
    assert np.nanmean(speed[:1600]) < 1
    rec = lm.Recording.from_arrays(
        time,
        1000,
        multiunit=spikes,
        speed=speed,
        pyramidal=np.arange(10),
        lfps=np.zeros(len(time)),  # the ripple windows are replaced below
        sleep_intervals=[[0, 1.599]],
    )
    monkeypatch.setitem(
        lm._IMPLEMENTATIONS,
        "muessig_2019_ripples",
        lambda *args: np.array([[0.9, 1.3], [4.9, 5.3]]),
    )
    events = lm.muessig_2019(rec, trial=trial)
    assert len(events) == 1  # The otherwise identical second burst is outside rest.
    assert events.start_time.iloc[0] < 1.07 < events.end_time.iloc[0]
    assert not len(lm.muessig_2019(rec, trial=trial, sample_speed_veto=True))
    # Entire events must fit the supplied state; crossing its end is excluded.
    clipped_state = replace(rec, sleep_intervals=np.array([[0, 1.1]]))
    assert not len(lm.muessig_2019(clipped_state, trial=trial))
    with pytest.raises(ValueError, match="sleep_intervals"):
        lm.muessig_2019(replace(rec, sleep_intervals=None), trial=trial)


def test_wikenheiser_run_lia_requires_low_theta_delta(measured):
    positive = lm.wikenheiser_2013(
        measured,
        branch="run_lia",
        window_anchor="peaks",
        theta_delta=-np.ones(len(measured.time)),
    )
    negative = lm.wikenheiser_2013(
        measured,
        branch="run_lia",
        window_anchor="peaks",
        theta_delta=np.ones(len(measured.time)),
    )
    assert len(positive)
    assert not len(negative)


def test_bhattarai_window_policy_changes_candidate_end(monkeypatch):
    time = np.arange(2000) / 1000
    spikes = np.zeros((len(time), 5))
    spikes[[500, 520, 540, 560, 580], np.arange(5)] = 1
    rec = lm.Recording.from_arrays(
        time,
        1000,
        lfps=np.zeros((len(time), 2)),  # the ripples are replaced below
        multiunit=spikes,
        place_cells=np.arange(5),
    )
    monkeypatch.setitem(
        lm._IMPLEMENTATIONS,
        "bhattarai_2020_ripples",
        lambda *args, **kwargs: np.array([[0.5, 0.8]]),
    )
    np.testing.assert_allclose(lm.bounds(lm.bhattarai_2020(rec)), [[0.5, 0.58]])
    np.testing.assert_allclose(
        lm.bounds(lm.bhattarai_2020(rec, window_end_rule="fixed")), [[0.5, 0.8]]
    )


def test_michon_order_changes_event_bounds():
    time = np.arange(45 * 1500) / 1500
    session = rd.simulate_session(
        time,
        [10, 25, 40],
        n_channels=3,
        n_units=20,
        baseline_rate=1,
        ripple_rate_gain=40,
        ripple_snr=8,
        ripple_duration=0.16,
        rng=21,
    )
    rec = lm.Recording.from_arrays(
        time, 1500, lfps=session.lfps, multiunit=session.multiunit, speed=np.zeros(len(time))
    )
    text = lm.michon_2021(rec, order="text")
    code = lm.michon_2021(rec, order="code")
    assert len(text) == 3
    assert len(code) == 3
    assert not np.array_equal(lm.bounds(text), lm.bounds(code))


def test_kudrimoti_threshold_changes_the_inventory(measured, monkeypatch):
    amplitude = np.ones(len(measured.time))
    amplitude[4500:4650] = 3
    amplitude[9000:9150] = 6
    monkeypatch.setattr(measured, "envelope", lambda *args: amplitude[:, None])
    moderate = lm.kudrimoti_1999(measured, threshold_sd=3)
    strict = lm.kudrimoti_1999(measured, threshold_sd=6)
    assert len(moderate) == 2
    assert len(strict) == 1
    np.testing.assert_allclose(lm.bounds(strict), measured.time[[9000, 9149]][None, :])


def test_wikenheiser_baseline_normalization_is_an_explicit_choice(measured):
    from dataclasses import replace

    rec = replace(measured, baseline_intervals=np.array([[0, 1.5]]))
    default = lm.wikenheiser_2013(rec, window_anchor="peaks")
    baseline = lm.wikenheiser_2013(rec, window_anchor="peaks", normalization="baseline")
    assert len(default)
    assert len(baseline)
    assert not default.equals(baseline)
    with pytest.raises(ValueError, match="baseline_intervals"):
        lm.wikenheiser_2013(
            replace(rec, baseline_intervals=None),
            window_anchor="peaks",
            normalization="baseline",
        )
    with pytest.raises(ValueError, match="normalization"):
        lm.wikenheiser_2013(rec, window_anchor="peaks", normalization="unknown")


def test_method_roles_are_independent_of_demonstration_grouping(measured):
    catalog = lm.list_methods().set_index("name")
    assert catalog.loc["harvey_2023_no_radiatum", "role"] == "candidate_detection"
    assert catalog.loc["harvey_2023_no_radiatum", "inventory"] == "additional"
    assert catalog.loc["widloski_2025_bursts", "role"] == "secondary"
    assert catalog.loc["widloski_2025", "inventory"] == "default"
    assert catalog.loc["widloski_2025", "role"] == "secondary"
    result = lm.mallory_2025_ripples(measured)
    assert result.attrs["role"] == "secondary"
    assert result.attrs["inventory"] == "additional"


@pytest.mark.parametrize("missing", ["timestamps", "nan"])
def test_gridchyn_feedback_clock_pauses_across_gaps(missing):
    full_time = np.arange(3000) / 1000
    spikes = np.zeros((3000, 1))
    spikes[::100] = 1
    valid = (full_time < 1) | (full_time >= 2)
    if missing == "timestamps":
        time, counts = full_time[valid], spikes[valid]
    else:
        time, counts = full_time, spikes.copy()
        counts[~valid] = np.nan
    rec = lm.Recording.from_arrays(time, 1000, multiunit=counts, baseline_intervals=[[0, 0.9]])
    compressed = lm.Recording.from_arrays(
        np.arange(2000) / 1000, 1000, multiunit=spikes[valid], baseline_intervals=[[0, 0.9]]
    )
    options = {"update_interval": 0.6003, "refractory": 0.05}
    gapped = lm.gridchyn_2020(rec, **options)
    expected = lm.gridchyn_2020(compressed, **options)
    assert len(gapped)
    assert len(expected)
    actual_updates = gapped.attrs["threshold_updates"]
    expected_updates = expected.attrs["threshold_updates"]
    assert len(actual_updates) == len(expected_updates) == 3
    for actual, reference in zip(actual_updates, expected_updates, strict=True):
        assert actual["multiplier"] == pytest.approx(reference["multiplier"])
        assert actual["observed_rate"] == pytest.approx(reference["observed_rate"])
    assert actual_updates[1]["time"] == pytest.approx(expected_updates[1]["time"] + 1)


@pytest.mark.parametrize("event", [[-0.1, 0.1], [0.1, 0.8], [0.8, 1.1]])
def test_both_merge_apis_reject_events_outside_valid_blocks(event):
    time = np.arange(1000) / 1000
    trace = np.ones(1000)
    trace[400:600] = np.nan
    rec = lm.Recording.from_arrays(time, 1000, multiunit=np.zeros((1000, 1)))
    native = lm.PopulationTrace(time, trace, np.zeros(1000), 1000)
    for merge in [
        lambda events: rec.merge(events, 0.01, trace),
        lambda events: native.merge(events, 0.01),
    ]:
        with pytest.raises(ValueError, match="valid detection block"):
            merge(np.array([event]))
        assert merge(np.empty((0, 2))).shape == (0, 2)
        np.testing.assert_allclose(merge(np.array([[0.1, 0.15], [0.151, 0.2]])), [[0.1, 0.2]])


def test_pfeiffer_2013_ripple_inventory_excludes_moving_and_unknown_speed(monkeypatch):
    time = np.arange(4000) / 1000
    speed = np.zeros(len(time))
    speed[1450:1650] = 10
    speed[2450:2650] = np.nan
    amplitude = np.ones(len(time))
    for start in [500, 1500, 2500]:
        amplitude[start : start + 80] = 20
    rec = lm.Recording.from_arrays(time, 1000, lfps=np.zeros(len(time)), speed=speed)
    monkeypatch.setattr(rec, "mean_envelope", lambda *args: amplitude)
    events = lm.pfeiffer_2013_ripples(rec)
    assert len(events) == 1
    assert events.start_time.iloc[0] < 0.54 < events.end_time.iloc[0]


def test_stella_wavelet_response_is_centered_and_symmetric(monkeypatch):
    time = np.arange(2001) / 1500
    impulse = np.zeros(len(time))
    impulse[1000] = 1
    rec = lm.Recording.from_arrays(time, 1500, lfps=impulse, sleep_intervals=[[0, time[-1]]])
    captured = []

    def capture(rec, trace, **kwargs):
        captured.append(trace)
        return pd.DataFrame(columns=["start_time", "end_time"])

    monkeypatch.setattr(lm, "_ripple_trace_events", capture)
    lm.stella_2019(rec, frequencies=[173], cycles=7)
    assert np.argmax(captured[0]) == 1000
    assert np.ptp(captured[0]) > 1
    np.testing.assert_allclose(captured[0], captured[0][::-1], atol=1e-12)


def _burst_trace(bursts, bin_width=0.01):
    """A 20 s, 1 kHz recording whose one unit fires every sample in each burst."""
    time = np.arange(20_000) / 1000
    spikes = np.zeros((len(time), 1))
    for start, stop in bursts:
        spikes[round(start * 1000) : round(stop * 1000)] = 1
    rec = lm.Recording.from_arrays(time, 1000, multiunit=spikes)
    return lm.population_trace(rec, bin_width=bin_width)


def _detect_bursts(trace, **options):
    return trace.detect(
        threshold=50,
        bound_threshold=50,
        normalization_method="none",
        minimum_duration=0.0,
        speed_threshold=np.inf,
        **options,
    )


def test_native_grid_events_are_reported_at_their_bins_samples():
    # Spikes fill 10.000-10.049 s: five complete 10 ms bins, 50 ms edge to edge,
    # reported at the first and last samples those bins counted.
    trace = _burst_trace([(10.0, 10.05)])
    events = _detect_bursts(trace)
    np.testing.assert_allclose(lm.bounds(events), [[10.0, 10.049]], atol=1e-9)
    np.testing.assert_allclose(events.duration, [0.049], atol=1e-9)
    # Duration limits count bins: five bins last 50 ms.
    assert len(_detect_bursts(trace, minimum_event_duration=0.05, maximum_duration=0.05))
    assert not len(_detect_bursts(trace, minimum_event_duration=0.06))
    assert not len(_detect_bursts(trace, maximum_duration=0.04))
    # The merge accepts the bounds detect reports.
    np.testing.assert_allclose(trace.merge(events, 0.0), [[10.0, 10.049]], atol=1e-9)


def test_native_grid_close_event_gaps_are_measured_between_edges():
    trace = _burst_trace([(10.0, 10.05), (10.08, 10.1)])  # 30 ms from edge to edge
    kept_apart = _detect_bursts(trace, close_event_threshold=0.03, close_event_rule="merge")
    np.testing.assert_allclose(
        lm.bounds(kept_apart), [[10.0, 10.049], [10.08, 10.099]], atol=1e-9
    )
    merged = _detect_bursts(trace, close_event_threshold=0.031, close_event_rule="merge")
    np.testing.assert_allclose(lm.bounds(merged), [[10.0, 10.099]], atol=1e-9)
    # merge measures between the bounds it is given: from the last sample of
    # one event to the first of the next, 31 ms here (30 ms edge to edge).
    np.testing.assert_allclose(trace.merge(kept_apart, 0.032), [[10.0, 10.099]], atol=1e-9)
    np.testing.assert_allclose(
        trace.merge(kept_apart, 0.031), lm.bounds(kept_apart), atol=1e-9
    )


def test_native_bins_leave_out_a_sample_on_the_final_edge():
    # 1001 samples: the last, at exactly 1.0 s, starts an incomplete bin.
    time = np.arange(1001) / 1000
    spikes = np.zeros((1001, 1))
    spikes[-1] = 1
    trace = lm.population_trace(
        lm.Recording.from_arrays(time, 1000, multiunit=spikes), bin_width=0.01
    )
    assert len(trace.time) == 100
    np.testing.assert_array_equal(trace.data, 0)


SHORT_BLOCK_TRANSFORMS = [
    (
        "tirole_2022",
        1500,
        100,
        ["Tirole's 41-point forward/backward kernel", "Tirole's ripple filter"],
    ),
    ("denovellis_2021", 1500, 300, ["the historical 101-tap ripple filter"]),
    ("bush_2022_ripples", 4800, 1000, ["Bush's 400th-order FIR"]),
    ("kaefer_2020", 1500, 300, ["Kaefer's 240 ms FFT chunk"]),
]


@pytest.mark.parametrize(("name", "fs", "spacing", "transforms"), SHORT_BLOCK_TRANSFORMS)
def test_blocks_too_short_for_a_transform_are_dropped_with_a_warning(
    name, fs, spacing, transforms
):
    inputs = _measured_inputs(fs)
    # A block of about 30 ms between two artifacts.
    rec = lm.Recording.from_arrays(**inputs, artifact_intervals=[[1.0, 1.1], [1.13, 1.2]])
    with pytest.warns(UserWarning, match="treated as missing") as record:
        lm.run_method(name, rec)
    messages = [str(warning.message) for warning in record]
    for transform in transforms:
        assert any(transform in m and m.startswith("1 block") for m in messages), messages


@pytest.mark.parametrize(("name", "fs", "spacing", "transforms"), SHORT_BLOCK_TRANSFORMS)
def test_a_recording_of_blocks_too_short_for_a_transform_raises(name, fs, spacing, transforms):
    inputs = _measured_inputs(fs)
    for key in ("lfps", "multiunit"):
        inputs[key] = inputs[key].astype(float)
        inputs[key][::spacing] = np.nan
    with pytest.raises(ValueError, match="No block of finite samples"):
        lm.run_method(name, lm.Recording.from_arrays(**inputs))


@pytest.mark.parametrize("name", ["muessig_2019_ripples", "harvey_2023_text"])
@pytest.mark.parametrize("problem", ["no finite sample", "zero scale"])
def test_caller_selected_baselines_must_have_a_finite_positive_scale(name, problem):
    inputs = _measured_inputs()
    if problem == "no finite sample":
        inputs["artifact_intervals"] = [[0.0, 2.0]]
        inputs["baseline_intervals"] = [[0.5, 1.5]]
    else:
        inputs["lfps"] = np.zeros_like(inputs["lfps"])
        inputs["sharp_wave_lfp"] = np.zeros_like(inputs["sharp_wave_lfp"])
    with pytest.raises(ValueError, match="baseline_intervals"):
        lm.run_method(name, lm.Recording.from_arrays(**inputs))


def test_a_fixed_channel_count_is_not_filled_from_fewer_channels(monkeypatch):
    inputs = _measured_inputs()
    inputs["lfps"] = inputs["lfps"][:, :2]
    rec = lm.Recording.from_arrays(**inputs)
    with pytest.raises(ValueError, match="3 selected LFP channels"):
        rec.mean_envelope((150.0, 250.0), channels=3)
    with pytest.raises(ValueError, match="3 selected LFP channels"):
        lm.berners_lee_2021(rec)
    # Michon averages the 1-3 selected tetrodes, so two are enough.
    requested = []
    original = lm.Recording.mean_envelope

    def record(self, band, channels=None):
        requested.append(channels)
        return original(self, band, channels)

    monkeypatch.setattr(lm.Recording, "mean_envelope", record)
    lm.michon_2021(rec)
    assert requested == [2]


@pytest.mark.parametrize(
    "name",
    ["chenani_2019", "olafsdottir_2017", "olafsdottir_2015", "diba_2007", "foster_2006"],
)
def test_measured_recordings_need_the_documented_behavior_intervals(measured, name):
    lm.run_method(name, measured, **_eligible_epochs(name, measured.time))  # runs with them
    meaning = next(
        need.meaning
        for need in lm._ENTRIES[name].requirements
        if need.input == "behavior_intervals"
    )
    with pytest.raises(ValueError, match=f"behavior_intervals: {meaning}"):
        lm.run_method(name, measured)


def test_krause_warns_when_it_skips_swrs():
    time = np.arange(4000) / 1000
    spikes = np.zeros((4000, 10))
    spikes[np.arange(1030, 1140, 6)] = 1
    spikes[2500] = np.nan  # missing spike data inside the second SWR
    rec = lm.Recording.from_arrays(
        time,
        1000,
        multiunit=spikes,
        place_cells=np.arange(10),
        external_ripples=[[1.0, 1.15], [2.45, 2.6], [3.0, 3.02]],
    )
    with pytest.warns(UserWarning, match=r"^2 of 3 SWR\(s\) skipped: 1 crossing"):
        result = lm.krause_2022(rec)
    assert len(result) == 1


def test_olafsdottir_2015_warns_for_small_templates_and_rejects_empty_ones():
    time = np.arange(2000) / 1000
    spikes = np.zeros((2000, 12))
    spikes[[500, 510, 520, 530, 540, 550, 560], np.arange(7)] = 1
    rec = lm.Recording.from_arrays(
        time,
        1000,
        multiunit=spikes,
        templates=[np.arange(7), np.arange(7, 12)],
    )
    with pytest.warns(UserWarning, match=r"^1 of 2 template\(s\) .* fewer than 7 cells"):
        events = lm.olafsdottir_2015(rec, minimum_active_units=7, behavior_intervals=[[0, 2]])
    np.testing.assert_allclose(lm.bounds(events), [[0.5, 0.56]])
    empty = lm.Recording.from_arrays(
        time, 1000, multiunit=spikes, templates=[np.arange(7), []]
    )
    with pytest.raises(ValueError, match="selects no cells"):
        lm.olafsdottir_2015(empty, behavior_intervals=[[0, 2]])


def test_mallory_flags_candidates_cut_by_a_block_edge():
    time = np.arange(400) / 1000
    z = np.full(400, -1.0)
    z[:30] = 1.0
    z[10] = 5.0
    z[200] = 5.0
    z[300:] = 1.0
    z[350] = 5.0
    result = lm._mallory_candidates(time, z)
    np.testing.assert_array_equal(result.clipped_start, [True, False, False])
    np.testing.assert_array_equal(result.clipped_end, [False, False, True])


def test_tirole_flags_bounds_set_by_the_search_limit_or_a_block_edge():
    time = np.arange(1000) / 1000
    z = np.full(1000, -1.0)
    z[100:900] = 1.0  # no bound level within 300 ms of the anchor at 0.5 s
    z[500] = 4
    z[:20] = 4  # an anchor at the recording's start
    events = lm._tirole_bounds(time, z)
    np.testing.assert_allclose(lm.bounds(events), [[0.0, 0.02], [0.2, 0.8]])
    np.testing.assert_array_equal(events.clipped_start, [True, True])
    np.testing.assert_array_equal(events.clipped_end, [False, True])


def test_tirole_output_keeps_clipped_flags(measured, monkeypatch):
    bounds_found = pd.DataFrame(
        {
            "start_time": [1.0005, 5.0005],  # 1 ms bin centers
            "end_time": [1.2005, 5.3005],
            "clipped_start": [True, False],
            "clipped_end": [False, True],
        }
    )
    monkeypatch.setattr(lm, "_tirole_bounds", lambda *args: bounds_found.copy())
    monkeypatch.setattr(lm.rd, "require_active_units", lambda events, *a, **k: events)
    monkeypatch.setattr(
        lm, "_tirole_ripple_amplitude", lambda rec: (rec.time, np.ones(len(rec.time)))
    )
    monkeypatch.setattr(lm, "_zscore", lambda values, *a, **k: np.full(len(values), 5.0))
    events = lm.tirole_2022(measured)
    # The first and last 1500 Hz samples of those bins.
    np.testing.assert_allclose(
        lm.bounds(events), [[1.0, 1.2 + 1 / 1500], [5.0, 5.3 + 1 / 1500]], atol=1e-9
    )
    np.testing.assert_array_equal(events.clipped_start, [True, False])
    np.testing.assert_array_equal(events.clipped_end, [False, True])


def test_local_peak_windows_flag_a_cut_at_a_block_edge(measured):
    trace = np.zeros(len(measured.time))
    trace[30] = 4  # 20 ms after the recording starts
    trace[3000] = 4
    events = lm._local_peaks(measured, trace, 3, before=0.05, after=0.05)
    np.testing.assert_allclose(
        lm.bounds(events)[0], [measured.time[0], measured.time[30] + 0.05]
    )
    np.testing.assert_array_equal(events.clipped_start, [True, False])
    np.testing.assert_array_equal(events.clipped_end, [False, False])


def test_gridchyn_warns_when_the_multiplier_reaches_zero():
    time = np.arange(3000) / 1000
    spikes = np.zeros((3000, 1))
    spikes[::100] = 1
    rec = lm.Recording.from_arrays(
        time, 1000, multiunit=spikes, baseline_intervals=[[0, 0.999]]
    )
    # Too few triggers for the target rate drive the multiplier below zero,
    # where the published rule, which has no floor, triggers on any spike.
    with pytest.warns(UserWarning, match="multiplier reached -"):
        events = lm.gridchyn_2020(rec, update_interval=1, target_rate=20)
    assert events.threshold_multiplier.min() <= 0
    lm.gridchyn_2020(rec, update_interval=1, target_rate=0)  # rises: no warning


@pytest.mark.parametrize(
    ("name", "limit", "strict"),
    [
        ("davidson_2009", 5.0, True),
        ("wu_2014", 5.0, True),
        ("silva_2015", 5.0, True),
        ("pfeiffer_2013", 5.0, True),
        ("ambrose_2016", 5.0, True),
        ("tang_2017", 4.0, True),
        ("jadhav_2016", 4.0, True),
        ("carr_2012", 4.0, True),
        ("karlsson_2009", 2.0, True),
        ("gillespie_2021", 4.0, True),
        ("gillespie_2021_mua", 4.0, True),
        ("shin_2019", 4.0, False),  # "immobility periods (<=4 cm/s)"
    ],
)
def test_speed_limits_follow_each_papers_inequality(name, limit, strict):
    inputs = _measured_inputs()
    time = inputs["time"]
    speed = np.zeros(len(time))
    speed[time < 0.5] = 20.0  # running, for Davidson's within-30-s-of-RUN rule
    inputs["speed"] = speed
    found = lm.bounds(lm.run_method(name, lm.Recording.from_arrays(**inputs)))
    assert len(found), name
    near = np.zeros(len(time), dtype=bool)
    for start, end in found:
        near |= (time >= start - 0.3) & (time <= end + 0.3)

    def overlapping(value):
        inputs["speed"] = np.where(near, value, speed)
        try:
            events = lm.bounds(lm.run_method(name, lm.Recording.from_arrays(**inputs)))
        except ValueError:  # speed_rule='restrict' with no slow sample left
            return 0
        return len(rd.require_overlap(events, found))

    assert overlapping(0.999 * limit), name
    assert bool(overlapping(limit)) is not strict, name


def test_wikenheiser_joins_overlapping_anchor_windows(measured, monkeypatch):
    amplitude = np.ones(len(measured.time))
    peaks = np.searchsorted(measured.time, [3.0, 3.1, 6.0])
    amplitude[peaks] = 20
    monkeypatch.setattr(measured, "mean_envelope", lambda *args, **kwargs: amplitude)
    events = lm.wikenheiser_2013(measured, window_anchor="peaks")
    # "Overlapping events were concatenated": 3.0 and 3.1 s share one window.
    np.testing.assert_allclose(lm.bounds(events), [[2.925, 3.175], [5.925, 6.075]], atol=1e-9)


def _signals(n_time=100, n_units=3):
    time = np.arange(n_time) / 1000
    return lm.RecordedSignals(
        time,
        1000.0,
        np.zeros((n_time, 1)),
        np.zeros(n_time),
        np.zeros(n_time),
        np.zeros((n_time, n_units)),
        np.zeros(n_time),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"place_cells": np.zeros(3)}, "Cell masks"),
        ({"pyramidal": np.ones(2, dtype=bool)}, "Cell masks"),
        ({"templates": (np.ones(4, dtype=bool),)}, "Cell masks"),
        ({"sleep_intervals": np.array([[0.05, 0.01]])}, "Intervals"),
        ({"baseline_intervals": np.array([0.0, 0.01])}, "Intervals"),
        ({"example_ripples": np.array([[0.02, 0.03], [0.025, 0.04]])}, "Intervals"),
        ({"external_ripples": np.array([[0.01, 0.02, 0.05]])}, "peaks"),
        ({"reference_lfp": np.zeros(5)}, "reference_lfp"),
    ],
)
def test_recording_constructor_validates_like_from_arrays(overrides, message):
    options = {"place_cells": np.zeros(3, dtype=bool), "pyramidal": np.zeros(3, dtype=bool)}
    options.update(overrides)
    with pytest.raises(ValueError, match=message):
        lm.Recording(_signals(), **options)


def test_recorded_signals_must_share_one_time_grid():
    signals = _signals()
    with pytest.raises(ValueError, match="one row per timestamp"):
        lm.RecordedSignals(
            signals.time,
            1000.0,
            signals.lfps,
            signals.raw_lfp[:50],
            signals.sharp_wave_lfp,
            signals.multiunit,
            signals.speed,
        )


def test_from_arrays_reads_integers_as_indices_and_rejects_repeats():
    options = {"time": np.arange(100) / 1000, "sampling_frequency": 1000}
    spikes = np.zeros((100, 3))
    # A 0/1 integer mask repeats indices; it is not silently read as [0, 1].
    with pytest.raises(ValueError, match="boolean mask"):
        lm.Recording.from_arrays(**options, multiunit=spikes, place_cells=[1, 0, 1])
    rec = lm.Recording.from_arrays(**options, multiunit=spikes, place_cells=[2, 0])
    np.testing.assert_array_equal(rec.place_cells, [True, False, True])


def test_from_arrays_raw_lfp_is_its_own_copy():
    rec = lm.Recording.from_arrays(
        np.arange(100) / 1000, 1000, lfps=np.ones((100, 2)), multiunit=np.zeros((100, 1))
    )
    assert not np.shares_memory(rec.session.raw_lfp, rec.session.lfps)


def test_only_simulated_sessions_allow_simulation_proxies(measured):
    from dataclasses import replace

    assert not measured.allows_simulation_proxies
    session = rd.simulate_session(np.arange(1500) / 1500, [0.5], n_units=3)
    simulated = replace(
        measured,
        session=session,
        place_cells=np.ones(3, dtype=bool),
        pyramidal=np.ones(3, dtype=bool),
        templates=(),
        sleep_intervals=None,
        baseline_intervals=None,
        reference_lfp=None,
    )
    assert simulated.allows_simulation_proxies


def test_registry_entries_and_native_traces_are_immutable():
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        lm.RECIPES[0].row = 3
    trace = lm.PopulationTrace(np.arange(3) / 100, np.zeros(3), None, 100.0)
    with pytest.raises(FrozenInstanceError):
        trace.data = np.ones(3)


def test_population_trace_arrays_share_the_bin_grid():
    with pytest.raises(ValueError, match="one value per bin"):
        lm.PopulationTrace(np.arange(3) / 100, np.zeros(2), None, 100.0)
    with pytest.raises(ValueError, match="one value per bin"):
        lm.PopulationTrace(np.arange(3) / 100, np.zeros(3), np.zeros(4), 100.0)


def test_registry_rejects_a_duplicate_name_without_changing_the_inventory():
    before = (len(lm.RECIPES), len(lm.VARIANTS), dict(lm._IMPLEMENTATIONS))
    with pytest.raises(ValueError, match="already registered"):
        lm._register(lm.VARIANTS, 0, "Mallory 2025", "MUA")(
            lm._IMPLEMENTATIONS["mallory_2025"]
        )
    assert (len(lm.RECIPES), len(lm.VARIANTS), dict(lm._IMPLEMENTATIONS)) == before


def test_each_entry_records_its_inventory():
    assert {entry.inventory for entry in lm.RECIPES} == {"default"}
    assert {entry.inventory for entry in lm.VARIANTS} == {"additional"}
    catalog = lm.list_methods().set_index("name")
    for entry in (*lm.RECIPES, *lm.VARIANTS):
        assert catalog.loc[entry.run.__name__, "inventory"] == entry.inventory


def test_grosmark_rejects_an_unknown_stage(measured):
    with pytest.raises(ValueError, match="stage='replay' - expected"):
        lm.grosmark_2016(
            lm.Recording.from_arrays(**_method_inputs("grosmark_2016")[0]),
            stage="replay",
            behavior_intervals=[[0, 20]],
        )


def test_within_intervals_rejects_unsorted_or_overlapping_intervals():
    events = np.array([[1.0, 1.5], [3.2, 3.4]])
    np.testing.assert_allclose(
        lm.within_intervals(events, [[0.5, 2.0], [3.0, 3.3]]), events[:1]
    )
    for intervals in ([[3.0, 3.5], [0.5, 2.0]], [[0.5, 2.0], [1.0, 3.5]]):
        with pytest.raises(ValueError, match="sorted, disjoint"):
            lm.within_intervals(events, intervals)


@pytest.mark.parametrize("entry", [*lm.RECIPES, *lm.VARIANTS], ids=lambda e: e.run.__name__)
def test_public_method_docstrings_list_their_actual_parameters(entry):
    doc = entry.run.__doc__
    assert "**options" not in doc
    parameters = [name for name in inspect.signature(entry.run).parameters if name != "rec"]
    section = doc.split("Parameters\n    ----------\n")[1].split("Returns")[0]
    listed = [line.split(" : ")[0].strip() for line in section.splitlines() if " : " in line]
    assert listed == ["rec", *parameters]


def test_private_helpers_are_not_public_names():
    for name in ["recipe", "variant", "only_in", "zugaro_ripple_peaks"]:
        assert not hasattr(lm, name), name


@pytest.mark.parametrize(
    ("fs", "width", "samples"),
    [
        (1500, 0.007, 11),
        (1500, 0.02, 31),
        (1500, 0.05, 75),
        (1000, 0.01, 11),
        (1000, 0.0001, 1),
    ],
)
def test_boxcar_rounds_half_up_to_an_odd_centered_window(fs, width, samples):
    time = np.arange(401) / fs
    impulse = np.zeros(401)
    impulse[200] = 1.0
    rec = lm.Recording.from_arrays(time, fs, lfps=impulse)
    smoothed = rec.boxcar(impulse, width)
    support = np.flatnonzero(smoothed > 0)
    assert len(support) == samples
    assert support[0] + support[-1] == 400  # centered on the impulse
    np.testing.assert_allclose(smoothed[support], 1 / samples)


@pytest.mark.parametrize("seed", range(5))
def test_interval_mask_matches_the_inclusive_loop(seed):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 10, 500))
    # Unsorted, overlapping, empty-range and exact-timestamp endpoints included.
    starts = np.r_[rng.uniform(-1, 11, 40), time[[10, 200]]]
    intervals = np.c_[starts, starts + rng.uniform(0, 1, len(starts))]
    intervals[-1, 1] = time[300]
    intervals = np.r_[intervals, [[5.0, 4.0]]]
    expected = np.zeros(len(time), dtype=bool)
    for start, end in intervals:
        expected |= (time >= start) & (time <= end)
    np.testing.assert_array_equal(lm._intervals_to_mask(time, intervals), expected)
    assert not lm._intervals_to_mask(time, np.empty((0, 2))).any()


UNIX_ORIGIN = 1_700_000_000.0


@pytest.mark.parametrize("entry", lm.RECIPES, ids=lambda x: x.run.__name__)
def test_every_recipe_gives_the_same_bounds_at_a_unix_clock_origin(entry):
    name = entry.run.__name__
    found = []
    for origin in (0.0, UNIX_ORIGIN):
        inputs, options = _method_inputs(name, origin)
        rec = lm.Recording.from_arrays(**inputs)
        found.append(lm.bounds(lm.run_method(name, rec, **options)) - origin)
    # Timestamps near 1.7e9 s carry about 2.4e-7 s of rounding each.
    tolerance = 16 * np.spacing(UNIX_ORIGIN + 20.0)
    assert found[0].shape == found[1].shape, name
    np.testing.assert_allclose(found[1], found[0], rtol=0, atol=tolerance, err_msg=name)


def test_carey_builds_its_template_from_the_supplied_example_ripples(monkeypatch):
    inputs, _ = _method_inputs("carey_2019")
    captured = []
    original = lm.rd.carey_spectral_ripple_score

    def capture(time, lfp, fs, examples, **kwargs):
        captured.append(np.asarray(examples))
        return original(time, lfp, fs, examples, **kwargs)

    monkeypatch.setattr(lm.rd, "carey_spectral_ripple_score", capture)
    events = lm.carey_2019(lm.Recording.from_arrays(**inputs))
    np.testing.assert_array_equal(captured[0], inputs["example_ripples"])
    assert len(events)
    assert rd.require_overlap(lm.bounds(events), inputs["example_ripples"]).size


CANDIDATES = np.array([[1.0, 1.2], [2.0, 2.2], [3.0, 3.2]])


@pytest.fixture
def population_candidates(monkeypatch):
    """Yang/Grosmark candidates fixed, so only the ripple and state gates act."""
    frame = pd.DataFrame(CANDIDATES, columns=["start_time", "end_time"])
    monkeypatch.setattr(lm, "_detect_population", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(lm.rd, "require_active_units", lambda events, *args, **kwargs: events)
    return _measured_inputs()


@pytest.mark.parametrize("name", ["yang_2024", "grosmark_2016"])
@pytest.mark.parametrize(
    ("external", "kept"),
    [
        # Three columns: the supplied peak must lie inside the candidate.
        ([[1.15, 1.3, 1.18], [2.3, 2.5, 2.4]], [0]),
        # A peak outside every candidate keeps none, although the ripple overlaps.
        ([[2.15, 2.4, 2.35]], []),
        # Two columns: the midpoint stands for the peak.
        ([[2.05, 2.15], [2.16, 2.36]], [1]),
        ([[0.9, 1.3], [2.95, 3.15]], [0, 2]),
    ],
)
def test_population_candidates_need_an_external_ripple_peak_inside(
    population_candidates, name, external, kept
):
    rec = lm.Recording.from_arrays(**population_candidates, external_ripples=external)
    np.testing.assert_allclose(
        lm.bounds(lm.run_method(name, rec, **_eligible_epochs(name, rec.time))),
        CANDIDATES[kept].reshape(-1, 2),
    )


def test_population_candidates_keep_only_eligible_behavior_epochs(population_candidates):
    external = np.c_[CANDIDATES, CANDIDATES.mean(axis=1)]
    rec = lm.Recording.from_arrays(**population_candidates, external_ripples=external)
    eligible = np.array([[0.5, 1.5], [2.9, 3.25]])
    np.testing.assert_allclose(
        lm.bounds(lm.yang_2024(rec, behavior_intervals=eligible)), CANDIDATES[[0, 2]]
    )
    # The eligible epochs are applied inside the method, not only by run_method.
    np.testing.assert_allclose(
        lm._population_with_ripple_peak(rec, np.array([[0, 20]]), eligible),
        CANDIDATES[[0, 2]],
    )
    with pytest.raises(ValueError, match="behavior_intervals"):
        lm.yang_2024(rec)


@pytest.mark.parametrize("name", ["farooq_2019_science", "drieu_2018"])
def test_sleep_frames_come_only_from_curated_sleep_intervals(name):
    # Long ripples, so the population stays above 2 SD for Farooq's 100 ms.
    inputs = _measured_inputs(ripple_duration=(0.25, 0.3))
    whole = lm.bounds(lm.run_method(name, lm.Recording.from_arrays(**inputs)))
    inputs["sleep_intervals"] = [[7.0, 13.5]]
    curated = lm.bounds(lm.run_method(name, lm.Recording.from_arrays(**inputs)))
    assert len(whole) > len(curated) > 0, name
    assert np.all((curated[:, 0] >= 7.0) & (curated[:, 1] <= 13.5)), name
    del inputs["sleep_intervals"]
    with pytest.raises(ValueError, match="sleep_intervals"):
        lm.run_method(name, lm.Recording.from_arrays(**inputs))


def _changed(fs=1500, drop=(), **changes):
    """Measured inputs at ``fs`` without ``drop`` and with ``changes``; a
    callable change receives the unchanged inputs."""

    def make():
        inputs = _measured_inputs(fs)
        for key in drop:
            del inputs[key]
        for key, value in changes.items():
            inputs[key] = value(inputs) if callable(value) else value
        return inputs

    return make


def _slow_recording():
    """Four samples per second: too slow for any 20 ms step."""
    time = np.arange(100) / 4
    return {
        "time": time,
        "sampling_frequency": 4,
        "lfps": np.random.default_rng(0).normal(size=(100, 1)),
        "reference_lfp": np.zeros(100),
        "baseline_intervals": [[0, time[-1]]],
    }


def _silent_baseline(inputs):
    spikes = inputs["multiunit"].astype(float)
    spikes[inputs["time"] < 2] = 0
    return spikes


ERROR_PATHS = [
    # Sampling-rate guards.
    ("denovellis_2021", _changed(fs=1000), {}, "sampled at 1500 Hz"),
    ("bush_2022_ripples", _changed(fs=1500), {}, "sampled at 4800 Hz"),
    ("olafsdottir_2017_ripples", _changed(fs=1500), {}, "sampled at 1200 Hz"),
    ("tirole_2022", _changed(fs=1234.5678), {}, "Resample LFP to 1000 Hz"),
    ("kaefer_2020", _slow_recording, {}, "too low for FFT windows"),
    ("stella_2019", _changed(), {"frequencies": [150, 800], "cycles": 7}, "Nyquist"),
    # Constant or empty baselines.
    (
        "ji_2007_ripples",
        _changed(lfps=lambda inputs: np.zeros_like(inputs["lfps"])),
        {},
        "positive finite SD",
    ),
    (
        "mou_2022",
        _changed(multiunit=lambda inputs: np.zeros_like(inputs["multiunit"])),
        {},
        "nonconstant population trace",
    ),
    (
        "gridchyn_2020",
        _changed(artifact_intervals=[[0, 2]], baseline_intervals=[[0.5, 1.5]]),
        {},
        "no valid spikes",
    ),
    (
        "gridchyn_2020",
        _changed(multiunit=_silent_baseline, baseline_intervals=[[0, 1.9]]),
        {},
        "positive population firing rate",
    ),
    ("gridchyn_2020", _changed(), {"gain": -1.0}, "Invalid adaptive detector parameters"),
    # Required inputs.
    ("gillespie_2021", _changed(drop=["lfps"]), {}, "lfps: every selected channel"),
    ("tirole_2022", _changed(drop=["lfps"]), {}, "lfps: the first selected channel"),
    ("kaefer_2020", _changed(drop=["reference_lfp"]), {}, "pass reference_lfp"),
    ("gridchyn_2020_ripples", _changed(drop=["reference_lfp"]), {}, "pass reference_lfp"),
    ("diba_2007_ripples", _changed(drop=["lfps"]), {}, "pass lfps"),
    (
        "chenani_2019_hfe",
        _changed(drop=["lfps"]),
        {"ar_coefficients": np.zeros((0, 2))},
        "pass lfps",
    ),
    (
        "bhattarai_2020_ripples",
        _changed(lfps=lambda inputs: inputs["lfps"][:, :1]),
        {},
        "at least 2 selected LFP channels, got 1",
    ),
    ("olafsdottir_2015", _changed(templates=[]), {}, "pass templates"),
    (
        "farooq_2019_science_awake",
        _changed(),
        {"behavior_intervals": None},
        "behavior_intervals",
    ),
    ("liu_2019_awake", _changed(), {"behavior_intervals": None}, "behavior_intervals"),
    ("wikenheiser_2013", _changed(), {"window_anchor": None}, "pass window_anchor="),
    ("wikenheiser_2013", _changed(), {"branch": "run_lia"}, "theta_delta: the caller"),
    (
        "wikenheiser_2013",
        _changed(),
        {"branch": "run_lia", "theta_delta": np.zeros(10)},
        "one value per input timestamp",
    ),
    # Unknown or invalid options.
    ("huelin_gorriz_2023", _changed(), {"interpretation": "other"}, "interpretation must be"),
    ("krause_2022_hse", _changed(), {"interpretation": "other"}, "interpretation must be"),
    ("harvey_2023_text", _changed(), {"sharp_wave_polarity": 0.0}, "sharp_wave_polarity"),
    ("mou_2022", _changed(), {"normalization": "zscore"}, "normalization must be"),
    ("michon_2021", _changed(), {"order": "reverse"}, "order must be"),
    ("muessig_2019", _changed(), {"trial": "sleep"}, "trial must be"),
    ("olafsdottir_2017", _changed(), {"analysis": "sequence"}, "analysis must be"),
    ("wikenheiser_2013", _changed(), {"branch": "sleep"}, "branch must be"),
    ("wikenheiser_2013", _changed(), {"window_anchor": "troughs"}, "window_anchor must be"),
    ("nadasdy_1999", _changed(), {"rms_window": -0.004}, "rms_window must be positive"),
    ("farooq_2019_science_ripples", _changed(), {"power_measure": "rms"}, "power_measure"),
    ("bhattarai_2020_ripples", _changed(), {"power_measure": "rms"}, "power_measure"),
    ("chenani_2019_hfe", _changed(), {"ar_coefficients": np.zeros((2, 2))}, "ar_coefficients"),
    ("liu_2019_ripples", _changed(), {"window": -0.1}, "window must be nonnegative"),
    ("drieu_2018_ripples", _changed(), {"signal_measure": "rms"}, "signal_measure must be"),
]


@pytest.mark.parametrize(
    ("name", "make_inputs", "options", "message"),
    ERROR_PATHS,
    ids=[f"{name}-{message}" for name, _, _, message in ERROR_PATHS],
)
def test_literature_method_error_paths(name, make_inputs, options, message):
    rec = lm.Recording.from_arrays(**make_inputs())
    options = (
        {**VARIANT_OPTIONS, **RECIPE_OPTIONS}.get(name, {})
        | _eligible_epochs(name, rec.time)
        | options
    )
    with pytest.raises(ValueError, match=message):
        lm.run_method(name, rec, **options)


@pytest.mark.parametrize(
    ("bin_width", "n_time", "message"),
    [(0.0, 100, "bin_width"), (np.nan, 100, "bin_width"), (0.06, 100, "two complete bins")],
)
def test_population_trace_error_paths(bin_width, n_time, message):
    rec = lm.Recording.from_arrays(
        np.arange(n_time) / 1000, 1000, multiunit=np.zeros((n_time, 1))
    )
    with pytest.raises(ValueError, match=message):
        lm.population_trace(rec, bin_width=bin_width)


def _burst_recording(origin, fs, *, burst=(10.0, 10.05), next_cell_spike=True, **kwargs):
    """Three cells fire at every sample of [burst start, burst end); a fourth
    fires once at the first sample of the following bin."""
    time = origin + np.arange(int(20 * fs)) / fs
    spikes = np.zeros((len(time), 4))
    first, stop = round(burst[0] * fs), round(burst[1] * fs)
    spikes[first:stop, :3] = 1
    if next_cell_spike:
        spikes[stop, 3] = 1
    rec = lm.Recording.from_arrays(
        time, fs, multiunit=spikes, place_cells=np.arange(4), **kwargs
    )
    return rec, first, stop


_EXACT_RATE = {
    "threshold": 1000.0,
    "bound_threshold": 1000.0,
    "normalization_method": "none",
    "minimum_duration": 0.0,
    "speed_threshold": np.inf,
}


@pytest.mark.parametrize("origin", [0.0, 1_700_000_000.0])
@pytest.mark.parametrize("fs", [1000.0, 1500.0])
def test_native_grid_bounds_are_the_first_and_last_samples_of_the_bins(origin, fs):
    rec, first, stop = _burst_recording(origin, fs)
    events = lm.population_trace(rec, bin_width=0.01).detect(**_EXACT_RATE)
    # Closed bounds on the recording's own timestamps: exactly the samples
    # counted in the event's bins, none from the next bin.
    np.testing.assert_array_equal(lm.bounds(events), [[rec.time[first], rec.time[stop - 1]]])
    active = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=4, units=np.arange(4)
    )
    assert not len(active)  # the fourth cell fired in the next bin
    assert len(
        rd.require_active_units(
            events, rec.multiunit, rec.time, minimum_active_units=3, units=np.arange(4)
        )
    )


@pytest.mark.parametrize("origin", [0.0, 1_700_000_000.0])
def test_events_matching_an_interval_are_contained_at_any_clock_origin(origin):
    rec, first, stop = _burst_recording(origin, 1000.0)
    events = lm.population_trace(rec, bin_width=0.01).detect(**_EXACT_RATE)
    edges = [[origin + 10.0, origin + 10.05]]
    assert len(lm.within_intervals(events, edges)) == 1
    # A bound off by less than the clock's resolution still counts as inside.
    ulp = float(np.spacing(origin + 10.0))
    shifted = lm.bounds(events) + np.array([[-ulp, ulp]])
    assert len(lm.within_intervals(shifted, [[rec.time[first], rec.time[stop - 1]]])) == 1


@pytest.mark.parametrize("origin", [0.0, 1_700_000_000.0])
@pytest.mark.parametrize("fs", [1000.0, 1500.0])
def test_interval_restricted_events_stay_inside_unaligned_intervals(origin, fs):
    rec, _, _ = _burst_recording(origin, fs, burst=(10.0, 10.3), next_cell_spike=False)
    interval = np.array([[origin + 10.0254, origin + 10.1808]])
    events = lm._detect_population_in(rec, interval, np.ones(4, bool), 0.0, **_EXACT_RATE)
    found = lm.bounds(events)
    assert len(found) == 1
    assert found[0, 0] >= interval[0, 0]
    assert found[0, 1] <= interval[0, 1]
    # Only bins lying wholly inside the interval are kept: the first bin that
    # starts at or after 10.0254 s, the last that ends at or before 10.1808 s.
    samples = rec.time[(rec.time >= interval[0, 0]) & (rec.time <= interval[0, 1])]
    assert samples[0] <= found[0, 0]
    assert found[0, 1] <= samples[-1]


@pytest.mark.parametrize("fs", [1000.0, 1500.0])
def test_duration_limits_on_sample_bounds_count_samples_inclusively(fs):
    rec, _, _ = _burst_recording(0.0, fs, next_cell_spike=False)
    events = lm.population_trace(rec, bin_width=0.001).detect(**_EXACT_RATE)
    # Fifty 1 ms bins last 50 ms by the package's inclusive sample count.
    assert len(lm.within_duration(events, 0.05, sampling_frequency=fs)) == 1
    assert not len(lm.within_duration(events, 0.05 + 1 / fs, sampling_frequency=fs))


def _uneven_time(origin, n):
    """1 kHz timestamps with every odd sample 0.3 ms early: steps of 0.7 and
    1.3 ms (median 1 ms, no gap), and every other 1 ms bin holds no sample."""
    return origin + np.arange(n) / 1000 - 0.0003 * (np.arange(n) % 2)


@pytest.mark.parametrize("origin", [0.0, 1_700_000_000.0])
@pytest.mark.parametrize("grid", ["500 Hz", "uneven 1 kHz"])
def test_bounds_stay_on_counted_samples_when_edge_bins_are_empty(origin, grid):
    if grid == "500 Hz":
        time = origin + np.arange(10_000) / 500
        fs = 500.0
    else:
        time = _uneven_time(origin, 20_001)  # an even number of steps: median 1 ms
        fs = 1000.0
    spikes = np.zeros((len(time), 3))
    burst = (time - origin >= 10.0) & (time - origin < 10.3)
    spikes[burst] = 1
    rec = lm.Recording.from_arrays(time, fs, multiunit=spikes)
    trace = lm.population_trace(rec, bin_width=0.001, smoothing_sigma=0.005)
    assert np.isnan(trace.first_sample).any()  # the grid has empty bins
    found = lm.bounds(
        trace.detect(**{**_EXACT_RATE, "threshold": 500.0, "bound_threshold": 500.0})
    )
    assert len(found)
    assert np.isfinite(found).all()
    # Every bound is a recorded timestamp inside the event's own bins.
    assert np.isin(found, time).all()
    assert (found[:, 0] <= found[:, 1]).all()


@pytest.mark.parametrize("origin", [0.0, 1_700_000_000.0])
def test_ji_frames_stay_inside_unaligned_sleep_intervals(origin, monkeypatch):
    sleep = [[origin + 10.024, origin + 10.176]]
    rec, _, _ = _burst_recording(
        origin, 1000.0, burst=(10.0, 10.3), next_cell_spike=False, sleep_intervals=sleep
    )
    # Isolate the sleep restriction from the histogram-minimum level.
    monkeypatch.setattr(lm.rd, "histogram_minimum_threshold", lambda *a, **k: 0.5)
    found = lm.bounds(lm.ji_2007(rec))
    assert len(found)
    assert (found[:, 0] >= sleep[0][0]).all()
    assert (found[:, 1] <= sleep[0][1]).all()


@pytest.mark.parametrize("entry", [*lm.RECIPES, *lm.VARIANTS], ids=lambda e: e.run.__name__)
def test_public_help_is_self_contained(entry):
    """help() and list_methods show the interpretation itself, never a pointer
    to a private helper the reader cannot see."""
    assert not re.search(r"\b_[a-z]", entry.note), entry.note
    assert len(entry.note) > 60, entry.note


def test_tirole_and_pfeiffer_help_state_their_rules_and_channels():
    catalog = lm.list_methods().set_index("name").interpretation
    assert "first selected LFP channel" in catalog["tirole_2022"]
    assert "41-point" in catalog["tirole_2022"]
    assert "every selected channel" in catalog["pfeiffer_2015"]
    assert "12.5 ms Gaussian" in catalog["pfeiffer_2015"]


SECONDARY_INVENTORIES = {
    "widloski_2025",
    "kaefer_2020",
    "widloski_2025_bursts",
    "mallory_2025_ripples",
    "bush_2022_ripples",
    "igata_2021_ripples",
    "gridchyn_2020_ripples",
    "xu_2019_ripples",
    "farooq_2019_neuron_ripples",
    "farooq_2019_science_ripples",
    "liu_2019_ripples",
    "drieu_2018_ripples",
    "olafsdottir_2017_ripples",
    "wu_2014_ripples",
    "pfeiffer_2013_ripples",
    "davidson_2009_ripples",
    "diba_2007_ripples",
    "ji_2007_ripples",
    "lee_2002_ripples",
    "foster_2006_ripples",
    "krause_2022_hse",
    "denovellis_2021_mua",
    "gillespie_2021_mua",
    "muessig_2019_ripples",
    "bhattarai_2020_ripples",
}


def test_secondary_is_a_role_not_part_of_the_output_description():
    catalog = lm.list_methods().set_index("name")
    assert set(catalog.index[catalog.role == "secondary"]) == SECONDARY_INVENTORIES
    assert set(catalog.role) == {"candidate_detection", "secondary", "candidate_gate"}
    for word in ("secondary", "separate"):
        assert not catalog.output.str.contains(word).any(), word


def test_paper_labels_use_the_surveys_spelling():
    survey = rd.load_literature_parameters()
    for entry in (*lm.RECIPES, *lm.VARIANTS):
        assert entry.paper.startswith(survey.loc[entry.row, "First Author"]), entry.paper


def test_not_reproduced_names_the_methods_that_remain():
    catalog = lm.list_methods().set_index("name")
    for row, (paper, reason, methods) in lm.NOT_REPRODUCED.items():
        assert paper
        assert reason
        for name in methods:
            assert lm._ENTRIES[name].row == row, name
            assert catalog.loc[name, "role"] in {"secondary", "candidate_gate"}, name
    assert lm.NOT_REPRODUCED[1][2] == ("widloski_2025", "widloski_2025_bursts")
    assert lm.NOT_REPRODUCED[9][2] == ()


def test_list_methods_defines_every_column_once():
    doc = inspect.getdoc(lm.list_methods)
    terms = [
        term
        for line in doc.splitlines()
        if re.fullmatch(r"    [a-z_]+(, [a-z_]+)*", line)
        for term in line.strip().split(", ")
    ]
    assert sorted(terms) == sorted(lm.list_methods().columns)


# ---------------------------------------------------------------- declared requirements


def _defaults(name):
    """The implementation's keyword defaults, for evaluating ``when`` conditions."""
    return {
        key: parameter.default
        for key, parameter in inspect.signature(lm._IMPLEMENTATIONS[name]).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _applies(need, options):
    return all(options.get(option) == value for option, value in need.when)


def _scenarios(entry):
    """Default options, then each option setting a requirement is conditional on."""
    conditions = sorted({need.when for need in entry.requirements if need.when}, key=repr)
    return [{}] + [dict(condition) for condition in conditions]


REQUIREMENT_CASES = [
    (entry.run.__name__, index, need.input)
    for entry in (*lm.RECIPES, *lm.VARIANTS)
    for index, scenario in enumerate(_scenarios(entry))
    for need in entry.requirements
    if _applies(need, _defaults(entry.run.__name__) | scenario)
]


def _declared_call(name, index, remove=None):
    """A measured recording and call holding exactly the method's declared
    requirements under scenario ``index``, less ``remove``."""
    entry = lm._ENTRIES[name]
    scenario = _scenarios(entry)[index]
    options_now = _defaults(name) | scenario
    active = [need for need in entry.requirements if _applies(need, options_now)]
    everything, table_options = _method_inputs(name)
    n_time = len(everything["time"])
    keep = {"time", "sampling_frequency"} | {need.input for need in active}
    inputs = {key: value for key, value in everything.items() if key in keep}
    options = {
        key: value
        for key, value in table_options.items()
        if key in keep or key not in _defaults(name)  # the signature requires it
    } | scenario
    if "theta_delta" in keep:
        options["theta_delta"] = -np.ones(n_time)
    options.pop("behavior_intervals", None)
    if "behavior_intervals" in keep:
        options["behavior_intervals"] = [[everything["time"][0], everything["time"][-1]]]
    need = next((need for need in active if need.input == remove), None)
    if remove == "multiunit":
        for key in ("multiunit", "place_cells", "pyramidal", "templates"):
            inputs.pop(key, None)
    elif remove == "lfps" and need.minimum > 1:
        inputs["lfps"] = inputs["lfps"][:, : need.minimum - 1]
    elif remove is not None:
        inputs.pop(remove, None)
        options.pop(remove, None)
    return lm.Recording.from_arrays(**inputs), options


@pytest.mark.parametrize(
    ("name", "index"),
    sorted({(name, index) for name, index, _ in REQUIREMENT_CASES}),
)
def test_each_method_runs_with_exactly_its_declared_requirements(name, index, monkeypatch):
    """The catalog cannot understate what a method needs: a recording holding
    only the declared inputs runs, and its population grid is the declared one."""
    widths = []
    original = lm.population_trace

    def spy(rec, *, bin_width, **kwargs):
        widths.append(bin_width)
        return original(rec, bin_width=bin_width, **kwargs)

    monkeypatch.setattr(lm, "population_trace", spy)
    rec, options = _declared_call(name, index)
    events = lm.run_method(name, rec, **options)
    assert isinstance(events, pd.DataFrame)
    declared = lm._ENTRIES[name].bin_width
    assert set(widths) <= {declared}, (name, widths, declared)


@pytest.mark.parametrize(("name", "index", "missing"), REQUIREMENT_CASES)
def test_removing_a_declared_requirement_fails_naming_it(name, index, missing):
    """The catalog cannot overstate either: without any one declared input the
    call fails naming it, and the method itself could not have run."""
    rec, options = _declared_call(name, index, remove=missing)
    with pytest.raises(ValueError, match=rf"(^|\n)- {missing}\b"):
        lm.run_method(name, rec, **options)
    implementation = lm._IMPLEMENTATIONS[name]
    parameters = inspect.signature(implementation).parameters
    if missing == "behavior_intervals" and missing not in parameters:
        return  # applied by the dispatcher as whole-event containment
    options.pop("behavior_intervals", None)
    if "behavior_intervals" in parameters:
        options["behavior_intervals"] = (
            None
            if missing == "behavior_intervals"
            else (np.array([[rec.time[0], rec.time[-1]]]))
        )
    # Without the pre-check a missing input fails, if only by a warning about
    # an empty channel set (every warning is an error here).
    with pytest.raises((ValueError, RuntimeWarning)):
        implementation(rec, **options)


@pytest.mark.parametrize(
    "entry", [e for e in (*lm.RECIPES, *lm.VARIANTS) if e.sampling_frequency]
)
def test_a_fixed_sampling_rate_is_checked_before_running(entry):
    name = entry.run.__name__
    inputs, options = _method_inputs(name)
    rate = 1000.0 if entry.sampling_frequency != 1000 else 1500.0
    time = np.arange(len(inputs["time"])) / rate
    rec = lm.Recording.from_arrays(**{**inputs, "time": time, "sampling_frequency": rate})
    with pytest.raises(ValueError, match=f"sampled at {entry.sampling_frequency:g} Hz"):
        lm.run_method(name, rec, **options)


def test_the_catalog_reports_the_declared_requirements():
    catalog = lm.list_methods().set_index("name")
    bush = catalog.loc["bush_2022_ripples"]
    assert bush.sampling_frequency == 4800
    assert bush.signals[0] == "lfps: the first selected channel: the highest theta SNR"
    assert "speed" in bush.signals
    assert catalog.loc["diba_2007", "intervals"] == (
        "behavior_intervals: track-end reward areas (measured data)",
    )
    assert catalog.loc["harvey_2023_code", "signals"][1].startswith(
        "sharp_wave_lfp: the stratum radiatum channel"
    )
    assert catalog.loc["mou_2022", "cells"] == (
        "place_cells: one template's cells (if stage='decoding_candidates')",
    )
    assert catalog.loc["mou_2022", "stages"] == ("detection", "decoding_candidates")
    assert catalog.loc["karlsson_2009", "stages"] == ("detection",)
    assert catalog.loc["wikenheiser_2013", "measured_options"][0].startswith("window_anchor")
    assert catalog.loc["carey_2019", "external_inputs"][0].startswith("example_ripples")
    assert catalog.loc["ji_2007", "bin_width"] == 0.01
    assert np.isnan(catalog.loc["karlsson_2009", "bin_width"])
    assert np.isnan(catalog.loc["karlsson_2009", "sampling_frequency"])
    krause = {need["input"]: need for need in catalog.loc["krause_2022", "requirements"]}
    assert krause["lfps"]["unless"] == "external_ripples"
    assert krause["place_cells"]["kind"] == "cells"
    assert catalog.loc["bendor_2012", "signals"] == ("multiunit",)


def test_check_method_reports_everything_a_call_lacks_at_once():
    """Yang on a bare measured recording used to take one run per missing input."""
    inputs = _measured_inputs()
    rec = lm.Recording.from_arrays(
        inputs["time"], 1500, lfps=inputs["lfps"], multiunit=inputs["multiunit"]
    )
    problems = lm.check_method("yang_2024", rec)
    named = [problem.split(" - ")[0].split(":")[0] for problem in problems]
    assert named == ["pyramidal", "sleep_intervals", "behavior_intervals", "external_ripples"]
    with pytest.raises(ValueError, match="cannot run") as error:
        lm.run_method("yang_2024", rec)
    for problem in problems:
        assert problem in str(error.value)


def test_check_method_is_empty_for_a_runnable_call_and_never_runs_it(monkeypatch):
    inputs, options = _method_inputs("yang_2024")
    rec = lm.Recording.from_arrays(**inputs)

    import functools

    @functools.wraps(lm._IMPLEMENTATIONS["yang_2024"])
    def never(*args, **kwargs):
        raise AssertionError

    monkeypatch.setitem(lm._IMPLEMENTATIONS, "yang_2024", never)
    assert lm.check_method("yang_2024", rec, **options) == []


def test_check_method_names_options_and_stages():
    rec = lm.Recording.from_arrays(**_measured_inputs())
    problems = lm.check_method("xu_2019_ripples", rec, window=0.1)
    assert [p.split(" - ")[0] for p in problems] == ["window", "rms_window", "bound_threshold"]
    assert "its options are: rms_window, bound_threshold" in problems[0]
    assert lm.check_method("shin_2019", rec, stage="replay") == [
        "stage='replay' - expected 'detection' or 'decoding_candidates'"
    ]
    # A signature problem is a TypeError that still lists the other problems.
    bare = lm.Recording.from_arrays(rec.time, 1500)
    with pytest.raises(TypeError, match=r"(?s)- window .*- lfps.*- speed"):
        lm.run_method("karlsson_2009", bare, window=0.1)
    with pytest.raises(KeyError, match="Unknown literature method"):
        lm.check_method("unrecognized", rec)


@pytest.mark.parametrize(
    ("query", "listed"),
    [
        ("Pfeiffer 2013", "pfeiffer_2013, pfeiffer_2013_ripples"),
        ("https://doi.org/10.1038/nature12112", "pfeiffer_2013, pfeiffer_2013_ripples"),
        ("10.1038/NATURE12112", "pfeiffer_2013, pfeiffer_2013_ripples"),
        ("Olafsdottir 2016", "olafsdottir_2016"),
        ("Ólafsdóttir 2016", "olafsdottir_2016"),
        ("nadasdy", "nadasdy_1999"),
    ],
)
def test_a_paper_or_doi_lists_its_methods_without_choosing_one(measured, query, listed):
    for function in (lm.run_method, lm.check_method):
        with pytest.raises(KeyError, match=f"methods for that paper: {listed}[.;]"):
            function(query, measured)


def test_an_unknown_name_suggests_close_ones(measured):
    with pytest.raises(KeyError, match="Did you mean karlsson_2009"):
        lm.run_method("karlson_2009", measured)
    with pytest.raises(KeyError, match=r"list_methods\(\)"):
        lm.run_method("no_such_method", measured)


def test_tuning_keywords_explain_that_inventories_are_fixed(measured):
    inputs, _ = _method_inputs("kaefer_2020")
    rec = lm.Recording.from_arrays(**inputs)
    with pytest.raises(TypeError, match="fixed published rules") as error:
        lm.run_method("kaefer_2020", rec, zscore_threshold=4)
    assert "its options are: none" in str(error.value)
    with pytest.raises(TypeError, match="its options are: stage, histogram_bins"):
        lm.ji_2007(measured, zscore_threshold=4)


def test_harvey_code_without_radiatum_names_the_channel_and_the_alternative():
    inputs = _measured_inputs()
    del inputs["sharp_wave_lfp"]
    rec = lm.Recording.from_arrays(**inputs)
    with pytest.raises(
        ValueError, match="sharp_wave_lfp: the stratum radiatum channel"
    ) as error:
        lm.harvey_2023_code(rec)
    assert "harvey_2023_no_radiatum" in str(error.value)
