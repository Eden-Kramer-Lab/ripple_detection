"""Behavioral checks for packaged literature methods, beyond simulation coverage."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.signal import filtfilt, remez

import ripple_detection as rd
from ripple_detection import literature_methods as lm

RIPPLE_TIMES = [3, 6, 9, 12, 15, 18]


def _measured_inputs(fs=1500, origin=0.0):
    """from_arrays inputs for 20 s of simulated signals, as measured data."""
    time = np.arange(int(20 * fs)) / fs
    session = rd.simulate_session(
        time,
        RIPPLE_TIMES,
        n_channels=3,
        n_units=20,
        baseline_rate=1,
        ripple_rate_gain=40,
        ripple_duration=(0.08, 0.16),
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
        "behavior_intervals": [[time[0], time[-1]]],
        "templates": [np.arange(10)],
    }


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
    np.testing.assert_allclose(lm._tirole_bounds(time, z), [[0.399, 0.601]])
    z[:] = 0.2
    z[450:501] = 4
    # No z<0 crossing exists; each side independently uses z<=0.25.
    np.testing.assert_allclose(lm._tirole_bounds(time, z), [[0.449, 0.501]])


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
    monkeypatch.setattr(lm, "_tirole_bounds", lambda *args: np.empty((0, 2)))
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
    return inputs, {**VARIANT_OPTIONS, **RECIPE_OPTIONS}.get(name, {})


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


@pytest.mark.parametrize("entry", lm.VARIANTS, ids=lambda x: x.run.__name__)
def test_every_added_inventory_runs_with_explicit_inputs(measured, entry):
    rec = measured
    if entry.run.__name__ in {"bush_2022_ripples", "olafsdottir_2017_ripples"}:
        fs = 4800 if entry.run.__name__ == "bush_2022_ripples" else 1200
        time = np.arange(int(20 * fs)) / fs
        session = rd.simulate_session(
            time, [3, 6, 9, 12, 15, 18], n_units=20, n_channels=3, ripple_rate_gain=40, rng=21
        )
        rec = lm.Recording.from_arrays(
            time,
            fs,
            lfps=session.lfps,
            multiunit=session.multiunit,
            speed=np.zeros(len(time)),
            pyramidal=np.arange(20),
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        events = lm.run_method(
            entry.run.__name__, rec, **VARIANT_OPTIONS.get(entry.run.__name__, {})
        )
    assert isinstance(events, pd.DataFrame)
    assert events.attrs["method"] == entry.run.__name__
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
        lm.run_method("yang_2024", measured)
    with pytest.raises(KeyError, match="Unknown literature method"):
        lm.run_method("unrecognized", measured)


def test_run_method_applies_behavior_containment_and_preserves_context(measured):
    original = measured.behavior_intervals
    measured.behavior_intervals = np.array([[5.0, 10.0]])
    try:
        result = lm.run_method("pfeiffer_2013_ripples", measured)
        assert len(result)
        assert (result.start_time >= 5).all()
        assert (result.end_time <= 10).all()
        assert result.attrs["behavior_intervals_applied"]
        assert "duration limits" in result.attrs["interpretation"]
    finally:
        measured.behavior_intervals = original


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
    lfps = measured.session.lfps.copy()
    if name in {"ji_2007_ripples", "lee_2002_ripples", "foster_2006_ripples"}:
        # These rectified-LFP thresholds need a stronger ripple than the
        # generic envelope-based demonstration supplies.
        relative = measured.time - 9
        lfps[:, 0] += (
            50 * np.cos(2 * np.pi * 180 * relative) * np.exp(-0.5 * (relative / 0.035) ** 2)
        )
    inputs = {
        "time": measured.time,
        "sampling_frequency": measured.fs,
        "lfps": lfps,
        "multiunit": measured.multiunit,
        "speed": measured.speed,
        "reference_lfp": measured.reference_lfp,
        "place_cells": measured.place_cells,
        "pyramidal": measured.pyramidal,
        "sleep_intervals": measured.sleep_intervals,
        "baseline_intervals": measured.baseline_intervals,
    }
    clean = lm.Recording.from_arrays(**inputs)
    rec = lm.Recording.from_arrays(**inputs, artifact_intervals=[[8.99, 9.01]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
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
    rec = lm.Recording.from_arrays(
        time, 1000, multiunit=spikes, templates=[[1, 3, 6]], behavior_intervals=[[0, 2]]
    )
    events = lm.run_method("olafsdottir_2015", rec)
    np.testing.assert_allclose(lm.bounds(events), [[0.5, 0.52]])
    assert not len(lm.run_method("olafsdottir_2015", rec, minimum_active_units=7))


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
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
        behavior_intervals=[[0, 2.999]],
    )
    candidates = np.array([[0.5, 0.575], [1.5, 1.7]])
    monkeypatch.setattr(lm, "_population_with_ripple_peak", lambda *args: candidates.copy())
    np.testing.assert_allclose(lm.bounds(lm.grosmark_2016(rec)), candidates)
    np.testing.assert_allclose(
        lm.bounds(lm.grosmark_2016(rec, stage="decoding_candidates")), candidates[1:]
    )
    frame = pd.DataFrame(candidates, columns=["start_time", "end_time"])
    monkeypatch.setattr(lm, "_detect_population", lambda *args, **kwargs: frame.copy())
    np.testing.assert_allclose(lm.bounds(lm.olafsdottir_2017(rec)), candidates)
    np.testing.assert_allclose(
        lm.bounds(lm.olafsdottir_2017(rec, analysis="trajectory")), candidates[1:]
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
    events = lm._tirole_bounds(time, z)
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
        ("yang_2024", {}, "external_ripples"),
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
    from dataclasses import replace

    unrestricted = lm.run_method(name, measured)
    assert len(unrestricted) > 1
    rec = replace(measured, behavior_intervals=np.array([[5, 10]]))
    direct = getattr(lm, name)(rec)
    dispatched = lm.run_method(name, rec)
    pd.testing.assert_frame_equal(direct, dispatched)
    assert direct.attrs == dispatched.attrs
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
        ({"behavior_intervals": [0, 1]}, "Intervals"),
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
    # Bins centered on 0.40-0.59 s; bounds are their outer edges.
    np.testing.assert_allclose(lm.bounds(lm.mou_2022(rec)), [[0.395, 0.595]])
    np.testing.assert_allclose(
        lm.bounds(lm.mou_2022(rec, normalization="maximum")), [[-0.005, 0.995]]
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
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
    rec = lm.Recording.from_arrays(time, 1000, multiunit=spikes, place_cells=np.arange(5))
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
    assert catalog.loc["widloski_2025_bursts", "role"] == "secondary_label"
    assert catalog.loc["widloski_2025", "inventory"] == "default"
    result = lm.mallory_2025_ripples(measured)
    assert result.attrs["role"] == "candidate_detection"
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


def test_native_grid_events_are_reported_at_bin_edges():
    # Spikes fill 10.000-10.049 s: five complete 10 ms bins, 50 ms edge to edge.
    trace = _burst_trace([(10.0, 10.05)])
    events = _detect_bursts(trace)
    np.testing.assert_allclose(lm.bounds(events), [[10.0, 10.05]], atol=1e-9)
    np.testing.assert_allclose(events.duration, [0.05], atol=1e-9)
    # Duration limits count bins, so they agree with the reported edges.
    assert len(_detect_bursts(trace, minimum_event_duration=0.05, maximum_duration=0.05))
    assert not len(_detect_bursts(trace, minimum_event_duration=0.06))
    assert not len(_detect_bursts(trace, maximum_duration=0.04))
    # The merge accepts the edges it reports.
    np.testing.assert_allclose(trace.merge(events, 0.0), [[10.0, 10.05]], atol=1e-9)


def test_native_grid_close_event_gaps_are_measured_between_edges():
    trace = _burst_trace([(10.0, 10.05), (10.08, 10.1)])  # 30 ms from edge to edge
    kept_apart = _detect_bursts(trace, close_event_threshold=0.03, close_event_rule="merge")
    np.testing.assert_allclose(
        lm.bounds(kept_apart), [[10.0, 10.05], [10.08, 10.1]], atol=1e-9
    )
    merged = _detect_bursts(trace, close_event_threshold=0.031, close_event_rule="merge")
    np.testing.assert_allclose(lm.bounds(merged), [[10.0, 10.1]], atol=1e-9)
    np.testing.assert_allclose(trace.merge(kept_apart, 0.031), [[10.0, 10.1]], atol=1e-9)
    np.testing.assert_allclose(trace.merge(kept_apart, 0.03), lm.bounds(kept_apart), atol=1e-9)


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
    from dataclasses import replace

    lm.run_method(name, measured)  # runs with them
    with pytest.raises(ValueError, match="behavior_intervals"):
        lm.run_method(name, replace(measured, behavior_intervals=None))
