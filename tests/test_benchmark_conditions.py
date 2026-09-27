"""The benchmark's simulation conditions (examples/benchmark/conditions.py):
the grid, the reference parameters, the running schedule and the common
random numbers that pair conditions by replicate."""

import copy
import inspect
import json
import re

import numpy as np
import pandas as pd
import pytest

import ripple_detection as rd

SHORT = {"session.duration_s": 60.0}

# Every condition, in order.
CONDITION_IDS = [
    "reference",
    "ripple_snr=low",
    "ripple_snr=high",
    "participation=low",
    "participation=high",
    "n_units=30",
    "n_units=120",
    "n_channels=1",
    "n_channels=16",
    "shared_noise_fraction=0.2",
    "shared_noise_fraction=0.8",
    "noise_type=brown",
    "event_rate=0.15",
    "event_rate=0.6",
    "type_mix=swr_only",
    "type_mix=hard",
    "burst_lag=0.0",
    "burst_lag=0.03",
    "ripple_chirp=none",
    "spike_leakage_rate=0",
    "spike_leakage_rate=6",
    "emg_rate=0",
    "emg_rate=3",
    "fast_gamma_rate=0",
    "fast_gamma_rate=6",
    "theta_burst_rate=0",
    "theta_burst_rate=18",
    "slow_amplitude=0",
    "slow_amplitude=8",
    "strength_correlation=coupled",
    "spatial_profile=local",
    "noise_modulation=varying",
    "fast_gamma_band=nearby",
    "spike_model=refractory",
    "envelope_power=quartic",
    "ripple_snr=low,participation=low",
    "ripple_snr=low,participation=high",
    "ripple_snr=high,participation=low",
    "ripple_snr=high,participation=high",
    "ripple_snr=low,spike_leakage_rate=0",
    "ripple_snr=low,spike_leakage_rate=6",
    "ripple_snr=high,spike_leakage_rate=0",
    "ripple_snr=high,spike_leakage_rate=6",
]

# The entries each factor sets, by dotted key.
FACTOR_KEYS = {
    "ripple_snr": {"events.ripple_snr"},
    "participation": {"events.participation"},
    "n_units": {"render.unit_counts"},
    "n_channels": {"render.n_channels", "non_events.n_channels"},
    "shared_noise_fraction": {"render.shared_noise_fraction"},
    "noise_type": {"render.noise_type"},
    "event_rate": {"events.event_rate"},
    "type_mix": {"events.type_probabilities"},
    "burst_lag": {"events.burst_lag"},
    "ripple_chirp": {"events.ripple_chirp"},
    "spike_leakage_rate": {"non_events.rates.spike_leakage"},
    "emg_rate": {"non_events.rates.emg"},
    "fast_gamma_rate": {"non_events.rates.fast_gamma"},
    "theta_burst_rate": {"non_events.rates.theta_burst"},
    "slow_amplitude": {"render.theta_amplitude", "render.delta_amplitude"},
    "strength_correlation": {"events.strength_correlation"},
    "spatial_profile": {
        "render.spatial_profile",
        "render.channel_occupancy",
        "render.channel_gain_range",
        "render.channel_delay",
    },
    "noise_modulation": {"render.noise_log_amplitude", "render.noise_modulation_period"},
    "fast_gamma_band": {"non_events.fast_gamma_frequency", "non_events.fast_gamma_band"},
    "spike_model": {"render.spike_model", "render.refractory_period"},
    "envelope_power": {"events.envelope_power"},
}

ALTERNATIVES = [
    "strength_correlation=coupled",
    "spatial_profile=local",
    "noise_modulation=varying",
    "fast_gamma_band=nearby",
    "spike_model=refractory",
    "envelope_power=quartic",
]

# The arguments simulate_condition supplies itself rather than from a section.
SUPPLIED = {"time", "events", "non_events", "running_intervals", "rng", "sampling_frequency"}
SECTION_FUNCTIONS = {
    "events": rd.draw_network_events,
    "non_events": rd.draw_non_events,
    "render": rd.simulate_network_session,
}


@pytest.fixture(scope="module")
def module(benchmark_import):
    return benchmark_import("conditions")


@pytest.fixture(scope="module")
def by_id(module):
    return {condition.condition_id: condition for condition in module.conditions()}


@pytest.fixture(scope="module")
def sessions(module, by_id):
    """Replicate 0 of the reference, ``ripple_snr=high``, ``event_rate=0.6``
    and the six alternative models, one minute each, and replicate 1 of the
    reference."""
    found = {
        (condition_id, 0): module.simulate_condition(by_id[condition_id], 0, SHORT)
        for condition_id in ["reference", "ripple_snr=high", "event_rate=0.6", *ALTERNATIVES]
    }
    found["reference", 1] = module.simulate_condition(by_id["reference"], 1, SHORT)
    return found


def _as_lists(value):
    """``value`` with every tuple a list, as JSON reads it back."""
    if isinstance(value, dict):
        return {key: _as_lists(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_lists(item) for item in value]
    return value


def _as_tuples(value):
    """``value`` with every list a tuple, as the simulator takes ranges."""
    if isinstance(value, dict):
        return {key: _as_tuples(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_as_tuples(item) for item in value)
    return value


def _assert_same_session(first, second):
    for name in ("lfps", "sharp_wave_lfp", "multiunit", "speed", "running_intervals"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    pd.testing.assert_frame_equal(first.events, second.events)
    pd.testing.assert_frame_equal(first.non_events, second.non_events)


def test_conditions_are_unique_and_complete(module, by_id):
    found = module.conditions()
    ids = [condition.condition_id for condition in found]
    assert ids == CONDITION_IDS
    assert len(set(ids)) == 43
    assert all(re.fullmatch(r"[A-Za-z0-9_.=,-]+", condition_id) for condition_id in ids)
    reference = found[0]
    assert (reference.factor, reference.level, reference.params) == (
        "reference",
        "reference",
        (),
    )
    for condition in found[1:]:
        keys = [key for key, _ in condition.params]
        assert len(keys) == len(set(keys)), condition.condition_id
        assert module.resolve(condition) != module.REFERENCE, condition.condition_id
        factors = condition.factor.split(",")
        labels = condition.level.split(",")
        assert condition.condition_id == ",".join(
            f"{factor}={label}" for factor, label in zip(factors, labels, strict=True)
        )
        assert set(keys) == set().union(*(FACTOR_KEYS[factor] for factor in factors))
        if len(factors) == 2:
            # a crossed cell is its two one-factor conditions together
            parts = [
                by_id[f"{factor}={label}"]
                for factor, label in zip(factors, labels, strict=True)
            ]
            assert dict(condition.params) == {**dict(parts[0].params), **dict(parts[1].params)}
            assert "reference" not in labels


def test_every_condition_simulates(module):
    """The simulator accepts every condition's values."""
    for condition in module.conditions():
        session = module.simulate_condition(condition, 0, {"session.duration_s": 5.0})
        n_channels = module.resolve(condition)["render"]["n_channels"]
        assert session.lfps.shape == (7500, n_channels), condition.condition_id


def test_reference_is_the_simulators_default(module):
    """``REFERENCE`` names every keyword of each call, and the calls with its
    values give what the simulator's defaults give."""
    reference = module.REFERENCE
    assert reference["session"] == {"duration_s": 600.0, "sampling_frequency": 1500.0}
    for section, function in SECTION_FUNCTIONS.items():
        keywords = set(inspect.signature(function).parameters) - SUPPLIED
        assert set(reference[section]) == keywords, section
    time = rd.simulate_time(60 * 1500, 1500.0)
    running = np.array([[20.0, 40.0]])
    events = rd.draw_network_events(time, running_intervals=running, rng=9)
    assert set(events.event_type) == set(rd.EVENT_TYPES)
    pd.testing.assert_frame_equal(
        rd.draw_network_events(time, running_intervals=running, rng=9, **reference["events"]),
        events,
    )
    non_events = rd.draw_non_events(time, running_intervals=running, rng=0)
    assert set(non_events.non_event_type) == set(rd.NON_EVENT_TYPES)
    pd.testing.assert_frame_equal(
        rd.draw_non_events(time, running_intervals=running, rng=0, **reference["non_events"]),
        non_events,
    )
    _assert_same_session(
        rd.simulate_network_session(
            time, events, non_events=non_events, running_intervals=running, rng=0
        ),
        rd.simulate_network_session(
            time,
            events,
            non_events=non_events,
            running_intervals=running,
            rng=0,
            **reference["render"],
        ),
    )


def test_common_random_numbers(module, sessions):
    assert [module.session_seed(k) for k in range(3)] == [20260924, 20260925, 20260926]
    reference = sessions["reference", 0]
    assert len(reference.running_intervals) == 1
    assert (reference.events.expression == "burst").sum() >= 5
    # another replicate is another draw
    assert not np.array_equal(
        sessions["reference", 1].running_intervals, reference.running_intervals
    )

    stronger = sessions["ripple_snr=high", 0]
    np.testing.assert_array_equal(stronger.running_intervals, reference.running_intervals)
    np.testing.assert_array_equal(stronger.events.center_time, reference.events.center_time)
    # ripple_snr sizes the swr and doublet ripples; weak ripples keep theirs
    ripples = reference.events.expression == "ripple"
    sized = ripples & reference.events.event_type.isin(["swr", "ripple_doublet"])
    assert sized.any()
    assert np.all(stronger.events.amplitude[sized] != reference.events.amplitude[sized])
    np.testing.assert_array_equal(
        stronger.events.amplitude[ripples & ~sized],
        reference.events.amplitude[ripples & ~sized],
    )

    # more events are more draws in the event stage alone: the other stages
    # draw from their own seeds
    busier = sessions["event_rate=0.6", 0]
    assert busier.events.event_id.nunique() > reference.events.event_id.nunique()
    np.testing.assert_array_equal(busier.running_intervals, reference.running_intervals)
    np.testing.assert_array_equal(busier.baseline_rates, reference.baseline_rates)
    pd.testing.assert_frame_equal(busier.non_events, reference.non_events)

    bursts = reference.events.expression == "burst"
    for condition_id in ALTERNATIVES:
        session = sessions[condition_id, 0]
        np.testing.assert_array_equal(session.running_intervals, reference.running_intervals)
        pd.testing.assert_frame_equal(
            session.events[["event_id", "event_type", "expression", "component"]],
            reference.events[["event_id", "event_type", "expression", "component"]],
        )
        np.testing.assert_array_equal(session.events.center_time, reference.events.center_time)
        np.testing.assert_array_equal(session.baseline_rates, reference.baseline_rates)
        participants = session.events.n_participants[bursts].to_numpy()
        expected = reference.events.n_participants[bursts].to_numpy()
        if condition_id != "strength_correlation=coupled":
            np.testing.assert_array_equal(participants, expected)
            continue
        # coupling redraws each burst's participation, which then recruits
        # from the same per-unit draws: more units where it rose, fewer
        # where it fell
        change = (
            session.events.participation[bursts].to_numpy()
            - reference.events.participation[bursts].to_numpy()
        )
        assert np.any(change != 0)
        assert np.all((participants - expected) * change >= 0)
        assert np.any(participants != expected)


def test_running_schedule(module):
    for duration in (0.0, 20.0, 34.9, 35.0, 60.0, 600.0, 3600.0):
        for seed in range(100):
            bouts = module.running_schedule(duration, np.random.default_rng(seed))
            assert bouts.shape[1] == 2
            if duration < 35.0:
                assert bouts.shape == (0, 2)
                continue
            edges = np.concatenate([[0.0], bouts.ravel(), [duration]])
            lengths = np.diff(edges)
            rests, final_rest = lengths[:-1:2], lengths[-1]
            assert np.all((rests >= 20.0) & (rests <= 40.0))
            assert np.all((lengths[1::2] >= 10.0) & (lengths[1::2] <= 20.0))
            # the last rest ends the session: at least 5 s, and short enough
            # that no further rest and bout could have fitted with 5 s to spare
            assert 5.0 <= final_rest < 40.0 + 20.0 + 5.0


def test_running_schedule_rejects_a_non_finite_duration(module):
    with pytest.raises(ValueError, match="duration_s must be finite"):
        module.running_schedule(np.inf, np.random.default_rng(0))


def test_resolve_applies_params_then_overrides(module, by_id):
    reference = copy.deepcopy(module.REFERENCE)
    resolved = module.resolve(by_id["emg_rate=3"], {"session.duration_s": 60.0})
    assert resolved["non_events"]["rates"] == {
        **reference["non_events"]["rates"],
        "emg": 3.0,
    }
    assert resolved["session"]["duration_s"] == 60.0
    for section in ("events", "render"):
        assert resolved[section] == reference[section]
    # an override comes after the condition's own value
    later = module.resolve(by_id["ripple_snr=high"], {"events.ripple_snr": (1.0, 2.0)})
    assert later["events"]["ripple_snr"] == (1.0, 2.0)
    # the result is a copy: changing it changes neither REFERENCE nor the condition
    condition = by_id["n_units=30"]
    resolved = module.resolve(condition)
    resolved["render"]["unit_counts"]["place"] = 0
    resolved["non_events"]["rates"]["emg"] = 0.0
    assert reference == module.REFERENCE
    assert module.resolve(condition)["render"]["unit_counts"]["place"] == 20
    # and each call of conditions() is its own copy
    changed = {c.condition_id: c for c in module.conditions()}["n_units=30"]
    dict(changed.params)["render.unit_counts"]["place"] = 0
    fresh = {c.condition_id: c for c in module.conditions()}["n_units=30"]
    assert dict(fresh.params)["render.unit_counts"]["place"] == 20


@pytest.mark.parametrize(
    ("key", "match"),
    [
        ("sessions.duration_s", "Unknown section 'sessions'"),
        ("", "Unknown section ''"),
        ("events", "Unknown parameter 'events'"),
        ("events.ripple_snrs", "Unknown parameter 'events.ripple_snrs'"),
        ("non_events.rates.lfp_artifact", "Unknown parameter 'non_events.rates.lfp_artifact'"),
        ("events.ripple_snr.low", "Unknown parameter 'events.ripple_snr.low'"),
        ("events.bursts.ripple_snr", "Unknown parameter 'events.bursts.ripple_snr'"),
        ("render.time", "Unknown parameter 'render.time'"),
    ],
)
def test_an_unknown_key_raises(module, by_id, key, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        module.resolve(by_id["reference"], {key: 1.0})
    condition = module.Condition("custom=1", "custom", "1", ((key, 1.0),))
    with pytest.raises(ValueError, match=re.escape(match)):
        module.resolve(condition)
    with pytest.raises(ValueError, match=re.escape(match)):
        module.simulate_condition(condition, 0)


def test_resolved_json_round_trips(module, by_id):
    texts = {}
    for condition in module.conditions():
        text = module.resolved_json(condition)
        loaded = json.loads(text)
        assert loaded == _as_lists(module.resolve(condition))
        assert text == json.dumps(loaded, sort_keys=True)
        texts[condition.condition_id] = text
    assert len(set(texts.values())) == len(texts)
    with pytest.raises(ValueError, match="not JSON compliant"):
        module.resolved_json(by_id["reference"], {"render.noise_amplitude": float("nan")})


def test_a_saved_specification_reproduces_the_session(module, by_id):
    """The resolved parameters alone, reloaded from JSON and applied to the
    reference, simulate the condition's session."""
    condition = by_id["spatial_profile=local"]
    overrides = {"session.duration_s": 10.0}
    loaded = _as_tuples(json.loads(module.resolved_json(condition, overrides)))
    flat = {
        f"{section}.{name}": value
        for section, values in loaded.items()
        for name, value in values.items()
    }
    _assert_same_session(
        module.simulate_condition(by_id["reference"], 2, flat),
        module.simulate_condition(condition, 2, overrides),
    )
