"""The benchmark's simulation conditions (examples/benchmark/conditions.py):
the grid, the reference parameters, the running schedule and the common
random numbers that pair conditions by replicate."""

import copy
import dataclasses
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
    # frozen, but a mapping among the values makes a condition unhashable
    with pytest.raises(TypeError, match="unhashable"):
        hash(by_id["n_units=30"])


def test_factor_levels(module):
    """Each factor's levels in the designed order, the reference's in place."""
    assert module.factor_levels("ripple_snr") == ("low", "reference", "high")
    assert module.factor_levels("n_units") == ("30", "reference", "120")
    assert module.factor_levels("noise_type") == ("reference", "brown")
    assert module.factor_levels("type_mix") == ("reference", "swr_only", "hard")
    assert module.factor_levels("ripple_chirp") == ("none", "reference")
    assert module.factor_levels("spike_model") == ("reference", "refractory")
    for factor in FACTOR_KEYS:
        levels = module.factor_levels(factor)
        assert levels.count("reference") == 1, factor
        one_factor = [c.level for c in module.conditions() if c.factor == factor]
        assert [level for level in levels if level != "reference"] == one_factor
    for name in ("reference", "ripple_snr,participation", "snr"):
        with pytest.raises(ValueError, match="Unknown factor"):
            module.factor_levels(name)


def test_every_condition_simulates(module):
    """The simulator accepts every condition's values."""
    for condition in module.conditions():
        session = module.simulate_condition(condition, 0, {"session.duration_s": 5.0})
        n_channels = module.resolve(condition)["render"]["n_channels"]
        assert session.lfps.shape == (7500, n_channels), condition.condition_id


def test_reference_is_the_simulators_default(module):
    """``REFERENCE`` names every keyword of each call, and the calls with its
    values, each recorded revision undone, give what the simulator's defaults
    give; each revision records the value it replaced."""
    reference = copy.deepcopy(module.REFERENCE)
    for revision in module.REFERENCE_REVISIONS:
        section, name = revision.key.split(".")
        assert reference[section][name] == revision.revised != revision.previous
        reference[section][name] = revision.previous
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


def test_select_conditions(module):
    assert module.select_conditions("all") == module.conditions()
    chosen = module.select_conditions("spatial_profile=local, reference")
    assert [c.condition_id for c in chosen] == ["reference", "spatial_profile=local"]
    # a crossed cell's id holds a comma: the longest known id wins
    crossed = module.select_conditions("ripple_snr=low,participation=low")
    assert [c.condition_id for c in crossed] == ["ripple_snr=low,participation=low"]
    both = module.select_conditions("ripple_snr=low,participation=low,ripple_snr=low")
    assert [c.condition_id for c in both] == [
        "ripple_snr=low",
        "ripple_snr=low,participation=low",
    ]
    for text in ("reference,nope", "ripple_snr=low,participation=none"):
        with pytest.raises(ValueError, match="Unknown condition"):
            module.select_conditions(text)
    with pytest.raises(ValueError, match="No condition id"):
        module.select_conditions(" , ")


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
        assert text == json.dumps(json.loads(text), sort_keys=True)
        # read back exactly: ranges as tuples again, mappings as mappings
        assert module.parameters_from_json(text) == module.resolve(condition)
        texts[condition.condition_id] = text
    assert len(set(texts.values())) == len(texts)
    loaded = module.parameters_from_json(texts["type_mix=hard"])
    assert loaded["events"]["ripple_snr"] == (2.5, 6.0)
    assert loaded["events"]["type_probabilities"]["weak_ripple"] == 0.3
    with pytest.raises(ValueError, match="not JSON compliant"):
        module.resolved_json(by_id["reference"], {"render.noise_amplitude": float("nan")})


def test_parameters_from_json_names_what_a_saved_set_lacks(module, by_id):
    saved = json.loads(module.resolved_json(by_id["reference"]))
    del saved["render"]["noise_type"]
    saved["events"]["ripple_snrs"] = [1.0, 2.0]
    with pytest.raises(ValueError, match=r"lack render\.noise_type; have events\.ripple_snrs"):
        module.parameters_from_json(json.dumps(saved))
    del saved["render"]
    with pytest.raises(ValueError, match="lack render, "):
        module.parameters_from_json(json.dumps(saved))


@pytest.mark.parametrize(
    ("section", "keyword", "entry"),
    [
        ("non_events", "rates", "emg"),
        ("render", "unit_counts", "interneuron"),
        ("render", "baseline_rate", "place"),
    ],
)
def test_parameters_from_json_names_a_lost_entry_of_a_mapping(
    module, by_id, section, keyword, entry
):
    """The simulator reads a left-out rate or unit count as 0, and a left-out
    baseline range as its own default: an entry lost from a saved set would
    change the session silently."""
    saved = json.loads(module.resolved_json(by_id["reference"]))
    del saved[section][keyword][entry]
    saved[section][keyword]["extra"] = 1.0
    key = f"{section}.{keyword}"
    with pytest.raises(ValueError, match=rf"lack {key}\.{entry}; have {key}\.extra"):
        module.parameters_from_json(json.dumps(saved))
    saved[section][keyword] = 1.0
    lost = rf"lack ({key}\.\w+, )+{key}\.\w+; have nothing else"
    with pytest.raises(ValueError, match=lost):
        module.parameters_from_json(json.dumps(saved))


def test_a_saved_mixture_of_event_types_may_leave_types_out(module, by_id):
    """A mixture is replaced whole (type_mix=swr_only), and a type it leaves
    out never occurs: a partial one is read as it is."""
    saved = json.loads(module.resolved_json(by_id["type_mix=swr_only"]))
    assert saved["events"]["type_probabilities"] == {"swr": 1.0}
    loaded = module.parameters_from_json(json.dumps(saved))
    assert loaded["events"]["type_probabilities"] == {"swr": 1.0}


def test_a_saved_specification_reproduces_the_session(module, by_id):
    """The resolved parameters alone, as conditions.csv saves them, simulate
    the condition's session."""
    condition = by_id["type_mix=swr_only"]
    overrides = {"session.duration_s": 40.0, "non_events.rates.emg": 3.0}
    parameters = module.parameters_from_json(module.resolved_json(condition, overrides))
    session = module.simulate_parameters(parameters, 2)
    _assert_same_session(session, module.simulate_condition(condition, 2, overrides))
    # the condition's and the override's values took effect
    assert set(session.events.event_type) == {"swr"}
    assert (session.non_events.non_event_type == "emg").sum() > 0


def test_reference_revisions_hold_the_current_values(module):
    """Each revision names a ``REFERENCE`` entry, and a key's latest revision
    holds its current value: a revision is recorded, never silent."""
    assert isinstance(module.REFERENCE_REVISIONS, tuple)
    revision = module.ReferenceRevision(
        "events.ripple_duration", (0.03, 0.15), (0.03, 0.1), "", ""
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        revision.revised = (0.0, 1.0)
    reference = module.resolve(module.conditions()[0])
    latest = {}
    for revision in module.REFERENCE_REVISIONS:
        assert isinstance(revision, module.ReferenceRevision)
        assert revision.previous != revision.revised, revision.key
        assert revision.reason, revision.key
        assert revision.evidence, revision.key
        latest[revision.key] = revision.revised
    for key, revised in latest.items():
        section, *path = key.split(".")
        value = reference[section]
        for name in path:
            value = value[name]
        assert value == revised, key
