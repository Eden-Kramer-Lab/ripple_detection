"""The benchmark's configurations of the packaged literature methods
(examples/benchmark/recipe_configs.py), run on a short simulated network
session through the installed API."""

import dataclasses
import gc
import io
import json
import re
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ripple_detection as rd
from ripple_detection import literature_methods
from ripple_detection.literature_methods import bounds

FS = 1500.0
DURATION = 60.0
RUNNING = np.array([[25.0, 35.0]])
UNIT_COUNTS = {"place": 20, "pyramidal": 5, "interneuron": 5}
UNIX_ORIGIN = 1_700_000_000.0
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# A methods.csv row's columns after session_id.
METHOD_COLUMNS = [
    "method",
    "setting",
    "doi",
    "role",
    "inventory",
    "stage",
    "primary_expression",
    "resolved_options",
    "input_policy",
    "assumptions",
    "interpretation",
]

# The expression each configuration is headlined against, reviewed per
# method against what its implemented events require.
PRIMARY_EXPRESSIONS = {
    "mallory_2025": "burst",
    "widloski_2025": "ripple",
    "yang_2024": "network",
    "huelin_gorriz_2023": "network",
    "harvey_2023_code": "ripple",
    "harvey_2023_text": "ripple",
    "liu_2023": "network",
    "tirole_2022": "network",
    "bush_2022": "burst",
    "berners_lee_2022": "burst",
    "krause_2022": "network",
    "mou_2022": "burst",
    "berners_lee_2021": "ripple",
    "denovellis_2021": "ripple",
    "gillespie_2021": "ripple",
    "michon_2021": "network",
    "igata_2021": "burst",
    "gridchyn_2020": "burst",
    "kaefer_2020": "ripple",
    "bhattarai_2020": "network",
    "stella_2019": "ripple",
    "xu_2019": "burst",
    "farooq_2019_neuron": "burst",
    "farooq_2019_science": "burst",
    "chenani_2019": "burst",
    "michon_2019": "network",
    "liu_2019": "burst",
    "shin_2019": "ripple",
    "carey_2019": "network",
    "muessig_2019": "network",
    "drieu_2018": "burst",
    "maboudi_2018": "burst",
    "olafsdottir_2017": "burst",
    "olafsdottir_2017.trajectory": "burst",
    "wu_2017": "burst",
    "yamamoto_2017": "network",
    "tang_2017": "ripple",
    "grosmark_2016": "network",
    "ambrose_2016": "ripple",
    "jadhav_2016": "ripple",
    "olafsdottir_2016": "burst",
    "silva_2015": "burst",
    "olafsdottir_2015": "burst",
    "olafsdottir_2015.bayesian_candidates": "burst",
    "pfeiffer_2015": "ripple",
    "wu_2014": "burst",
    "wikenheiser_2013": "ripple",
    "pfeiffer_2013": "burst",
    "carr_2012": "ripple",
    "bendor_2012": "burst",
    "gupta_2010": "ripple",
    "karlsson_2009": "ripple",
    "davidson_2009": "burst",
    "diba_2007": "burst",
    "ji_2007": "burst",
    "foster_2006": "burst",
    "lee_2002": "burst",
    "nadasdy_1999": "ripple",
    "kudrimoti_1999": "ripple",
    "harvey_2023_no_radiatum": "ripple",
    "mallory_2025_ripples": "ripple",
    "igata_2021_ripples": "ripple",
    "wu_2014_ripples": "ripple",
    "pfeiffer_2013_ripples": "ripple",
    "davidson_2009_ripples": "ripple",
    "ji_2007_ripples": "ripple",
    "lee_2002_ripples": "ripple",
    "foster_2006_ripples": "ripple",
    "widloski_2025_bursts": "burst",
    "krause_2022_hse": "burst",
    "denovellis_2021_mua": "burst",
    "gillespie_2021_mua": "burst",
    "maboudi_2018_open_field": "burst",
    "muessig_2019_ripples": "ripple",
    "bhattarai_2020_ripples": "ripple",
    "farooq_2019_science_awake": "burst",
    "liu_2019_awake": "burst",
}

# Inputs that are observations of the session; any other input a recording is
# given stands in for something the simulation lacks, and must be stated.
OBSERVED = {"lfps", "sharp_wave_lfp", "multiunit", "speed"}

# Configurations that exercise each stand-in: the external ripple inventory,
# Carey's example ripples, and the unreported options at their demonstration
# values, with rest as the sleep and baseline epochs.
STAND_INS = (
    "yang_2024",
    "grosmark_2016",
    "carey_2019",
    "stella_2019",
    "nadasdy_1999",
    "kudrimoti_1999",
    "wikenheiser_2013",
)

# One or more configurations per grid, input kind and expression, for the
# checks too slow to run on every configuration.
REPRESENTATIVE = (
    "karlsson_2009",
    "bendor_2012",
    "yang_2024",
    "carey_2019",
    "krause_2022",
    "kaefer_2020",
    "gridchyn_2020",
    "olafsdottir_2015",
    "wikenheiser_2013",
    "stella_2019",
)


@pytest.fixture(scope="module")
def recipe_configs(benchmark_import):
    return benchmark_import("recipe_configs")


def _simulate() -> rd.SimulatedSession:
    """One minute with a running bout: two stretches of rest, ripples strong
    enough for every threshold rule, and units of every type."""
    time = np.arange(int(DURATION * FS)) / FS
    events = rd.draw_network_events(
        time, running_intervals=RUNNING, ripple_snr=(6.0, 10.0), event_rate=0.5, rng=1
    )
    return rd.simulate_network_session(
        time, events, running_intervals=RUNNING, unit_counts=UNIT_COUNTS, rng=2
    )


@pytest.fixture(scope="module")
def session():
    return _simulate()


@pytest.fixture(scope="module")
def configs(recipe_configs):
    return {config.config_id: config for config in recipe_configs.RECIPES}


@pytest.fixture(scope="module")
def catalog():
    return literature_methods.list_methods().set_index("name")


def _run(module, session, config):
    recording = module.make_recording(session, config)
    return module.run_recipe(config, recording, module.behavior_intervals(session, config))


@pytest.fixture(scope="module")
def results(recipe_configs, session):
    return {
        config.config_id: _run(recipe_configs, session, config)
        for config in recipe_configs.RECIPES
    }


def _unconfigured(module, method):
    """A configuration of ``method`` with no options, as an excluded one would be."""
    return module.configure(method, "ripple")


def _supplied(recording) -> set[str]:
    """The ``Recording.from_arrays`` inputs a recording was given."""
    signals = recording.session
    given = {
        "lfps": signals.lfps.shape[1] > 0,
        "sharp_wave_lfp": np.isfinite(signals.sharp_wave_lfp).any(),
        "multiunit": signals.multiunit.shape[1] > 0,
        "speed": signals.speed is not None,
        "place_cells": recording.place_cells.any(),
        "pyramidal": recording.pyramidal.any(),
        "templates": bool(recording.templates),
        "sleep_intervals": recording.sleep_intervals is not None,
        "baseline_intervals": recording.baseline_intervals is not None,
        "reference_lfp": recording.reference_lfp is not None,
        "example_ripples": recording.example_ripples is not None,
        "external_ripples": recording.external_ripples is not None,
    }
    return {name for name, supplied in given.items() if supplied}


def _assert_same_recording(recording, other):
    for owner, other_owner in ((recording, other), (recording.session, other.session)):
        for field in dataclasses.fields(owner):
            if field.name == "session":
                continue
            first, second = getattr(owner, field.name), getattr(other_owner, field.name)
            np.testing.assert_array_equal(first, second, err_msg=field.name)


def test_configurations_and_exclusions_cover_the_catalog_once(recipe_configs, catalog):
    configured = {config.method for config in recipe_configs.RECIPES}
    excluded = set(recipe_configs.EXCLUSIONS)
    assert not configured & excluded
    assert configured | excluded == set(catalog.index)
    assert all(reason.strip() for reason in recipe_configs.EXCLUSIONS.values())


def test_config_ids_are_unique_and_name_their_method(recipe_configs):
    ids = [config.config_id for config in recipe_configs.RECIPES]
    assert len(ids) == len(set(ids))
    for config in recipe_configs.RECIPES:
        assert re.fullmatch(r"[a-z0-9_]+(\.[a-z0-9_]+)?", config.config_id)
        assert config.config_id.split(".")[0] == config.method
        assert config.input_policy == recipe_configs.INPUT_POLICY
    variants = {
        config.config_id: dict(config.options)
        for config in recipe_configs.RECIPES
        if config.config_id != config.method
    }
    assert variants == {
        "olafsdottir_2015.bayesian_candidates": {"minimum_active_units": 7},
        "olafsdottir_2017.trajectory": {"analysis": "trajectory"},
    }
    assert {"olafsdottir_2015", "olafsdottir_2017"} <= set(ids)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (("Karlsson_2009", "Karlsson_2009", "ripple"), "config_id"),
        (("karlsson_2009-x", "karlsson_2009", "ripple"), "config_id"),
        (("kay", "karlsson_2009", "ripple"), "config_id"),
        (("karlsson_2009x", "karlsson_2009", "ripple"), "config_id"),
        (("karlsson_2009.", "karlsson_2009", "ripple"), "config_id"),
        (("karlsson_2009..x", "karlsson_2009", "ripple"), "config_id"),
        (("karlsson_2009.a.b", "karlsson_2009", "ripple"), "config_id"),
        (("karlsson_2009", "karlsson_2009", "ripples"), "primary_expression"),
    ],
)
def test_a_malformed_configuration_raises(recipe_configs, fields, message):
    with pytest.raises(ValueError, match=message):
        recipe_configs.RecipeConfig(*fields)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ([("stage", "detection")], "tuple"),
        ((("stage",),), "pairs"),
        (((1, "detection"),), "pairs"),
        ((("stage", "detection"), ("stage", "detection")), "stage.*more than once"),
        ((("frequencies", [150.0, 250.0]),), "frequencies.*hashable"),
        ((("rec", None),), "rec"),
        ((("behavior_intervals", ((0.0, 1.0),)),), "behavior_intervals"),
    ],
)
def test_malformed_options_raise(recipe_configs, options, message):
    with pytest.raises(ValueError, match=message):
        recipe_configs.RecipeConfig("mou_2022", "mou_2022", "burst", options)


def test_every_configuration_is_hashable(recipe_configs):
    assert len({hash(config) for config in recipe_configs.RECIPES}) == len(
        recipe_configs.RECIPES
    )


def test_every_method_with_stages_runs_its_detection_stage(recipe_configs, catalog):
    for config in recipe_configs.RECIPES:
        options = dict(config.options)
        if "decoding_candidates" in catalog.loc[config.method, "stages"]:
            assert options.get("stage") == "detection", config.config_id
        else:
            assert "stage" not in options, config.config_id


def test_every_configuration_can_run_under_the_policy(recipe_configs, session):
    for config in recipe_configs.RECIPES:
        recording = recipe_configs.make_recording(session, config)
        eligible = recipe_configs.behavior_intervals(session, config)
        assert recipe_configs.check_recipe(config, recording, eligible) == [], config.config_id


def test_run_recipe_is_the_direct_public_call(recipe_configs, session, results):
    for config in recipe_configs.RECIPES:
        recording = recipe_configs.make_recording(session, config)
        direct = getattr(literature_methods, config.method)(
            recording,
            behavior_intervals=recipe_configs.behavior_intervals(session, config),
            **dict(config.options),
        )
        found = results[config.config_id]
        pd.testing.assert_frame_equal(found, direct, check_exact=True, obj=config.config_id)
        assert found.attrs == direct.attrs, config.config_id


def test_resolved_options_are_the_options_a_call_records(recipe_configs, results):
    for config in recipe_configs.RECIPES:
        assert (
            recipe_configs.resolved_options(config)
            == results[config.config_id].attrs["options"]
        ), config.config_id


def test_resolved_options_follow_the_packages_json_convention(recipe_configs, session):
    # NumPy scalars become Python numbers and non-finite values None, as in
    # a result's attrs["options"], whatever type the configuration holds.
    config = recipe_configs.configure(
        "olafsdottir_2015", "burst", ("minimum_active_units", np.int64(7))
    )
    resolved = recipe_configs.resolved_options(config)
    recorded = _run(recipe_configs, session, config).attrs["options"]
    assert resolved == recorded
    assert type(resolved["minimum_active_units"]) is type(recorded["minimum_active_units"])
    json.dumps(resolved, allow_nan=False)
    unbounded = recipe_configs.RecipeConfig(
        "nadasdy_1999",
        "nadasdy_1999",
        "ripple",
        (("rms_window", np.float64(0.004)), ("bound_threshold", -np.inf)),
    )
    assert recipe_configs.resolved_options(unbounded) == {
        "rms_window": 0.004,
        "bound_threshold": None,
    }
    assert type(recipe_configs.resolved_options(unbounded)["rms_window"]) is float


def test_every_configured_path_finds_events(results):
    # Every stand-in (rest as sleep, baseline and eligible epochs, the zero
    # reference, one place-cell template, the external and example ripple
    # inventories, the demonstration values), every stage and protocol variant
    # and every expression finds events on this session. muessig_2019 needs
    # pyramidal bursts above 3 SD for 100 ms, longer than any simulated here.
    assert {name for name, events in results.items() if events.empty} == {"muessig_2019"}


def test_each_exclusion_is_what_check_method_reports(recipe_configs, session):
    for method, reason in recipe_configs.EXCLUSIONS.items():
        config = _unconfigured(recipe_configs, method)
        problems = recipe_configs.check_recipe(
            config,
            recipe_configs.make_recording(session, config),
            recipe_configs.behavior_intervals(session, config),
        )
        # A reason starts with what it names ("rms_window, bound_threshold: ..."
        # or "input sampled at 4800 Hz: ..."): exactly what check_method reports.
        named = set(reason.split(": ", 1)[0].split(", "))
        assert named == {problem.split(" - ")[0] for problem in problems}, method


def test_primary_expressions_are_the_reviewed_table(recipe_configs):
    assert {
        config.config_id: config.primary_expression for config in recipe_configs.RECIPES
    } == PRIMARY_EXPRESSIONS


def test_primary_expressions_have_the_inputs_their_expression_needs(recipe_configs):
    for config in recipe_configs.RECIPES:
        inputs = set(recipe_configs.policy_inputs(config))
        ripple = bool(inputs & {"lfps", "external_ripples"})
        burst = "multiunit" in inputs
        expected = {
            "ripple": ripple,
            "burst": burst and not ripple,
            "network": ripple and burst,
        }[config.primary_expression]
        assert expected, config.config_id


def test_an_unknown_method_raises(recipe_configs, session, configs):
    config = recipe_configs.RecipeConfig(
        "not_a_method", "not_a_method", "ripple", input_policy=recipe_configs.INPUT_POLICY
    )
    recording = recipe_configs.make_recording(session, configs["bendor_2012"])
    for call in (
        lambda: recipe_configs.make_recording(session, config),
        lambda: recipe_configs.resolved_options(config),
        lambda: recipe_configs.run_recipe(config, recording),
    ):
        with pytest.raises(KeyError, match="not_a_method"):
            call()


def test_a_missing_option_raises(recipe_configs, session):
    config = _unconfigured(recipe_configs, "gridchyn_2020_ripples")
    recording = recipe_configs.make_recording(session, config)
    with pytest.raises(TypeError, match="rms_window"):
        recipe_configs.run_recipe(config, recording)


def test_an_option_the_method_does_not_take_raises(recipe_configs):
    config = recipe_configs.RecipeConfig(
        "karlsson_2009", "karlsson_2009", "ripple", (("threshold", 2.0),)
    )
    for call in (recipe_configs.policy_inputs, recipe_configs.resolved_options):
        with pytest.raises(TypeError, match="threshold"):
            call(config)


def test_a_missing_input_raises(recipe_configs, session, configs):
    chenani = configs["chenani_2019"]
    with pytest.raises(ValueError, match="behavior_intervals"):
        recipe_configs.run_recipe(chenani, recipe_configs.make_recording(session, chenani))
    bendor_inputs = recipe_configs.make_recording(session, configs["bendor_2012"])
    with pytest.raises(ValueError, match="lfps"):
        recipe_configs.run_recipe(configs["karlsson_2009"], bendor_inputs)


def test_another_input_policy_is_refused(recipe_configs, session, configs):
    config = dataclasses.replace(configs["karlsson_2009"], input_policy="measured")
    with pytest.raises(ValueError, match="input policy"):
        recipe_configs.make_recording(session, config)


def test_a_configuration_whose_provenance_is_stale_raises(recipe_configs, session, configs):
    # Changing the stage adds place_cells, whose stand-in the copied
    # assumptions do not state; a bare configuration names no input policy.
    stale = dataclasses.replace(
        configs["grosmark_2016"], options=(("stage", "decoding_candidates"),)
    )
    bare = recipe_configs.RecipeConfig("karlsson_2009", "karlsson_2009", "ripple")
    for config, message in ((stale, "assumptions"), (bare, "input policy")):
        for call in (
            recipe_configs.input_policy,
            recipe_configs.method_record,
            lambda config: recipe_configs.make_recording(session, config),
        ):
            with pytest.raises(ValueError, match=message):
                call(config)


def test_unit_selections_are_the_simulators_labels(recipe_configs, session, configs):
    types = session.unit_types
    assert set(types) == set(rd.UNIT_TYPES)
    recording = recipe_configs.make_recording(session, configs["farooq_2019_science"])
    np.testing.assert_array_equal(recording.place_cells, types == "place")
    np.testing.assert_array_equal(recording.pyramidal, np.isin(types, ["place", "pyramidal"]))
    templates = recipe_configs.make_recording(session, configs["olafsdottir_2015"]).templates
    assert len(templates) == 1
    np.testing.assert_array_equal(templates[0], types == "place")


def test_rest_is_the_recorded_samples_outside_every_running_bout(recipe_configs, session):
    rest = recipe_configs.rest_intervals(session)
    in_rest = rd.intervals_to_mask(session.time, rest)
    running = rd.intervals_to_mask(session.time, session.running_intervals)
    assert (in_rest != running).all()
    assert np.isin(rest, session.time).all()
    assert np.all(rest[1:, 0] > rest[:-1, 1])
    assert len(rest) == 2
    still = dataclasses.replace(session, running_intervals=np.empty((0, 2)))
    np.testing.assert_array_equal(
        recipe_configs.rest_intervals(still), [[session.time[0], session.time[-1]]]
    )


def test_a_session_without_rest_raises_rather_than_passing_no_epochs(
    recipe_configs, session, configs
):
    # Running throughout leaves no rest to stand in for sleep, baseline or
    # eligible epochs; an empty stand-in would give zero events, not an error.
    running = dataclasses.replace(
        session, running_intervals=np.array([[session.time[0], session.time[-1]]])
    )
    for call in (
        lambda: recipe_configs.rest_intervals(running),
        lambda: recipe_configs.make_recording(running, configs["yang_2024"]),
        lambda: recipe_configs.make_recording(running, configs["gridchyn_2020"]),
        lambda: recipe_configs.behavior_intervals(running, configs["chenani_2019"]),
    ):
        with pytest.raises(ValueError, match="no rest"):
            call()


def test_a_recording_holds_exactly_the_declared_inputs(recipe_configs, session):
    rest = recipe_configs.rest_intervals(session)
    configs = [
        *recipe_configs.RECIPES,
        *(_unconfigured(recipe_configs, method) for method in recipe_configs.EXCLUSIONS),
    ]
    for config in configs:
        recording = recipe_configs.make_recording(session, config)
        eligible = recipe_configs.behavior_intervals(session, config)
        declared = set(recipe_configs.policy_inputs(config))
        supplied = _supplied(recording) | (
            {"behavior_intervals"} if eligible is not None else set()
        )
        assert supplied == declared, config.config_id
        assert recording.fs == session.sampling_frequency
        np.testing.assert_array_equal(recording.time, session.time)
        for name, value in (
            ("sleep_intervals", recording.sleep_intervals),
            ("baseline_intervals", recording.baseline_intervals),
            ("behavior_intervals", eligible),
        ):
            if name in declared:
                np.testing.assert_array_equal(value, rest, err_msg=config.config_id)
        if "reference_lfp" in declared:
            assert not recording.reference_lfp.any()
        if "lfps" in declared:
            np.testing.assert_array_equal(recording.session.lfps, session.lfps)
        if "multiunit" in declared:
            np.testing.assert_array_equal(recording.multiunit, session.multiunit)


@pytest.mark.parametrize(
    ("method", "options", "inputs", "absent"),
    [
        # Krause's own SWRs need lfps and speed unless external ripples are
        # supplied, and the policy supplies none it does not declare.
        ("krause_2022", {}, {"lfps", "speed"}, {"external_ripples"}),
        ("harvey_2023_text", {"stage": "detection"}, {"baseline_intervals"}, {"multiunit"}),
        (
            "harvey_2023_text",
            {"stage": "decoding_candidates"},
            {"multiunit", "place_cells"},
            set(),
        ),
        ("wikenheiser_2013", {}, {"sleep_intervals"}, {"speed", "baseline_intervals"}),
        ("wikenheiser_2013", {"branch": "run_lia"}, {"speed"}, {"sleep_intervals"}),
        ("muessig_2019", {}, {"sleep_intervals"}, {"speed"}),
        ("muessig_2019", {"sample_speed_veto": True}, {"speed"}, set()),
    ],
)
def test_requirements_apply_under_their_own_conditions(
    recipe_configs, method, options, inputs, absent
):
    config = recipe_configs.RecipeConfig(
        method,
        method,
        "ripple",
        tuple(options.items()),
        recipe_configs.INPUT_POLICY,
    )
    declared = set(recipe_configs.policy_inputs(config))
    assert inputs <= declared
    assert not absent & declared


def test_unlabeled_units_are_not_selected(recipe_configs, session, configs):
    unlabeled = dataclasses.replace(session, unit_types=np.empty(0, dtype="<U11"))
    for name, missing in (
        ("olafsdottir_2015", {"templates"}),
        ("farooq_2019_science", {"pyramidal", "place_cells"}),
    ):
        config = configs[name]
        recording = recipe_configs.make_recording(unlabeled, config)
        assert not recording.place_cells.any()
        assert not recording.pyramidal.any()
        assert recording.templates == ()
        eligible = recipe_configs.behavior_intervals(unlabeled, config)
        problems = recipe_configs.check_recipe(config, recording, eligible)
        assert {problem.split(":")[0].split(" - ")[0] for problem in problems} == missing
        with pytest.raises(ValueError, match=next(iter(missing))):
            recipe_configs.run_recipe(config, recording, eligible)


def test_the_policy_reads_no_truth(recipe_configs, session, configs, results):
    blind = dataclasses.replace(
        session,
        events=session.events.iloc[:0],
        non_events=session.non_events.iloc[:0],
        ripple_times=np.empty(0),
        ripple_durations=np.empty(0),
        ripple_frequencies=np.empty(0),
        artifact_times=np.empty(0),
        baseline_rates=np.empty(0),
        ripple_channels=session.ripple_channels.iloc[:0],
    )
    for config in recipe_configs.RECIPES:
        _assert_same_recording(
            recipe_configs.make_recording(session, config),
            recipe_configs.make_recording(blind, config),
        )
        np.testing.assert_array_equal(
            recipe_configs.behavior_intervals(session, config),
            recipe_configs.behavior_intervals(blind, config),
        )
    for name in STAND_INS:
        pd.testing.assert_frame_equal(
            _run(recipe_configs, blind, configs[name]), results[name]
        )


def test_external_inventories_are_the_stated_detectors(recipe_configs, session, configs):
    # The settings are pinned here, not read from the module: the package's
    # Zugaro stand-in on channel 0 and the five largest default Kay events.
    filtered = rd.filter_ripple_band(session.lfps, FS, band=(130.0, 200.0), time=session.time)
    zugaro = rd.Zugaro_ripple_detector(
        session.time,
        filtered[:, :1],
        session.speed,
        FS,
        low_threshold=2.0,
        high_threshold=5.0,
        maximum_duration=0.2,
        speed_threshold=np.inf,
    )
    external = zugaro[["start_time", "end_time", "peak_time"]].to_numpy()
    kay = rd.Kay_ripple_detector(
        session.time,
        rd.filter_ripple_band(session.lfps, FS, time=session.time),
        session.speed,
        FS,
    )
    examples = kay.nlargest(5, "max_zscore")[["start_time", "end_time"]].to_numpy()
    assert len(external)
    assert len(examples) == 5
    for name, source, detector, expected in (
        ("yang_2024", "external_ripples", "Zugaro_ripple_detector", external),
        ("grosmark_2016", "external_ripples", "Zugaro_ripple_detector", external),
        ("carey_2019", "example_ripples", "Kay_ripple_detector", examples),
    ):
        recording = recipe_configs.make_recording(session, configs[name])
        np.testing.assert_array_equal(getattr(recording, source), expected, err_msg=name)
        record = recipe_configs.method_record(configs[name])
        assert json.loads(record["input_policy"])["recording"][source]["detector"] == detector
        assert any(
            assumption.startswith(source) and detector in assumption
            for assumption in json.loads(record["assumptions"])
        )


def test_input_policy_records_every_setting_of_each_detector(recipe_configs, configs):
    recorded = {}
    for name, source in (("yang_2024", "external_ripples"), ("carey_2019", "example_ripples")):
        policy = json.loads(recipe_configs.method_record(configs[name])["input_policy"])
        spec = policy["recording"][source]
        assert set(spec["options"]) == set(rd.get_detector(spec["detector"]).parameters)
        recorded[source] = spec["options"]
    # Configured values replace the defaults; a non-finite value is its repr,
    # since None is already a setting ("no limit", "no mask").
    assert recorded["external_ripples"] == {
        "speed_threshold": "inf",
        "low_threshold": 2.0,
        "high_threshold": 5.0,
        "minimum_inter_ripple_interval": 0.03,
        "minimum_duration": 0.02,
        "maximum_duration": 0.2,
        "smoothing_window": 0.0088,
        "normalization_mask": None,
    }
    assert recorded["example_ripples"] == {
        "speed_threshold": 4.0,
        "minimum_duration": 0.015,
        "zscore_threshold": 2.0,
        "smoothing_sigma": 0.004,
        "close_ripple_threshold": 0.0,
        "normalization_method": "zscore",
        "normalization_mask": None,
        "maximum_duration": None,
    }


@pytest.mark.parametrize(
    ("settings", "detector"),
    [
        ("EXTERNAL_RIPPLES", "Kay_ripple_detector"),
        ("EXAMPLE_RIPPLES", "Karlsson_ripple_detector"),
    ],
)
def test_the_recorded_detector_is_the_one_that_runs(
    recipe_configs, session, monkeypatch, settings, detector
):
    spec = getattr(recipe_configs, settings)
    monkeypatch.setitem(spec, "detector", detector)
    monkeypatch.setitem(spec, "options", {})
    filtered = rd.filter_ripple_band(session.lfps, FS, band=spec["band"], time=session.time)
    if settings == "EXTERNAL_RIPPLES":
        events = getattr(rd, detector)(session.time, filtered[:, :1], session.speed, FS)
        expected = events[["start_time", "end_time", "peak_time"]].to_numpy()
        found = recipe_configs.external_ripples(session)
    else:
        events = getattr(rd, detector)(session.time, filtered, session.speed, FS)
        expected = bounds(events.nlargest(5, "max_zscore"))
        found = recipe_configs.example_ripples(session)
    np.testing.assert_array_equal(found, expected)


def test_the_stand_ins_are_the_packages_simulation_proxies(
    recipe_configs, session, configs, results
):
    # The package's simulation fallbacks run when a Recording wraps the
    # SimulatedSession and the stand-in is not given: its assumed ripple
    # detector, its Kay examples and its demonstration values.
    rest = recipe_configs.rest_intervals(session)
    simulated = literature_methods.Recording(
        session,
        place_cells=session.unit_types == "place",
        pyramidal=np.isin(session.unit_types, ["place", "pyramidal"]),
        sleep_intervals=rest,
        baseline_intervals=rest,
    )
    for name in STAND_INS:
        config = configs[name]
        stage = {key: value for key, value in config.options if key == "stage"}
        proxy = literature_methods.run_method(
            config.method,
            simulated,
            behavior_intervals=recipe_configs.behavior_intervals(session, config),
            **stage,
        )
        assert len(proxy), name
        np.testing.assert_array_equal(bounds(proxy), bounds(results[name]), err_msg=name)


def test_missing_lfp_samples_end_events_and_ripple_inventories(
    recipe_configs, session, configs, results
):
    # 20 ms missing inside a ripple every method below finds (and the
    # external inventory holds, 13.564-13.621 s) on the complete session.
    missing = (session.time >= 13.58) & (session.time <= 13.60)
    lfps = session.lfps.copy()
    lfps[missing] = np.nan
    gapped = dataclasses.replace(session, lfps=lfps, raw_lfp=lfps[:, 0].copy())
    first, last = session.time[missing][[0, -1]]
    before, after = session.time[np.flatnonzero(missing)[[0, -1]] + [-1, 1]]

    def spanning(found):
        return np.any((found[:, 0] <= last) & (found[:, 1] >= first))

    for name in (
        "karlsson_2009",
        "gillespie_2021",
        "stella_2019",
        "nadasdy_1999",
        "harvey_2023_no_radiatum",
    ):
        assert spanning(bounds(results[name])), name
        events = _run(recipe_configs, gapped, configs[name])
        assert not spanning(bounds(events)), name
        if events.attrs["clipping_tracked"]:
            ending, starting = events.end_time == before, events.start_time == after
            assert ending.any() or starting.any(), name
            assert events.clipped_end[ending].all(), name
            assert events.clipped_start[starting].all(), name
    assert spanning(recipe_configs.external_ripples(session)[:, :2])
    assert not spanning(recipe_configs.external_ripples(gapped)[:, :2])
    assert len(_run(recipe_configs, gapped, configs["yang_2024"]))


def test_results_shift_with_the_clock_origin(recipe_configs, session, configs, results):
    shifted = dataclasses.replace(
        session,
        time=session.time + UNIX_ORIGIN,
        running_intervals=session.running_intervals + UNIX_ORIGIN,
    )
    # Timestamps near 1.7e9 s carry about 2.4e-7 s of rounding each.
    tolerance = 16 * np.spacing(UNIX_ORIGIN + DURATION)
    for name in REPRESENTATIVE:
        events = _run(recipe_configs, shifted, configs[name])
        expected = bounds(results[name])
        assert len(events) == len(expected), name
        np.testing.assert_allclose(
            bounds(events) - UNIX_ORIGIN, expected, rtol=0, atol=tolerance, err_msg=name
        )


def test_a_discarded_recording_is_collected(recipe_configs, session, configs):
    config = configs["yang_2024"]
    recording = recipe_configs.make_recording(session, config)
    events = recipe_configs.run_recipe(
        config, recording, recipe_configs.behavior_intervals(session, config)
    )
    reference = weakref.ref(recording)
    del recording
    gc.collect()
    assert reference() is None
    assert len(events)


def test_the_package_and_the_standalone_demo_import_no_benchmark_code():
    demo = (EXAMPLES / "literature_recipes.py").read_text()
    assert "benchmark" not in demo
    assert "recipe_configs" not in demo
    for path in Path(rd.__file__).parent.rglob("*.py"):
        assert "recipe_configs" not in path.read_text(), path


def test_method_records_are_methods_csv_rows(recipe_configs, catalog):
    records = []
    for config in recipe_configs.RECIPES:
        record = recipe_configs.method_record(config)
        assert list(record) == METHOD_COLUMNS
        assert all(isinstance(value, str) for value in record.values())
        assert record["method"] == f"recipe:{config.config_id}"
        assert record["setting"] == "literature"
        entry = catalog.loc[config.method]
        for column in ("doi", "role", "inventory", "interpretation"):
            assert record[column] == entry[column]
        assert record["primary_expression"] == config.primary_expression
        options = json.loads(record["resolved_options"])
        assert options == recipe_configs.resolved_options(config)
        assert record["stage"] == options.get("stage", "detection")
        assert json.loads(record["assumptions"]) == list(config.assumptions)
        policy = json.loads(record["input_policy"])
        assert policy["name"] == recipe_configs.INPUT_POLICY
        assert set(policy["recording"]) | (
            {"behavior_intervals"} if policy["behavior_intervals"] else set()
        ) == set(recipe_configs.policy_inputs(config))
        records.append(record)
    table = pd.DataFrame(records)
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    buffer.seek(0)
    pd.testing.assert_frame_equal(pd.read_csv(buffer, dtype=str, keep_default_na=False), table)


def test_assumptions_state_every_stand_in_and_unreported_value(recipe_configs, catalog):
    for config in recipe_configs.RECIPES:
        named = {re.match(r"[a-z_]+", text).group() for text in config.assumptions}
        stand_ins = set(recipe_configs.policy_inputs(config)) - OBSERVED
        unreported = {
            requirement["input"]
            for requirement in catalog.loc[config.method, "requirements"]
            if requirement["kind"] == "option" and requirement["measured_only"]
        }
        assert named == stand_ins | unreported, config.config_id
        options = dict(config.options)
        for name in unreported:
            assert f"{name}={options[name]!r}: unreported" in " ".join(config.assumptions)


def test_the_readme_lists_every_exclusion(recipe_configs):
    readme = (EXAMPLES / "benchmark" / "README.md").read_text()
    section = readme.split("## Exclusions", 1)[1].split("\n## ", 1)[0]
    items = re.findall(
        r"^- `([a-z0-9_]+)`: (.*?)(?=^- `|\Z)", section, re.MULTILINE | re.DOTALL
    )
    listed = {name: " ".join(reason.split()) for name, reason in items}
    assert listed == recipe_configs.EXCLUSIONS
