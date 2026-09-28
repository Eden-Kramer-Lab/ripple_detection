"""The benchmark's analyses (examples/benchmark/analyze.py): the paired bootstrap
and sign-flip test, the held-out split and the results size limit."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def analyze(benchmark_import):
    return benchmark_import("analyze")


def _design_bootstrap(frame, statistic, *, key, n_resamples, seed=0, level=0.95):
    """The paired bootstrap written as plainly as possible: one concatenated
    frame per resample, each drawn value's rows relabelled with its draw."""
    rng = np.random.default_rng(seed)
    values = frame[key].unique()
    groups = dict(tuple(frame.groupby(key)))
    draws = []
    for _ in range(n_resamples):
        pick = rng.choice(values, size=values.size, replace=True)
        resampled = [
            groups[v].assign(
                session_id=groups[v].session_id.astype(str) + f"#{k}",
                replicate=groups[v].replicate.astype(str) + f"#{k}",
            )
            for k, v in enumerate(pick)
        ]
        draws.append(statistic(pd.concat(resampled, ignore_index=True)))
    draws = pd.DataFrame(draws)
    alpha = (1 - level) / 2
    return pd.DataFrame(
        {
            "estimate": statistic(frame),
            "low": draws.quantile(alpha),
            "high": draws.quantile(1 - alpha),
        }
    )


def _per_session(values_a, shift):
    """Two methods on each session, the second ``shift`` above the first."""
    n = len(values_a)
    return pd.DataFrame(
        {
            "session_id": [f"reference/{k}" for k in range(n)] * 2,
            "replicate": list(range(n)) * 2,
            "method": ["a"] * n + ["b"] * n,
            "value": [*values_a, *(v + shift for v in values_a)],
        }
    )


def _mean_by_method(frame):
    return frame.groupby("method")["value"].mean()


def test_paired_bootstrap_by_hand(analyze):
    frame = _per_session([1.0, 4.0, 2.0, 8.0, 5.0, 3.0], shift=10.0)
    got = analyze.paired_bootstrap(frame, _mean_by_method, key="session_id", n_resamples=400)
    # the estimate is the full sample's statistic, inside its interval
    assert got.estimate.tolist() == [23 / 6, 23 / 6 + 10]
    assert (got.low < got.estimate).all()
    assert (got.estimate < got.high).all()
    # every method gets the same draws: b's interval is a's moved by the shift
    np.testing.assert_allclose(
        got.loc["b", ["low", "high"]], got.loc["a", ["low", "high"]] + 10
    )
    # and the draws are the plain algorithm's, value for value
    pd.testing.assert_frame_equal(
        got, _design_bootstrap(frame, _mean_by_method, key="session_id", n_resamples=400)
    )


def test_paired_bootstrap_counts_a_session_drawn_twice_twice(analyze):
    frame = _per_session([1.0, 2.0, 3.0], shift=0.0)
    # sessions per resample, grouped by the relabelled id: always all three draws
    sizes = analyze.paired_bootstrap(
        frame,
        lambda f: pd.Series(
            {"sessions": f.session_id.nunique(), "replicates": f.replicate.nunique()}
        ),
        key="session_id",
        n_resamples=50,
    )
    assert sizes.low.tolist() == [3, 3]
    assert sizes.high.tolist() == [3, 3]


def _mean_difference(frame):
    by_condition = frame.groupby("condition_id")["value"].mean()
    return pd.Series({"difference": by_condition["b"] - by_condition["a"]})


def test_paired_bootstrap_keeps_replicates_paired(analyze):
    values = [3, 9, 4, 1, 7, 6, 2, 8]
    n = len(values)
    frame = pd.DataFrame(
        {
            "session_id": [f"a/{k}" for k in range(n)] + [f"b/{k}" for k in range(n)],
            "condition_id": ["a"] * n + ["b"] * n,
            "replicate": list(range(n)) * 2,
            "value": [*values, *(v + 1 for v in values)],
        }
    )
    paired = analyze.paired_bootstrap(
        frame, _mean_difference, key="replicate", n_resamples=300
    )
    assert paired.loc["difference"].tolist() == [1.0, 1.0, 1.0]
    # drawing sessions instead breaks the pairs, and the interval opens
    unpaired = analyze.paired_bootstrap(
        frame, _mean_difference, key="session_id", n_resamples=300
    )
    assert unpaired.loc["difference", "low"] < 1 < unpaired.loc["difference", "high"]


def test_sign_flip_exact(analyze):
    assert analyze.sign_flip_test([1, 1, 1, 1]) == 2 / 16
    assert analyze.sign_flip_test([1, -1]) == 1.0
    # 17 sessions and more: random flips, never 0 (no flip of 999 matches all 20)
    assert analyze.sign_flip_test(np.ones(20), n_resamples=999) == 1 / 1000


def test_sign_flip_needs_finite_pairs(analyze):
    for differences in ([np.nan, 0.0], [np.nan, np.nan], [np.inf, 1.0]):
        with pytest.raises(ValueError, match="finite paired differences"):
            analyze.sign_flip_test(differences)
    assert np.isnan(analyze.sign_flip_test([]))


def test_held_out_membership_is_the_same_in_every_condition(analyze):
    reference, other = range(20), range(10)
    held_out = {k for k in reference if analyze.is_held_out(k)}
    assert {k for k in other if analyze.is_held_out(k)} == {1, 3, 5, 7, 9}
    assert all(analyze.is_held_out(k) == (k in held_out) for k in other)
    assert len(held_out) == 10
    assert sum(analyze.is_held_out(k) for k in other) == 5


def test_results_size_limit(analyze, tmp_path):
    path = tmp_path / "table.csv"
    with pytest.raises(ValueError, match="over the 1,000,000-byte limit"):
        analyze.write_result(path, b"x" * (analyze.SIZE_LIMIT + 1))
    assert not path.exists()
    analyze.write_result(path, b"x" * analyze.SIZE_LIMIT)
    assert path.stat().st_size == analyze.SIZE_LIMIT
