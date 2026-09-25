"""The shipped survey of published detection parameters."""

import re
from pathlib import Path

import pandas as pd
import pytest

from ripple_detection import load_literature_parameters

README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.fixture(scope="module")
def parameters():
    return load_literature_parameters()


def test_loads_a_dataframe_of_papers(parameters):
    assert isinstance(parameters, pd.DataFrame)
    assert len(parameters) == 57


def test_every_paper_has_an_author_a_year_and_a_doi(parameters):
    for column in ("First Author", "Year", "DOI"):
        assert parameters[column].notna().all()
        assert (parameters[column].astype(str).str.strip() != "").all()


def test_numeric_columns_are_numbers_or_missing(parameters):
    """`#N/A` reads as missing, so a column of thresholds is usable as one."""
    thresholds = parameters["SWR Z-score Thresh. (STD)"]

    assert pd.api.types.is_numeric_dtype(thresholds.dropna())
    assert thresholds.notna().any()


def test_two_papers_share_an_author_and_year(parameters):
    """Farooq & Dragoi 2019 appears twice, so rows are keyed by more than that."""
    duplicated = parameters.duplicated(subset=["First Author", "Year"], keep=False)

    assert parameters.loc[duplicated, "Journal"].nunique() == 2


def _readme_parameter_table():
    """The rows of the README's published-values table, keyed by parameter."""
    text = README.read_text()
    section = text.split("### Published parameter values")[1].split("Three cautions")[0]
    rows = {}
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| Parameter"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


@pytest.mark.parametrize(
    ("label", "column"),
    [
        ("`zscore_threshold` (ripple)", "SWR Z-score Thresh. (STD)"),
        ("`zscore_threshold` (multiunit)", "MUA Z-score Thresh. (STD)"),
        ("smoothing width (ripple)", "SWR smooth (ms)"),
        ("smoothing width (multiunit)", "MUA smooth (ms)"),
        ("`speed_threshold`", "Animal Speed (cm/s)"),
        ("`minimum_duration`", "Min. Duration (ms)"),
        ("`maximum_duration`", "Max Duration (ms)"),
        ("`minimum_active_units`", "Min. Cells (#)"),
        ("event grouping interval", "Combine Events Thresh. (ms)"),
    ],
)
def test_readme_table_matches_the_shipped_data(parameters, label, column):
    """The documented range cannot drift away from the data behind it."""
    table = _readme_parameter_table()
    assert label in table, f"{label} missing from the README table"
    stated_count, stated_range, stated_median = (
        table[label][0],
        table[label][1],
        table[label][2],
    )

    values = pd.to_numeric(parameters[column], errors="coerce").dropna()

    assert int(stated_count) == len(values)
    low, high = (float(x) for x in re.findall(r"\d+\.?\d*", stated_range)[:2])
    assert low == values.min()
    assert high == values.max()
    assert float(re.findall(r"\d+\.?\d*", stated_median)[0]) == values.median()
    stated_modes = [float(x) for x in re.findall(r"\d+\.?\d*", table[label][3])]
    assert stated_modes == values.mode().tolist()


def test_readme_ripple_bands_match_data(parameters):
    stated = _readme_parameter_table()["ripple band"]
    bands = parameters[["SWR Low Band (Hz)", "SWR High Band (Hz)"]].dropna()
    assert int(stated[0]) == len(bands)
    assert [float(x) for x in re.findall(r"\d+\.?\d*", stated[1])] == [
        bands.iloc[:, 0].min(),
        bands.iloc[:, 0].max(),
        bands.iloc[:, 1].min(),
        bands.iloc[:, 1].max(),
    ]
    assert [float(x) for x in re.findall(r"\d+\.?\d*", stated[2])] == bands.median().tolist()
    counts = bands.value_counts()
    most_common = counts.index[0]
    assert [float(x) for x in re.findall(r"\d+\.?\d*", stated[3])] == [
        *most_common,
        counts.iloc[0],
    ]


def test_readme_channel_counts_exclude_unknowns(parameters):
    stated = _readme_parameter_table()["ripple channels sampled"]
    channels = parameters["SWR electrodes (#)"].dropna()
    numeric = pd.to_numeric(channels, errors="coerce")
    one = int((numeric == 1).sum())
    multiple = int(((numeric > 1) | (channels == ">1")).sum())
    ranges = int(channels.str.fullmatch(r"\d+-\d+").sum())
    assert int(stated[0]) == one + multiple + ranges
    assert [int(x) for x in re.findall(r"\d+", stated[1])] == [one, multiple, ranges]
