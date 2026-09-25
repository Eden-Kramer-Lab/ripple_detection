"""The shipped survey of published detection parameters."""

import csv
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
import pytest

from ripple_detection import load_literature_parameters

README = Path(__file__).resolve().parents[1] / "README.md"
LITERATURE = README.parent / "docs" / "literature"


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


def test_evidence_covers_every_field_once_and_links_each_paper(parameters):
    """Evidence joins to the table by DOI/column without caching its values."""
    with (LITERATURE / "evidence.csv").open() as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == [
            "doi",
            "column",
            "status",
            "source_location",
            "paper_note",
        ]
        evidence = list(reader)

    expected = {(doi, column) for doi in parameters.DOI for column in parameters}
    keys = [(entry["doi"], entry["column"]) for entry in evidence]
    assert len(keys) == len(set(keys))
    assert set(keys) == expected

    statuses = {
        "checked_paper",
        "checked_code",
        "checked_metadata",
        "derived",
        "inferred",
        "unresolved",
        "not_reported",
        "not_applicable",
        "reviewed_context",
    }
    paper_notes = {}
    for entry in evidence:
        assert entry["status"] in statuses
        assert entry["source_location"].strip()
        note = LITERATURE / entry["paper_note"]
        assert note.is_file()
        assert f"[Paper]({entry['doi']})" in note.read_text()
        paper_notes.setdefault(entry["doi"], set()).add(note)
    assert all(len(notes) == 1 for notes in paper_notes.values())
    assert {note for notes in paper_notes.values() for note in notes} == set(
        (LITERATURE / "papers").glob("*.md")
    )


def test_literature_navigation_has_no_missing_files_or_anchors():
    """Moving a source or note must not strand its evidence/navigation links."""
    for document in LITERATURE.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            link = urlsplit(target)
            if link.scheme or link.netloc:
                continue
            path = (document.parent / unquote(link.path)) if link.path else document
            assert path.exists(), f"{document.name}: {target}"
            if link.fragment:
                headings = re.findall(r"^#+ (.+)$", path.read_text(), re.MULTILINE)
                anchors = {
                    re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
                    for heading in headings
                }
                assert unquote(link.fragment) in anchors, f"{document.name}: {target}"


def test_source_fingerprints_identify_distinct_artifacts():
    text = (LITERATURE / "sources.md").read_text()
    table = text.split("## Artifact fingerprints\n")[1].split("\n## ")[0]
    rows = [line for line in table.splitlines() if line.startswith("| `")]
    assert rows
    names = []
    for row in rows:
        artifact, source, digest, size = [part.strip() for part in row.strip("|").split("|")]
        names.append(artifact)
        assert re.fullmatch(r"`[0-9a-f]{64}`", digest), artifact
        assert int(size) > 0, artifact
        assert re.fullmatch(r"\[[^]]+\]\(#[^)]+\)", source), artifact
    assert len(names) == len(set(names))
