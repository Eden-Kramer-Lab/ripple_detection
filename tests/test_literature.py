"""The shipped survey of published detection parameters."""

import csv
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
import pytest

from ripple_detection import load_literature_datasets, load_literature_parameters

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


@pytest.fixture(scope="module")
def datasets():
    return load_literature_datasets()


def test_dataset_catalog_has_unique_relationships_to_surveyed_papers(datasets, parameters):
    assert isinstance(datasets, pd.DataFrame)
    assert not datasets.empty
    assert list(datasets) == [
        "paper_doi",
        "dataset_name",
        "dataset_url",
        "relationship",
        "available_inputs",
        "event_annotations",
        "verification_status",
        "source_note",
        "notes",
    ]
    assert not datasets.duplicated(["paper_doi", "dataset_url"]).any()
    joined = datasets.merge(
        parameters, left_on="paper_doi", right_on="DOI", validate="many_to_one"
    )
    assert len(joined) == len(datasets), "Every dataset must link to a surveyed DOI."
    assert datasets.paper_doi.duplicated().any(), "One paper can use multiple deposits."
    assert datasets.dataset_url.duplicated().any(), "Several papers can share a deposit."
    for column in datasets:
        assert (
            datasets[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        )
    for column in ("dataset_url", "source_note"):
        for url in datasets[column]:
            parsed = urlsplit(url)
            assert parsed.scheme == "https", url
            assert parsed.netloc, url
            assert not re.search(r"\s", url), url


def test_dataset_catalog_distinguishes_unverified_from_scoped_inspection(datasets):
    assert set(datasets.relationship) <= {"original", "reused", "not_verified"}
    assert set(datasets.verification_status) <= {
        "reported",
        "metadata_inspected",
        "files_inspected",
    }
    assert set(datasets.event_annotations) <= {
        "present",
        "reported_present",
        "not_found_in_inspected_scope",
        "not_verified",
    }
    for inputs in datasets.available_inputs:
        kinds = inputs.split(";")
        assert set(kinds) <= {"lfp", "spikes", "behavior", "electrophysiology", "not_verified"}
        assert len(kinds) == len(set(kinds))
        assert "not_verified" not in kinds or len(kinds) == 1
    inspected = datasets.event_annotations == "present"
    assert inspected.any()
    assert (datasets.loc[inspected, "verification_status"] == "files_inspected").all()
    scoped_absence = datasets.event_annotations == "not_found_in_inspected_scope"
    assert not (datasets.loc[scoped_absence, "verification_status"] == "reported").any()


def test_dataset_evidence_links_resolve_in_the_repository(datasets):
    prefix = "https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/"
    for entry in datasets.itertuples(index=False):
        assert entry.source_note.startswith(prefix)
        link = urlsplit(entry.source_note.removeprefix(prefix))
        path = README.parent / unquote(link.path)
        assert path.is_file(), entry.source_note
        text = path.read_text()
        if path.parent.name == "papers":
            assert f"[Paper]({entry.paper_doi})" in text
        if link.fragment:
            headings = re.findall(r"^#+ (.+)$", text, re.MULTILINE)
            anchors = {
                re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
                for heading in headings
            }
            assert unquote(link.fragment) in anchors, entry.source_note


def test_dataset_catalog_keeps_reused_recordings_and_trial_events_distinct(
    datasets, parameters
):
    linked = datasets.merge(parameters, left_on="paper_doi", right_on="DOI")
    bush = linked.loc[(linked["First Author"] == "Bush") & (linked.Year == 2022)].iloc[0]
    original = linked.loc[linked.dataset_url == bush.dataset_url]
    assert bush.relationship == "reused"
    assert ((original.Year == 2016) & (original.relationship == "original")).any()
    bhattarai = linked.loc[linked["First Author"] == "Bhattarai"].iloc[0]
    assert bhattarai.event_annotations == "not_found_in_inspected_scope"
    assert "trial/delay/reward" in bhattarai.notes
    # A code-only release must not be mistaken for Widloski's data deposit.
    widloski = linked.loc[(linked["First Author"] == "Widloski") & (linked.Year == 2025)]
    assert set(widloski.dataset_url) == {"https://zenodo.org/records/16916108"}


def test_dataset_loader_returns_independent_tables(datasets):
    other = load_literature_datasets()
    other.loc[0, "notes"] = "caller annotation"
    pd.testing.assert_frame_equal(load_literature_datasets(), datasets)


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
        ("minimum event duration", "Min. Duration (ms)"),
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
