"""Published detection parameters and associated public datasets, as tables.

The values every detector in this package needs are chosen differently in
every paper that reports them. :func:`load_literature_parameters` ships the
survey those choices were read from, so a threshold can be picked against what
the field does rather than against one lab's default.
"""

from importlib.resources import files

import pandas as pd

DATA_FILE = "literature_detection_parameters.csv"
DATASETS_FILE = "literature_datasets.csv"

NUMERIC_COLUMNS = (
    "Year",
    "MUA Z-score Thresh. (STD)",
    "SWR Z-score Thresh. (STD)",
    "Animal Speed (cm/s)",
    "MUA smooth (ms)",
    "SWR smooth (ms)",
    "Min. Cells (#)",
    "SWR Low Band (Hz)",
    "SWR High Band (Hz)",
    "Min. Duration (ms)",
    "Max Duration (ms)",
    "Combine Events Thresh. (ms)",
    "Time bin (ms)",
    "Time bin step (ms)",
    "Position bin (cm)",
    "Place field smooth (cm)",
    "Shuffles (#)",
)
"""Columns parsed as numbers. Entries that are not a bare number, such as
``">4"`` or ``"33%"``, become missing rather than being coerced to a value the
paper did not state."""


def load_literature_parameters() -> pd.DataFrame:
    """Detection and analysis parameters from 57 replay/reactivation papers.

    One row per paper, 1999-2025, compiled from Methods, supplements and released
    code. Field evidence and source limitations are documented in the
    literature guide,
    https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/docs/literature/README.md.
    Columns cover the trigger (ripple power, multiunit activity, both, or the
    decoded posterior), thresholds, smoothing widths, the ripple band,
    duration and merge limits, active-cell minima, the
    decoding settings, and the shuffle and significance procedure.

    Secondary/control detectors remain in the numeric columns; the detection
    notes distinguish them from the primary trigger and describe protocol
    variants. Smoothing widths are not uniformly Gaussian standard deviations,
    and channel counts are not minimum participation requirements.

    The columns in :data:`NUMERIC_COLUMNS` are numeric, with ``#N/A`` and any
    entry that is not a bare number read as missing, so a column can be
    summarized directly. The rest are left as text, since they hold phrases
    such as "Not reported" that mean something different from a missing value.

    Two rows share a first author and year, Farooq & Dragoi's 2019 Neuron and
    Science papers, so a row is identified by DOI rather than by author alone.
    :func:`load_literature_datasets` supplies associated public data links,
    joined on this table's ``DOI`` and its ``paper_doi``.

    Returns
    -------
    parameters : pd.DataFrame, shape (57, 35)
        One row per paper.

    Examples
    --------
    >>> from ripple_detection import load_literature_parameters
    >>> parameters = load_literature_parameters()
    >>> float(parameters["SWR Z-score Thresh. (STD)"].median())
    3.0

    """
    with (files("ripple_detection.data") / DATA_FILE).open(encoding="utf-8-sig") as file:
        parameters = pd.read_csv(file, dtype=str, keep_default_na=False)
    parameters = parameters.replace({"#N/A": None})
    for column in NUMERIC_COLUMNS:
        parameters[column] = pd.to_numeric(parameters[column], errors="coerce")
    return parameters


def load_literature_datasets() -> pd.DataFrame:
    """Public recording and event-data references for the surveyed papers.

    Load the packaged catalog without network access. This is a partial
    catalog, not an availability census: a paper without a row has no
    cataloged dataset, not necessarily no public data. Code-only archives
    are excluded; repositories containing inspected event data are included.

    Returns
    -------
    datasets : pandas.DataFrame
        One row per ``(paper_doi, dataset_url)`` relationship, all columns
        read as text. A paper may use several deposits, and a deposit may
        support several papers. Different archives can mirror the same
        recordings; rows are not independent experimental datasets.

        ``paper_doi`` joins to ``load_literature_parameters()["DOI"]``.
        ``dataset_name`` labels the deposit and ``dataset_url`` links to
        its landing page or archived files. ``relationship`` is
        ``original``, ``reused`` or ``not_verified`` (see ``notes`` for
        subsets and derived data). ``available_inputs`` lists established
        input types separated by semicolons, or ``not_verified``; an
        omitted type is not evidence of its absence.

        ``event_annotations`` distinguishes ``present`` (files inspected),
        ``reported_present`` (metadata only), ``not_found_in_inspected_scope``
        and ``not_verified``. These refer to ripple, population-candidate or
        replay events, not trial timing; none is a ground-truth guarantee.
        ``verification_status`` is ``reported``, ``metadata_inspected`` or
        ``files_inspected``. ``source_note`` links to the supporting audit
        note, and ``notes`` records the inspection scope and limitations.

    See Also
    --------
    load_literature_parameters : Detection and analysis parameters by paper DOI.

    Examples
    --------
    >>> from ripple_detection import load_literature_datasets, load_literature_parameters
    >>> datasets = load_literature_datasets()
    >>> papers = load_literature_parameters()[["DOI", "First Author", "Year"]]
    >>> linked = datasets.merge(papers, left_on="paper_doi", right_on="DOI", validate="many_to_one")
    >>> len(linked) == len(datasets)
    True
    >>> datasets.loc[datasets.event_annotations == "present", "dataset_url"].empty
    False

    """
    with (files("ripple_detection.data") / DATASETS_FILE).open(encoding="utf-8") as file:
        return pd.read_csv(file, dtype=str, keep_default_na=False)
