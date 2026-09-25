"""Published detection parameters, as a table.

The values every detector in this package needs are chosen differently in
every paper that reports them. :func:`load_literature_parameters` ships the
survey those choices were read from, so a threshold can be picked against what
the field does rather than against one lab's default.
"""

from importlib.resources import files

import pandas as pd

DATA_FILE = "literature_detection_parameters.csv"

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
