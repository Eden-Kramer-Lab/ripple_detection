# Public datasets associated with the survey

The packaged [literature_datasets.csv](../../src/ripple_detection/data/literature_datasets.csv)
owns dataset links and availability summaries. The parameter CSV retains its
35-column schema. Source notes explain the supporting evidence and inspection
limits; edit the dataset table when a link or availability finding changes.

This is a **partial catalog**, seeded from references in the existing audit,
with selected landing-page checks documented in the
[source catalog](sources.md#dataset-catalog-checks). A paper without a row has
no cataloged deposit; this does not mean its data are unavailable. A link does
not establish that the deposit contains every signal, curated selection or
original event needed to reproduce a paper. Some repositories require an
account or impose use conditions; consult their landing pages.

## Load and join

The table ships with the package and loads without network access:

```python
from ripple_detection import load_literature_datasets, load_literature_parameters

datasets = load_literature_datasets()
papers = load_literature_parameters()[["DOI", "First Author", "Year"]]
linked = datasets.merge(
    papers, left_on="paper_doi", right_on="DOI", validate="many_to_one"
)
dandi = linked.loc[linked.dataset_url.str.contains("dandiarchive.org", regex=False)]
event_files = linked.loc[linked.event_annotations == "present"]
```

Each row describes one `(paper_doi, dataset_url)` relationship. Several rows
can belong to one paper or point to the same deposit. Separate archives can
also mirror the same recordings: Grosmark's CRCNS hc-11 and DANDI 000044 are
one such pair. Do not treat rows or URLs as counts of independent datasets.
The table includes released event files and example data embedded in code
repositories; code-only releases remain in the source catalog.

## Columns

All columns load as text. Unknown information is explicit rather than a blank
or a numeric missing value.

| Column | Meaning |
|---|---|
| `paper_doi` | Exact DOI URL used by the parameter table's `DOI` column. |
| `dataset_name` | Human-readable label for the deposit or released data artifact. |
| `dataset_url` | Dataset landing page or archived file/repository URL; pinned to the inspected version where established. |
| `relationship` | `original`: recordings associated with the paper's original data; `reused`: recordings from a prior study; `not_verified`: that distinction is not established. A new event-file release can describe reused recordings. Subset and session limitations are in `notes`. |
| `available_inputs` | Semicolon-separated established input types: `lfp`, `spikes`, `behavior`, or broader `electrophysiology` when LFP/sorted-unit availability is not established. `not_verified` means input contents were not established. An omitted type is unknown, not necessarily absent. Availability may cover only the inspected subset. |
| `event_annotations` | Status of ripple, population-candidate or replay annotations, defined below. Trial/reward timing alone does not qualify. |
| `verification_status` | Depth of deposit inspection, defined below. It does not assert full reproduction or complete coverage of every file. |
| `source_note` | Web link to the supporting paper/source note in this repository; accessible from an installed package without a local checkout. |
| `notes` | Contents, reuse or mirror relationships, exact inspection scope, and unresolved details. |

`event_annotations` has four values:

- `present`: event data were inspected; `notes` names the inventory and scope.
- `reported_present`: metadata or an availability statement identifies event
  annotations, but their contents were not inspected.
- `not_found_in_inspected_scope`: the documented inventory or metadata inspection
  did not establish such annotations. This is **not** a claim that every file
  variable was checked or that annotations cannot exist elsewhere.
- `not_verified`: no sufficiently scoped finding is established.

`verification_status` distinguishes `reported` (a recorded paper/note reference),
`metadata_inspected` (landing metadata, file listings or headers), and
`files_inspected` (selected data contents inspected). For example, a ZIP listing
and README do not establish that every array or MATLAB variable was read.
These statuses record the cited audit scope, not a continuous availability check.
The [link check](sources.md#dataset-catalog-checks) records external URL
resolution separately. `source_note` URLs target GitHub `master`; in an
unmerged checkout, newly added notes or anchors may work locally before they
have been published at those URLs.

Published events are reference outputs, not validated ground truth. Initial
candidates, final replay labels and binned event support are different
inventories. Read `notes` before using them to compare detection methods.

## Maintenance

Add a row only when a deposit and its relationship to a surveyed paper have a
source. Record unsupported contents as `not_verified`. Keep the same canonical
URL for the same deposit/version across papers, and prefer a versioned URL when
the inspected version is known. Do not create placeholder rows for every paper
or fill unknowns from another study's similarly named dataset.

Keep links and availability summaries in this CSV, detailed evidence in its
`source_note`, and detection parameters in the parameter CSV. Dataset columns
are not parameter fields and do not add rows to `evidence.csv`. Git preserves
earlier states. Run `pytest tests/test_literature.py tests/test_public_api.py -q
--no-cov` after editing.
