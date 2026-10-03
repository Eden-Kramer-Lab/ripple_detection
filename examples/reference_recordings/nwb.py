"""Open DANDI NWB files remotely, reading only the slices asked for.

Needs ``remfile`` and ``h5py``, which are not dependencies of the package; run as::

    uv run --with remfile --with h5py python examples/reference_recordings/nwb.py \\
        DANDISET VERSION PATH_OR_ASSET_ID [--rows N]

No test imports this module; ``remfile`` and ``h5py`` are imported inside functions.
"""

from __future__ import annotations

import argparse
from typing import Any

import fetch
import numpy as np


def open_nwb(url: str) -> Any:
    """Open an NWB (HDF5) file over HTTPS without downloading it.

    Parameters
    ----------
    url : str
        The asset's S3 content URL (``fetch.dandi_asset(...)["content_url"]``).

    Returns
    -------
    h5py.File
        Read-only; only the chunks that are read are transferred. Close it when done.
    """
    import h5py
    import remfile

    return h5py.File(remfile.File(url), "r")


def describe(file: Any, n_rows: int = 4000) -> list[dict[str, Any]]:
    """Shape, dtype and chunking of each 2-D ``data`` dataset, and its first rows' range.

    Parameters
    ----------
    file : h5py.File
        An open NWB file.
    n_rows : int
        Rows read from the start of each dataset (a few thousand at most).

    Returns
    -------
    list of dict
        ``path``, ``shape``, ``dtype``, ``chunks`` and the minimum and maximum of the
        rows read.
    """
    out = []
    for group in ("acquisition", "processing"):
        if group not in file:
            continue

        def visit(name: str, obj: Any, group: str = group) -> None:
            if name.endswith("data") and getattr(obj, "ndim", 0) == 2:
                head = obj[: min(n_rows, obj.shape[0])]
                out.append(
                    {
                        "path": f"/{group}/{name}",
                        "shape": tuple(obj.shape),
                        "dtype": str(obj.dtype),
                        "chunks": obj.chunks,
                        "head_min": float(np.nanmin(head)),
                        "head_max": float(np.nanmax(head)),
                    }
                )

        file[group].visititems(visit)
    return out


def main(argv: list[str] | None = None) -> None:
    """Record a DANDI asset in the manifest, open it remotely and print its datasets."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("dandiset")
    parser.add_argument("version")
    parser.add_argument("path_or_asset_id")
    parser.add_argument("--rows", type=int, default=4000)
    args = parser.parse_args(argv)
    record = fetch.dandi_asset(args.dandiset, args.version, args.path_or_asset_id)
    print(f"asset {record['asset_path']} ({record['bytes']} bytes) sha256 {record['sha256']}")
    file = open_nwb(record["content_url"])
    try:
        for entry in describe(file, args.rows):
            print(entry)
    finally:
        file.close()


if __name__ == "__main__":
    main()
