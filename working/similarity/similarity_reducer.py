#!/usr/bin/env python3
"""Reducer for PSCAN structural similarity (PCSS) — MapReduce Phase 2.

Folder layout assumed
---------------------
project/
    data/           datasets.py  (Shikha)
    similarity/     similarity_mapper.py
                    similarity_reducer.py  ← this file
                    similarity_main.py
    run/            main.py
                    job.sh

Responsibility
--------------
Consume mapper records grouped by canonical edge key (u, v) and compute
the final PCSS score for every edge:

    sim(u, v) = |N[u] ∩ N[v]| / sqrt(|N[u]| * |N[v]|)

Expects exactly TWO records per edge (one per endpoint) from similarity_mapper.py.

Output TSV format
-----------------
    u <TAB> v <TAB> similarity

where u < v — this is the format consumed by clustering.py (Van, Division 3).

Usage
-----
    # From a mapper file
    python similarity_reducer.py \
        --input  ../data/output/adjlists/lfr_5000.mapper.tsv \
        --output ../data/output/adjlists/lfr_5000.sim.tsv

    # As a library (called from similarity_main.py)
    from similarity.similarity_reducer import reduce_from_records, write_similarities
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Set, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
SimMap    = Dict[Tuple[int, int], float]
MapRecord = Tuple[Tuple[int, int], int, Set[int], int]


# ---------------------------------------------------------------------------
# 1. Parse mapper records
# ---------------------------------------------------------------------------

def parse_mapper_line(line: str) -> MapRecord | None:
    """Parse one TSV line from the mapper output.

    Format::

        u <TAB> v <TAB> node <TAB> nbr1 nbr2 ... <TAB> closed_degree

    Returns None for empty / comment lines.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split("\t")
    if len(parts) != 5:
        return None

    edge_u = int(parts[0])
    edge_v = int(parts[1])
    node   = int(parts[2])
    nbrs   = set(int(x) for x in parts[3].split()) if parts[3].strip() else set()
    deg    = int(parts[4])

    return ((edge_u, edge_v), node, nbrs, deg)


def iter_mapper_records(stream: Iterable[str]) -> Generator[MapRecord, None, None]:
    """Yield parsed MapRecords from any iterable of lines."""
    for line in stream:
        record = parse_mapper_line(line)
        if record is not None:
            yield record


# ---------------------------------------------------------------------------
# 2. Core reducer logic
# ---------------------------------------------------------------------------

def pcss_score(nbrs_u, nbrs_v) -> float:
    """Apply the PCSS formula given two closed neighbourhood sets.

    sim(u, v) = |N[u] ∩ N[v]| / sqrt(|N[u]| * |N[v]|)

    Parameters
    ----------
    nbrs_u, nbrs_v : Set[int] or List[int]
        Closed neighbourhoods. Converted to sets internally so this works
        whether the in-memory pipeline passes sets or lists.

    Returns
    -------
    float
        Similarity in (0, 1].  Returns 0.0 if either set is empty.
    """
    nbrs_u = set(nbrs_u)    # safe conversion — works for set, list, or generator
    nbrs_v = set(nbrs_v)
    denom_sq = len(nbrs_u) * len(nbrs_v)
    if denom_sq == 0:
        return 0.0
    intersection = len(nbrs_u & nbrs_v)
    return intersection / math.sqrt(denom_sq)


def reduce_from_records(records: Iterable[MapRecord]) -> SimMap:
    """Group mapper records by edge key and compute PCSS for each edge.

    Each canonical edge (u, v) needs exactly two records — one per endpoint.
    Incomplete groups (only one record) are skipped defensively.

    Parameters
    ----------
    records : Iterable[MapRecord]
        Stream of parsed mapper records (order does not matter).

    Returns
    -------
    SimMap
        ``{(u, v): similarity}`` for every edge, with u < v.
    """
    grouped: Dict[Tuple[int, int], List[Tuple[int, Set[int]]]] = defaultdict(list)

    for (edge_u, edge_v), node, nbrs, _deg in records:
        edge = (min(edge_u, edge_v), max(edge_u, edge_v))
        grouped[edge].append((node, nbrs))

    sim: SimMap = {}
    for edge, entries in grouped.items():
        if len(entries) != 2:
            continue   # skip incomplete groups

        (node_a, nbrs_a), (node_b, nbrs_b) = entries
        u, v = edge

        if node_a == u:
            nbrs_u, nbrs_v = nbrs_a, nbrs_b
        else:
            nbrs_u, nbrs_v = nbrs_b, nbrs_a

        sim[edge] = pcss_score(nbrs_u, nbrs_v)

    return sim


# ---------------------------------------------------------------------------
# 3. File helpers
# ---------------------------------------------------------------------------

def reduce_from_file(input_path: str | Path) -> SimMap:
    """Read a mapper TSV file and return the full similarity map."""
    with Path(input_path).open("r", encoding="utf-8") as fh:
        return reduce_from_records(iter_mapper_records(fh))


def write_similarities(sim: SimMap, path: str | Path) -> None:
    """Write similarity scores to a TSV file for clustering.py (Van).

    Format::

        u <TAB> v <TAB> similarity

    sorted by (u, v), u < v.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("u\tv\tsimilarity\n")
        for (u, v), score in sorted(sim.items()):
            fh.write(f"{u}\t{v}\t{score:.10f}\n")


def load_similarities(path: str | Path) -> SimMap:
    """Load a similarity TSV written by write_similarities()."""
    sim: SimMap = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        next(fh)  # skip header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            u_s, v_s, sc_s = line.split("\t")
            sim[(int(u_s), int(v_s))] = float(sc_s)
    return sim


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reducer: compute PCSS from mapper records and write similarity TSV."
    )
    parser.add_argument("--input",  "-i", type=Path, default=None,
                        help="Mapper TSV file. Omit or use '-' for stdin.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output similarity TSV. Default: <input_stem>.sim.tsv.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input is None or str(args.input) == "-":
        if args.verbose:
            print("[reducer] Reading from stdin ...", file=sys.stderr)
        sim = reduce_from_records(iter_mapper_records(sys.stdin))
        default_stem = "output"
    else:
        if args.verbose:
            print(f"[reducer] Reading {args.input} ...", file=sys.stderr)
        sim = reduce_from_file(args.input)
        default_stem = args.input.stem.replace(".mapper", "")

    if args.verbose:
        print(f"[reducer]   Reduced {len(sim):,} similarities.", file=sys.stderr)

    if args.output is None:
        if args.input is not None and str(args.input) != "-":
            out = args.input.parent / f"{default_stem}.sim.tsv"
        else:
            out = Path("output.sim.tsv")
        write_similarities(sim, out)
        if args.verbose:
            print(f"[reducer] Wrote to {out}", file=sys.stderr)
    elif str(args.output) == "-":
        sys.stdout.write("u\tv\tsimilarity\n")
        for (u, v), score in sorted(sim.items()):
            sys.stdout.write(f"{u}\t{v}\t{score:.10f}\n")
    else:
        write_similarities(sim, args.output)
        if args.verbose:
            print(f"[reducer] Wrote to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()