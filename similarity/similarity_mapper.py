#!/usr/bin/env python3
"""Mapper for PSCAN structural similarity (PCSS) — MapReduce Phase 1.

Responsibility
--------------
Read a chunk (or full) adjacency-list file and emit one key-value pair per
directed edge (u → v) containing the information that the reducer needs to
compute the intersection |N[u] ∩ N[v]|:

    key   : (min(u,v), max(u,v))   — canonical undirected edge
    value : (closed_neighbourhood_of_emitting_node, degree_of_emitting_node)

Each edge therefore produces TWO records (one from u's perspective, one from
v's perspective).  The reducer collects both, computes the intersection, and
applies the PCSS formula.

Output format (TSV, one record per line)
-----------------------------------------
    u  <TAB>  v  <TAB>  node  <TAB>  closed_nbrs_space_separated  <TAB>  closed_degree

where u < v is the canonical edge key, `node` is the emitting node (u or v),
and `closed_nbrs` is the space-separated sorted closed neighbourhood N[node].

Usage
-----
    # Standalone (writes to stdout or --output file)
    python similarity_mapper.py --input data/adjlists/lfr_5000.adjlist

    # As a library
    from similarity_mapper import load_adjacency_list, mapper, emit_records
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Generator, List, Set, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
AdjList    = Dict[int, Set[int]]
MapRecord  = Tuple[Tuple[int, int], int, List[int], int]
# (canonical_edge, emitting_node, sorted_closed_nbrs, closed_degree)


# ---------------------------------------------------------------------------
# 1. Load adjacency list (same format as datasets.py output)
# ---------------------------------------------------------------------------

def load_adjacency_list(path: str | Path) -> AdjList:
    """Parse an adjacency-list file produced by datasets.py.

    Each non-empty, non-comment line has the form::

        u  v1  v2  ...  vk

    Parameters
    ----------
    path : str | Path
        Path to the ``.adjlist`` file.

    Returns
    -------
    AdjList
        Mapping node ID → set of neighbour IDs (open neighbourhood).
        Self-loops are removed defensively.
    """
    adj: AdjList = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            u = int(parts[0])
            nbrs: Set[int] = set()
            for token in parts[1:]:
                v = int(token)
                if v != u:
                    nbrs.add(v)
            adj[u] = nbrs
    return adj


# ---------------------------------------------------------------------------
# 2. Mapper logic
# ---------------------------------------------------------------------------

def closed_neighbourhood(node: int, adj: AdjList) -> Set[int]:
    """Return N[node] = N(node) ∪ {node} (closed neighbourhood)."""
    return adj.get(node, set()) | {node}


def mapper(adj: AdjList) -> Generator[MapRecord, None, None]:
    """Yield one MapRecord per directed edge endpoint.

    For each undirected edge (u, v) with u < v, two records are emitted:
      - one carrying N[u] (from u's perspective)
      - one carrying N[v] (from v's perspective)

    The reducer receives both records for the same canonical edge key and can
    then compute the intersection.

    Parameters
    ----------
    adj : AdjList
        Open adjacency list.

    Yields
    ------
    MapRecord
        ``(canonical_edge, emitting_node, sorted_closed_nbrs, closed_degree)``
    """
    for u, neighbours in adj.items():
        nu = closed_neighbourhood(u, adj)
        nu_sorted = sorted(nu)
        nu_deg    = len(nu)

        for v in neighbours:
            if u < v:                           # emit each undirected edge once per side
                edge = (u, v)
                yield (edge, u, nu_sorted, nu_deg)

                nv = closed_neighbourhood(v, adj)
                yield (edge, v, sorted(nv), len(nv))


# ---------------------------------------------------------------------------
# 3. Serialise / write mapper output
# ---------------------------------------------------------------------------

def emit_records(
    adj: AdjList,
    out_stream=None,
) -> None:
    """Run the mapper and write records to *out_stream* (default: stdout).

    Output TSV format per line::

        u <TAB> v <TAB> node <TAB> nbr1 nbr2 ... <TAB> closed_degree

    Parameters
    ----------
    adj : AdjList
        Adjacency list to map over.
    out_stream :
        File-like object to write to.  Defaults to ``sys.stdout``.
    """
    if out_stream is None:
        out_stream = sys.stdout

    for (edge_u, edge_v), node, nbrs, deg in mapper(adj):
        nbrs_str = " ".join(str(x) for x in nbrs)
        out_stream.write(f"{edge_u}\t{edge_v}\t{node}\t{nbrs_str}\t{deg}\n")


def write_mapper_output(adj: AdjList, path: str | Path) -> None:
    """Write mapper records to a file.

    Parameters
    ----------
    adj : AdjList
        Adjacency list.
    path : str | Path
        Destination path for the mapper output TSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        emit_records(adj, out_stream=fh)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mapper: emit per-edge neighbourhood records for PCSS reducer."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Adjacency-list file (.adjlist) from datasets.py.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help=(
            "Output file for mapper records. "
            "Defaults to <input_stem>.mapper.tsv in the same directory. "
            "Use '-' for stdout."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress to stderr.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        print(f"[mapper] Loading graph from {args.input} ...", file=sys.stderr)

    adj = load_adjacency_list(args.input)

    n_nodes = len(adj)
    n_edges = sum(len(nbrs) for nbrs in adj.values()) // 2

    if args.verbose:
        print(
            f"[mapper]   Nodes: {n_nodes:,}   Edges: {n_edges:,}",
            file=sys.stderr,
        )
        print(f"[mapper] Emitting records ...", file=sys.stderr)

    if args.output is None:
        out_path = args.input.parent / (args.input.stem + ".mapper.tsv")
        write_mapper_output(adj, out_path)
        if args.verbose:
            print(f"[mapper] Wrote records to {out_path}", file=sys.stderr)
    elif str(args.output) == "-":
        emit_records(adj, out_stream=sys.stdout)
    else:
        write_mapper_output(adj, args.output)
        if args.verbose:
            print(f"[mapper] Wrote records to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()