#!/usr/bin/env python3
"""Mapper for PSCAN structural similarity (PCSS) — MapReduce Phase 1.

Folder layout assumed
---------------------
project/
    data/           datasets.py  (Shikha)
    similarity/     similarity_mapper.py   ← this file
                    similarity_reducer.py
                    similarity_main.py
    run/            main.py
                    job.sh

Responsibility
--------------
Read an adjacency-list file produced by data/datasets.py and emit one
key-value pair per directed edge endpoint containing the closed neighbourhood
that the reducer needs to compute the intersection |N[u] ∩ N[v]|:

    key   : (min(u,v), max(u,v))   — canonical undirected edge
    value : (closed_neighbourhood_of_emitting_node, degree)

Each undirected edge produces TWO records (one from u, one from v).

Output TSV format (one record per line)
----------------------------------------
    u <TAB> v <TAB> node <TAB> closed_nbrs_space_separated <TAB> closed_degree

Usage
-----
    # Standalone
    python similarity_mapper.py --input ../data/output/adjlists/lfr_5000.adjlist

    # As a library (called from similarity_main.py)
    from similarity.similarity_mapper import load_adjacency_list, mapper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Generator, List, Set, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
AdjList   = Dict[int, Set[int]]
MapRecord = Tuple[Tuple[int, int], int, List[int], int]
# (canonical_edge, emitting_node, sorted_closed_nbrs, closed_degree)


# ---------------------------------------------------------------------------
# 1. Load adjacency list  (matches write_adjacency_list in data/datasets.py)
# ---------------------------------------------------------------------------

def load_adjacency_list(path: str | Path) -> AdjList:
    """Parse an adjacency-list file written by data/datasets.py.

    Each non-empty, non-comment line has the form::

        u  v1  v2  ...  vk

    where u is the node ID and v1…vk are its sorted neighbours.

    Parameters
    ----------
    path : str | Path
        Path to the ``.adjlist`` file under data/output/adjlists/.

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
    """Return N[node] = N(node) ∪ {node}."""
    return adj.get(node, set()) | {node}


def mapper(adj: AdjList) -> Generator[MapRecord, None, None]:
    """Yield one MapRecord per directed edge endpoint.

    For each undirected edge (u, v) with u < v, two records are emitted:
      - one carrying N[u]
      - one carrying N[v]

    Parameters
    ----------
    adj : AdjList
        Open adjacency list from load_adjacency_list().

    Yields
    ------
    MapRecord
        (canonical_edge, emitting_node, sorted_closed_nbrs, closed_degree)
    """
    for u, neighbours in adj.items():
        nu        = closed_neighbourhood(u, adj)
        nu_sorted = sorted(nu)
        nu_deg    = len(nu)

        for v in neighbours:
            if u < v:
                edge = (u, v)
                yield (edge, u, nu_sorted, nu_deg)

                nv = closed_neighbourhood(v, adj)
                yield (edge, v, sorted(nv), len(nv))


# ---------------------------------------------------------------------------
# 3. Write mapper output to file
# ---------------------------------------------------------------------------

def emit_records(adj: AdjList, out_stream=None) -> None:
    """Run the mapper and write TSV records to out_stream (default: stdout)."""
    if out_stream is None:
        out_stream = sys.stdout
    for (eu, ev), node, nbrs, deg in mapper(adj):
        nbrs_str = " ".join(str(x) for x in nbrs)
        out_stream.write(f"{eu}\t{ev}\t{node}\t{nbrs_str}\t{deg}\n")


def write_mapper_output(adj: AdjList, path: str | Path) -> None:
    """Write mapper records to a TSV file."""
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
    parser.add_argument("--input",  "-i", type=Path, required=True,
                        help="Adjacency-list file from data/datasets.py.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output mapper TSV. Default: <input_stem>.mapper.tsv. Use '-' for stdout.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        print(f"[mapper] Loading {args.input} ...", file=sys.stderr)

    adj = load_adjacency_list(args.input)
    n_nodes = len(adj)
    n_edges = sum(len(n) for n in adj.values()) // 2

    if args.verbose:
        print(f"[mapper]   Nodes: {n_nodes:,}  Edges: {n_edges:,}", file=sys.stderr)

    if args.output is None:
        out = args.input.parent / (args.input.stem + ".mapper.tsv")
        write_mapper_output(adj, out)
        if args.verbose:
            print(f"[mapper] Wrote to {out}", file=sys.stderr)
    elif str(args.output) == "-":
        emit_records(adj)
    else:
        write_mapper_output(adj, args.output)
        if args.verbose:
            print(f"[mapper] Wrote to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()