#!/usr/bin/env python3
"""
similarity.py

Compute structural similarity for each unique undirected edge in a graph.

This module implements the structural similarity stage of PSCAN.
The input graph is expected in adjacency-list format, where each line is:

    u v1 v2 v3 ... vk

meaning node `u` is connected to neighbors `v1, v2, ..., vk`.

Structural similarity for an edge (u, v) is defined as:

                 |N(u) ∩ N(v)|
    sim(u, v) = -----------------------
                sqrt(|N(u)| * |N(v)|)

where:
- N(u) is the neighbor set of u
- N(v) is the neighbor set of v

Notes
-----
1. The graph is treated as a simple undirected graph.
2. Self-loops are ignored.
3. Each undirected edge is processed exactly once using canonical ordering:
       (min(u, v), max(u, v))
4. The output contains one line per unique edge.

Example
-------
Input adjacency list:
    1 2 3
    2 1 3
    3 1 2 4
    4 3

Edges:
    (1,2), (1,3), (2,3), (3,4)

Output example:
    1 2 0.500000
    1 3 0.408248
    2 3 0.408248
    3 4 0.000000

Usage
-----
Write results to stdout:
    python similarity.py --input data/adjlists/lfr_5000.adjlist

Write results to a file:
    python similarity.py --input data/adjlists/lfr_5000.adjlist --output data/similarity/lfr_5000.sim.tsv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing input/output paths.
    """
    parser = argparse.ArgumentParser(
        description="Compute structural similarity for each unique undirected edge."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the adjacency-list input file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. If omitted, results are written to stdout.",
    )
    return parser.parse_args()


def load_adjacency_list(path: Path) -> Dict[int, Set[int]]:
    """
    Load an undirected graph from an adjacency-list file.

    Each line must have the format:
        u v1 v2 v3 ... vk

    This function:
    - creates neighbor sets for each node
    - removes self-loops
    - enforces undirected symmetry, so if u lists v, then v also contains u

    Parameters
    ----------
    path : Path
        Path to the adjacency-list file.

    Returns
    -------
    Dict[int, Set[int]]
        Mapping from node -> set of neighbors.

    Raises
    ------
    ValueError
        If a non-integer token is found.
    """
    adjacency: Dict[int, Set[int]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # Skip empty lines safely.
            if not line:
                continue

            parts = line.split()

            try:
                values = [int(token) for token in parts]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer token on line {line_num}: {raw_line.rstrip()}"
                ) from exc

            node = values[0]
            neighbors = values[1:]

            if node not in adjacency:
                adjacency[node] = set()

            for nbr in neighbors:
                # Ignore self-loops in a simple graph.
                if nbr == node:
                    continue

                adjacency[node].add(nbr)

                # Enforce undirected symmetry.
                if nbr not in adjacency:
                    adjacency[nbr] = set()
                adjacency[nbr].add(node)

    # Ensure isolated nodes are preserved if they appeared alone on a line.
    for node in list(adjacency.keys()):
        adjacency.setdefault(node, set())

    return adjacency


def get_unique_undirected_edges(adjacency: Dict[int, Set[int]]) -> List[Tuple[int, int]]:
    """
    Extract all unique undirected edges from the adjacency structure.

    An undirected edge is stored once in canonical form:
        (min(u, v), max(u, v))

    Parameters
    ----------
    adjacency : Dict[int, Set[int]]
        Node -> neighbor set mapping.

    Returns
    -------
    List[Tuple[int, int]]
        Sorted list of unique undirected edges.
    """
    edges: Set[Tuple[int, int]] = set()

    for u, neighbors in adjacency.items():
        for v in neighbors:
            if u == v:
                continue
            edge = (u, v) if u < v else (v, u)
            edges.add(edge)

    return sorted(edges)


def structural_similarity(
    u: int,
    v: int,
    adjacency: Dict[int, Set[int]],
) -> float:
    """
    Compute structural similarity for one undirected edge (u, v).

    Formula:
                     |N(u) ∩ N(v)|
        sim(u, v) = -----------------------
                    sqrt(|N(u)| * |N(v)|)

    Parameters
    ----------
    u : int
        First endpoint.
    v : int
        Second endpoint.
    adjacency : Dict[int, Set[int]]
        Node -> neighbor set mapping.

    Returns
    -------
    float
        Structural similarity score in [0, 1] for normal simple graphs.
        Returns 0.0 if one endpoint has degree 0.
    """
    neighbors_u = adjacency.get(u, set())
    neighbors_v = adjacency.get(v, set())

    degree_u = len(neighbors_u)
    degree_v = len(neighbors_v)

    if degree_u == 0 or degree_v == 0:
        return 0.0

    common_neighbors = neighbors_u.intersection(neighbors_v)
    numerator = len(common_neighbors)
    denominator = math.sqrt(degree_u * degree_v)

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def compute_all_similarities(
    adjacency: Dict[int, Set[int]]
) -> List[Tuple[int, int, float]]:
    """
    Compute structural similarity for all unique undirected edges.

    Parameters
    ----------
    adjacency : Dict[int, Set[int]]
        Node -> neighbor set mapping.

    Returns
    -------
    List[Tuple[int, int, float]]
        List of (u, v, similarity) tuples, sorted by edge.
    """
    results: List[Tuple[int, int, float]] = []
    edges = get_unique_undirected_edges(adjacency)

    for u, v in edges:
        sim = structural_similarity(u, v, adjacency)
        results.append((u, v, sim))

    return results


def write_results(
    results: Iterable[Tuple[int, int, float]],
    output_path: Path | None = None,
) -> None:
    """
    Write similarity results either to stdout or to a file.

    Output format:
        u<TAB>v<TAB>similarity

    Parameters
    ----------
    results : Iterable[Tuple[int, int, float]]
        Similarity results.
    output_path : Path | None
        Output file path. If None, write to stdout.
    """
    lines = [f"{u}\t{v}\t{sim:.6f}" for u, v, sim in results]

    if output_path is None:
        for line in lines:
            print(line)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("u\tv\tsimilarity\n")
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    """
    Main entry point.

    Workflow:
    1. Read adjacency-list graph.
    2. Reconstruct undirected neighbor sets.
    3. Extract unique undirected edges.
    4. Compute structural similarity for each edge.
    5. Write results.
    """
    args = parse_args()

    adjacency = load_adjacency_list(args.input)
    results = compute_all_similarities(adjacency)
    write_results(results, args.output)


if __name__ == "__main__":
    main()