#!/usr/bin/env python3
"""
similarity_reducer.py

Reducer for PSCAN structural similarity computation.

Input format
------------
Lines from the mapper in the form:

    (u, v)    ('ADJ', source_node, neighbor_list)

For each unique undirected edge (u, v), the reducer should ideally receive
two adjacency records:
- one carrying N(u)
- one carrying N(v)

Output format
-------------
Tab-separated edge similarity records:

    u    v    similarity

Similarity formula
------------------
                 |N(u) ∩ N(v)|
    sim(u, v) = -----------------------
                sqrt(|N(u)| * |N(v)|)

Notes
-----
- If one side is missing, the reducer skips the edge.
- If one endpoint has degree 0, similarity is reported as 0.0.
"""

from __future__ import annotations

import ast
import math
import sys
from typing import Dict, Iterable, List, Optional, Tuple


def emit_similarity(
    edge: Tuple[int, int],
    records: List[Tuple[str, int, List[int]]],
) -> None:
    """
    Compute and print structural similarity for one edge.

    Parameters
    ----------
    edge : Tuple[int, int]
        Unique undirected edge (u, v).
    records : List[Tuple[str, int, List[int]]]
        All mapper records associated with this edge.
    """
    u, v = edge
    adjacency: Dict[int, set[int]] = {}

    for tag, source_node, neighbor_list in records:
        if tag != "ADJ":
            continue
        adjacency[source_node] = set(neighbor_list)

    # Need both endpoint neighborhoods to compute similarity.
    if u not in adjacency or v not in adjacency:
        return

    neighbors_u = adjacency[u]
    neighbors_v = adjacency[v]

    degree_u = len(neighbors_u)
    degree_v = len(neighbors_v)

    if degree_u == 0 or degree_v == 0:
        similarity = 0.0
    else:
        common_neighbors = neighbors_u.intersection(neighbors_v)
        similarity = len(common_neighbors) / math.sqrt(degree_u * degree_v)

    print(f"{u}\t{v}\t{similarity:.6f}")


def main() -> None:
    current_edge: Optional[Tuple[int, int]] = None
    current_records: List[Tuple[str, int, List[int]]] = []

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            edge_text, value_text = line.split("\t", 1)
            edge = ast.literal_eval(edge_text)
            value = ast.literal_eval(value_text)
        except (ValueError, SyntaxError):
            continue

        if current_edge is None:
            current_edge = edge

        if edge != current_edge:
            emit_similarity(current_edge, current_records)
            current_edge = edge
            current_records = []

        current_records.append(value)

    if current_edge is not None:
        emit_similarity(current_edge, current_records)


if __name__ == "__main__":
    main()