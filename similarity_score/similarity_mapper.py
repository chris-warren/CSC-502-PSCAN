#!/usr/bin/env python3
"""
similarity_mapper.py

Mapper for PSCAN structural similarity computation.

Input format
------------
Adjacency-list input, one node per line:

    u v1 v2 v3 ... vk

where:
- u is the node ID
- v1, v2, ..., vk are neighbors of u

Output format
-------------
For each undirected edge (u, v), emit:

    (min(u,v), max(u,v))    ("LEFT", source_node, neighbor_list)

The reducer will receive two records for each valid undirected edge:
one from u's side and one from v's side. It will then reconstruct
N(u) and N(v) and compute structural similarity.

Notes
-----
- Self-loops are ignored.
- Duplicate undirected edges are avoided by canonical ordering.
- Neighbor lists are emitted as Python lists so they can be parsed
  safely by ast.literal_eval in the reducer.
"""

from __future__ import annotations

import sys


def main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()

        try:
            values = [int(token) for token in parts]
        except ValueError:
            continue

        u = values[0]
        neighbors = values[1:]

        # Remove self-loops if present and deduplicate neighbors.
        neighbor_set = sorted(set(v for v in neighbors if v != u))

        for v in neighbor_set:
            edge = (u, v) if u < v else (v, u)

            # Emit this node's neighborhood information for the edge.
            print(f"{edge}\t{('ADJ', u, neighbor_set)}")


if __name__ == "__main__":
    main()