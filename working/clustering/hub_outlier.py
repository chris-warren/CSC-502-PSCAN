#!/usr/bin/env python3
"""
Hub and Outlier Detection for PSCAN.

After LPCC clustering, nodes are classified as:
  - Core    : degree in pruned graph >= μ
  - Hub     : not core, connected to 2+ distinct clusters
  - Outlier : not core, connected to 0 or 1 cluster

Mirrors Section 3.4 of the PSCAN paper.

Usage (as library):
    from hub_outlier import detect, write_classification, summary

Usage (CLI):
    python hub_outlier.py \
        --adjlist  data/output/adjlists/lfr_5000.adjlist \
        --clusters data/output/clusters/lfr_5000_clusters.csv \
        --mu       5 \
        --output   data/output/classifications/lfr_5000.types.tsv
"""

from __future__ import annotations

import argparse
import csv
import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


CORE    = "core"
HUB     = "hub"
OUTLIER = "outlier"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect(
    node_labels: Dict[int, int],
    adj: Dict[int, List[int]],
    mu: int = 1,
) -> Dict[int, str]:
    """Classify every node as core, hub, or outlier.

    Parameters
    ----------
    node_labels : dict
        {node_id: cluster_label} for all nodes assigned by LPCC.
    adj : dict
        Pruned adjacency list {node_id: [neighbor_ids]} (after epsilon filtering).
    mu : int
        Minimum number of neighbors in the pruned graph required to be a core node.

    Returns
    -------
    dict
        {node_id: "core" | "hub" | "outlier"} for every node in adj.
    """
    classification: Dict[int, str] = {}

    # Step 1: classify cores — degree in pruned graph >= mu
    for node in adj:
        neighbors = adj.get(node, [])
        if len(neighbors) >= mu:
            classification[node] = CORE

    # Step 2 & 3: classify remaining nodes as hub or outlier
    for node in adj:
        if classification.get(node) == CORE:
            continue

        # Collect distinct cluster labels from ALL neighbors (not just cores)
        neighbour_clusters: Set[int] = set()
        for nbr in adj.get(node, []):
            if nbr in node_labels:
                neighbour_clusters.add(node_labels[nbr])

        if len(neighbour_clusters) >= 2:
            classification[node] = HUB
        else:
            classification[node] = OUTLIER

    return classification


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summary(classification: Dict[int, str]) -> Dict[str, int]:
    """Return counts: {core, hub, outlier}."""
    counts: Dict[str, int] = {CORE: 0, HUB: 0, OUTLIER: 0}
    for t in classification.values():
        counts[t] = counts.get(t, 0) + 1
    return counts


def nodes_by_type(classification: Dict[int, str]) -> Dict[str, List[int]]:
    """Return {type: sorted list of node ids}."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for node, t in classification.items():
        groups[t].append(node)
    return {k: sorted(v) for k, v in groups.items()}


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_classification(classification: Dict[int, str], path: str | Path) -> None:
    """Write TSV: node <TAB> type."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("node\ttype\n")
        for node in sorted(classification):
            f.write(f"{node}\t{classification[node]}\n")
    print(f"[hub_outlier] Saved → {path}")


def load_classification(path: str | Path) -> Dict[int, str]:
    """Load a classification TSV written by write_classification()."""
    result: Dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if line:
                node_s, t = line.split("\t")
                result[int(node_s)] = t
    return result


def load_adjlist(path: str | Path) -> Dict[int, List[int]]:
    """Load adjacency list file into {node: [neighbors]}."""
    adj: Dict[int, List[int]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            u = int(parts[0])
            adj[u] = [int(v) for v in parts[1:] if int(v) != u]
    return adj


def load_clusters_csv(path: str | Path) -> Dict[int, int]:
    """Load cluster CSV (from LPSS_main) → {node: cluster_label}."""
    node_labels: Dict[int, int] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            cluster_id = int(row[0])
            nodes = ast.literal_eval(row[1])
            for node in nodes:
                node_labels[int(node)] = cluster_id
    return node_labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PSCAN hub and outlier detection."
    )
    parser.add_argument("--adjlist",  "-a", type=Path, required=True,
                        help="Pruned adjacency list file (.adjlist) after epsilon filtering.")
    parser.add_argument("--clusters", "-c", type=Path, required=True,
                        help="Cluster CSV from LPSS_main.")
    parser.add_argument("--mu",       "-m", type=int, default=1,
                        help="Minimum neighbors in pruned graph to be a core node.")
    parser.add_argument("--output",   "-o", type=Path, default=None,
                        help="Output classification TSV.")
    parser.add_argument("--verbose",  "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        print(f"[hub_outlier] Loading adjacency list: {args.adjlist}")
    adj = load_adjlist(args.adjlist)

    if args.verbose:
        print(f"[hub_outlier] Loading clusters: {args.clusters}")
    node_labels = load_clusters_csv(args.clusters)

    if args.verbose:
        print(f"[hub_outlier] mu = {args.mu}")

    classification = detect(node_labels, adj, mu=args.mu)
    counts = summary(classification)

    print(f"[hub_outlier] Core: {counts[CORE]:,}  "
          f"Hub: {counts[HUB]:,}  "
          f"Outlier: {counts[OUTLIER]:,}")

    out = args.output or (
        args.adjlist.parent.parent / "classifications" / (args.adjlist.stem + ".types.tsv")
    )
    write_classification(classification, out)


if __name__ == "__main__":
    main()