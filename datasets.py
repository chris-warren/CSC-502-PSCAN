#!/usr/bin/env python3
"""Dataset generation utilities for PSCAN experiments.

This script generates:
1. LFR benchmark graphs for clustering-accuracy evaluation.
2. Barabasi-Albert (BA) graphs for runtime/scalability evaluation.
3. Ground-truth node labels for LFR graphs.
4. Adjacency-list files for PSCAN input.
5. Metadata JSON files and a dataset manifest for reproducibility.

The implementation mirrors the paper's dataset split:
- LFR graphs are used for accuracy because they have planted communities.
- BA graphs are used for runtime because they scale well but do not provide
  trustworthy ground-truth communities.

Example:
    python datasets.py --output-dir data --paper-scales --seed 42

    python datasets.py \
        --output-dir data \
        --lfr-sizes 5000 10000 \
        --ba-sizes 100000 200000 \
        --tau1 3.0 --tau2 1.5 --mu 0.1 \
        --average-degree 15 --max-degree 75 \
        --min-community 20 --max-community 100 \
        --ba-m 7 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx


PAPER_LFR_SIZES = [5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
PAPER_BA_SIZES = [1_000_000, 2_000_000, 3_000_000, 4_000_000]


@dataclass
class DatasetStats:
    name: str
    family: str
    num_nodes: int
    num_edges: int
    average_degree_realized: float
    density: float
    connected_components: int
    largest_component_size: int
    num_clusters: Optional[int] = None
    isolated_vertices: Optional[int] = None
    parameters: Optional[dict] = None
    files: Optional[dict] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PSCAN datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--paper-scales",
        action="store_true",
        help="Use the graph sizes reported in the PSCAN paper.",
    )
    parser.add_argument(
        "--lfr-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Node counts for LFR graphs.",
    )
    parser.add_argument(
        "--ba-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Node counts for BA graphs.",
    )

    # LFR parameters
    parser.add_argument("--tau1", type=float, default=3.0, help="Degree exponent.")
    parser.add_argument("--tau2", type=float, default=1.5, help="Community-size exponent.")
    parser.add_argument("--mu", type=float, default=0.1, help="Mixing parameter.")
    parser.add_argument("--average-degree", type=int, default=15)
    parser.add_argument("--max-degree", type=int, default=75)
    parser.add_argument("--min-community", type=int, default=20)
    parser.add_argument("--max-community", type=int, default=100)
    parser.add_argument(
        "--lfr-max-retries",
        type=int,
        default=10,
        help="Retry budget for LFR generation, since the generator may fail for some seeds.",
    )

    # BA parameters
    parser.add_argument(
        "--ba-m",
        type=int,
        default=7,
        help="Number of edges to attach from each new node in BA generation.",
    )

    parser.add_argument(
        "--skip-lfr",
        action="store_true",
        help="Skip LFR generation.",
    )
    parser.add_argument(
        "--skip-ba",
        action="store_true",
        help="Skip BA generation.",
    )

    return parser.parse_args()


def resolve_sizes(args: argparse.Namespace) -> Tuple[List[int], List[int]]:
    lfr_sizes = args.lfr_sizes
    ba_sizes = args.ba_sizes

    if args.paper_scales:
        if lfr_sizes is None:
            lfr_sizes = PAPER_LFR_SIZES
        if ba_sizes is None:
            ba_sizes = PAPER_BA_SIZES

    if lfr_sizes is None:
        lfr_sizes = [5_000, 10_000]
    if ba_sizes is None:
        ba_sizes = [100_000, 200_000]

    return sorted(set(lfr_sizes)), sorted(set(ba_sizes))


def ensure_simple_undirected_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple undirected graph with self-loops removed."""
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        graph = nx.Graph(graph)
    elif isinstance(graph, nx.DiGraph):
        graph = nx.Graph(graph)
    else:
        graph = graph.copy()

    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def generate_lfr_graph(
    n: int,
    *,
    tau1: float,
    tau2: float,
    mu: float,
    average_degree: int,
    max_degree: int,
    min_community: int,
    max_community: int,
    base_seed: int,
    max_retries: int,
) -> Tuple[nx.Graph, int]:
    """Generate an LFR graph, retrying with different seeds if needed."""
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        seed = base_seed + attempt
        try:
            graph = nx.generators.community.LFR_benchmark_graph(
                n=n,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=average_degree,
                max_degree=max_degree,
                min_community=min_community,
                max_community=max_community,
                seed=seed,
            )
            graph = ensure_simple_undirected_graph(graph)
            return graph, seed
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to generate LFR graph for n={n} after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def extract_lfr_ground_truth(graph: nx.Graph) -> Tuple[Dict[int, int], int]:
    """Extract one planted community label per node from the LFR 'community' attribute.

    NetworkX stores the LFR ground truth in node attribute 'community'. In the
    non-overlapping case, this is a single community represented as a set of node
    IDs. To be robust, we canonicalize each community as a sorted frozenset and
    map equal communities to the same integer label.
    """
    community_to_label: Dict[frozenset, int] = {}
    labels: Dict[int, int] = {}

    for node, attrs in graph.nodes(data=True):
        if "community" not in attrs:
            raise ValueError("LFR graph is missing the 'community' node attribute.")

        community_attr = attrs["community"]
        if isinstance(community_attr, (set, frozenset, list, tuple)):
            if len(community_attr) == 0:
                raise ValueError(f"Node {node} has an empty community attribute.")

            first = next(iter(community_attr))
            if isinstance(first, (set, frozenset, list, tuple)):
                candidates = [frozenset(c) for c in community_attr]
                canonical = min(candidates, key=lambda c: (len(c), tuple(sorted(c))))
            else:
                canonical = frozenset(community_attr)
        else:
            canonical = frozenset([community_attr])

        if canonical not in community_to_label:
            community_to_label[canonical] = len(community_to_label)
        labels[int(node)] = community_to_label[canonical]

    return labels, len(community_to_label)


def write_adjacency_list(graph: nx.Graph, path: Path) -> None:
    """Write graph as one line per vertex: u v1 v2 ... vk."""
    with path.open("w", encoding="utf-8") as f:
        for node in sorted(graph.nodes()):
            neighbors = sorted(graph.neighbors(node))
            line = " ".join(str(x) for x in [node, *neighbors])
            f.write(line + "\n")


def write_labels(labels: Dict[int, int], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("node\tlabel\n")
        for node in sorted(labels):
            f.write(f"{node}\t{labels[node]}\n")


def graph_density(num_nodes: int, num_edges: int) -> float:
    if num_nodes <= 1:
        return 0.0
    return (2.0 * num_edges) / (num_nodes * (num_nodes - 1))


def basic_graph_stats(graph: nx.Graph) -> Tuple[int, int, float, int, int, int]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    avg_deg = (2.0 * m / n) if n else 0.0
    density = graph_density(n, m)
    components = list(nx.connected_components(graph))
    num_cc = len(components)
    largest_cc = max((len(c) for c in components), default=0)
    isolated = sum(1 for node in graph.nodes() if graph.degree(node) == 0)
    return n, m, avg_deg, density, num_cc, largest_cc, isolated


def dump_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def generate_ba_graph(n: int, *, m: int, seed: int) -> nx.Graph:
    graph = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    return ensure_simple_undirected_graph(graph)


def create_output_dirs(root: Path) -> Dict[str, Path]:
    subdirs = {
        "root": root,
        "adjlists": root / "adjlists",
        "labels": root / "labels",
        "metadata": root / "metadata",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def main() -> None:
    args = parse_args()
    lfr_sizes, ba_sizes = resolve_sizes(args)
    output_dirs = create_output_dirs(args.output_dir)

    random.Random(args.seed)
    manifest: List[dict] = []

    if not args.skip_lfr:
        for n in lfr_sizes:
            name = f"lfr_{n}"
            print(f"[LFR] Generating {name} ...")
            graph, used_seed = generate_lfr_graph(
                n=n,
                tau1=args.tau1,
                tau2=args.tau2,
                mu=args.mu,
                average_degree=args.average_degree,
                max_degree=args.max_degree,
                min_community=args.min_community,
                max_community=args.max_community,
                base_seed=args.seed + n,
                max_retries=args.lfr_max_retries,
            )
            labels, num_clusters = extract_lfr_ground_truth(graph)

            adj_path = output_dirs["adjlists"] / f"{name}.adjlist"
            labels_path = output_dirs["labels"] / f"{name}.labels.tsv"
            meta_path = output_dirs["metadata"] / f"{name}.meta.json"

            write_adjacency_list(graph, adj_path)
            write_labels(labels, labels_path)

            n_nodes, n_edges, avg_deg, density, num_cc, largest_cc, isolated = basic_graph_stats(graph)
            stats = DatasetStats(
                name=name,
                family="LFR",
                num_nodes=n_nodes,
                num_edges=n_edges,
                average_degree_realized=avg_deg,
                density=density,
                connected_components=num_cc,
                largest_component_size=largest_cc,
                num_clusters=num_clusters,
                isolated_vertices=isolated,
                parameters={
                    "generator": "networkx.generators.community.LFR_benchmark_graph",
                    "tau1": args.tau1,
                    "tau2": args.tau2,
                    "mu": args.mu,
                    "average_degree": args.average_degree,
                    "max_degree": args.max_degree,
                    "min_community": args.min_community,
                    "max_community": args.max_community,
                    "seed": used_seed,
                },
                files={
                    "adjlist": relative_to_root(adj_path, args.output_dir),
                    "labels": relative_to_root(labels_path, args.output_dir),
                    "metadata": relative_to_root(meta_path, args.output_dir),
                },
            )
            dump_json(asdict(stats), meta_path)
            manifest.append(asdict(stats))

    if not args.skip_ba:
        for idx, n in enumerate(ba_sizes):
            name = f"ba_{n}"
            seed = args.seed + 10_000 + idx
            print(f"[BA] Generating {name} ...")
            graph = generate_ba_graph(n=n, m=args.ba_m, seed=seed)

            adj_path = output_dirs["adjlists"] / f"{name}.adjlist"
            meta_path = output_dirs["metadata"] / f"{name}.meta.json"

            write_adjacency_list(graph, adj_path)

            n_nodes, n_edges, avg_deg, density, num_cc, largest_cc, isolated = basic_graph_stats(graph)
            stats = DatasetStats(
                name=name,
                family="Barabasi-Albert",
                num_nodes=n_nodes,
                num_edges=n_edges,
                average_degree_realized=avg_deg,
                density=density,
                connected_components=num_cc,
                largest_component_size=largest_cc,
                num_clusters=None,
                isolated_vertices=isolated,
                parameters={
                    "generator": "networkx.barabasi_albert_graph",
                    "m": args.ba_m,
                    "seed": seed,
                },
                files={
                    "adjlist": relative_to_root(adj_path, args.output_dir),
                    "labels": None,
                    "metadata": relative_to_root(meta_path, args.output_dir),
                },
            )
            dump_json(asdict(stats), meta_path)
            manifest.append(asdict(stats))

    manifest_path = args.output_dir / "manifest.json"
    dump_json({"datasets": manifest}, manifest_path)
    print(f"Wrote dataset manifest to {manifest_path}")


if __name__ == "__main__":
    main()