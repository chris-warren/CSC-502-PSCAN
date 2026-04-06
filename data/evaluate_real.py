import ast
import os
import sys
import time

import networkx as nx
from tqdm import tqdm

GRAPH_TSV_PATH = "filtered_edge_orkut_eps06.tsv"   # change as needed
CLUSTERS_PATH = "orkut_eps06.csv"


def read_graph_tsv(graph_tsv_path: str) -> nx.Graph:
    """
    Expected format per line:
      u<TAB>v1 v2 v3 ...
    """
    G = nx.Graph()
    total_bytes = os.path.getsize(graph_tsv_path)

    with open(graph_tsv_path, "r", encoding="utf-8") as f, tqdm(
        total=total_bytes,
        desc="Loading graph TSV",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        file=sys.stdout,
        mininterval=2.0,
    ) as pbar:
        for line_num, line in enumerate(f, start=1):
            pbar.update(len(line.encode("utf-8")))

            line = line.strip()
            if not line:
                continue

            if "\t" not in line:
                continue

            u_str, nbrs_str = line.split("\t", 1)

            try:
                u = int(u_str.strip())
            except ValueError:
                continue

            G.add_node(u)

            nbrs_str = nbrs_str.strip()
            if not nbrs_str:
                continue

            for v_str in nbrs_str.split():
                try:
                    v = int(v_str)
                except ValueError:
                    continue

                if u != v:
                    G.add_edge(u, v)

    return G


def read_pscan_clusters(clusters_path: str) -> dict[int, set[int]]:
    """
    Expected format per line:
      1,"[1, 2, 3, ...]"
    Returns:
      cluster_to_nodes: cluster_id -> set(nodes)
    """
    cluster_to_nodes = {}
    total_bytes = os.path.getsize(clusters_path)

    with open(clusters_path, "r", encoding="utf-8") as f, tqdm(
        total=total_bytes,
        desc="Loading PSCAN clusters",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        file=sys.stdout,
        mininterval=2.0,
    ) as pbar:
        for row_idx, line in enumerate(f, start=1):
            pbar.update(len(line.encode("utf-8")))

            line = line.strip()
            if not line:
                continue

            first_comma = line.find(",")
            if first_comma == -1:
                continue

            cluster_id = int(line[:first_comma].strip())
            members_str = line[first_comma + 1 :].strip()

            if members_str.startswith('"') and members_str.endswith('"'):
                members_str = members_str[1:-1]

            members = set(int(x) for x in ast.literal_eval(members_str))
            cluster_to_nodes[cluster_id] = members

    return cluster_to_nodes


def partition_for_modularity(
    G: nx.Graph,
    cluster_to_nodes: dict[int, set[int]],
) -> list[set[int]]:
    """
    Build a full partition:
      - PSCAN clusters as given
      - any node missing from cluster output becomes a singleton
    """
    communities = []

    cluster_items = list(cluster_to_nodes.items())
    for _, nodes in tqdm(
        cluster_items,
        desc="Building partition",
        file=sys.stdout,
        mininterval=2.0,
    ):
        if nodes:
            communities.append(set(nodes))

    assigned = set().union(*communities) if communities else set()
    missing = set(G.nodes()) - assigned

    for u in tqdm(
        missing,
        desc="Adding singleton communities",
        file=sys.stdout,
        mininterval=2.0,
    ):
        communities.append({u})

    return communities


def average_conductance(
    G: nx.Graph,
    cluster_to_nodes: dict[int, set[int]],
    min_size: int = 2,
) -> float:
    vals = []

    cluster_items = list(cluster_to_nodes.items())
    for _, nodes in tqdm(
        cluster_items,
        desc="Computing conductance",
        file=sys.stdout,
        mininterval=2.0,
    ):
        S = set(nodes)
        if len(S) < min_size or len(S) == G.number_of_nodes():
            continue
        try:
            c = nx.algorithms.cuts.conductance(G, S)
            vals.append(c)
        except ZeroDivisionError:
            pass

    return float("nan") if not vals else sum(vals) / len(vals)


def main():
    print("Loading graph...", flush=True)
    t0 = time.perf_counter()
    G = read_graph_tsv(GRAPH_TSV_PATH)
    t1 = time.perf_counter()
    print(f"Graph loaded in {t1 - t0:.2f} seconds", flush=True)

    print("Loading PSCAN clusters...", flush=True)
    t0 = time.perf_counter()
    cluster_to_nodes = read_pscan_clusters(CLUSTERS_PATH)
    t1 = time.perf_counter()
    print(f"Clusters loaded in {t1 - t0:.2f} seconds", flush=True)

    print(f"Graph nodes: {G.number_of_nodes()}", flush=True)
    print(f"Graph edges: {G.number_of_edges()}", flush=True)
    print(f"Clusters loaded: {len(cluster_to_nodes)}", flush=True)

    print("Preparing partition for modularity...", flush=True)
    t0 = time.perf_counter()
    communities = partition_for_modularity(G, cluster_to_nodes)
    t1 = time.perf_counter()
    print(f"Partition prepared in {t1 - t0:.2f} seconds", flush=True)

    print("Computing modularity...", flush=True)
    t0 = time.perf_counter()
    modularity = nx.algorithms.community.quality.modularity(G, communities)
    t1 = time.perf_counter()
    print(f"Modularity computed in {t1 - t0:.2f} seconds", flush=True)

    print("Computing average conductance...", flush=True)
    t0 = time.perf_counter()
    avg_cond = average_conductance(G, cluster_to_nodes, min_size=2)
    t1 = time.perf_counter()
    print(f"Average conductance computed in {t1 - t0:.2f} seconds", flush=True)

    print("\n=== Graph-Structure Metrics ===", flush=True)
    print(f"Number of PSCAN clusters: {len(cluster_to_nodes)}", flush=True)
    print(f"Modularity:               {modularity:.6f}", flush=True)
    print(f"Average conductance:      {avg_cond:.6f}", flush=True)


if __name__ == "__main__":
    main()