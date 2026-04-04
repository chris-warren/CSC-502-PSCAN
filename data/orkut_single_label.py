from collections import defaultdict


def read_nodes_from_adjlist(adjlist_path):
    nodes = set()
    with open(adjlist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            u = int(parts[0])
            nodes.add(u)
            for v in parts[1:]:
                nodes.add(int(v))
    return nodes


def read_communities(cmty_path):
    communities = []
    with open(cmty_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nodes = [int(x) for x in line.split()]
            communities.append(nodes)
    return communities


def assign_single_labels_smallest_comm(all_nodes, communities):
    node_to_comms = defaultdict(list)
    comm_sizes = {}

    for cid, members in enumerate(communities):
        comm_sizes[cid] = len(members)
        for node in members:
            node_to_comms[node].append(cid)

    labels = {}

    for node in sorted(all_nodes):
        comms = node_to_comms.get(node, [])

        if not comms:
            labels[node] = -1
            continue

        # Pick the smallest community by size; break ties by community ID
        best_cid = min(comms, key=lambda cid: (comm_sizes[cid], cid))
        labels[node] = best_cid

    return labels, comm_sizes


def write_labels(labels, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("node\tlabel\n")
        for node in sorted(labels):
            f.write(f"{node}\t{labels[node]}\n")


if __name__ == "__main__":
    adjlist_path = "orkut.adjlist"
    cmty_path = "com-orkut.top5000.cmty.txt"   # or com-orkut.all.cmty.txt
    output_path = "orkut_single_labels.tsv"

    all_nodes = read_nodes_from_adjlist(adjlist_path)
    communities = read_communities(cmty_path)

    labels, comm_sizes = assign_single_labels_smallest_comm(all_nodes, communities)
    write_labels(labels, output_path)

    unlabeled = sum(1 for x in labels.values() if x == -1)
    used_labels = sorted(set(x for x in labels.values() if x != -1))

    print(f"Wrote: {output_path}")
    print(f"Total nodes: {len(labels)}")
    print(f"Assigned labels: {len(labels) - unlabeled}")
    print(f"Unlabeled nodes: {unlabeled}")
    print(f"Number of used labels: {len(used_labels)}")