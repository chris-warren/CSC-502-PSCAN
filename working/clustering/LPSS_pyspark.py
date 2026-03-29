#!/usr/bin/env python3
"""
LPCC (Label Propagation Connected Components) — Pure Python version.

Replaces Spark-based implementation with iterative label propagation.

Input format (from LPSS.py):
    node,True,label,neighbor1 neighbor2 ...

Output:
    dict: {node: cluster_label}
"""


def load_data(input_path):
    """Load LPCC input file into memory."""
    data = {}

    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            node = parts[0]
            label = int(parts[2])

            if len(parts) > 3 and parts[3]:
                neighbors = parts[3].split(" ")
            else:
                neighbors = []

            data[node] = {
                "label": label,
                "neighbors": neighbors,
            }

    return data


def run_lpcc(input_path):
    """
    Run label propagation until convergence.

    Returns:
        dict: node -> cluster_label
    """

    data = load_data(input_path)

    iteration = 0

    while True:
        print(f"--- Iteration {iteration} ---")

        changed = False
        new_data = {}

        for node, info in data.items():
            current_label = info["label"]

            # collect neighbor labels
            neighbor_labels = [
                data[n]["label"] for n in info["neighbors"] if n in data
            ]

            # include self label
            min_label = min([current_label] + neighbor_labels)

            if min_label < current_label:
                changed = True

            new_data[node] = {
                "label": min_label,
                "neighbors": info["neighbors"],
            }

        data = new_data
        iteration += 1

        print(f"active change: {changed}")

        if not changed:
            print("finish lpcc")
            break

    # convert to node -> label mapping
    return {int(node): info["label"] for node, info in data.items()}