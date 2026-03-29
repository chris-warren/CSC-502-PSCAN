#!/usr/bin/env python3
from pathlib import Path
import sys
import ast
from typing import Dict, Iterable, List, Set, Tuple
from collections import defaultdict
import os


def create_filtered_adjlist_and_LPCC_emitter(
    project_root, tsv_file, EPSILON: float = 0.5
) -> str:
    parsed_input_path = f"{project_root}/data/output/parsed_input/"
    # tsv_files = list(Path("./similarity/data/data/adjlists/").glob("*.tsv"))
    tsv_files = [tsv_file]

    for f in tsv_files:
        print(f)

    for f in tsv_files:
        current_dict = defaultdict(list)
        with open(f) as file:
            next(file)
            for line in file:
                current_row = line.strip().split("\t")
                u, v, current_similarity = current_row
                u = u
                v = v
                if float(current_similarity) >= EPSILON:
                    current_dict[u].append(v)
                    current_dict[v].append(u)

        adjlist_path = f"{project_root}/data/output/filtered_adjlists/"
        os.makedirs(os.path.dirname(adjlist_path), exist_ok=True)
        with open(
            f"{adjlist_path}filtered_edge_{f.stem}.tsv",
            "w",
        ) as file:

            for key in current_dict:
                neighbours = " ".join(map(str, current_dict[key]))
                file.write(f"{key}\t{neighbours}\n")

        os.makedirs(os.path.dirname(parsed_input_path), exist_ok=True)
        with open(f"{parsed_input_path}parse_{f.stem}.tsv", "w") as file:

            for key, current_adjlists in current_dict.items():
                current_adjlists = " ".join(current_adjlists)

                file.write(f"{key},{True},{key},{current_adjlists}\n")

    return f"{parsed_input_path}parse_{f.stem}.tsv"


# if __name__ == "__main__":
#     create_filtered_adjlist_and_LPCC_emitter()
