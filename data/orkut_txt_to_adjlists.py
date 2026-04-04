from collections import defaultdict
from pathlib import Path

input_path = "com-orkut.ungraph.txt"   # change this
output_path = "orkut.adjlist"

adj = defaultdict(set)

line_num = 0
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if line_num % 1000000 == 0:
            print(line_num)
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        u, v = parts[0], parts[1]
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
        line_num += 1

with open(output_path, "w", encoding="utf-8") as f:
    for u in sorted(adj, key=lambda x: int(x) if x.isdigit() else x):
        nbrs = sorted(adj[u], key=lambda x: int(x) if x.isdigit() else x)
        f.write(" ".join([u] + nbrs) + "\n")

print(f"Wrote {output_path} with {len(adj)} nodes")