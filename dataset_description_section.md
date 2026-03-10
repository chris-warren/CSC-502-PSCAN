## Dataset Description

We use two families of synthetic graphs in our PSCAN implementation, following the evaluation design in the paper. LFR benchmark graphs are used for clustering-accuracy evaluation because they contain planted community structure, which provides ground-truth labels for external validation. Barabási-Albert (BA) graphs are used for runtime and scalability analysis because they are large scale-free graphs, but they do not come with reliable ground-truth communities.

### 1. LFR Benchmark Graphs for Accuracy Evaluation

For accuracy experiments, we generate synthetic graphs using the LFR benchmark generator. The LFR model is appropriate because both the degree distribution and the community-size distribution follow power laws, which makes the graphs structurally closer to real networks than simpler random graph models. Each generated LFR graph contains built-in community memberships, which we extract and store as node labels for evaluation with clustering-quality metrics such as ARI and NMI.

The paper reports the following LFR graph sizes for accuracy evaluation:

| Graph name | Number of nodes |
|---|---:|
| Graph-5k | 5,000 |
| Graph-10k | 10,000 |
| Graph-20k | 20,000 |
| Graph-40k | 40,000 |
| Graph-80k | 80,000 |
| Graph-160k | 160,000 |

In our implementation, ground-truth labels are extracted from the `community` attribute returned by the LFR generator. Since PSCAN itself expects only the graph structure as input, the labels are written separately and used only during evaluation.

### 2. Barabási-Albert Graphs for Runtime Analysis

For runtime experiments, we generate Barabási-Albert graphs. The BA model produces scale-free networks through preferential attachment, which makes it suitable for stress-testing PSCAN on large sparse graphs. Unlike the LFR benchmark, BA graphs do not provide true community labels, so they are used only for efficiency experiments and not for clustering-accuracy evaluation.

The paper reports the following BA graph sizes for scalability experiments:

| Graph name | Number of nodes |
|---|---:|
| Barabasi-1M | 1,000,000 |
| Barabasi-2M | 2,000,000 |
| Barabasi-3M | 3,000,000 |
| Barabasi-4M | 4,000,000 |

### 3. Graph Representation

PSCAN assumes that the input graph is an undirected graph represented in adjacency-list format. Therefore, all generated datasets are converted into adjacency-list files before running the algorithm.

Each line of the adjacency-list file has the form:

`u v1 v2 v3 ... vk`

where `u` is a node ID and `v1, v2, ..., vk` are the neighbors of `u`.

The generated graphs are stored as simple undirected graphs. Self-loops are removed, and the adjacency lists are written in sorted order for reproducibility and easier debugging.

### 4. Ground-Truth Labels

Ground-truth labels are available only for the LFR datasets. For each node, we extract its planted community membership from the generator output and assign an integer cluster label. These labels are stored in a separate tab-separated file with the format:

`node<TAB>label`

This separation keeps the graph input compatible with PSCAN while preserving the community assignments required for external evaluation.

### 5. Dataset Parameters

The dataset generation script (`datasets.py`) documents all generation parameters explicitly. For LFR graphs, these include:

- `tau1`: exponent of the degree distribution
- `tau2`: exponent of the community-size distribution
- `mu`: mixing parameter
- `average_degree`
- `max_degree`
- `min_community`
- `max_community`
- random seed

For BA graphs, the main generation parameter is:

- `m`: number of edges attached by each new node
- random seed

The default parameters used in our script are:

#### LFR defaults
- `tau1 = 3.0`
- `tau2 = 1.5`
- `mu = 0.1`
- `average_degree = 15`
- `max_degree = 75`
- `min_community = 20`
- `max_community = 100`

#### BA defaults
- `m = 7`

### 6. Reproducibility Notes

The PSCAN paper clearly specifies the two synthetic dataset families and the graph sizes used in evaluation, but it does not fully specify every LFR generator parameter needed for exact regeneration of the benchmark instances. Therefore, in our implementation, we reproduce the experimental design of the paper and explicitly document the parameter values used in `datasets.py`.

For reproducibility, our script outputs:

- an adjacency-list file for each graph,
- a ground-truth label file for each LFR graph,
- a metadata JSON file containing the generation parameters and realized graph statistics,
- and a manifest file summarizing all generated datasets.

This makes the dataset pipeline transparent and easy to rerun.