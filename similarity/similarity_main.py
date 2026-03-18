#!/usr/bin/env python3
"""Main orchestrator for the PSCAN structural similarity (PCSS) pipeline.

This script ties together the mapper and reducer into a single runnable
pipeline:

    adjacency-list file
          ↓
    similarity_mapper  →  MapRecords (in memory or via temp file)
          ↓
    similarity_reducer →  SimMap  {(u,v): score}
          ↓
    similarity TSV file  (consumed by clustering.py)

The pipeline can run in two modes:

1. **In-memory** (default, fast for single-machine runs):
   Mapper yields records as a Python generator; the reducer consumes them
   directly without writing intermediate data to disk.

2. **File-based** (``--use-temp-file``):
   Mapper writes records to a temporary TSV; reducer reads that file.
   Useful for debugging or for inspecting the intermediate mapper output.

Usage
-----
    # Basic run
    python similarity_main.py --input data/adjlists/lfr_5000.adjlist

    # Specify output explicitly
    python similarity_main.py \
        --input  data/adjlists/lfr_5000.adjlist \
        --output data/similarities/lfr_5000.sim.tsv \
        --verbose

    # File-based pipeline (keeps intermediate mapper output)
    python similarity_main.py \
        --input        data/adjlists/lfr_5000.adjlist \
        --output       data/similarities/lfr_5000.sim.tsv \
        --use-temp-file \
        --mapper-output data/adjlists/lfr_5000.mapper.tsv \
        --verbose

    # As a library
    from similarity_main import run_pipeline
    sim = run_pipeline("data/adjlists/lfr_5000.adjlist")
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Internal modules
from similarity_mapper import load_adjacency_list, mapper, write_mapper_output
from similarity_reducer import (
    iter_mapper_records,
    reduce_from_records,
    reduce_from_file,
    write_similarities,
    load_similarities,
)


# ---------------------------------------------------------------------------
# Type alias (re-exported for convenience)
# ---------------------------------------------------------------------------
SimMap = Dict[Tuple[int, int], float]


# ---------------------------------------------------------------------------
# 1. In-memory pipeline (default)
# ---------------------------------------------------------------------------

def run_pipeline_in_memory(
    input_path: str | Path,
    *,
    verbose: bool = False,
) -> SimMap:
    """Run the full mapper → reducer pipeline in memory (no temp files).

    Parameters
    ----------
    input_path : str | Path
        Adjacency-list file produced by datasets.py.
    verbose : bool
        If True, print timing and progress to stderr.

    Returns
    -------
    SimMap
        ``{(u, v): similarity}`` with u < v for every edge.
    """
    t0 = time.perf_counter()

    if verbose:
        print(f"[main] Loading graph from {input_path} ...", file=sys.stderr)

    adj = load_adjacency_list(input_path)

    n_nodes = len(adj)
    n_edges = sum(len(nbrs) for nbrs in adj.values()) // 2

    if verbose:
        t1 = time.perf_counter()
        print(
            f"[main]   Nodes: {n_nodes:,}   Edges: {n_edges:,}   "
            f"(load: {t1 - t0:.3f}s)",
            file=sys.stderr,
        )
        print("[main] Running mapper ...", file=sys.stderr)

    t_map0 = time.perf_counter()

    # Mapper yields MapRecords; reducer consumes them as a generator
    # so neither side needs to materialise the full record list.
    map_records = mapper(adj)
    sim = reduce_from_records(map_records)

    t_map1 = time.perf_counter()

    if verbose:
        print(
            f"[main]   Mapped + reduced {len(sim):,} edges in {t_map1 - t_map0:.3f}s",
            file=sys.stderr,
        )
        print(f"[main] Total pipeline time: {t_map1 - t0:.3f}s", file=sys.stderr)

    return sim


# ---------------------------------------------------------------------------
# 2. File-based pipeline (useful for debugging / inspecting intermediate data)
# ---------------------------------------------------------------------------

def run_pipeline_file_based(
    input_path: str | Path,
    *,
    mapper_output_path: Optional[str | Path] = None,
    verbose: bool = False,
) -> Tuple[SimMap, Path]:
    """Run the pipeline through an intermediate mapper TSV file.

    Parameters
    ----------
    input_path : str | Path
        Adjacency-list file.
    mapper_output_path : str | Path, optional
        Where to write the mapper TSV.  Defaults to
        ``<input_stem>.mapper.tsv`` in the same directory.
    verbose : bool
        Print timing/progress to stderr.

    Returns
    -------
    (SimMap, mapper_path)
        The computed similarity map and the path to the mapper file.
    """
    input_path = Path(input_path)

    if mapper_output_path is None:
        mapper_output_path = input_path.parent / (input_path.stem + ".mapper.tsv")
    mapper_output_path = Path(mapper_output_path)

    t0 = time.perf_counter()

    # --- Map phase ---
    if verbose:
        print(f"[main] Loading graph from {input_path} ...", file=sys.stderr)

    adj = load_adjacency_list(input_path)
    n_nodes = len(adj)
    n_edges = sum(len(nbrs) for nbrs in adj.values()) // 2

    if verbose:
        t1 = time.perf_counter()
        print(
            f"[main]   Nodes: {n_nodes:,}   Edges: {n_edges:,}   "
            f"(load: {t1 - t0:.3f}s)",
            file=sys.stderr,
        )
        print(f"[main] Running mapper → {mapper_output_path} ...", file=sys.stderr)

    t_map0 = time.perf_counter()
    write_mapper_output(adj, mapper_output_path)
    t_map1 = time.perf_counter()

    if verbose:
        print(f"[main]   Mapper done in {t_map1 - t_map0:.3f}s", file=sys.stderr)

    # --- Reduce phase ---
    if verbose:
        print(f"[main] Running reducer from {mapper_output_path} ...", file=sys.stderr)

    t_red0 = time.perf_counter()
    sim = reduce_from_file(mapper_output_path)
    t_red1 = time.perf_counter()

    if verbose:
        print(
            f"[main]   Reducer done in {t_red1 - t_red0:.3f}s   "
            f"({len(sim):,} edges)",
            file=sys.stderr,
        )
        print(f"[main] Total pipeline time: {t_red1 - t0:.3f}s", file=sys.stderr)

    return sim, mapper_output_path


# ---------------------------------------------------------------------------
# 3. Public API entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    use_temp_file: bool = False,
    mapper_output_path: Optional[str | Path] = None,
    verbose: bool = False,
) -> SimMap:
    """Run the full PCSS pipeline and optionally write the result to disk.

    This is the primary entry point for use in ``main.py`` or notebooks.

    Parameters
    ----------
    input_path : str | Path
        Adjacency-list file produced by datasets.py.
    output_path : str | Path, optional
        Where to write the similarity TSV for clustering.py.
        If None, the result is returned but not written to disk.
    use_temp_file : bool
        If True, use the file-based pipeline (writes mapper TSV to disk).
        Default is the faster in-memory pipeline.
    mapper_output_path : str | Path, optional
        Only used when ``use_temp_file=True``.  Path for the intermediate
        mapper TSV.
    verbose : bool
        Print progress and timing.

    Returns
    -------
    SimMap
        ``{(u, v): similarity}`` with u < v for every edge.
    """
    if use_temp_file:
        sim, _ = run_pipeline_file_based(
            input_path,
            mapper_output_path=mapper_output_path,
            verbose=verbose,
        )
    else:
        sim = run_pipeline_in_memory(input_path, verbose=verbose)

    if output_path is not None:
        if verbose:
            print(f"[main] Writing similarities to {output_path} ...", file=sys.stderr)
        t_w0 = time.perf_counter()
        write_similarities(sim, output_path)
        t_w1 = time.perf_counter()
        if verbose:
            print(f"[main]   Write done in {t_w1 - t_w0:.3f}s", file=sys.stderr)

    return sim


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PSCAN Structural Similarity Pipeline: "
            "runs mapper + reducer and writes a similarity TSV."
        )
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Adjacency-list file (.adjlist) from datasets.py.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help=(
            "Output similarity TSV for clustering.py. "
            "Defaults to <input_stem>.sim.tsv in the same directory."
        ),
    )
    parser.add_argument(
        "--use-temp-file",
        action="store_true",
        help=(
            "Use the file-based pipeline (mapper writes to disk before reducer). "
            "Useful for debugging.  Default: in-memory pipeline."
        ),
    )
    parser.add_argument(
        "--mapper-output",
        type=Path,
        default=None,
        help=(
            "Path for the intermediate mapper TSV (only with --use-temp-file). "
            "Defaults to <input_stem>.mapper.tsv."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress and timing to stderr.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve default output path
    output_path = args.output
    if output_path is None:
        output_path = args.input.parent / (args.input.stem + ".sim.tsv")

    sim = run_pipeline(
        args.input,
        output_path=output_path,
        use_temp_file=args.use_temp_file,
        mapper_output_path=args.mapper_output,
        verbose=args.verbose,
    )

    print(f"[main] Done. {len(sim):,} edge similarities written to {output_path}")


if __name__ == "__main__":
    main()