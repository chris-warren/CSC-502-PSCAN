#!/usr/bin/env python3
"""Similarity pipeline orchestrator for PSCAN — Division 2.

Folder layout assumed
---------------------
project/
    data/           datasets.py  (Shikha)
    similarity/     similarity_mapper.py
                    similarity_reducer.py
                    similarity_main.py   ← this file
    run/            main.py
                    job.sh

This module is the public API for the similarity division.
run/main.py calls run_pipeline() to compute all similarity scores.

Two modes
---------
1. In-memory (default, fast):
   mapper → generator → reducer  (nothing written to disk in between)

2. File-based (--use-temp-file, for debugging):
   mapper → .mapper.tsv → reducer

Usage
-----
    # Standalone CLI
    python similarity_main.py \
        --input  ../data/output/adjlists/lfr_5000.adjlist \
        --output ../data/output/adjlists/lfr_5000.sim.tsv \
        --verbose

    # As a library (from run/main.py)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from similarity.similarity_main import run_pipeline
    sim = run_pipeline(input_path, output_path, verbose=True)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from similarity_mapper import load_adjacency_list, mapper, write_mapper_output
from similarity_reducer import (
    iter_mapper_records,
    reduce_from_records,
    reduce_from_file,
    write_similarities,
    load_similarities,
)

SimMap = Dict[Tuple[int, int], float]


# ---------------------------------------------------------------------------
# 1. In-memory pipeline (default — fastest, no temp files)
# ---------------------------------------------------------------------------

def run_pipeline_in_memory(
    input_path: str | Path,
    *,
    verbose: bool = False,
) -> SimMap:
    """Run mapper → reducer fully in memory.

    Parameters
    ----------
    input_path : str | Path
        Adjacency-list file from data/datasets.py.
    verbose : bool
        Print timing to stderr.

    Returns
    -------
    SimMap
        ``{(u, v): similarity}`` with u < v for every edge.
    """
    t0 = time.perf_counter()

    if verbose:
        print(f"[similarity] Loading {input_path} ...", file=sys.stderr)

    adj = load_adjacency_list(input_path)
    n_nodes = len(adj)
    n_edges = sum(len(n) for n in adj.values()) // 2

    t1 = time.perf_counter()
    if verbose:
        print(f"[similarity]   Nodes: {n_nodes:,}  Edges: {n_edges:,}  (load: {t1-t0:.3f}s)", file=sys.stderr)
        print(f"[similarity] Running mapper + reducer ...", file=sys.stderr)

    t2 = time.perf_counter()
    sim = reduce_from_records(mapper(adj))
    t3 = time.perf_counter()

    if verbose:
        print(f"[similarity]   {len(sim):,} similarities in {t3-t2:.3f}s", file=sys.stderr)

    return sim


# ---------------------------------------------------------------------------
# 2. File-based pipeline (debug mode — writes mapper TSV to disk)
# ---------------------------------------------------------------------------

def run_pipeline_file_based(
    input_path: str | Path,
    *,
    mapper_output_path: Optional[str | Path] = None,
    verbose: bool = False,
) -> Tuple[SimMap, Path]:
    """Run mapper → .mapper.tsv → reducer.

    Parameters
    ----------
    input_path : str | Path
        Adjacency-list file.
    mapper_output_path : str | Path, optional
        Where to write mapper TSV. Defaults to <input_stem>.mapper.tsv.
    verbose : bool
        Print timing to stderr.

    Returns
    -------
    (SimMap, mapper_path)
    """
    input_path = Path(input_path)

    if mapper_output_path is None:
        mapper_output_path = input_path.parent / (input_path.stem + ".mapper.tsv")
    mapper_output_path = Path(mapper_output_path)

    t0 = time.perf_counter()

    if verbose:
        print(f"[similarity] Loading {input_path} ...", file=sys.stderr)

    adj = load_adjacency_list(input_path)
    n_nodes = len(adj)
    n_edges = sum(len(n) for n in adj.values()) // 2

    t1 = time.perf_counter()
    if verbose:
        print(f"[similarity]   Nodes: {n_nodes:,}  Edges: {n_edges:,}  (load: {t1-t0:.3f}s)", file=sys.stderr)
        print(f"[similarity] Mapper → {mapper_output_path} ...", file=sys.stderr)

    t2 = time.perf_counter()
    write_mapper_output(adj, mapper_output_path)
    t3 = time.perf_counter()

    if verbose:
        print(f"[similarity]   Mapper done in {t3-t2:.3f}s", file=sys.stderr)
        print(f"[similarity] Reducer ...", file=sys.stderr)

    t4 = time.perf_counter()
    sim = reduce_from_file(mapper_output_path)
    t5 = time.perf_counter()

    if verbose:
        print(f"[similarity]   Reducer done in {t5-t4:.3f}s  ({len(sim):,} edges)", file=sys.stderr)
        print(f"[similarity] Total: {t5-t0:.3f}s", file=sys.stderr)

    return sim, mapper_output_path


# ---------------------------------------------------------------------------
# 3. Public API — called by run/main.py
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

    This is the single entry point called by run/main.py.

    Parameters
    ----------
    input_path : str | Path
        Adjacency-list file from data/datasets.py.
    output_path : str | Path, optional
        Where to write the .sim.tsv for clustering.py (Van).
        If None, result is returned in memory only.
    use_temp_file : bool
        Use file-based pipeline (writes mapper TSV). Default: in-memory.
    mapper_output_path : str | Path, optional
        Intermediate mapper TSV path (only used with use_temp_file=True).
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
            print(f"[similarity] Writing → {output_path} ...", file=sys.stderr)
        t0 = time.perf_counter()
        write_similarities(sim, output_path)
        if verbose:
            print(f"[similarity]   Write done in {time.perf_counter()-t0:.3f}s", file=sys.stderr)

    return sim


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PSCAN similarity pipeline: mapper + reducer → .sim.tsv"
    )
    parser.add_argument("--input",  "-i", type=Path, required=True,
                        help="Adjacency-list file from data/datasets.py.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output .sim.tsv. Default: <input_stem>.sim.tsv.")
    parser.add_argument("--use-temp-file", action="store_true",
                        help="Write mapper TSV to disk before reducing (debug mode).")
    parser.add_argument("--mapper-output", type=Path, default=None,
                        help="Intermediate mapper TSV path (only with --use-temp-file).")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    print(f"[similarity] Done. {len(sim):,} similarities → {output_path}")


if __name__ == "__main__":
    main()