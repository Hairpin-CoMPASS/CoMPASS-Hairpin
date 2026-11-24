"""Disk-backed memoization for expensive, deterministic pipeline computations.

The cache lives inside the *project's own data folder* (next to `figures/`,
`results/`, etc.) rather than this git repo, so it is specific to each
dataset and never gets committed. `joblib.Memory` keys a cached result by the
wrapped function's source code plus its argument values, so a result is
reused until one of those actually changes -- editing a cell's code or
passing a different dataframe/genotype set/zone set/bin size invalidates
exactly the affected entries, nothing else.
"""
from pathlib import Path

from joblib import Memory


def get_memory(config: dict) -> Memory:
    cache_dir = Path(config["project_path_full"]) / "cache" / "joblib"
    return Memory(location=cache_dir, verbose=0)
