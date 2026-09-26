"""CLI: normalise the raw TSVs into the binary record store.

    python -m src.build_cache --data-dir ../../dataset --cache-dir ../../work/cache
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.store import build_store

SETS = {
    "train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"],
    "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../dataset")
    ap.add_argument("--cache-dir", default="../../work/cache")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(","):
        split = split.strip()
        for fname in SETS[split]:
            tag = Path(fname).stem  # train_source1
            prefix = cache_dir / tag
            if (cache_dir / (tag + ".meta.json")).is_file():
                print(f"  [skip] {tag} already cached")
                continue
            t0 = time.time()
            print(f"[build] {tag}")
            build_store(data_dir / split / fname, prefix, workers=args.workers)
            print(f"[done]  {tag} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
