"""Compact, memory-mapped record store.

The raw challenge TSVs total ~2.5 GB and hold 24M records; normalising them is
pure-Python work that is far too slow to repeat on every experiment.  This module
normalises once into a binary layout that can be memory-mapped:

*   variable-length UTF-8 text is stored as one flat ``uint8`` blob plus an
    ``int64`` offset table (avoids the 4-bytes-per-char cost of numpy ``U`` dtypes),
*   ``entity_id`` is stored as a fixed-width byte array (the IDs are short ASCII),
*   ``country`` is stored as ``uint8`` codes with the label list in a JSON sidecar.

The country vocabulary is built from the data rather than hard-coded, so the
unseen ``France`` partition in the test set flows through the pipeline untouched.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import normalize

FIELDS = ("name", "addr", "nskel", "askel")
ID_WIDTH = 24
CHUNK = 400_000
NORM_BATCH = 50_000


def _norm_init():
    global _NORMALIZE
    _NORMALIZE = normalize


def _norm_batch(batch):
    """``batch`` is a list of ``(name, address)``; returns a list of 4-tuples."""
    return [_NORMALIZE(n, a) for n, a in batch]


class BlobWriter:
    """Appends UTF-8 strings into a flat blob + offset table."""

    def __init__(self, blob_path: Path, off_path: Path, capacity_hint: int = 1 << 22):
        self._fh = open(blob_path, "wb")
        self._offs = [0]
        self._pos = 0
        self._cap = capacity_hint
        self._buf = []
        self._buf_len = 0
        self.off_path = off_path

    def write(self, values) -> None:
        ap = self._buf.append
        for v in values:
            b = v.encode("utf-8", "replace")
            ap(b)
            self._buf_len += len(b)
            self._pos += len(b)
            self._offs.append(self._pos)
        if self._buf_len >= self._cap:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            self._fh.write(b"".join(self._buf))
            self._buf = []
            self._buf_len = 0

    def close(self) -> None:
        self.flush()
        self._fh.close()
        np.asarray(self._offs, dtype=np.int64).tofile(self.off_path)


class RecordStore:
    """Read-only memory-mapped view over a normalised record set."""

    def __init__(self, prefix):
        self.prefix = Path(prefix)
        meta_path = self.prefix.parent / (self.prefix.name + ".meta.json")
        self.meta = json.loads(meta_path.read_text("utf-8"))
        # The country vocabulary is *global* (one code per label for every store),
        # so codes from different files are always comparable.
        vocab_path = self.prefix.parent / "countries.json"
        if vocab_path.is_file():
            vocab = json.loads(vocab_path.read_text("utf-8"))
            self.countries = [None] * len(vocab)
            for k, v in vocab.items():
                self.countries[v] = k
        else:
            self.countries = self.meta["countries"]
        self.n = int(self.meta["n"])
        self.id_len = int(self.meta.get("id_len", ID_WIDTH))
        self.ids = np.memmap(
            self.prefix.with_suffix(".ids"), dtype=f"S{self.id_len}", mode="r", shape=(self.n,)
        )
        self.country = np.memmap(
            self.prefix.with_suffix(".country"), dtype=np.uint8, mode="r", shape=(self.n,)
        )
        self._blobs = {}
        self._offs = {}
        for field in FIELDS:
            self._blobs[field] = np.memmap(self.prefix.with_suffix("." + field), dtype=np.uint8, mode="r")
            self._offs[field] = np.memmap(
                self.prefix.with_suffix("." + field + ".off"), dtype=np.int64, mode="r"
            )

    # -- single record -----------------------------------------------------------
    def get(self, field: str, index: int) -> str:
        o = self._offs[field]
        return self._blobs[field][o[index]:o[index + 1]].tobytes().decode("utf-8", "replace")

    # -- scattered indices -------------------------------------------------------
    def get_many(self, field: str, indices: np.ndarray) -> list:
        o = self._offs[field]
        blob = self._blobs[field]
        out = []
        ap = out.append
        for i in indices:
            ap(blob[o[i]:o[i + 1]].tobytes().decode("utf-8", "replace"))
        return out

    def id_of(self, index: int) -> str:
        return self.ids[index].decode("ascii", "replace")

    def ids_of(self, indices: np.ndarray) -> list:
        return [x.decode("ascii", "replace") for x in self.ids[indices]]

    def country_code(self, label: str) -> int:
        return self.countries.index(label)

    def release(self) -> None:
        self.ids = None
        self.country = None
        self._blobs = {}
        self._offs = {}


def _load_or_create_vocab(cache_dir: Path, found: dict) -> dict:
    """Country codes must agree across *all* stores, otherwise comparing the
    ``country`` byte of a Source-1 record with a pool record compares codes from
    two different vocabularies.  The first store to be built writes the canonical
    vocabulary and every later store extends it."""
    path = cache_dir / "countries.json"
    vocab = {}
    if path.is_file():
        vocab = json.loads(path.read_text("utf-8"))
    changed = False
    for key in found:
        if key not in vocab:
            vocab[key] = len(vocab)
            changed = True
    if changed or not path.is_file():
        path.write_text(json.dumps(vocab, ensure_ascii=False), "utf-8")
    return vocab


def build_store(tsv_path, prefix, workers: int = 0, log=print) -> RecordStore:
    """Normalise a raw ``*_sourceN.tsv`` into a :class:`RecordStore` on disk."""
    tsv_path = Path(tsv_path)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # pass 1 -- country vocabulary + row count
    found, n_rows = {}, 0
    for chunk in pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNK):
        for c in chunk["country"].unique():
            key = unicodedata.normalize("NFKC", str(c)).strip()
            found.setdefault(key, 0)
        n_rows += len(chunk)
    vocab = _load_or_create_vocab(prefix.parent, found)
    countries = [None] * len(vocab)
    for k, v in vocab.items():
        countries[v] = k
    log(f"    {tsv_path.name}: {n_rows:,} rows, countries={countries}")

    id_arr = np.memmap(prefix.with_suffix(".ids"), dtype=f"S{ID_WIDTH}", mode="w+", shape=(n_rows,))
    cc_arr = np.memmap(prefix.with_suffix(".country"), dtype=np.uint8, mode="w+", shape=(n_rows,))
    writers = {f: BlobWriter(prefix.with_suffix("." + f), prefix.with_suffix("." + f + ".off")) for f in FIELDS}

    workers = workers or min(12, mp.cpu_count())
    pool = None
    if workers > 1:
        pool = mp.Pool(workers, initializer=_norm_init)

    pos = 0
    for chunk in pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNK):
        size = len(chunk)
        id_arr[pos:pos + size] = np.array(chunk["entity_id"].to_numpy(), dtype=f"S{ID_WIDTH}")
        keys = [unicodedata.normalize("NFKC", c).strip() for c in chunk["country"].to_numpy()]
        cc_arr[pos:pos + size] = np.fromiter((vocab[c] for c in keys), dtype=np.uint8, count=size)

        pairs = list(zip(chunk["business_name"].tolist(), chunk["business_address"].tolist()))
        if pool is not None:
            batches = [pairs[i:i + NORM_BATCH] for i in range(0, len(pairs), NORM_BATCH)]
            normed = [r for batch in pool.imap(_norm_batch, batches, chunksize=1) for r in batch]
        else:
            normed = _norm_batch(pairs)
        for field, idx in (("name", 0), ("addr", 1), ("nskel", 2), ("askel", 3)):
            writers[field].write([r[idx] for r in normed])
        del normed, pairs
        pos += size
    if pool is not None:
        pool.close()
        pool.join()

    for w in writers.values():
        w.close()
    id_arr.flush()
    cc_arr.flush()
    del id_arr, cc_arr

    meta = {
        "n": n_rows,
        "countries": countries,
        "id_len": ID_WIDTH,
        "source_tsv": tsv_path.name,
    }
    prefix.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), "utf-8"
    )
    return RecordStore(prefix)
