"""Candidate generation (blocking) with IDF-weighted token index.

Scale: ~10M pool records, CPU-only. Dense vector search is infeasible.
Candidates from a token inverted index per country partition with IDF scoring.
"""

from __future__ import annotations

import pickle
from array import array
from pathlib import Path

import numpy as np

from src.preprocess import ADDRESS_BLOCK_STOPWORDS, skeleton

SPACE_X = "X"  # whole-name skeleton
SPACE_N = "N"  # name token skeleton
SPACE_T = "T"  # original name token (longest)
SPACE_A = "A"  # address token (longest)
SPACE_D = "D"  # digit run

DEFAULT_MAX_DF = 50000


def record_keys(name: str, addr: str, nskel: str, n_name=3, n_addr=3, n_digit=2):
    """Emit (key, weight) for one record. Weights filled in by IDF after index build."""
    keys = []

    # Whole-name consonant skeleton
    skel = nskel.replace(" ", "")
    if len(skel) >= 6:
        keys.append(SPACE_X + skel)

    # Per-token skeletons from name (more robust than full skeleton)
    ntok = [t for t in name.split() if len(t) >= 3]
    ntok.sort(key=len, reverse=True)
    for t in ntok[:n_name]:
        keys.append(SPACE_N + skeleton(t))

    # Original longest name tokens (exact match signal)
    for t in ntok[:n_name]:
        keys.append(SPACE_T + t)

    # Address tokens (longest, excluding stopwords)
    atok = [t for t in addr.split() if len(t) >= 3 and t not in ADDRESS_BLOCK_STOPWORDS]
    atok.sort(key=len, reverse=True)
    for t in atok[:n_addr]:
        keys.append(SPACE_A + t)

    # Longest digit runs
    digits, cur = [], []
    for ch in addr:
        if ch.isdigit():
            cur.append(ch)
        elif cur:
            digits.append("".join(cur))
            cur = []
    if cur:
        digits.append("".join(cur))
    digits.sort(key=len, reverse=True)
    for d in digits[:n_digit]:
        if len(d) >= 2:
            keys.append(SPACE_D + d)
    return keys


class InvertedIndex:
    def __init__(self):
        self.vocab = {}
        self.postings = np.zeros(0, dtype=np.int32)
        self.token_start = np.zeros(1, dtype=np.int64)
        self.idf = np.zeros(0, dtype=np.float32)
        self.df = np.zeros(0, dtype=np.int32)
        self.n_rows = 0

    def build(self, key_rows, n_rows: int, max_df: int, log=None):
        vocab = self.vocab
        inv = []
        tids = array("i")
        rows = array("i")
        for row, keys in enumerate(key_rows):
            for k in keys:
                t = vocab.get(k)
                if t is None:
                    t = len(vocab)
                    vocab[k] = t
                    inv.append(k)
                tids.append(t)
                rows.append(row)
        tids = np.frombuffer(tids, dtype=np.int32)
        rows = np.frombuffer(rows, dtype=np.int32)
        if log:
            log(f"      {len(tids):,} postings, {len(vocab):,} keys")

        counts = np.bincount(tids, minlength=len(vocab))
        keep = counts <= max_df
        if not keep.all():
            kept = np.flatnonzero(keep)
            remap = np.full(len(vocab), -1, dtype=np.int32)
            remap[kept] = np.arange(len(kept), dtype=np.int32)
            mask = keep[tids]
            rows = rows[mask]
            tids = remap[tids[mask]]
            counts = counts[kept]
            vocab = {inv[old]: int(remap[old]) for old in kept}
            if log:
                log(f"      max_df={max_df}: {len(tids):,} postings, {len(vocab):,} keys")
        else:
            # Ensure vocab is consistent with inv (already is, but be explicit)
            vocab = {inv[old]: old for old in range(len(inv))}
            if log:
                log(f"      no max_df filter: {len(tids):,} postings, {len(vocab):,} keys")

        nk = len(counts)
        starts = np.zeros(nk + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        order = np.argsort(tids, kind="stable")
        self.postings = rows[order].astype(np.int32, copy=False)
        self.token_start = starts
        self.df = counts.astype(np.int32, copy=False)
        self.n_rows = n_rows
        # IDF = log((N+1)/(df+1)) + 1
        self.idf = (np.log((n_rows + 1.0) / (counts.astype(np.float32) + 1.0)) + 1.0).astype(np.float32)
        del order, tids, rows, inv

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"vocab": self.vocab, "n_rows": self.n_rows}, fh, pickle.HIGHEST_PROTOCOL)
        np.savez(str(path) + ".npz", postings=self.postings, token_start=self.token_start,
                 idf=self.idf, df=self.df)

    @classmethod
    def load(cls, path):
        path = Path(path)
        idx = cls()
        with open(path, "rb") as fh:
            meta = pickle.load(fh)
        idx.vocab = meta["vocab"]
        idx.n_rows = meta["n_rows"]
        z = np.load(str(path) + ".npz")
        idx.postings = z["postings"]
        idx.token_start = z["token_start"]
        idx.idf = z["idf"]
        idx.df = z["df"]
        return idx

    def postings_for(self, key: str):
        tid = self.vocab.get(key)
        if tid is None:
            return None, 0.0
        s, e = self.token_start[tid], self.token_start[tid + 1]
        if s == e:
            return None, 0.0
        return self.postings[s:e], self.idf[tid]

    def expand(self, query_keys, max_post=500, max_total=2_000_000, log=None):
        """Union of postings with caps to avoid OOM."""
        exp_q, exp_c, exp_w = [], [], []
        total = capped = 0
        for q, keys in enumerate(query_keys):
            for k in keys:
                post, w = self.postings_for(k)
                if post is None:
                    continue
                if len(post) > max_post:
                    post = post[:max_post]
                    capped += 1
                n = len(post)
                total += n
                if total > max_total:
                    break
                exp_q.append(np.full(n, q, dtype=np.int32))
                exp_c.append(post)
                exp_w.append(np.full(n, w, dtype=np.float32))
            if total > max_total:
                break
        if log:
            log(f"      expansion {total:,} pairs ({capped:,} capped keys)")
        if not exp_c:
            z32, zf = np.zeros(0, np.int32), np.zeros(0, np.float32)
            return z32, z32, zf
        return np.concatenate(exp_q), np.concatenate(exp_c), np.concatenate(exp_w)


def aggregate(q, c, w, n_pool, min_votes=1):
    """One IDF-weighted score per (query, pool) pair."""
    if len(q) == 0:
        z32, zf = np.zeros(0, np.int32), np.zeros(0, np.float32)
        return z32, z32, zf
    combined = q.astype(np.int64) * np.int64(n_pool) + c.astype(np.int64)
    order = np.argsort(combined, kind="stable")
    cs = combined[order]
    ws = w[order]
    first = np.empty(len(cs), dtype=bool)
    first[0] = True
    np.not_equal(cs[1:], cs[:-1], out=first[1:])
    starts = np.flatnonzero(first)
    sums = np.add.reduceat(ws, starts).astype(np.float32)
    votes = np.diff(np.append(starts, len(cs)))
    uniq = cs[starts]
    uq = (uniq // n_pool).astype(np.int32)
    uc = (uniq % n_pool).astype(np.int32)
    keep = votes >= min_votes
    return uq[keep], uc[keep], sums[keep]


def top_k_per_query(uq, uc, score, k):
    if len(uq) == 0:
        return uq, uc, score
    order = np.lexsort((uc, -score, uq))
    uq_s, uc_s, sc_s = uq[order], uc[order], score[order]
    bnd = np.flatnonzero(np.r_[True, uq_s[1:] != uq_s[:-1]])
    sizes = np.diff(np.append(bnd, len(uq_s)))
    rank = np.arange(len(uq_s)) - np.repeat(bnd, sizes)
    keep = rank < k
    return uq_s[keep], uc_s[keep], sc_s[keep]