"""End-to-end training pipeline using cached data."""
import sys
sys.path.insert(0, r"C:\Users\Aditya\OneDrive\Desktop\AMAZON ML CHANLLENGES\work\erlib")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score

from src.store import RecordStore
from src.blocking import InvertedIndex, record_keys, aggregate, top_k_per_query
from src.preprocess import name_tokens, address_tokens, digits_of, digit_runs_of

from rapidfuzz import fuzz

CACHE = Path(r"C:\Users\Aditya\OneDrive\Desktop\AMAZON ML CHANLLENGES\work\cache")
OUT = Path(r"C:\Users\Aditya\OneDrive\Desktop\AMAZON ML CHANLLENGES\output")
OUT.mkdir(parents=True, exist_ok=True)

def log(*a):
    print(*a, flush=True)

def pool_text(stores, global_rows, field):
    n2 = stores[0].n
    out = [""] * len(global_rows)
    m2 = global_rows < n2
    if m2.any():
        for j, v in zip(np.flatnonzero(m2), stores[0].get_many(field, global_rows[m2])):
            out[j] = v
    if (~m2).any():
        for j, v in zip(np.flatnonzero(~m2), stores[1].get_many(field, global_rows[~m2] - n2)):
            out[j] = v
    return out

def compute_features(s1_row, pool_row, s1_store, pool_stores):
    """Compute features for one candidate pair."""
    # Get texts
    s1_name = s1_store.get("name", s1_row)
    s1_addr = s1_store.get("addr", s1_row)
    s1_nskel = s1_store.get("nskel", s1_row)
    
    if pool_row < pool_stores[0].n:
        p_name = pool_stores[0].get("name", pool_row)
        p_addr = pool_stores[0].get("addr", pool_row)
        p_nskel = pool_stores[0].get("nskel", pool_row)
    else:
        p_name = pool_stores[1].get("name", pool_row - pool_stores[0].n)
        p_addr = pool_stores[1].get("addr", pool_row - pool_stores[0].n)
        p_nskel = pool_stores[1].get("nskel", pool_row - pool_stores[0].n)
    
    feats = {}
    
    # Rapidfuzz name features
    feats['name_token_set'] = fuzz.token_set_ratio(s1_name, p_name)
    feats['name_token_sort'] = fuzz.token_sort_ratio(s1_name, p_name)
    feats['name_partial'] = fuzz.partial_ratio(s1_name, p_name)
    feats['name_wratio'] = fuzz.WRatio(s1_name, p_name)
    feats['name_qratio'] = fuzz.QRatio(s1_name, p_name)
    
    # Rapidfuzz address features
    feats['addr_token_set'] = fuzz.token_set_ratio(s1_addr, p_addr)
    feats['addr_token_sort'] = fuzz.token_sort_ratio(s1_addr, p_addr)
    feats['addr_partial'] = fuzz.partial_ratio(s1_addr, p_addr)
    
    # Jaccard on tokens
    n1 = set(name_tokens(s1_name))
    n2 = set(name_tokens(p_name))
    feats['name_jaccard'] = len(n1 & n2) / max(len(n1 | n2), 1)
    
    a1 = set(address_tokens(s1_addr))
    a2 = set(address_tokens(p_addr))
    feats['addr_jaccard'] = len(a1 & a2) / max(len(a1 | a2), 1)
    
    # Digit overlap
    d1 = set(digits_of(s1_addr))
    d2 = set(digits_of(p_addr))
    feats['digit_overlap'] = len(d1 & d2)
    feats['digit_run_overlap'] = len(set(digit_runs_of(s1_addr)) & set(digit_runs_of(p_addr)))
    
    # Skeleton features
    s1_skel = s1_nskel.replace(" ", "")
    p_skel = p_nskel.replace(" ", "")
    feats['skel_ratio'] = fuzz.ratio(s1_skel, p_skel)
    feats['skel_token_set'] = fuzz.token_set_ratio(s1_nskel, p_nskel)
    
    # Length diff
    feats['name_len_diff'] = abs(len(s1_name) - len(p_name))
    feats['addr_len_diff'] = abs(len(s1_addr) - len(p_addr))
    
    return feats

def build_index_for_country(s1_store, pool_stores, country_code, log=log):
    """Build blocking index for one country."""
    pool_country = np.concatenate([pool_stores[0].country, pool_stores[1].country])
    pool_rows = np.flatnonzero(pool_country == country_code)
    log(f"  Building index for {len(pool_rows):,} pool rows...")
    
    names = pool_text(pool_stores, pool_rows, "name")
    addrs = pool_text(pool_stores, pool_rows, "addr")
    nskels = pool_text(pool_stores, pool_rows, "nskel")
    
    idx = InvertedIndex()
    idx.build(
        (record_keys(n, a, s) for n, a, s in zip(names, addrs, nskels)),
        n_rows=len(pool_rows), max_df=10_000_000, log=log
    )
    return idx, pool_rows

def query_candidates(s1_store, idx, pool_rows, s1_rows, k=50, log=log):
    """Generate top-k candidates for S1 rows in batches to avoid OOM cutoffs."""
    cand_map = {}
    batch_size = 10000
    for start_idx in range(0, len(s1_rows), batch_size):
        end_idx = min(start_idx + batch_size, len(s1_rows))
        batch_s1_rows = s1_rows[start_idx:end_idx]
        
        qnames = s1_store.get_many("name", batch_s1_rows)
        qaddrs = s1_store.get_many("addr", batch_s1_rows)
        qnskels = s1_store.get_many("nskel", batch_s1_rows)
        
        qkeys = [record_keys(n, a, s) for n, a, s in zip(qnames, qaddrs, qnskels)]
        # Allow enough total postings per batch of 10k
        q, c, w = idx.expand(qkeys, max_post=500, max_total=5000000, log=None)
        
        if len(q) == 0:
            continue
            
        uq, uc, score = aggregate(q, c, w, n_pool=idx.n_rows, min_votes=1)
        uqk, uck, _ = top_k_per_query(uq, uc, score, k)
        
        for q_local, c_local in zip(uqk, uck):
            s1_idx = int(batch_s1_rows[q_local])
            pool_global = int(pool_rows[c_local])
            cand_map.setdefault(s1_idx, []).append(pool_global)
            
    return cand_map

def main():
    t0 = time.time()
    log("Loading stores...")
    s1_train = RecordStore(CACHE / "train_source1")
    s2_train = RecordStore(CACHE / "train_source2")
    s3_train = RecordStore(CACHE / "train_source3")
    s1_test = RecordStore(CACHE / "test_source1")
    s2_test = RecordStore(CACHE / "test_source2")
    s3_test = RecordStore(CACHE / "test_source3")
    
    pool_train = (s2_train, s3_train)
    pool_test = (s2_test, s3_test)
    log(f"Loaded in {time.time()-t0:.1f}s")
    
    # Load ground truth
    gt = pd.read_csv(r"C:\Users\Aditya\OneDrive\Desktop\AMAZON ML CHANLLENGES\dataset\train\train_ground_truth.tsv", 
                     sep="\t", dtype=str, keep_default_na=False)
    s1_id_to_row = {x.decode("ascii"): i for i, x in enumerate(s1_train.ids)}
    gt_pairs = {}
    for sid, mids in zip(gt["source1_entity_id"].to_numpy(), gt["matched_entity_ids"].to_numpy()):
        r = s1_id_to_row.get(sid)
        if r is None:
            continue
        cset = set()
        for m in (mids.split(",") if mids else []):
            if m:
                cset.add(m)
        gt_pairs[r] = cset
    
    # Build pool ID index (entity_id <-> global index)
    pool_id_index = {}
    pool_idx_to_id = {}
    for off, st in ((0, s2_train), (s2_train.n, s3_train)):
        for i, x in enumerate(st.ids):
            sid = x.decode("ascii")
            global_idx = off + i
            pool_id_index[sid] = global_idx
            pool_idx_to_id[global_idx] = sid
    
    # Test pool index
    test_pool_idx_to_id = {}
    for off, st in ((0, s2_test), (s2_test.n, s3_test)):
        for i, x in enumerate(st.ids):
            test_pool_idx_to_id[off + i] = x.decode("ascii")
    
    def pool_id_to_str(idx):
        return pool_idx_to_id.get(idx, f"S2-{idx}")
    
    def test_pool_id_to_str(idx):
        return test_pool_idx_to_id.get(idx, f"S2-{idx}")
    
    # Train on a subset of S1 (e.g., 100k rows) for speed
    log("\nGenerating training candidates...")
    train_candidates = {}
    all_features = []
    all_labels = []
    
    for ci, clabel in enumerate(s1_train.countries):
        if clabel not in ["US", "India"]:  # Only train countries
            continue
        log(f"\n=== Country {clabel} ===")
        
        # Build index
        idx, pool_rows = build_index_for_country(s1_train, pool_train, ci)
        
        # Sample S1 rows for training
        s1_rows = np.flatnonzero(s1_train.country == ci)
        np.random.seed(42)
        train_rows = np.random.choice(s1_rows, size=min(50000, len(s1_rows)), replace=False)
        
        cand_map = query_candidates(s1_train, idx, pool_rows, train_rows, k=30)
        
        # Build feature matrix
        for s1_row, cand_globals in cand_map.items():
            for pool_global in cand_globals:
                feats = compute_features(s1_row, pool_global, s1_train, pool_train)
                label = 1 if pool_id_index.get(s1_train.ids[s1_row].decode(), 0) in gt_pairs.get(s1_row, set()) else 0
                # Check if this is a true match
                s1_id = s1_train.ids[s1_row].decode()
                true_matches = gt_pairs.get(s1_row, set())
                label = 1 if pool_id_index.get(s1_train.ids[s1_row].decode(), None) in [x for x in true_matches] else 0
                # Actually check pool_global against true matches
                true_pool_ids = set()
                for m in gt_pairs.get(s1_row, []):
                    true_pool_ids.add(pool_id_index.get(m))
                label = 1 if pool_global in true_pool_ids else 0
                
                all_features.append(feats)
                all_labels.append(label)
    
    log(f"\nTraining data: {len(all_features)} pairs, {sum(all_labels)} positives")
    
    if len(all_features) == 0:
        log("ERROR: No training data!")
        return
    
    # Convert to DataFrame
    X = pd.DataFrame(all_features)
    y = np.array(all_labels)
    
    # Train LightGBM
    log("Training LightGBM...")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    thresholds = []
    oof_preds = np.zeros(len(X))
    
    for fold, (tr, va) in enumerate(kf.split(X)):
        log(f"  Fold {fold+1}/3")
        clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1)
        clf.fit(X.iloc[tr], y[tr])
        va_preds = clf.predict_proba(X.iloc[va])[:, 1]
        oof_preds[va] = va_preds
        
        # Find best threshold for F0.5
        best_f05 = 0
        best_th = 0.5
        for th in np.arange(0.1, 0.9, 0.05):
            preds = (va_preds >= th).astype(int)
            f = fbeta_score(y[va], preds, beta=0.5, zero_division=0)
            if f > best_f05:
                best_f05 = f
                best_th = th
        thresholds.append(best_th)
        log(f"    Best threshold: {best_th:.2f}, F0.5: {best_f05:.4f}")
    
    avg_th = np.mean(thresholds)
    log(f"Average threshold: {avg_th:.4f}")
    
    # Train final model on all data
    final_clf = lgb.LGBMClassifier(n_estimators=150, random_state=42, n_jobs=-1, verbosity=-1)
    final_clf.fit(X, y)
    
    # Inference on test
    log("\nRunning inference on test set...")
    test_matches = {}
    test_candidates = {}
    
    for ci, clabel in enumerate(s1_test.countries):
        log(f"\n=== Test Country {clabel} ===")
        idx, pool_rows = build_index_for_country(s1_test, pool_test, ci)
        
        s1_rows = np.flatnonzero(s1_test.country == ci)
        cand_map = query_candidates(s1_test, idx, pool_rows, s1_rows, k=30)
        
        # Score candidates
        for s1_row, cand_globals in cand_map.items():
            s1_id = s1_test.ids[s1_row].decode()
            test_candidates[s1_id] = cand_globals
            
            # Build features and predict
            probs = []
            for pool_global in cand_globals:
                feats = compute_features(s1_row, pool_global, s1_test, pool_test)
                probs.append(feats)
            
            if probs:
                X_test = pd.DataFrame(probs)
                pred_probs = final_clf.predict_proba(X_test)[:, 1]
                matches = [test_pool_id_to_str(g) for g, p in zip(cand_globals, pred_probs) if p >= avg_th]
            else:
                matches = []
            
            test_matches[s1_id] = matches
    
    # Write output
    log("\nWriting output...")
    
    # matching_results.tsv
    s1_test_ids = [x.decode() for x in s1_test.ids]
    with open(OUT / "matching_results.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_test_ids:
            matches = test_matches.get(sid, [])
            f.write(f"{sid}\t{','.join(matches)}\n")
    
    # candidate_pairs.tsv
    with open(OUT / "candidate_pairs.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_test_ids:
            cands = test_candidates.get(sid, [])
            cands_str = [test_pool_id_to_str(g) for g in cands]
            f.write(f"{sid}\t{','.join(cands_str)}\n")
    
    log(f"Done in {time.time()-t0:.1f}s")
    log(f"Output written to {OUT}")

if __name__ == "__main__":
    main()