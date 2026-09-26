# DECISIONS.md — Architectural & Implementation Decisions

## 1. Dense Embeddings → Token Inverted Index (REJECTED → CHOSEN)

**Decision**: Do NOT use sentence-transformers + FAISS for candidate generation.

**Why**: 
- 10M pool records × 384-dim MiniLM = 3.8 GB embeddings (CPU RAM)
- Encoding 10M texts on 16-core CPU ≈ 2–4 hours
- FAISS `IndexFlatIP` over 10M × 384 for 1.7M queries ≈ 8 PFLOP (infeasible on CPU)
- Challenge constraint: no GPU, final model ≤8B params, MIT/Apache license

**Alternative**: Token-level inverted index with IDF weighting
- 50M postings (int32) = 200 MB
- Build: 3–4 min (single pass, numpy bincount + argsort)
- Query: sub-second per 1000 S1 entities
- Scales linearly, CPU-only, deterministic

---

## 2. Country Partitioning — Dynamic Vocabulary (CHOSEN)

**Decision**: Partition candidate generation strictly by `country` label, but build a **global country vocabulary** shared across all 6 stores.

**Why**:
- Training: `US`, `India` — Test adds unseen `France`
- Hard-coding `{"US":0, "India":1}` breaks on test
- Per-file vocabularies caused code mismatches (e.g., train_source1: US=0, train_source2: India=0)
- Single `countries.json` in cache root guarantees consistent codes across all stores

**Implementation**: First store writes `countries.json`; subsequent stores extend it. Each `RecordStore` loads this global vocab at init.

---

## 3. Cross-Script Matching — Consonant Skeleton (CHOSEN)

**Decision**: Map both Devanagari and Latin names to a **consonant-only skeleton** (`"राम मार्केटिंग"` → `"rm mrktng"`, `"Ram Marketing"` → `"rm mrktng"`).

**Why**:
- ~4% of GT pairs are cross-script (Devanagari ↔ Latin transliteration)
- Full transliteration (ISO 15919) is fragile: vowels differ (`राम` → `"raama"` vs `"ram"`)
- Consonant skeleton is deterministic, hand-written static table (no external data), and bridges the scripts exactly

**Keys emitted per record**:
- `X{whole_name_skeleton}` — high-weight whole-name match
- `N{token_skeleton}` — per-token skeleton (robust to reordering)
- `T{original_token}` — exact token match signal
- `A{addr_token}` — address tokens (longest, stopword-filtered)
- `D{digit_run}` — house/postal numbers

---

## 4. Blocking Weights — IDF (not fixed per-space)

**Decision**: Weight = `log((N+1)/(df+1)) + 1` (IDF), not fixed `{X:3.0, N:1.0, ...}`.

**Why**:
- Fixed weights can't adapt to corpus statistics (e.g., "services" appears millions of times)
- IDF naturally down-weights common tokens, up-weights rare discriminative ones
- Computed once per index build from `df` array (zero overhead)

---

## 5. Candidate Scoring — Weighted Key Overlap

**Decision**: Score = sum of IDF weights for each **distinct shared key** between S1 and pool record. Require `min_votes ≥ 1` (at least one shared key).

**Why**:
- A pair sharing `X` + `N` + `A` is stronger than one sharing only `A`
- `min_votes=1` keeps recall high; `min_votes=2` dropped true pairs that share only the skeleton
- Top-K per S1 entity (K=30) limits candidate set size for feature extraction

---

## 6. Feature Set — RapidFuzz + Skeleton + Digits

**Decision**: 18 features per candidate pair:
- 5 RapidFuzz name metrics (token_set, token_sort, partial, WRatio, QRatio)
- 3 RapidFuzz address metrics (token_set, token_sort, partial)
- 2 Jaccard (name tokens, address tokens)
- 2 digit overlap (all digits, digit runs ≥3)
- 2 skeleton (full-skeleton ratio, token-set skeleton)
- 2 length diffs (name, address)
- 2 derived (but not used: country equality constant within partition)

**Why**:
- RapidFuzz is fast C++ implementation (thousands of pairs/sec)
- No learned embeddings needed — works on raw strings
- Skeleton features catch cross-script pairs
- Digit overlap critical for address matching
- Feature dimensionality low → fast LightGBM training

---

## 7. Model — LightGBM (Gradient Boosted Trees)

**Decision**: `LGBMClassifier(n_estimators=150, random_state=42, n_jobs=-1)`.

**Why**:
- Handles mixed feature scales natively
- Fast training on 8.5k rows (3-fold CV < 30 sec)
- Built-in handling of class imbalance (559 pos / 8.5k total ≈ 6.6%)
- Apache 2.0 license, ≤8B params (tree ensemble)
- No GPU required

**Threshold tuning**: Macro-F1 per fold (F₀.₅ proxy), then average threshold across folds (0.33). Final model retrained on all data.

---

## 8. Training Data — Sampled S1 Entities (not all 2.2M)

**Decision**: Sample 50k S1 entities per country for training (100k total).

**Why**:
- Full 2.2M S1 × 30 candidates = 66M pairs → too large for feature extraction + training
- Candidate generation + feature computation for 100k S1 ≈ 5 min
- Stratified by country (US, India) preserves distribution
- 559 positives sufficient for stable LightGBM training

---

## 9. Output Format — Strict TSV (no quotes, no spaces)

**Decision**: Tab delimiter (`\t`), comma-separated ID lists, no quoting, no spaces after commas.

**Why**: Challenge specification mandates this; validator rejects any deviation.
- `S1-00001\tS2-00047,S3-00812`
- Empty match → `S1-00002\t` (trailing tab, nothing after)

---

## 10. Cache Format — Blob + Offset (not numpy `U` strings)

**Decision**: Variable-length UTF-8 stored as flat `uint8` blob + `int64` offset table.

**Why**:
- numpy `dtype='U60'` = 4 bytes/char = 240 bytes/record → 24M × 240 = 5.8 GB per field
- Blob+offset = 1 byte/char (UTF-8) → ~40 bytes/record → 24M × 40 = 1 GB per field
- 6 fields × 2 splits ≈ 4 GB total (fits in 16 GB RAM, fast mmap)

---

## 11. Memory Caps — `max_post=500`, `max_total=500k` per query batch

**Decision**: Cap expansion in `InvertedIndex.expand()`.

**Why**:
- Without caps, common tokens (e.g., "wayne" df=17k) explode expansion to 456M pairs (3.4 GB int64)
- Caps keep peak memory < 2 GB, query latency < 2 sec per 50k S1 entities
- Trade-off: some recall loss for very common tokens, but high-IDF keys dominate scoring

---

## 12. Validation — Run `validate_submission.py` Before Submit

**Decision**: Always run the provided validator locally.

**Why**: Catches format errors (missing S1 rows, duplicate IDs, wrong prefixes, non-subset candidates) that would waste a submission. The validator is strict on format but does not compute F₀.₅.

---

## 13. Rejected Approaches (Tried & Abandoned)

| Approach | Reason |
|----------|--------|
| Sentence-BERT + FAISS | CPU encoding too slow; FAISS flat index O(NQ×NP) infeasible |
| TF-IDF char n-gram cosine | Sparse matrix 2.2M × 10M too large; per-pair dot product slow |
| MinHash LSH | Implementation complexity; recall hard to guarantee |
| Fixed per-space weights | IDF is strictly better and free |
| `min_votes=2` | Killed recall (true pairs often share only skeleton) |
| Full 2.2M training pairs | OOM / too slow; 100k sample sufficient |

---

## 14. Known Limitations / Future Improvements

1. **Cross-country GT pairs**: Current code assumes GT pairs share country. If any GT pair crosses countries, it's missed. (Validator showed 0 cross-country in sample, but unconfirmed at full scale.)
2. **Singleton calibration**: Threshold tuned on candidate pairs (no negatives for singletons). Could use a separate singleton classifier.
3. **Feature engineering**: Could add TF-IDF cosine on name (sparse), or cross-encoder re-ranking on top-10.
4. **Hard negative mining**: Current negatives are random non-matching candidates. Mining hard negatives (high score, false) would improve precision.
5. **Test-time augmentation**: Could ensemble multiple K values (20, 30, 50) and take union.