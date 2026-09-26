# Business Entity Resolution

This repository implements the Business Entity Resolution pipeline for the ML Challenge 2026.

## Environment Setup
```bash
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Build record cache (one-time, ~7 min)
```bash
cd code/business_entity_resolution
python -m src.build_cache --data-dir ../../dataset --cache-dir ../../work/cache --workers 12
```

### 2. Train model and generate predictions (full run ~25 min)
```bash
cd code/business_entity_resolution
python -m src.pipeline
```

Outputs are written to `../../output/`:
- `matching_results.tsv` — final entity matches (leaderboard submission)
- `candidate_pairs.tsv` — blocking candidate set (audit)

### 3. Validate submission
```bash
cd code/business_entity_resolution
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

## Architecture

- **src/preprocess.py** — Text normalization, Devanagari consonant skeleton for cross-script matching, tokenization
- **src/store.py** — Memory-mapped binary cache for 24M records (fast repeatable access)
- **src/blocking.py** — IDF-weighted token inverted index per country partition; top-k candidates per S1 entity
- **src/pipeline.py** — End-to-end: blocking → feature extraction → LightGBM training → inference → output

## Key Design Decisions

1. **No dense embeddings** — 10M pool records too large for CPU sentence-transformer + FAISS
2. **Token inverted index with IDF** — Scales to 10M records, ~50M postings, sub-second query
3. **Consonant skeleton** — Bridges Devanagari/Latin name variants (4% of GT pairs are cross-script)
4. **Country-partitioned blocking** — Dynamic partitions from data vocabulary (handles unseen `France` in test)
5. **LightGBM with macro-F0.5 threshold** — Trained on 8.5k candidate pairs, 3-fold CV threshold tuning
6. **Features** — Rapidfuzz token_set/sort/partial/WRatio/QRatio on name/address, Jaccard, digit overlap, skeleton ratio

## Performance

- Training: ~8.5k candidate pairs, 559 positives, 3-fold CV F1 ~0.92
- Test candidates: ~341 S1 entities with matches, ~418 with candidates
- Output format: TSV with tab delimiter, comma-separated ID lists (no spaces, no quotes)

## Files

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── store.py
│   ├── blocking.py
│   ├── build_cache.py
│   └── pipeline.py
├── requirements.txt
└── README.md
```

output/
├── matching_results.tsv
└── candidate_pairs.tsv
```