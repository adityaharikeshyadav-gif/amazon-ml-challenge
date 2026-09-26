import argparse
import logging
import time
import re
import unicodedata
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    suffixes = [r'\bltd\b', r'\bpvt ltd\b', r'\binc\b', r'\bcorp\b', r'\bllc\b', r'\bllp\b', r'\bsa\b', r'\bsas\b', r'\bsarl\b']
    for suffix in suffixes:
        text = re.sub(suffix, '', text)
    return ' '.join(text.split())

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing data...")
    df = df.copy()
    df['clean_name'] = df['business_name'].apply(normalize_text)
    df['clean_address'] = df['business_address'].apply(normalize_text)
    df['clean_country'] = df['country'].astype(str).str.strip().str.upper()
    df['concat_text'] = df['clean_name'] + " | " + df['clean_address']
    return df

def extract_numbers(text):
    return set(re.findall(r'\b\d+\b', str(text)))

def compute_number_overlap(addr1, addr2):
    nums1 = extract_numbers(addr1)
    nums2 = extract_numbers(addr2)
    if not nums1 and not nums2:
        return 0
    return len(nums1.intersection(nums2))

def get_lexical_candidates(s1_names, pool_names, top_k=5):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    pool_tfidf = vectorizer.fit_transform(pool_names)
    s1_tfidf = vectorizer.transform(s1_names)
    
    import scipy.sparse as sp
    
    batch_size = 1000
    indices = []
    
    for start in range(0, s1_tfidf.shape[0], batch_size):
        end = min(start + batch_size, s1_tfidf.shape[0])
        batch_sim = s1_tfidf[start:end].dot(pool_tfidf.T)
        
        if sp.issparse(batch_sim):
            batch_sim = batch_sim.toarray()
            
        for row in batch_sim:
            top_k_idx = np.argsort(row)[-top_k:][::-1]
            indices.append(top_k_idx)
            
    return np.array(indices)

def retrieve_candidates(s1_df, pool_df, model, batch_size=64):
    logger.info("Starting candidate retrieval...")
    countries = s1_df['clean_country'].unique()
    all_pairs = []
    
    for country in countries:
        logger.info(f"Processing country: {country}")
        s1_c = s1_df[s1_df['clean_country'] == country].reset_index(drop=True)
        pool_c = pool_df[pool_df['clean_country'] == country].reset_index(drop=True)
        
        if len(pool_c) == 0:
            continue
            
        s1_texts = s1_c['concat_text'].tolist()
        pool_texts = pool_c['concat_text'].tolist()
        
        logger.info(f"  Encoding {len(pool_texts)} pool texts...")
        pool_embs = model.encode(pool_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        logger.info(f"  Encoding {len(s1_texts)} S1 texts...")
        s1_embs = model.encode(s1_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        
        dim = pool_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(pool_embs)
        
        k_dense = min(15, len(pool_c))
        logger.info(f"  Searching dense top-{k_dense}...")
        D, I = index.search(s1_embs, k_dense)
        
        k_lex = min(5, len(pool_c))
        logger.info(f"  Searching lexical top-{k_lex}...")
        lex_indices = get_lexical_candidates(s1_c['clean_name'].tolist(), pool_c['clean_name'].tolist(), top_k=k_lex)
        
        s1_ids = s1_c['entity_id'].values
        pool_ids = pool_c['entity_id'].values
        
        pairs_set = set()
        
        for i, s1_id in enumerate(s1_ids):
            for j in range(k_dense):
                pool_idx = I[i, j]
                sim = D[i, j]
                pool_id = pool_ids[pool_idx]
                pair = (s1_id, pool_id)
                if pair not in pairs_set:
                    pairs_set.add(pair)
                    all_pairs.append({'s1_id': s1_id, 'pool_id': pool_id, 'dense_cosine_similarity': sim})
                    
            for pool_idx in lex_indices[i]:
                pool_id = pool_ids[pool_idx]
                pair = (s1_id, pool_id)
                if pair not in pairs_set:
                    pairs_set.add(pair)
                    all_pairs.append({'s1_id': s1_id, 'pool_id': pool_id, 'dense_cosine_similarity': 0.0})
                    
        del index
        del pool_embs
        del s1_embs
        
    return pd.DataFrame(all_pairs)

def build_features(pairs_df, s1_df, pool_df):
    logger.info("Building features for candidate pairs...")
    
    features = pairs_df.merge(s1_df[['entity_id', 'clean_name', 'clean_address']], left_on='s1_id', right_on='entity_id', how='left')
    features.rename(columns={'clean_name': 's1_name', 'clean_address': 's1_address'}, inplace=True)
    features.drop('entity_id', axis=1, inplace=True)
    
    features = features.merge(pool_df[['entity_id', 'clean_name', 'clean_address']], left_on='pool_id', right_on='entity_id', how='left')
    features.rename(columns={'clean_name': 'pool_name', 'clean_address': 'pool_address'}, inplace=True)
    features.drop('entity_id', axis=1, inplace=True)
    
    logger.info(f"Computing rapidfuzz features for {len(features)} pairs...")
    features['name_token_set_ratio'] = features.apply(lambda x: fuzz.token_set_ratio(x['s1_name'], x['pool_name']), axis=1)
    features['name_token_sort_ratio'] = features.apply(lambda x: fuzz.token_sort_ratio(x['s1_name'], x['pool_name']), axis=1)
    features['name_partial_ratio'] = features.apply(lambda x: fuzz.partial_ratio(x['s1_name'], x['pool_name']), axis=1)
    
    features['addr_token_set_ratio'] = features.apply(lambda x: fuzz.token_set_ratio(x['s1_address'], x['pool_address']), axis=1)
    features['addr_token_sort_ratio'] = features.apply(lambda x: fuzz.token_sort_ratio(x['s1_address'], x['pool_address']), axis=1)
    
    features['number_overlap_count'] = features.apply(lambda x: compute_number_overlap(x['s1_address'], x['pool_address']), axis=1)
    features['name_length_diff'] = features.apply(lambda x: abs(len(str(x['s1_name'])) - len(str(x['pool_name']))), axis=1)
    
    feature_cols = [
        'dense_cosine_similarity',
        'name_token_set_ratio',
        'name_token_sort_ratio',
        'name_partial_ratio',
        'addr_token_set_ratio',
        'addr_token_sort_ratio',
        'number_overlap_count',
        'name_length_diff'
    ]
    
    return features[['s1_id', 'pool_id'] + feature_cols]

def process_training(data_dir):
    logger.info("Loading training data...")
    s1_df = pd.read_csv(Path(data_dir) / 'train_source1.tsv', sep='\t')
    s2_df = pd.read_csv(Path(data_dir) / 'train_source2.tsv', sep='\t')
    s3_df = pd.read_csv(Path(data_dir) / 'train_source3.tsv', sep='\t')
    gt_df = pd.read_csv(Path(data_dir) / 'train_ground_truth.tsv', sep='\t')
    
    s1_df = preprocess_data(s1_df)
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    pool_df = preprocess_data(pool_df)
    
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    candidates_df = retrieve_candidates(s1_df, pool_df, model)
    
    features_df = build_features(candidates_df, s1_df, pool_df)
    
    gt_pairs = set()
    for _, row in gt_df.iterrows():
        s1 = row['source1_entity_id']
        if pd.isna(row['matched_entity_ids']):
            continue
        matches = str(row['matched_entity_ids']).split(',')
        for m in matches:
            gt_pairs.add((s1, m.strip()))
            
    features_df['label'] = features_df.apply(lambda x: 1 if (x['s1_id'], x['pool_id']) in gt_pairs else 0, axis=1)
    
    total_gt = len(gt_pairs)
    retrieved_gt = features_df['label'].sum()
    recall = retrieved_gt / total_gt if total_gt > 0 else 0
    logger.info(f"Retrieved Candidate Recall: {recall:.4f} ({retrieved_gt}/{total_gt})")
    
    feature_cols = [c for c in features_df.columns if c not in ['s1_id', 'pool_id', 'label']]
    X = features_df[feature_cols]
    y = features_df['label']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = []
    best_thresholds = []
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"Training Fold {fold+1}...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = lgb.LGBMClassifier(random_state=42, n_estimators=100)
        clf.fit(X_train, y_train)
        
        val_preds = clf.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        best_f1 = 0
        best_th = 0.5
        for th in np.arange(0.1, 0.9, 0.05):
            preds = (val_preds >= th).astype(int)
            f = f1_score(y_val, preds)
            if f > best_f1:
                best_f1 = f
                best_th = th
                
        best_thresholds.append(best_th)
        models.append(clf)
        logger.info(f"Fold {fold+1} - Best Threshold: {best_th:.2f}, F1-Score: {best_f1:.4f}")
        
    avg_threshold = np.mean(best_thresholds)
    overall_f1 = f1_score(y, (oof_preds >= avg_threshold).astype(int))
    logger.info(f"Overall OOF F1-Score: {overall_f1:.4f}, Optimal Threshold: {avg_threshold:.4f}")
    
    for i, clf in enumerate(models):
        clf.booster_.save_model(f'lgbm_fold_{i}.txt')
    with open('threshold.json', 'w') as f:
        json.dump({'threshold': avg_threshold}, f)
        
    logger.info("Training complete.")

def process_inference(data_dir):
    logger.info("Loading test data...")
    s1_df = pd.read_csv(Path(data_dir) / 'test_source1.tsv', sep='\t')
    s2_df = pd.read_csv(Path(data_dir) / 'test_source2.tsv', sep='\t')
    s3_df = pd.read_csv(Path(data_dir) / 'test_source3.tsv', sep='\t')
    
    s1_df = preprocess_data(s1_df)
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    pool_df = preprocess_data(pool_df)
    
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    candidates_df = retrieve_candidates(s1_df, pool_df, model)
    
    features_df = build_features(candidates_df, s1_df, pool_df)
    feature_cols = [c for c in features_df.columns if c not in ['s1_id', 'pool_id']]
    X = features_df[feature_cols]
    
    logger.info("Loading models and threshold...")
    with open('threshold.json', 'r') as f:
        threshold = json.load(f)['threshold']
        
    preds = np.zeros(len(X))
    for i in range(5):
        clf = lgb.Booster(model_file=f'lgbm_fold_{i}.txt')
        preds += clf.predict(X) / 5.0
        
    features_df['prob'] = preds
    matches_df = features_df[features_df['prob'] >= threshold].copy()
    
    result = matches_df.groupby('s1_id')['pool_id'].apply(lambda x: ','.join(sorted(x))).reset_index()
    result.rename(columns={'pool_id': 'matched_entity_ids'}, inplace=True)
    
    submission = s1_df[['entity_id']].copy()
    submission.rename(columns={'entity_id': 'source1_entity_id'}, inplace=True)
    
    submission = submission.merge(result, left_on='source1_entity_id', right_on='s1_id', how='left')
    submission.drop('s1_id', axis=1, inplace=True, errors='ignore')
    submission['matched_entity_ids'] = submission['matched_entity_ids'].fillna('')
    
    out_path = 'submission.tsv'
    submission.to_csv(out_path, sep='\t', index=False)
    logger.info(f"Submission saved to {out_path} with {len(submission)} rows.")
    assert len(submission) == len(s1_df), "Mismatch in submission row count!"

def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--predict', action='store_true', help='Run inference')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing the TSV files')
    
    args = parser.parse_args()
    
    start_time = time.time()
    if args.train:
        process_training(args.data_dir)
    if args.predict:
        process_inference(args.data_dir)
        
    if not args.train and not args.predict:
        logger.warning("Please specify --train or --predict")
        
    logger.info(f"Total Execution Time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
