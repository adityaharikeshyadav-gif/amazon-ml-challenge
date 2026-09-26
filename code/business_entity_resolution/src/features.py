import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import re

def compute_number_overlap_ratio(addr1, addr2):
    nums1 = set(re.findall(r'\b\d+\b', str(addr1)))
    nums2 = set(re.findall(r'\b\d+\b', str(addr2)))
    if not nums1 and not nums2:
        return 1.0 # Both empty
    if not nums1 or not nums2:
        return 0.0
    return len(nums1.intersection(nums2)) / max(len(nums1), len(nums2))

def build_features(pairs_df, s1_df, pool_df):
    features = pairs_df.merge(s1_df[['entity_id', 'clean_name', 'clean_address']], left_on='s1_id', right_on='entity_id', how='left')
    features.rename(columns={'clean_name': 's1_name', 'clean_address': 's1_address'}, inplace=True)
    features.drop('entity_id', axis=1, inplace=True)
    
    features = features.merge(pool_df[['entity_id', 'clean_name', 'clean_address']], left_on='pool_id', right_on='entity_id', how='left')
    features.rename(columns={'clean_name': 'pool_name', 'clean_address': 'pool_address'}, inplace=True)
    features.drop('entity_id', axis=1, inplace=True)
    
    features['s1_name'] = features['s1_name'].fillna('')
    features['pool_name'] = features['pool_name'].fillna('')
    features['s1_address'] = features['s1_address'].fillna('')
    features['pool_address'] = features['pool_address'].fillna('')
    
    features['name_token_sort_ratio'] = features.apply(lambda x: fuzz.token_sort_ratio(x['s1_name'], x['pool_name']), axis=1)
    features['name_token_set_ratio'] = features.apply(lambda x: fuzz.token_set_ratio(x['s1_name'], x['pool_name']), axis=1)
    features['name_jaro_winkler'] = features.apply(lambda x: JaroWinkler.normalized_similarity(x['s1_name'], x['pool_name']), axis=1)
    
    features['addr_token_sort_ratio'] = features.apply(lambda x: fuzz.token_sort_ratio(x['s1_address'], x['pool_address']), axis=1)
    features['addr_token_set_ratio'] = features.apply(lambda x: fuzz.token_set_ratio(x['s1_address'], x['pool_address']), axis=1)
    
    features['number_overlap_ratio'] = features.apply(lambda x: compute_number_overlap_ratio(x['s1_address'], x['pool_address']), axis=1)
    
    features['name_length_diff_ratio'] = features.apply(lambda x: abs(len(x['s1_name']) - len(x['pool_name'])) / max(1, max(len(x['s1_name']), len(x['pool_name']))), axis=1)
    
    feature_cols = [
        'dense_cosine_similarity',
        'name_token_sort_ratio',
        'name_token_set_ratio',
        'name_jaro_winkler',
        'addr_token_sort_ratio',
        'addr_token_set_ratio',
        'number_overlap_ratio',
        'name_length_diff_ratio'
    ]
    
    return features[['s1_id', 'pool_id'] + feature_cols]
