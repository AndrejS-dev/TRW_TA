import random

def select_features(feature_list: list, min_features: int, max_features: int):
    temp_features = feature_list.copy()
    n_features = random.randint(min_features, max_features)
    out = []
    for _ in range(n_features):
        index = random.randint(0, len(temp_features) - 1)
        out.append(temp_features.pop(index))
    return out

def df_columns_tolist(df) -> list:
    all_cols = list(df.columns)
    to_remove = ['open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in all_cols if col not in to_remove]
    return feature_cols