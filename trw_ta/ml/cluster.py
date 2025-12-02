from sklearn.cluster import KMeans

def fit_predict_KMeans(train_df, test_df, train_scaled, test_scaled, n_clusters: int = 2, random_state: int = 42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(train_scaled)

    train_df['cluster'] = kmeans.predict(train_scaled)
    test_df['cluster'] = kmeans.predict(test_scaled)

    return train_df, test_df