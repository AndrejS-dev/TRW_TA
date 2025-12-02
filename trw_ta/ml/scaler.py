from sklearn.preprocessing import StandardScaler

def Standardize(train_X, test_X):
    scaler = StandardScaler()
    return scaler.fit_transform(train_X), scaler.transform(test_X)
