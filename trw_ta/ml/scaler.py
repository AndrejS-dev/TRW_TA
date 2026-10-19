from sklearn.preprocessing import StandardScaler

def standardize(train_X, test_X):
    """
    Standardize Train/Test Features.
    
    Fit a StandardScaler on train_X and transform train/test.
    
    Parameters
    ----------
    train_X : Any
        Training feature matrix.
    test_X : Any
        Test feature matrix.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(train_X), scaler.transform(test_X)

# Backward-compatible alias
Standardize = standardize
