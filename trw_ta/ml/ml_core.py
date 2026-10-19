import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss

def euclidean_distance(a, b):
    """
    Euclidean distance.
    
    Standard Euclidean (L2) distance.
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    """
    return distance.euclidean(a, b)

def manhattan_distance(a, b):
    """
    Manhattan distance.
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    """
    return distance.cityblock(a, b)

def minkowski_distance(a, b, p=3):
    """
    Minkowski distance.
    
    Generalized Minkowski distance (p-norm).
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    p : Any, default 3
        P.
    """
    return distance.minkowski(a, b, p)

def cosine_distance(a, b):
    """
    Cosine distance.
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    """
    return distance.cosine(a, b)

def kl_divergence(p, q):
    """
    Kl divergence.
    
    Kullback–Leibler divergence between two distributions.
    
    Parameters
    ----------
    p : Any
        P.
    q : Any
        Q.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

def js_divergence(p, q):
    """
    Js divergence.
    
    Jensen–Shannon divergence (symmetric KL).
    
    Parameters
    ----------
    p : Any
        P.
    q : Any
        Q.
    """
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

def mahalanobis_distance(a, b, cov):
    """
    Mahalanobis distance.
    
    Mahalanobis distance with covariance matrix.
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    cov : Any
        Cov.
    """
    return distance.mahalanobis(a, b, np.linalg.inv(cov))

def lorentzian_distance(a, b):
    """
    Lorentzian distance.
    
    Lorentzian (logarithmic) distance, robust to outliers.
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    """
    a, b = np.asarray(a), np.asarray(b)
    return np.sum(np.log1p(np.abs(a - b)))

def sigmoid(x):
    """
    Sigmoid.
    
    Sigmoid activation.
    
    Parameters
    ----------
    x : Any
        Numeric input.
    """
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """
    Tanh.
    
    Hyperbolic tangent activation.
    
    Parameters
    ----------
    x : Any
        Numeric input.
    """
    return np.tanh(x)

def relu(x):
    """
    Relu.
    
    Rectified Linear Unit activation.
    
    Parameters
    ----------
    x : Any
        Numeric input.
    """
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """
    Leaky relu.
    
    Leaky ReLU to avoid dead neurons.
    
    Parameters
    ----------
    x : Any
        Numeric input.
    alpha : Any, default 0.01
        Alpha.
    """
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    """
    Softmax.
    
    Softmax for multiclass probabilities.
    
    Parameters
    ----------
    x : Any
        Numeric input.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def gelu(x):
    """
    Gelu.
    
    Gaussian Error Linear Unit (used in transformers).
    
    Parameters
    ----------
    x : Any
        Numeric input.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def swish(x):
    """
    Swish.
    
    Swish activation (x * sigmoid(x)).
    
    Parameters
    ----------
    x : Any
        Numeric input.
    """
    return x * sigmoid(x)

def mse(y_true, y_pred):
    """
    Mse.
    
    Mean Squared Error.
    
    Parameters
    ----------
    y_true : Any
        Y true.
    y_pred : Any
        Y pred.
    """
    return mean_squared_error(y_true, y_pred)

def mae(y_true, y_pred):
    """
    Mae.
    
    Mean Absolute Error.
    
    Parameters
    ----------
    y_true : Any
        Y true.
    y_pred : Any
        Y pred.
    """
    return mean_absolute_error(y_true, y_pred)

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber loss.
    
    Parameters
    ----------
    y_true : Any
        Y true.
    y_pred : Any
        Y pred.
    delta : Any, default 1.0
        Delta.
    """
    err = np.abs(y_true - y_pred)
    return np.where(err <= delta, 0.5 * err**2, delta * (err - 0.5 * delta)).mean()

def binary_crossentropy(y_true, y_pred):
    """
    Binary crossentropy.
    
    Binary cross-entropy loss.
    
    Parameters
    ----------
    y_true : Any
        Y true.
    y_pred : Any
        Y pred.
    """
    return log_loss(y_true, y_pred, labels=[0, 1])

def categorical_crossentropy(y_true, y_pred):
    """
    Categorical crossentropy.
    
    Categorical cross-entropy loss.
    
    Parameters
    ----------
    y_true : Any
        Y true.
    y_pred : Any
        Y pred.
    """
    y_true = np.asarray(y_true)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def hinge_loss(y_true, y_pred):
    """
    Hinge loss.
    
    Parameters
    ----------
    y_true : Any
        Y true.
    y_pred : Any
        Y pred.
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

def minmax_scaling(x):
    """
    Minmax scaling.
    
    Scale features to [0, 1] range.
    
    Parameters
    ----------
    x : Any
        Numeric input.
    """
    return minmax_scale(x)

def l2_normalize(x):
    """
    L2 normalize.
    
    Normalize vector to unit L2 norm.
    
    Parameters
    ----------
    x : Any
        Numeric input.
    """
    return normalize([x])[0]

def log_scale(x):
    """
    Log scale.
    
    Apply log(1+x) scaling (for skewed data).
    
    Parameters
    ----------
    x : Any
        Numeric input.
    """
    return np.log1p(x)

def linear_kernel(a, b):
    """
    Linear kernel.
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    """
    return np.dot(a, b)

def polynomial_kernel(a, b, degree=3, c=1):
    """
    Polynomial kernel.
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    degree : Any, default 3
        Degree.
    c : Any, default 1
        C.
    """
    return (np.dot(a, b) + c) ** degree

def rbf_kernel(a, b, gamma=0.1):
    """
    Rbf kernel.
    
    Radial Basis Function (Gaussian) kernel.
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    gamma : Any, default 0.1
        Gamma.
    """
    return np.exp(-gamma * np.linalg.norm(a - b)**2)

def laplacian_kernel(a, b, gamma=0.1):
    """
    Laplacian kernel.
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    gamma : Any, default 0.1
        Gamma.
    """
    return np.exp(-gamma * np.linalg.norm(a - b, 1))

def sigmoid_kernel(a, b, alpha=0.01, c=0):
    """
    Sigmoid kernel.
    
    Parameters
    ----------
    a : Any
        Numeric input.
    b : Any
        Numeric input.
    alpha : Any, default 0.01
        Alpha.
    c : Any, default 0
        C.
    """
    return np.tanh(alpha * np.dot(a, b) + c)

def l1_regularization(weights, lam=0.01):
    """
    L1 regularization.
    
    L1 regularization penalty.
    
    Parameters
    ----------
    weights : Any
        Weights.
    lam : Any, default 0.01
        Lam.
    """
    return lam * np.sum(np.abs(weights))

def l2_regularization(weights, lam=0.01):
    """
    L2 regularization.
    
    L2 regularization penalty.
    
    Parameters
    ----------
    weights : Any
        Weights.
    lam : Any, default 0.01
        Lam.
    """
    return lam * np.sum(weights**2)

def soft_threshold(x, threshold):
    """
    Soft threshold.
    
    Soft thresholding operator.
    
    Parameters
    ----------
    x : Any
        Numeric input.
    threshold : Any
        Threshold.
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def dropout(x, rate=0.5):
    """
    Dropout.
    
    Randomly zero out inputs (Dropout).
    
    Parameters
    ----------
    x : Any
        Numeric input.
    rate : Any, default 0.5
        Rate.
    """
    mask = np.random.binomial(1, 1 - rate, size=x.shape)
    return x * mask / (1 - rate)
