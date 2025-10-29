import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss

def euclidean_distance(a, b):
    """Standard Euclidean (L2) distance."""
    return distance.euclidean(a, b)

def manhattan_distance(a, b):
    """Manhattan (L1) distance."""
    return distance.cityblock(a, b)

def minkowski_distance(a, b, p=3):
    """Generalized Minkowski distance (p-norm)."""
    return distance.minkowski(a, b, p)

def cosine_distance(a, b):
    """Cosine distance (1 - cosine similarity)."""
    return distance.cosine(a, b)

def kl_divergence(p, q):
    """Kullback–Leibler divergence between two distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

def js_divergence(p, q):
    """Jensen–Shannon divergence (symmetric KL)."""
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

def mahalanobis_distance(a, b, cov):
    """Mahalanobis distance with covariance matrix."""
    return distance.mahalanobis(a, b, np.linalg.inv(cov))

def lorentzian_distance(a, b):
    """Lorentzian (logarithmic) distance, robust to outliers."""
    a, b = np.asarray(a), np.asarray(b)
    return np.sum(np.log1p(np.abs(a - b)))

def sigmoid(x):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Hyperbolic tangent activation."""
    return np.tanh(x)

def relu(x):
    """Rectified Linear Unit activation."""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU to avoid dead neurons."""
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    """Softmax for multiclass probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def gelu(x):
    """Gaussian Error Linear Unit (used in transformers)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def swish(x):
    """Swish activation (x * sigmoid(x))."""
    return x * sigmoid(x)

def mse(y_true, y_pred):
    """Mean Squared Error."""
    return mean_squared_error(y_true, y_pred)

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss (smooth L1)."""
    err = np.abs(y_true - y_pred)
    return np.where(err <= delta, 0.5 * err**2, delta * (err - 0.5 * delta)).mean()

def binary_crossentropy(y_true, y_pred):
    """Binary cross-entropy loss."""
    return log_loss(y_true, y_pred, labels=[0, 1])

def categorical_crossentropy(y_true, y_pred):
    """Categorical cross-entropy loss."""
    y_true = np.asarray(y_true)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def hinge_loss(y_true, y_pred):
    """Hinge loss (used in SVMs)."""
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

def minmax_scaling(x):
    """Scale features to [0, 1] range."""
    return minmax_scale(x)

def l2_normalize(x):
    """Normalize vector to unit L2 norm."""
    return normalize([x])[0]

def log_scale(x):
    """Apply log(1+x) scaling (for skewed data)."""
    return np.log1p(x)

def linear_kernel(a, b):
    """Linear kernel (dot product)."""
    return np.dot(a, b)

def polynomial_kernel(a, b, degree=3, c=1):
    """Polynomial kernel."""
    return (np.dot(a, b) + c) ** degree

def rbf_kernel(a, b, gamma=0.1):
    """Radial Basis Function (Gaussian) kernel."""
    return np.exp(-gamma * np.linalg.norm(a - b)**2)

def laplacian_kernel(a, b, gamma=0.1):
    """Laplacian kernel (L1-based)."""
    return np.exp(-gamma * np.linalg.norm(a - b, 1))

def sigmoid_kernel(a, b, alpha=0.01, c=0):
    """Sigmoid kernel (tanh of dot product)."""
    return np.tanh(alpha * np.dot(a, b) + c)

def l1_regularization(weights, lam=0.01):
    """L1 regularization penalty."""
    return lam * np.sum(np.abs(weights))

def l2_regularization(weights, lam=0.01):
    """L2 regularization penalty."""
    return lam * np.sum(weights**2)

def soft_threshold(x, threshold):
    """Soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def dropout(x, rate=0.5):
    """Randomly zero out inputs (Dropout)."""
    mask = np.random.binomial(1, 1 - rate, size=x.shape)
    return x * mask / (1 - rate)
