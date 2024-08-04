import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def perform_pca(data, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on data.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data

def linear_regression(X, y):
    """
    Perform linear regression.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    model = LinearRegression()
    model.fit(X, y)
    return model

def ridge_regression(X, y, alpha=1.0):
    """
    Perform ridge regression.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

def lasso_regression(X, y, alpha=1.0):
    """
    Perform lasso regression.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model
