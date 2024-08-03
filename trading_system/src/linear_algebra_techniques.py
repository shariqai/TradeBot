# src/linear_algebra_techniques.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def perform_pca(data, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on data.
    """
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.transform(data)

def linear_regression(X, y):
    """
    Perform linear regression.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def ridge_regression(X, y, alpha=1.0):
    """
    Perform ridge regression.
    """
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

def lasso_regression(X, y, alpha=1.0):
    """
    Perform lasso regression.
    """
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model
