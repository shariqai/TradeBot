# src/mathematical_techniques.py
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def monte_carlo_simulation(S, K, T, r, sigma, num_simulations=10000):
    """
    Monte Carlo simulation for option pricing.
    """
    dt = T / num_simulations
    S_paths = np.zeros((num_simulations + 1))
    S_paths[0] = S
    for i in range(1, num_simulations + 1):
        z = np.random.standard_normal()
        S_paths[i] = S_paths[i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return np.mean(np.maximum(S_paths[-1] - K, 0))
