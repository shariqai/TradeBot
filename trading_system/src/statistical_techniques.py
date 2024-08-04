import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

def arima_forecasting(data, order=(5, 1, 0)):
    """
    ARIMA model for time series forecasting.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Data must be a pandas Series or DataFrame.")
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit.forecast(steps=1)

def garch_volatility(data, p=1, q=1):
    """
    GARCH model for volatility modeling.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Data must be a pandas Series or DataFrame.")
    model = arch_model(data, vol='Garch', p=p, q=q)
    model_fit = model.fit(disp='off')
    return model_fit.conditional_volatility[-1]
