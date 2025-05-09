from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class Regresion_of_Features:
    """
    This class implements Linear Regresion Model to the assets features
    like RSI-is_high, price_over_the_mean, is_December, volatility,...
    I want to maximize function=CAGR-volatility

    Method choice:
    -scipy minimize

    Input data:
    -features_dict: dict with features df(datetime.index, columns=tickers)

    Output data:
    -Coeficients of regresion (so factor to multiply each feature ): array(n_tickers, n_features)

    """

    def __init__(self,features_dict,tickers_returns,volat_target):

        x_len=len(features_dict)+1
        x0 = np.ones(x_len)/x_len
        self.results = self.compute_minimize(features_dict,tickers_returns,volat_target,x0)

    def compute_minimize(self,features_dict,tickers_returns,volat_target,x0):

        #Bounds
        bnds=[(-3,3)]*(len(features_dict)+1)

        # opt_fun arguments in adition to x
        args = (tickers_returns,features_dict,volat_target)

        # compute optimisation
        return minimize(objective_function, x0, args=args,
                         bounds=bnds, tol=0.0001, options={'maxiter': 100})




def objective_function(x,tickers_returns,features_dict,volat_target):
    """
    Objective function for minimization with x as an array.

    Args:
        x: Input array.
        args: more args

    Returns:
        The value of the objective function.
    """

    dict_df_index=list(features_dict.values())[-1].index
    tickers_returns=tickers_returns.reindex(dict_df_index)
    returns_x = pd.Series((x[0]*tickers_returns).sum(axis=1),index=dict_df_index)
    i=1
    for df in features_dict.values():
        returns_x=returns_x+(tickers_returns*df*x[i]).sum(axis=1)
        i=i+1

    cagr_x= np.mean(returns_x)*252
    volat_x = np.std(returns_x)*16
    #penalties_x = np.where(volat_x > volat_target, volat_x / volat_target - 1, 0)
    #objective_fun=volat_x-cagr_x #+ penalties_x
    objective_fun = volat_x/cagr_x
    return objective_fun

