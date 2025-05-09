import numpy as np
import pandas as pd

from scipy.optimize import minimize


class MarkowitzWeights:
    """
       This class implements portfolio optimization using the Markowitz model.

       It calculates optimal portfolio weights based on the following factors:

       - Expected returns of individual assets
       - Covariance matrix of asset returns
       - User-specified settings like volatility target and weight bounds

       Attributes:
           tickers_returns: A DataFrame containing historical returns for the assets.
           settings: A dictionary containing various settings for the strategy.
           size: The number of assets in the portfolio.
           CAGR: A Series containing annualized expected returns for the assets.
           covariance_matrix: A NumPy array representing the covariance matrix of asset returns.
           portfolio: A NumPy array containing the calculated optimal portfolio weights.
           slice_weights: A list of NumPy arrays, where each array contains the optimization date, end date, objective function value, and weights for a specific time window.
       """

    def __init__(self, tickers_returns, volatility_target, settings, x0):
        self.tickers_returns = tickers_returns
        self.settings = settings
        self.size = len(tickers_returns.columns)
        self.tickers=tickers_returns.columns
        self.CAGR = np.mean(tickers_returns) * 252
        contango_list=[settings['contango'][ticker] for ticker in self.tickers]
        self.CAGR = self.CAGR -np.array(contango_list)/100
        self.covariance_matrix = np.cov(tickers_returns.T) * 252


        self.results=self.compute_portfolio(x0, settings['tickers_bounds'], volatility_target=volatility_target)



    def compute_portfolio(self, x0,tickers_bounds, volatility_target=None):
        """
                Calculates the optimal portfolio weights using the Markowitz model with constraints.

                The objective function minimizes portfolio volatility while considering expected returns and a potential volatility target penalty.

                Args:
                    x0: An initial guess for the portfolio weights.
                    tickers_bounds: A dictionary containing upper and lower bounds for each asset weight.
                    volatility_target: An optional volatility target for the portfolio.

                Returns:
                    None (updates the `weights` and `slice_weights` attributes).
                """
        # Portfolio calculation Minimizing opt_fun
        # Constraints & Bounds
        weight_sum_lim = self.settings['w_sum_max']
        cons = [
            {"type": "ineq", "fun": lambda x: weight_sum_lim - sum(abs(x))},  # Weights abs sum < limit
            {"type": "ineq", "fun": lambda x: sum(x)},  # Weights sum > 0
        ]

        #Bounds
        bnds = [tickers_bounds[tick] for tick in self.tickers]

        #opt_fun arguments in adition to x
        args=(self.CAGR, self.covariance_matrix, volatility_target)

        # compute optimisation
        res=minimize(mkwtz_opt_fun, x0, args=args,constraints=cons, bounds=bnds, tol=0.0001, options={'maxiter': 100})

        return res

def mkwtz_opt_fun(x, CAGR, covariance_matrix,volatility_target):
    """
        Objective function for the Markowitz optimization problem.

        Args:
            x: A NumPy array representing the portfolio weights.
            CAGR: A Series containing annualized expected returns for the assets.
            covariance_matrix: A NumPy array representing the covariance matrix of asset returns.
            volatility_target: An optional volatility target for the portfolio.

        Returns:
            A float representing the combined objective function value.
        """

    #x contains weights vector
    variance=max(np.dot(x.T, np.dot(covariance_matrix, x)).item(),0)
    volatility_x=(variance**0.5)+0.0001
    returns_x = np.dot(CAGR.T, x)
    penalties_x = np.where(volatility_x > volatility_target, volatility_x /volatility_target-1, 0)

    opt_fun = volatility_x - returns_x + penalties_x

    return opt_fun