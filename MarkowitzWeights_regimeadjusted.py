import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture


class MarkowitzWeights:
    def __init__(self, tickers_returns, volatility_target, settings, x0):
        self.tickers_returns = tickers_returns
        self.settings = settings
        self.size = len(tickers_returns.columns)
        self.tickers = tickers_returns.columns

        # 1. Regime Detection (Machine Learning Change of View)
        # We look at the volatility of the first asset (usually NQ) to determine the 'state'
        self.regime = self._detect_regime(tickers_returns)

        # 2. Expected Returns calculation
        raw_cagr = tickers_returns.mean() * 252
        contango_list = [settings['contango'].get(ticker, 0) for ticker in self.tickers]
        self.CAGR = raw_cagr - np.array(contango_list) / 100

        # 3. Regime-Adjusted Covariance Matrix
        # If Regime 1 (High Stress), we increase 'shrinkage' to be more conservative
        sample_cov = np.cov(tickers_returns.T) * 252
        shrinkage = 0.15 if self.regime == 1 else 0.02
        prior = np.eye(self.size) * np.mean(np.diag(sample_cov))
        self.covariance_matrix = (1 - shrinkage) * sample_cov + shrinkage * prior

        self.results = self.compute_portfolio(x0, settings['tickers_bounds'], volatility_target)

    def _detect_regime(self, returns):
        """Uses GMM (ML) to identify if we are in a High or Low volatility state."""
        try:
            # Analyze the volatility of the portfolio's main driver (NQ)
            data = returns.iloc[:, 0].values.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=42).fit(data)
            # Returns 0 for 'Quiet' or 1 for 'Turbulent'
            return gmm.predict(data[-1].reshape(-1, 1))[0]
        except:
            return 0

    def compute_portfolio(self, x0, tickers_bounds, volatility_target=None):
        # Constraints: Formalized for SLSQP speed
        cons = [
            {"type": "ineq", "fun": lambda x: self.settings.get('w_sum_max', 1.0) - np.sum(np.abs(x))},
            {"type": "ineq", "fun": lambda x: np.sum(x)},
        ]

        if volatility_target:
            cons.append({
                "type": "ineq",
                "fun": lambda x: volatility_target - np.sqrt(max(np.dot(x.T, np.dot(self.covariance_matrix, x)), 0))
            })

        # Bounds: Filter assets with low/negative expected returns
        bnds = [
            (tickers_bounds[tick][0], 0.0) if self.CAGR[i] <= 0.001
            else tickers_bounds[tick]
            for i, tick in enumerate(self.tickers)
        ]

        # Optimization
        res = minimize(
            mkwtz_opt_fun,
            x0,
            args=(self.CAGR, self.covariance_matrix),
            constraints=cons,
            bounds=bnds,
            method='SLSQP',
            tol=1e-5
        )
        return res


def mkwtz_opt_fun(x, CAGR, covariance_matrix):
    """Refined Objective: Risk-Adjusted Return."""
    variance = np.dot(x.T, np.dot(covariance_matrix, x))
    volatility_x = np.sqrt(max(variance, 0))
    returns_x = np.dot(CAGR, x)

    # Minimizing this leads to the highest return per unit of risk
    return volatility_x - returns_x