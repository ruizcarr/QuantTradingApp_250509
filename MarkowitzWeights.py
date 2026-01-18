import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture


class MarkowitzWeights:
    # GLOBAL STATE: Persists throughout the backtest
    _gmm_model = None
    _counter = 0  # Replaces randomness with a deterministic counter
    _last_weights = None

    @classmethod
    def reset_state(cls):
        """Resets the global state for a fresh backtest run."""
        cls._gmm_model = None
        cls._counter = 0
        cls._last_weights = None

    def __init__(self, tickers_returns, volatility_target, settings, x0):
        self.tickers_returns = tickers_returns
        self.settings = settings
        self.tickers = tickers_returns.columns
        self.size = len(self.tickers)

        # 1. Deterministic Regime Detection
        # Retrain every 12 iterations (approx. once per quarter if weekly)
        self.regime = self._get_regime_deterministic(tickers_returns, retrain_step=12)

        # 2. Expected Returns calculation
        raw_cagr = tickers_returns.mean() * 252
        contango = np.array([settings['contango'].get(t, 0) for t in self.tickers]) / 100
        self.CAGR = raw_cagr - contango

        # 3. Covariance with Regime-Based Shrinkage
        sample_cov = np.cov(tickers_returns.T) * 252
        # Use 20% shrinkage in turbulence to prevent "over-optimization"
        shrinkage = 0.20 if self.regime == 1 else 0.05
        prior = np.eye(self.size) * np.mean(np.diag(sample_cov))
        self.covariance_matrix = (1 - shrinkage) * sample_cov + shrinkage * prior

        # 4. Optimization
        # We ALWAYS use the same starting point logic for 100% consistency
        # Warm start is used for speed, but 'x0' is used if it's the first run
        start_x = MarkowitzWeights._last_weights if MarkowitzWeights._last_weights is not None else x0

        self.results = self.compute_portfolio(start_x, settings['tickers_bounds'], volatility_target)

        # Save state for the next weekly window
        MarkowitzWeights._last_weights = self.results.x
        MarkowitzWeights._counter += 1

    def _get_regime_deterministic(self, returns, retrain_step=12):
        """Identifies regime using a GMM that updates on a fixed schedule."""
        try:
            # Feature: 20-day rolling annualized volatility
            recent_vol = returns.iloc[-20:, 0].std() * np.sqrt(252)

            # DETERMINISTIC UPDATE: No randomness.
            # If counter is 0 or hits the step (e.g., every 12th week), re-fit.
            if MarkowitzWeights._gmm_model is None or (MarkowitzWeights._counter % retrain_step == 0):
                # Annualized rolling vol history
                history_vol = returns.iloc[:, 0].rolling(20).std().dropna().values.reshape(-1, 1) * np.sqrt(252)
                # Fixed random_state ensures the GMM always clusters the same way
                MarkowitzWeights._gmm_model = GaussianMixture(n_components=2, random_state=42).fit(history_vol)

            # Predict
            state = MarkowitzWeights._gmm_model.predict(np.array([[recent_vol]]))[0]
            # Ensure index 1 is always the state with the higher average volatility
            high_vol_idx = np.argmax(MarkowitzWeights._gmm_model.means_.flatten())
            return 1 if state == high_vol_idx else 0
        except:
            return 0

    def compute_portfolio(self, x0, tickers_bounds, volatility_target=None):
        w_sum_max = self.settings.get('w_sum_max', 1.0)

        # Fixed Constraints for SLSQP
        cons = [
            {"type": "ineq", "fun": lambda x: w_sum_max - np.sum(np.abs(x))},
            {"type": "ineq", "fun": lambda x: np.sum(x)},
        ]

        if volatility_target:
            cons.append({
                "type": "ineq",
                "fun": lambda x: volatility_target - np.sqrt(max(np.dot(x.T, np.dot(self.covariance_matrix, x)), 0))
            })

        bnds = [tickers_bounds[tick] for tick in self.tickers]

        # Use 1e-8 for high precision consistency
        res = minimize(
            mkwtz_opt_fun,
            x0,
            args=(self.CAGR, self.covariance_matrix),
            method='SLSQP',
            constraints=cons,
            bounds=bnds,
            tol=1e-8,
            options={'ftol': 1e-8, 'maxiter': 150}
        )
        return res


def mkwtz_opt_fun(x, CAGR, cov):
    # Quadratic Utility: Maximize (Return - 0.5 * Risk_Aversion * Variance)
    # This is the most stable formulation for 5 assets
    port_var = np.dot(x.T, np.dot(cov, x))
    port_ret = np.dot(CAGR, x)
    utility = port_ret - 0.5  * port_var
    return -utility