# ddn_ltd_portfolio.py
import pandas as pd
import numpy as np


class DDNLimitedPortfolio:
    """
    Portfolio constructor focused on Drawdown (DDN) mitigation.
    Uses a power-law penalty function to deleverage assets as
    predicted risk exceeds user-defined thresholds.
    """

    def __init__(self, settings):
        self.settings = settings
        # Intermediate attributes for analysis/plotting
        self.drawdown_df = None
        self.risk_metric = None
        self.risk_ratio = None
        self.penalty = None
        self.utility = None
        self.final_weights = None

    def _get_drawdown(self, returns, window):
        """Calculates historical drawdown for risk prediction."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window=window, min_periods=1).max()
        return (cumulative / rolling_max) - 1

    def get_ddn_penalty(self, tickers_returns):
        s = self.settings

        # 1. Risk Prediction (3 x  DDN Std )
        self.drawdown_df = self._get_drawdown(tickers_returns, s['ddn_w'])
        self.risk_metric = 3 * self.drawdown_df.rolling(s['ddn_std_w']).std()

        # 2. Risk Ratio & Penalty Function
        self.risk_ratio = self.risk_metric / s['lower_ddn_limit']
        # The x^4 penalty creates an aggressive 'exit' signal
        self.penalty = (1 / (0.5 + self.risk_ratio) ** 4).clip(upper=1).fillna(1)

    def compute_weights(self, tickers_returns):
        """
        Calculates the portfolio weights based on CAGR momentum
        and the Drawdown/Volatility risk blend.
        """
        s = self.settings

        self.get_ddn_penalty(tickers_returns)

        # 3. CAGR Utility (Arithmetic mean based on window)
        cagr = tickers_returns.rolling(s['d_cagr_w']).mean() * 252
        self.utility = cagr.clip(lower=0, upper=1)

        # 4. Apply Penalty to Utility
        # Only penalize assets where the risk ratio > 1.0
        utility_final = self.utility.where(~(self.risk_ratio > 1), self.utility * self.penalty)

        # 5. Apply Constraints & Scaling
        weights = utility_final.copy() #Create weights df

        # Ticker-specific caps (e.g., Crypto/Risky assets)
        for ticker in s['d_risky_tickers']:
            if ticker in weights.columns:
                weights[ticker] = weights[ticker] / 2
                weights[ticker] = weights[ticker].clip(upper=s['d_max_risky_tickers_weight'])

        #Cash Bonus
        if 'cash' in weights.columns:
            weights['cash'] = weights['cash']*3

        # Global asset caps
        weights = weights.clip(upper=s['d_max_asset_weight'])

        #Excluded Tickers
        available_excl = [t for t in s['d_excluded_tickers'] if t in weights.columns]
        weights[available_excl] = 0

        # Apply fix_mult
        self.final_weights = (weights * s['d_fix_mult'])

        # 6. Total Leverage Guard (The "Safety Valve")
        if 'd_max_total_leverage' in s:
            # Calculate the sum of weights for each day
            current_total_leverage = self.final_weights.sum(axis=1)

            # Determine the scaling factor:
            # If current leverage is 2.0 and max is 1.0, factor is 2.0.
            # We clip at lower=1.0 so we never "scale up" a small portfolio.
            scaling_factor = (current_total_leverage / s['d_max_total_leverage']).clip(lower=1.0)

            # Divide all weights by that factor to bring the total down to the cap
            self.final_weights = self.final_weights.div(scaling_factor, axis=0)

        # 7. Shift to avoid lookahead bias
        self.final_weights = self.final_weights.shift(1)

        return self.final_weights

    @property
    def total_leverage(self):
        """Returns the daily total portfolio exposure."""
        if self.final_weights is not None:
            return self.final_weights.sum(axis=1)
        return None