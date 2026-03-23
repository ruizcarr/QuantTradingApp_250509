import pandas as pd
import numpy as np

def compute_minmax_mean_portfolio(tickers_returns,settings):
    cum_rets = (1 + tickers_returns).cumprod()
    #Compute Min, Max & Mean Bands
    max_band = cum_rets.rolling(settings['minmax_w']).max()
    min_band= cum_rets.rolling(settings['minmax_w']).min()
    mean_minmax = (max_band+min_band)/2

    #Compute Trend weights
    rets_over_mean=cum_rets>mean_minmax*1.03
    trend_weight=rets_over_mean.clip(lower=0)
    trend_weight=trend_weight.rolling(22).mean()
    trend_weight=trend_weight.shift(1).fillna(0)

    #Filters & Fine Tunning
    from ddn_ltd_portfolio import DDNLimitedPortfolio
    portfolio_manager = DDNLimitedPortfolio(settings)
    trend_weights = portfolio_manager.apply_constraints(trend_weight,settings)

    return trend_weights,cum_rets,max_band,min_band,mean_minmax
