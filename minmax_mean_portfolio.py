import pandas as pd
import numpy as np


def compute_minmax_mean(tickers_returns, settings):
    cum_rets = (1 + tickers_returns).cumprod()
    # Compute Min, Max & Mean Bands
    max_band = cum_rets.rolling(settings['minmax_w']).max()
    min_band = cum_rets.rolling(settings['minmax_w']).min()
    mean_minmax = (max_band + min_band) / 2


    return mean_minmax, cum_rets, max_band, min_band

def compute_trend_weights(tickers_returns, settings):

    mean_minmax, cum_rets, max_band, min_band = compute_minmax_mean(tickers_returns, settings)
    mean_minmax = mean_minmax * settings['minmax_delta']
    rets_over_mean=cum_rets>mean_minmax
    trend_weights = (rets_over_mean*1).clip(lower=0.0).shift(1).fillna(0)
    return trend_weights,mean_minmax, cum_rets, max_band, min_band



def compute_minmax_mean_portfolio(tickers_returns,settings):

    #Compute Trend weights
    trend_weights,mean_minmax, cum_rets, max_band, min_band =compute_trend_weights(tickers_returns, settings)

    trend_weights = trend_weights.rolling(22).mean()  # 22


    #Filters & Fine Tunning
    from ddn_ltd_portfolio import DDNLimitedPortfolio
    portfolio_manager = DDNLimitedPortfolio(settings)
    trend_weights = portfolio_manager.apply_constraints(trend_weights,settings)


    return trend_weights,cum_rets,max_band,min_band,mean_minmax


