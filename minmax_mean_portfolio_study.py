import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import settings,utils
t_settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

import quantstats_lumi as quantstats
import os


import Market_Data_Feed as mdf

#Get Data
data_ind = mdf.Data_Ind_Feed(t_settings).data_ind
data, _ = data_ind
data_dict = data.data_dict

tickers_returns=data.tickers_returns
cum_rets=(1+tickers_returns).cumprod()

def compute_minmax_mean_portfolio(tickers_returns,settings):
    cum_rets = (1 + tickers_returns).cumprod()
    #Compute Min, Max & Mean Bands
    max_band = cum_rets.rolling(settings['minmax_w']).max()
    min_band= cum_rets.rolling(settings['minmax_w']).min()
    mean_minmax = (max_band+min_band)/2

    #Compute Trend weights
    rets_over_mean=cum_rets>mean_minmax
    trend_weight=rets_over_mean.clip(lower=0)
    trend_weight=trend_weight.rolling(5).mean()
    trend_weight=trend_weight.shift(1).fillna(0)

    #Filters & Fine Tunning
    from ddn_ltd_portfolio import DDNLimitedPortfolio
    portfolio_manager = DDNLimitedPortfolio(t_settings)
    trend_weights = portfolio_manager.apply_constraints(trend_weight,t_settings)

    return trend_weights,cum_rets,max_band,min_band,mean_minmax

trend_weight,cum_rets,max_band,min_band,mean_minmax=compute_minmax_mean_portfolio(tickers_returns,t_settings)

#BACKTEST
trend_ret=tickers_returns*trend_weight
trend_ret['sum']=(tickers_returns*trend_weight).sum(axis=1)
trend_cumret=(1+trend_ret).cumprod()

#Plots
for ticker in tickers_returns.columns:

    plot_df=pd.DataFrame()
    plot_df['cum_ret']=cum_rets[ticker]
    plot_df['max_band'] = max_band[ticker]
    plot_df['min_band'] = min_band[ticker]
    plot_df['mean_minmax'] = mean_minmax[ticker]

    plot_df['trend_weight'] = trend_weight[ticker]
    plot_df['trend_cumret'] = trend_cumret[ticker]
    plot_df.plot(title=ticker)

plot_df2=pd.DataFrame()
plot_df2['benchmark']=cum_rets['ES=F']
plot_df2['strategy']=trend_cumret['sum']
plot_df2.plot(title='Trend Startegy Results')

q_title = 'Cash Backtest Trend Strategy'
path = "results\\"
q_filename = os.path.abspath(path + q_title + '.html')
q_returns = trend_cumret['sum']
q_returns.index = pd.to_datetime(q_returns.index, utc=True).tz_convert(None)

q_benchmark_ticker = 'ES=F'
q_benchmark = cum_rets['ES=F']
q_benchmark.index = pd.to_datetime(q_benchmark.index, utc=True).tz_convert(None)

quantstats.reports.html(
    q_returns, title=q_title,
    benchmark=q_benchmark, benchmark_title=q_benchmark_ticker,
    output=q_filename
)
import webbrowser

webbrowser.open(q_filename)


plt.show()