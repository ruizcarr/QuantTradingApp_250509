import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import settings,utils
t_settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

import quantstats_lumi as quantstats
import os


import Market_Data_Feed as mdf

from minmax_mean_portfolio import compute_minmax_mean_portfolio

#Get Data
data_ind = mdf.Data_Ind_Feed(t_settings).data_ind
data, _ = data_ind
data_dict = data.data_dict

tickers_returns=data.tickers_returns
cum_rets=(1+tickers_returns).cumprod()

trend_weight,cum_rets,max_band,min_band,mean_minmax=compute_minmax_mean_portfolio(tickers_returns,t_settings)

#Apply Strong Trend Boost & Penalty when repeated new max
trend_change_counter=np.sign(max_band.diff()).rolling(10).sum()
strong_trend_mask=trend_change_counter.shift(1)>=2
strong_trend_boost=np.where(strong_trend_mask,1.5,0.5) #Best sharpe 1.25-1.4,0.5
trend_weight=trend_weight*strong_trend_boost

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
    plot_df['trend_change_counter'] = trend_change_counter[ticker]
    #plot_df['lateral_ind'] = lateral_ind[ticker]
    plot_df['strong_trend_mask'] = strong_trend_mask[ticker]*1

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