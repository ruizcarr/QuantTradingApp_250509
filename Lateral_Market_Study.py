import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import settings,utils
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

import quantstats_lumi as quantstats
import os

#Update Settings
#settings['start']='2015-01-01'

import Market_Data_Feed as mdf

#Get Data
data_ind = mdf.Data_Ind_Feed(settings).data_ind
data, _ = data_ind
data_dict = data.data_dict

tickers_returns=data.tickers_returns
cum_rets=(1+tickers_returns).cumprod()
#print('tickers_returns.describe()',tickers_returns.describe())

def get_OHLC(data_dict):
    # Get Open, High, Low, Closes from data_dict - optimize by using list comprehension
    desired_order = list(data_dict.keys())

    # Optimize by creating DataFrames directly without intermediate dictionary
    opens = pd.concat([data_dict[key]['Open'] for key in desired_order], axis=1, keys=desired_order)
    highs = pd.concat([data_dict[key]['High'] for key in desired_order], axis=1, keys=desired_order)
    lows = pd.concat([data_dict[key]['Low'] for key in desired_order], axis=1, keys=desired_order)
    closes = pd.concat([data_dict[key]['Close'] for key in desired_order], axis=1, keys=desired_order)

    return opens, highs, lows, closes

opens, highs, lows, closes = get_OHLC(data_dict)


def get_minmax_pct(highs, lows, minmax_w=22):
    highs_rolling_max=highs.rolling(minmax_w).max()
    lows_rolling_min=lows.rolling(minmax_w).min()
    minmax_pct=highs_rolling_max/lows_rolling_min-1
    return highs_rolling_max,lows_rolling_min,minmax_pct

minmax_w=22*9
highs_rolling_max,lows_rolling_min,minmax_pct=get_minmax_pct(cum_rets, cum_rets, minmax_w=minmax_w)
#norm_minmax_pct=(minmax_pct.max()-minmax_pct)/(minmax_pct.max()-minmax_pct.min())
#wide_minmax_pct=get_minmax_pct(highs, lows, minmax_w=minmax_w)
#lateral_indicator=narrow_minmax_pct/wide_minmax_pct

mean_minmax=(highs_rolling_max+lows_rolling_min)/2
mean_minmax_pct=mean_minmax.pct_change()
mean_minmax_cum=(1+mean_minmax_pct).cumprod()

mean_minmax_cagr=mean_minmax_pct.rolling(5).mean()*252
cagr=tickers_returns.rolling(5).mean()*252

trend_indicator=np.sign(mean_minmax_cagr) #1,0,-1 values for bullish, lateral, and bearish markets.

last_trend = trend_indicator.replace(0, np.nan).ffill()  # last known non-zero trend
previous_trend_indicator = last_trend.where(trend_indicator == 0, other=0) #0 (lateral) → previous_trend_indicator = 1 or -1 (last trend before this lateral period)

trend_weight=trend_indicator*1.0+previous_trend_indicator*0.5
trend_weight=trend_weight.clip(lower=0)

rets_over_mean=cum_rets>mean_minmax
#trend_weight=trend_weight.where(rets_over_mean, other=0)
trend_weight=rets_over_mean.clip(lower=0)

trend_weight=trend_weight.rolling(5).mean()
trend_weight=trend_weight*1.0
trend_weight=trend_weight.shift(1)



weights=trend_weight.copy()

#Excluded Tickers
available_excl = [t for t in settings['d_excluded_tickers'] if t in weights.columns]
weights[available_excl] = 0

#Scale
weights=weights/len(available_excl)

# Ticker-specific caps (e.g., Crypto/Risky assets)
settings['d_max_risky_tickers_weight']=0.1
for ticker in settings['d_risky_tickers']:
    if ticker in weights.columns:
        weights[ticker] = weights[ticker] / 2
        weights[ticker] = weights[ticker].clip(upper=settings['d_max_risky_tickers_weight'])


# Global asset caps
settings['d_max_asset_weight']=0.20
weights = weights.clip(upper=settings['d_max_asset_weight'])

settings['d_fix_mult']=1.0
weights = weights*settings['d_fix_mult']

# 6. Total Leverage Guard (The "Safety Valve")
settings['d_max_total_leverage']=1.5
if 'd_max_total_leverage' in settings:
    # Calculate the sum of weights for each day
    current_total_leverage = weights.sum(axis=1)

    # Determine the scaling factor:
    # If current leverage is 2.0 and max is 1.0, factor is 2.0.
    # We clip at lower=1.0 so we never "scale up" a small portfolio.

    scaling_factor = (current_total_leverage / settings['d_max_total_leverage']).clip(lower=1.0)

    # Divide all weights by that factor to bring the total down to the cap
    weights = weights.div(scaling_factor, axis=0)

#BACKTEST
trend_weight=weights.copy()
trend_ret=tickers_returns*trend_weight

#trend_ret['mean']=trend_ret.mean(axis=1)
trend_ret['mean']=(tickers_returns*weights).sum(axis=1)

trend_cumret=(1+trend_ret).cumprod()
cum_rets=cum_rets.reindex(highs.index)

for ticker in tickers_returns.columns:
    plot_df1=pd.DataFrame()
    plot_df1['highs']=highs[ticker]
    plot_df1['lows'] = lows[ticker]
    plot_df1['highs_rolling_max']=highs_rolling_max[ticker]
    plot_df1['lows_rolling_min'] = lows_rolling_min[ticker]
    plot_df1['mean_minmax'] = mean_minmax[ticker]
    plot_df1.plot(title=ticker)

    plot_df=pd.DataFrame()
    plot_df['cum_ret']=cum_rets[ticker]
    plot_df['mean_minmax_cum'] = mean_minmax_cum[ticker]
    #plot_df['mean_minmax_cagr'] = mean_minmax_cagr[ticker]
    #plot_df['trend_indicator'] = trend_indicator[ticker]
    #plot_df['previous_trend_indicator'] = previous_trend_indicator[ticker]
    plot_df['trend_weight'] = trend_weight[ticker]
    plot_df['trend_cumret'] = trend_cumret[ticker]
    #plot_df['cagr'] = cagr[ticker]
    #plot_df['norm_minmax_pct']=norm_minmax_pct[ticker]
    plot_df.plot(title=ticker)

plot_df2=pd.DataFrame()
plot_df2['benchmark']=cum_rets['ES=F']
plot_df2['strategy']=trend_cumret['mean']
plot_df2.plot(title='Trend Startegy Results')

q_title = 'Cash Backtest Trend Strategy'
path = "results\\"
q_filename = os.path.abspath(path + q_title + '.html')
q_returns = trend_cumret['mean']
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