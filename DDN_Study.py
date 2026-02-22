import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis



#Get SETTINGS
from config import settings,utils
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

#Update Settings
#settings['start']='1996-01-01'

import Market_Data_Feed as mdf


#Get Data
data_ind = mdf.Data_Ind_Feed(settings).data_ind
data, ind = data_ind
data_dict = data.data_dict

tickers_returns=data.tickers_returns
#tickers_returns['mean']=tickers_returns.mean(axis=1)
cum_rets=(1+tickers_returns).cumprod()

# Import your specific class from your new file
from ddn_ltd_portfolio import DDNLimitedPortfolio

# Initialize the Class
portfolio_manager = DDNLimitedPortfolio(settings)

# Generate the Weights
# This runs the math and stores the intermediates inside the object
ddn_weights = portfolio_manager.compute_weights(tickers_returns)

print(ddn_weights)

if settings['do_BT']:
    from Backtest_Vectorized_Class import compute_backtest_vectorized
    from Training_Markowitz import process_log_data
    #positions=ddn_weights.copy()
    #settings['tickers'] = list(positions.columns)
    _, log_history = compute_backtest_vectorized(ddn_weights, settings, data.data_dict)

    # End Of day Values From Log History
    eod_log_history, trading_history = process_log_data(log_history, settings)

    if settings['verbose']:
        # Print Backtrader Results
        print('\nCash BackTest with Backtrader ')
        print("log_history\n", log_history.tail(30))

        print("ddn_weights\n",ddn_weights.tail(20))
        print("eod_log_history\n", eod_log_history.tail(20))

        eod_log_history[ddn_weights.columns].plot(title='n_contracts')

#Backtest
ddn_returns=tickers_returns*ddn_weights
ddn_returns['mean'] = ddn_returns.sum(axis=1)

ddn_cumret=(1+ddn_returns).cumprod()


ddn_weights['sum'] = ddn_weights.sum(axis=1)
ddn_weights.plot(title='ddn_weights')

#ddn_cumret.plot(title='ddn_cumret')

results=pd.DataFrame()
results['ddn_cumret']=ddn_cumret['mean']
results['benchmarl'] = cum_rets['ES=F']
results.plot(title='Results DDN Weights Strategy ')

q_title = 'DDN Strategy Backtest'
path = "results\\"
import os
q_filename = os.path.abspath(path + q_title + '.html')
q_returns = ddn_returns['mean']
q_returns.index = pd.to_datetime(q_returns.index, utc=True).tz_convert(None)

q_benchmark_ticker = 'ES=F'
q_benchmark = tickers_returns[q_benchmark_ticker]
q_benchmark.index = pd.to_datetime(q_benchmark.index, utc=True).tz_convert(None)

import quantstats_lumi as quantstats

quantstats.reports.html(q_returns, title=q_title, benchmark=q_benchmark, benchmark_title=q_benchmark_ticker, output=q_filename)
import webbrowser

webbrowser.open(q_filename)


plt.show()