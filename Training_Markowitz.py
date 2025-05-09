#IMPORT RELEVANT MODULES

#Import libraries and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import os.path

# Wider print limits
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
# Silence warnings
import warnings
warnings.filterwarnings('ignore')

from Backtest_Vectorized import compute_backtest_vectorized
from Markowitz_Vectorized import compute_optimized_markowitz_d_w_m
from WalkForwardTraining import WalkForwardTraining
import Market_Data_Feed as mdf
from utils import mean_positions


#Get SETTINGS
from config import settings,utils
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

# MAIN CODE
def run(settings):

    times={}

    #DATA & INDICATORS

    start = time.time()

    data_ind=mdf.Data_Ind_Feed(settings).data_ind
    data, indicators_dict = data_ind
    tickers_returns=data.tickers_returns

    end = time.time()
    times['get_data']= round(end - start,3)

    if settings['verbose']:
        print(tickers_returns.iloc[:-settings['add_days']])
        print('Data & Indicators Ok',times['get_data'])

    # TRAINING

    start = time.time()

    # Get Trained Optimized Parameters
    wft = WalkForwardTraining(data_ind, settings) #Get wft Instance
    print('tt_windows\n', wft.tt_windows)
    params_train = wft.get_params_train(data_ind, settings)

    # Save settings as training_settings to make sure same settings are used at trading
    utils.settings_to_JASON(settings)

    end = time.time()
    times['training'] =  round(end - start,3)

    if settings['verbose']:
        print('Training Ok',times['training'])


    #BACKTEST & TRADING

    #Trading/Test:Apply Trained Params to Test
    start = time.time()
    wft.Test(indicators_dict,params_train,settings['do_annalytics'])

    #Get Test Positions
    positions = wft.test_positions

    positions.plot(title='Test Positions')

    #After Test Optimization
    if settings['apply_after_test_opt']:
        from AfterTestOptimization import AfterTestOptimization
        ATO = AfterTestOptimization(wft.test_positions, wft.test_returns, mean_w=22 * 9,over_mean_pct=0.04, lookback=22 * 2, up_f=1.3, dn_f=1.0, plotting=True)
        positions = ATO.after_test_positions

    #Apply Exposition Constraints
    #Exponential factor,Mult factor & Limit maximum/minimum individual position
    if settings['apply_pos_consraints']:
        positions = apply_pos_constrain(positions,settings )


    end = time.time()
    times['test'] =  round(end - start,3)

    if settings['verbose']:
        print('Test Ok',times['training'])

    #Cash BackTest with Backtest_Vectorized
    start = time.time()

    if settings['do_BT'] :
        settings['tickers'] = list(positions.columns)
        #log_history,sell_stop_price,bt_returns_eur=bt.run(positions, settings, data.data_dict)
        _, log_history = compute_backtest_vectorized(positions, settings, data.data_dict)

        # End Of day Values From Log History
        eod_log_history, trading_history = process_log_data(log_history, settings)

        if settings['verbose']:
             # Print Backtrader Results
            print('\nCash BackTest with Backtrader ')
            print("log_history\n", log_history.tail(20))
            print("eod_log_history\n", eod_log_history.tail(20))

    else:
        log_history, sell_stop_price, bt_returns_eur =None,None,None


    end = time.time()
    times['backtrader'] =  round(end - start,3)

    # endregion & TRADING

    #region Prints

    # Execution Times
    times['total'] = sum(times.values())
    times= {k: round(v, 2) for k, v in times.items()}
    print('\ntimes',times)

    if False:

        # Print Settings
        print('Default Settings:')
        for k in settings.keys(): print(k,settings[k])
        print('\nParameters Bounds:')
        for k in params.keys(): print(k, params_bounds[k])


    #endregion

    #region Plots
    #mkt.mkwtz_weights.plot(title='weights')
    plot_df=positions.copy()
    plot_df['sum']=plot_df.sum(axis=1)
    plot_df['cum_ret']=(1+eod_log_history['portfolio_value_eur'].pct_change()).cumprod()-1
    plot_df.plot(title='positions')
    if settings['do_BT'] :
        eod_log_history[positions.columns].plot(title='n_contracts')


    # endregion

    plt.show()

    return log_history,positions

def multiple_charts(charts_dict,chart_title=''):
    n_subplots=len(list(charts_dict.keys()))
    fig, axs = plt.subplots(n_subplots, sharex=True)
    fig.suptitle(chart_title)
    for i, (k, v) in enumerate(charts_dict.items()):
        axs[i].plot(v)
        axs[i].set_ylabel(k)
    axs[0].legend(list(charts_dict.values())[0].columns, loc="lower left")
    return

def process_log_data(log_history,settings):

    # Overwrite Drawdown YTD EUR
    max_portfolio_value_eur = log_history['portfolio_value_eur'].rolling(252, min_periods=5).max()
    log_history['ddn_eur'] = round(1 - max_portfolio_value_eur / log_history['portfolio_value_eur'], 3)

    #End of day Portfolio data

    eod_log_history=log_history.drop_duplicates(subset='date', keep='last').set_index('date')[settings["tickers"]+["portfolio_value","portfolio_value_eur","pos_value","ddn","exchange_rate" , "dayly_profit","dayly_profit_eur"]]

    # Add Drawdown YTD EUR
    max_portfolio_value_eur = eod_log_history['portfolio_value_eur'].rolling(252, min_periods=5).max()
    eod_log_history['ddn_eur'] = round(1 - max_portfolio_value_eur / eod_log_history['portfolio_value_eur'], 3)

    #Filter days where trading
    eod_log_history['keep_day']=(eod_log_history[settings["tickers"]] == eod_log_history[settings["tickers"]].shift(1)).all(axis=1).astype(int)
    trading_history = eod_log_history[eod_log_history['keep_day']!=1]

    #Add some annalytics
    eod_log_history["portfolio_return"]=eod_log_history["portfolio_value_eur"].pct_change()
    eod_log_history["cagr"]=eod_log_history["portfolio_return"].rolling(252).sum()
    eod_log_history["weekly_return"]=eod_log_history["portfolio_return"].rolling(5).sum()
    eod_log_history["monthly_return"]=eod_log_history["portfolio_return"].rolling(22).sum()

    #Index to datetime
    eod_log_history.index = pd.to_datetime(eod_log_history.index)
    trading_history.index = pd.to_datetime(trading_history.index)

    return eod_log_history,trading_history

def get_vector_positions(tickers_returns, settings, data):

    # Compute Markowitz And Optimize with Utility Factor and Strategy Weights
    vector_positions, _, _, _, _, _ = compute_optimized_markowitz_d_w_m(tickers_returns, settings, data)

    return vector_positions



def apply_pos_constrain(positions,settings ):

    #Update pos_mult_factor when add_cash
    #if settings['add_cash']:
    #    settings['pos_mult_factor'] = 2 * settings['pos_mult_factor'] * settings['tickers_bounds']['cash'][1] * 10

    # Apply Exponential factor keeping position sign
    positions =np.sign(positions) *positions.abs() ** settings['pos_exp_factor']

    # Apply Mult factor
    positions = positions * settings['pos_mult_factor']


    # Limit Position to maximum of Exposition Allowed
    positions_sum=positions.sum(axis=1)
    positions_sum_series = pd.Series(positions_sum, index=positions.index)
    positions_sum_is_high=positions_sum>settings['exposition_lim']
    reduced_position=positions.div(positions_sum_series, axis=0)
    positions.loc[positions_sum_is_high,:]=reduced_position

    # Limit Position to maximum/minimum individual position
    positions = positions.clip(upper=settings['w_upper_lim'],lower=settings['w_lower_lim'])

    # Limit Upper Cash position by available cash not used in futures guaranties
    if 'cash' in positions.columns:
        no_cash_pos_sum=positions.sum(axis=1)-positions['cash']
        futures_guaranties=0.20*no_cash_pos_sum
        available_cash = 1 - futures_guaranties
        positions['cash'] = positions['cash'].clip(upper=available_cash)

    return positions


if __name__ == '__main__':
    run(settings)