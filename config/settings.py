"""
Dictionaries with settings for QuantTradingApp
To access the settings dictionary within the config module, you need to import it
from config.settings import settings
"""

from datetime import date
from datetime import timedelta


settings={

    #Settings for Data Feed
    'tickers': [ 'ES=F','NQ=F', 'GC=F','CL=F', 'EURUSD=X'],#
    'start': '1996-01-01',#TRADING#'2019-01-01' #TRAINING#'1996-01-01'
    'end':  (date.today() + timedelta(days=1)).isoformat(),  #'2023-04-01',# '2023-01-18',# '2004-01-18',#
    'add_days': 5,  # Additional Business days for next days position estimation
    'contango': {'ES=F': 1.85, 'NQ=F': 2.35, 'GC=F': 20.0, 'CL=F': 1.64, 'EURUSD=X': 1.00,'cash':0},#Yearly Contango % -  Dif beetween Cash and next Future value 'GC=F': 6.26
    'add_cash':True,

    #Settings for Portfolio Optimization
    #'tickers_bounds': {'ES=F': (0,0.25), 'NQ=F': (-0,0.25), 'GC=F': (0.00,0.5), 'CL=F': (0,0.1), 'EURUSD=X': (-0,0.0),'cash':(0,0.30)},  # Default Weights Bounds Upper Limit for Each asset at optimization {'ES=F': 0.50, 'NQ=F': 0.50, 'GC=F': 0.50, 'EURUSD=X': 0.25, 'CL=F': 0.25}
    'tickers_bounds': {'ES=F': (-0.0, 0.0), 'NQ=F': (-0, 0.5), 'GC=F': (0.00, 0.5), 'CL=F': (0, 0.1), 'EURUSD=X': (-0.00, 0.0), 'cash': (0, 0.3)},
    'volatility_target': 0.110,#0.110,  #0.113, # 0.135, #0.124,  # 0.125,  # 0.135,  # 0.24#0.115

    #Settings for Cash Back Test
    'do_BT': True,
    'startcash': 68000, # 60000, #starting cash EUR
    'mults':{'ES=F': 5, 'NQ=F': 2, 'GC=F': 10, 'CL=F': 500, 'EURUSD=X': 12500,'cash':20}, # multipliers for e-micro futures (CL mini x500, CL micro x100,
    'EURUSD_hedge_factor': 0.0, #Percentage Hedge Exchange Rate Risk 0-1
    'btprint': False,
    'commision': 5, #USD by B/S contract

    #Exposition Constraints
    'w_sum_max': 2.5,  # Max sum of weights at markovitz calc
    'exposition_lim': 2.5, #2.5, Max exposition allowed aka Max strategy_pos sum. When no futures set to 1
    'w_upper_lim': 0.75,  #0.66, #1.0,  #0.9, #  Individual Upper Weight Limit
    'w_lower_lim': -0.1,  # Individual Lower Weight Limit
    'pos_exp_factor': 1.00, #1.05 # Position Exponential factor; positions = positions ** settings['pos_exp_factor']
    'pos_mult_factor': 1.5, #1.5, #1.25, #1.15 # Position Multiplicative factor; positions = positions * settings['pos_mult_factor']
    'apply_pos_consraints': True,

    #Markowitz Windows Parameters [fast,mid,slow]
    'mkwtz_ws': [44,180,360],# Markowitz Lookback Window days [fast,mid,slow]
    'mkwtz_mean_fs': [0.9,1,1],  # factor to apply when mean of diferent windows weights[fast,mid,slow]
    'mkwtz_ps':  ['W-FRI','W-FRI','M'], #Rebalance Period 'W-FRI','M','Q','Y' [fast,mid,slow]

    #Walk Forward Training & Test Parameters
    'train_length': 245 * 7,   #Lookback of data for training
    'train_length_min': 245 * 3,#3  # Min Lookback of data for training
    'test_length': 'Y',  #  Period 'W-FRI','M','Q','Y'

    #Indicators Paramters
    'rsi_upp': 65,  # 65 #100 means desactivated
    'rsi_low': 35,  # 35 #0 means desactivated
    'rsi_w': 14,  # 14
    'rsi_window': 22,#22

    #Post Opt Parameters
    'apply_post_opt': True,  # Calculate Post Optimization and get post_factor
    'trading_volatility_delta': -0.02,  # -0.02,  # trading_volatility_target= volatility_target + trading_volatility_delta
    'volatility_factor_diff_delta': {'up':1,'dn':1},#0.25,  # .25 #volatility_factor maximum abs diff
    'vol_factor_max': 8.0, #2.0 #1.85 #1.65, #1.75

    #After Test Optimization Params
    'apply_after_test_opt': False,  # Calculate After Test Optimization


    # Parameters Markowitz
    'mkwtz_scipy': True,  # Get Scipy Calculated optimal weights
    'mkwtz_vectorized': True,  # Get Vectorized Calculated results to make mean with Minimize calc
    'cov_w': 10, #10  # Optimized Windows in n - days
    'cagr_w': [20,160,250,390],# Optimized Windows in n - days
    'param_to_loop': 'cagr_w',
    'strat_periods':  [
                'dayly',
               'weekly',
               #'monthly',
    ],
    #Weekly params. Window in n - weeks
    'cov_w_weekly': 15, #15
    'cagr_w_weekly': [32, 50, 78] ,
    # Monthly params. Window in n - months
    'cov_w_monthly': 9,
    'cagr_w_monthly': [9, 12],
    #Weights for mean
    'mean_weights_d_w_m':[3.0 , 1 , 0], #dayly, weekly, monthly weights
    'apply_utility_factor': True,

    # Default Settings
    'trading_app_only': False,
    'weekly_trading_only': False,
    'apply_opt_fun_predict_factor':False, #Apply opt_fun Predictibity factor. Better when Prediction is good
    'apply_strategy_weights': True,  # Apply RSI and other additional Strategy Weights of top of Markowitz weights
     #'apply_strategy': True,  #Apply Strategy Weights at Porfolio Optimized Weights
    'use_train_csv': False,  # Use train csv for test without recalculate train again
    'apply_tickers_returns_EUR': False,  #Assets in EUR before Training & Test
    'apply_post_opt_in_EUR': False,  # Portfolio Returns in EUR before Post Optimization
    'qstats': True,  #Plot html quantstats summary
    'verbose': True,
    'do_annalytics': True,
    'offline': False, #Work from data csv without internet connection

    #Folder paths
    'path_train': "trained_models/",


    # Default System Hyperpameters to be optimized
    'params': {},

    # Parameters Bounds for Tunning
    'params_bounds': {
            'volatility_target': (0.10,0.22),
            'w_sum_max': (1.0, 1.4),
            'mkwtz_w': (120, 280),
            'mkwtz_w_slow': (350, 410),
            'mkwtz_w_fast': (5, 55),
            'mkwtz_f': (0, 2),
            'mkwtz_slow_f': (0,2),
            'mkwtz_fast_f': (0,2),
            'rsi_upp': (55, 75),
            'rsi_low': (25, 45),
            'rsi_len': (10, 22),
            'rsi_window': (10, 44),
            'bnd_max': (0.35, 1),
            'bnd_min': (0.00,0.35),

            }

}

def get_settings():
    tickers = settings['tickers'].copy()
    #Add cash
    if settings['add_cash']:
        tickers.append('cash')

    # Order  as tickers
    for setting in ['tickers_bounds','contango','mults']:
        settings[setting] = {ticker: settings[setting][ticker] for ticker in tickers}

    return settings