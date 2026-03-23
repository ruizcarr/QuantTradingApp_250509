import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import time
from datetime import date
from datetime import timedelta

from utils import weighted_mean_of_dfs_dict
#from namespace import Namespace

import Market_Data_Feed as mdf

np.random.seed(1234)

# Wider print limits
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# Silence warnings
import warnings
warnings.filterwarnings('ignore')
warnings.warn = lambda *a, **kw: False


#FUNCTIONS

def compute_optimized_markowitz_d_w(tickers_returns, settings):

    # Weights Combination xs
    tickers_bounds = {ticker: settings['tickers_bounds'][ticker] for ticker in tickers_returns.columns}

    weight_sum_lim = settings['exposition_lim']

    # Generate Fix weights Combinations to find the best

    xs = generate_xs_combinations(tickers_bounds, weight_sum_lim, step=0.10) #0.10

    # Compute Dayly Markowitz Looping  over Selected Parameter
    dayly_weights_df, d_returns, returns_p,metrics_df, rolling_metrics_dict,markowitz_metrics_dicts = compute_markowitz_loop_over_ps(tickers_returns,xs, settings,strat_period='dayly')

    #Compute Weekly Markowitz
    weekly_weights_df, w_returns, weekly_metrics_df, weekly_rolling_metrics_dict,w_k=compute_weekly_markowitz(tickers_returns,xs, settings,strat_period='weekly')

   #Weighed Mean Dayly / Weekly

    # Apply Markowitz to get optimal Startegy
    #Get strat_periods
    strats =settings['strat_periods']
    #Create d_w_returns_df with n_strats columns
    d_w_returns_df=pd.DataFrame({strat: df for strat,df in zip(strats,[d_returns,w_returns])})
    # Concat d_w_ weights array (n_strats,n_days,n_tickers)
    d_w_weights = np.array([np.array(df) for strat, df in zip(strats, [dayly_weights_df, weekly_weights_df])])

    volat_target=settings['volatility_target']
    weight_lim=0.8 #0.8
    weight_sum_lim = 1.2 #1.2
    cagr_w=250*4 #250*4
    timeframe = 'dayly'

    weights_comb_array, weights_by_strategy_df = get_combined_strategy_by_markowitz(d_w_returns_df, d_w_weights, volat_target,weight_lim,weight_sum_lim,cagr_w, timeframe)

    weights_df = pd.DataFrame(weights_comb_array, index=dayly_weights_df.index, columns=dayly_weights_df.columns, dtype=float)

    # Make fast mean for smooth  curve
    w = 6  # 6
    weights_df = weights_df.rolling(w).mean().fillna(0)

    #Get Metrics for Mean D/W Returns
    strategy_returns, metrics, rolling_metrics = get_strategy_metrics(weights_df, tickers_returns, 'dayly')
    metrics_df['d_w']=metrics.T
    rolling_metrics_dict['d_w']=rolling_metrics

    # Optimization by Utility Up factor
    weights_df, metrics_df, rolling_metrics_dict, returns_p = compute_utility_factor(settings['apply_utility_factor'],tickers_returns, weights_df, metrics_df, rolling_metrics_dict, returns_p,weight_sum_lim,'dayly')

    return weights_df, metrics_df, rolling_metrics_dict, weekly_metrics_df, markowitz_metrics_dicts

def compute_optimized_markowitz(tickers_returns, settings):
    # Compute Markowitz Looping  over Selected Parameter
    weights_df, returns_p, metrics_df, rolling_metrics_dict = compute_markowitz_loop_over_ps(tickers_returns, settings)

    # Optimization by Utility Up factor
    weights_df, metrics_df, rolling_metrics_dict, returns_p = compute_utility_factor(settings['apply_utility_factor'],tickers_returns, weights_df, metrics_df, rolling_metrics_dict, returns_p)

    return weights_df, metrics_df, rolling_metrics_dict,returns_p


def compute_markowitz_loop_over_ps(tickers_ret,xs,settings,strat_period='dayly'):

    #Compute markowitz metrics invariant by p parameter loop

    # Parameters from Dict
    volat_target = settings['volatility_target']

    if strat_period=='dayly':
        cov_w=settings['cov_w']
        ps = settings[settings['param_to_loop']]
        year=250



    elif strat_period=='weekly':
        cov_w = settings['cov_w_weekly']
        ps = settings[str(settings['param_to_loop']+'_weekly')]
        year=52

    # Substract Contango from tickers_ret
    contango_list = [settings['contango'][ticker] for ticker in tickers_ret.columns]
    tickers_ret = tickers_ret - np.array(contango_list) / 100 / 252

    #Get Markowitz Metrics
    markowitz_metrics_dict=compute_markowitz_cov_metrics(tickers_ret,xs,cov_w,volat_target,strat_period)

    #Initialize df and dict to store loop resuts
    weights_p_df = pd.DataFrame(index=tickers_ret.index)
    returns_p=pd.DataFrame(index=tickers_ret.index)
    metrics_p=pd.DataFrame()
    rolling_metrics_dict={}
    markowitz_metrics_p_dicts={}

    weights_p_array=[]


    for cagr_w in ps:

        #Update markowitz metrics function of p parameter loop
        markowitz_metrics_p_dict=update_markowitz_cagr_metrics(cagr_w,markowitz_metrics_dict,strat_period)

        #Compute Markowitz
        weights_df,metrics_opt_df,_ = compute_mkwtz_vectorized_local(markowitz_metrics_p_dict)
        # Shift for weights to use today
        weights_df = weights_df.shift(1).fillna(0)
        #Save Weights
        weights_p_array.append(np.array(weights_df ))

        #Get Startegy Metrics
        returns_w, m, rolling_metrics = get_strategy_metrics(weights_df,tickers_ret, strat_period)

        # Save Results for this Parameter
        metrics_p[str(cagr_w)]=m.T
        rolling_metrics_dict[str(cagr_w)]=rolling_metrics
        returns_p[str(cagr_w)] = returns_w

        #Add opt_fun values at minimum selected
        markowitz_metrics_p_dict['opt_fun_min']=metrics_opt_df['opt_fun']
        markowitz_metrics_p_dicts[str(cagr_w)] = markowitz_metrics_p_dict

    # np Array of weights by parameter
    weights_p_array=np.array(weights_p_array)

    #Combined Strategy

    mean= True

    if mean:
        # Simple mean of p Strategies
        weights_comb_array=np.mean(weights_p_array, axis=0)

    else:

        #Apply Markowitz to get optimal Startegy
        weight_lim, weight_sum_lim,cagr_w   =0.35, 1.2, 250*10
        weights_comb_array, weights_by_strategy_df = get_combined_strategy_by_markowitz(returns_p, weights_p_array, volat_target, weight_lim, weight_sum_lim, cagr_w, strat_period)

        plot_df = weights_by_strategy_df.copy()
        plot_df['sum'] = plot_df.sum(axis=1)
        plot_df.plot(title='weights_by_strategy_df')


    #Weights Array to df
    weights_comb_df = pd.DataFrame(weights_comb_array,index=weights_df.index, columns=weights_df.columns, dtype=float)

    #print('weights_comb_df',weights_comb_df )

    #weights_comb_df .plot(title='weights_comb_df ')


    # Get Combined Startegy Metrics
    strat_returns, m, rolling_metrics = get_strategy_metrics(weights_comb_df, tickers_ret, strat_period)

    #save Results for Combined Strategy
    metrics_p['combined']=m.T
    rolling_metrics_dict['combined']=rolling_metrics

    return weights_comb_df,strat_returns,returns_p,metrics_p,rolling_metrics_dict,markowitz_metrics_p_dicts

def get_combined_strategy_by_markowitz(returns_p, weights_p_array, volat_target,weight_lim, weight_sum_lim, cagr_w,strat_period ):
        """"
        :param returns_p: Dataframe (n_days,n_params)
        :param weights_p_array: np.array (n_params,n_days,n_tickers)
        :param volat_target: float
        :return:weights_by_ticker_array:(n_days,n_tickers)
        """

        #Get Dimensions values
        n_params,n_days,n_tickers=weights_p_array.shape

        #print('returns_p', returns_p)
        #print('weights_p_array.shape',weights_p_array.shape)

        #Get xs
        # Generate Fix weights Combinations to find the best
        p_bounds={key:(0,weight_lim) for key in returns_p.columns}

        xs = generate_xs_combinations(p_bounds, weight_sum_lim, step=0.15)

        # Get Markowitz Metrics
        cov_w=10
        markowitz_metrics_dict = compute_markowitz_cov_metrics(returns_p, xs, cov_w, volat_target, strat_period)

        #Update markowitz metrics function of p parameter loop
        markowitz_metrics_p_dict=update_markowitz_cagr_metrics(cagr_w,markowitz_metrics_dict,strat_period)

        #Compute Markowitz
        weights_by_strategy_df,_,_ = compute_mkwtz_vectorized_local(markowitz_metrics_p_dict)

        # Work with yesterday weights_by_strategy_df to avoid knowledge of the future
        weights_by_strategy_df=weights_by_strategy_df.shift(1).fillna(0)

        #Use the mean only
        #weights_by_strategy_df.loc[:, :]  = np.array([np.array(weights_by_strategy_df.mean())]* n_days)

        #Get Weights by Ticker

        #print('weights_by_strategy_df.shape',weights_by_strategy_df.shape)

        # Ensure that weights_by_strategy_df is a NumPy array
        weights_by_strategy_array = np.array(weights_by_strategy_df)
        #print('weights_by_strategy_array.shape', weights_by_strategy_array.shape)

        # Reshape weights_by_strategy_array to (n_params, n_days, n_tickers) for efficient broadcasting
        weights_by_strategy_array_reshaped = weights_by_strategy_array.T.reshape(n_params, n_days, 1)
        weights_by_strategy_array_reshaped = np.repeat(weights_by_strategy_array_reshaped, n_tickers, axis=2)

        #print('weights_by_strategy_array_reshaped.shape', weights_by_strategy_array_reshaped.shape)

        # Multiply and sum along axis 0
        weights_by_ticker_array = np.sum(weights_p_array * weights_by_strategy_array_reshaped, axis=0)



        return weights_by_ticker_array,weights_by_strategy_df

def generate_xs_combinations(bounds_dict,weight_sum_lim,step=0.1):
    """
      Generates an array of all possible combinations of n_assets weights with limited bounds at fixed step

      Args:
        bounds_dict: A dictionary containing keys (element names) and values as tuples specifying bounds (lower, upper) for each element.
        weight_sum_lim: The maximum allowed sum of elements in a combination.
        step: The step size for the grid of values generated for each element (default: 0.1).bounds_dict: A dictionary containing keys (element names) and values as tuples specifying bounds (lower, upper) for each element.

      Returns:
          A NumPy array containing all possible combinations (shape: (number of combinations, n_asstets) )
          with n_asstets=len(bounds_dict.keys()).
      """
    lower_bounds, upper_bounds = zip(*bounds_dict.values())  # Separate lower and upper bounds
    grids = np.meshgrid(*[np.linspace(lb, ub, int((ub - lb) / step) + 1) for lb, ub in zip(lower_bounds, upper_bounds)])  # Create grids with bounds

    # Combine grids into single array using advanced indexing
    xs = np.stack([grid.ravel() for grid in grids], axis=-1)

    # Limited Weight Sum
    xs = xs[np.sum(xs, axis=1) <= weight_sum_lim]

    return xs

def limit_xs_diff(xs, max_diff=0.1):
    diffs = np.diff(xs, axis=1)
    xs = xs[np.all(np.abs(diffs) <= max_diff, axis=1)]
    return xs

def compute_markowitz_cov_metrics(returns,xs,cov_w,volat_target,strat_period):
    if strat_period=='dayly':  year,week=252,5
    elif strat_period=='weekly': year,week=53,1
    else: print('strat_period not defined')

    #Covariance Matrix
    # Get Covariance Matrices
    np_cov_matrices = mdf.get_np_cov_matrices(returns, cov_w)

    # Variances for all weights
    variances_xs = np.sum(np.multiply(np.dot(np_cov_matrices, xs.T), xs.T), axis=1)


    #Annualized Volatility for each Weights Combination xs
    volatility_xs = np.sqrt(variances_xs* 252*(week/5)) + 0.0001

     # Rolling Std Dev of Volatility
    volatility_xs_df = pd.DataFrame(volatility_xs)
    volat_xs_std = np.array(volatility_xs_df.rolling(int(year/12)).std())

    # Rolling Drawdawn
    returns_xs = np.dot(np.array(returns), xs.T)
    cum_ret = (1 + pd.DataFrame(returns_xs)).cumprod()
    rolling_ddn = cum_ret.rolling(year * 3, min_periods=year).max() / cum_ret - 1
    ddn_xs = np.array(rolling_ddn)

    # Rolling Std Dev of Ddn
    ddn_xs_std_df = rolling_ddn.rolling(year).std()
    ddn_xs_std = np.array(ddn_xs_std_df)

    # penalties
    high_volat_xs = np.where(volatility_xs > volat_target, volatility_xs / volat_target - 1, 0)
    lower_volat = 0.01
    low_volat_xs = np.where(volatility_xs < lower_volat, lower_volat / volatility_xs - 1, 0)

    penalties_xs = np.clip(high_volat_xs + low_volat_xs,0,2)

    #Save values in a dict
    dict={
        'returns':returns,
        'xs':xs,
        'volatility_xs':volatility_xs,
        'volat_xs_std':volat_xs_std,
        'returns_xs': returns_xs,
        'cum_ret_xs': cum_ret,
        'ddn_xs': ddn_xs,
        'ddn_xs_std': ddn_xs_std,
        'penalties_xs': penalties_xs,
          }

    return dict

def update_markowitz_cagr_metrics(cagr_w, dict,strat_period='dayly'):

    if strat_period=='dayly': year, week=252,5
    elif strat_period=='weekly': year, week =53,1
    else: print('strat_period not defined')

    #CAGR for all weights xs (Alternate)
    returns_xs_df=pd.DataFrame(dict['returns_xs'])
    cagr_xs_df = returns_xs_df.rolling(cagr_w, min_periods=week).mean().fillna(0.0001) * year
    dict['cagr_xs'] = np.array(cagr_xs_df)

    # Get Function to minimize
    use_normalized=True

    #def get_opt_fun
    if use_normalized:

        #Normalize values
        # Normalized Metrics Clip (0,1)
        for key in list(dict.keys()):
            dict[key + "_norm"] = np.clip(dict[key], 0.0001, 1)

        dict['cagr_xs_norm'] = minmax_normalize(np.clip(dict['cagr_xs'], -1, 1))  # value -1 to 1

        #Alternate with components of function normalized
        #volat_f,ddn_std_f,volat_std_f = 1.0, 1.0, 0.5 #Weighted Mean factor
        #dict['risk_xs'] = (dict['volatility_xs_norm'] * volat_f + dict['ddn_xs_std_norm'] * ddn_std_f + dict['volat_xs_std_norm'] * volat_std_f)/(volat_f+ddn_std_f+volat_std_f )
        risk_xs_components=['volatility_xs_norm','ddn_xs_std_norm','volat_xs_std_norm']
        dfs_dict={key: dict [key] for key in risk_xs_components}
        weights_list=[1.0, 1.0, 0.5]
        dict['risk_xs_norm'] = weighted_mean_of_dfs_dict(dfs_dict, weights_list)

        dict['opt_fun_xs'] = dict['risk_xs_norm'] - dict['cagr_xs_norm'] + dict['penalties_xs_norm']


    else:

        dict['risk_xs'] = dict['volatility_xs'] * 1 + dict['ddn_xs_std']*0.0 + dict['volat_xs_std'] * 0.0

        dict['opt_fun_xs'] = dict['risk_xs'] - dict['cagr_xs'] + dict['penalties_xs']

    return dict

def softmax(x):
    """Applies softmax normalization to a NumPy array row-wise.

    Args:
        x: A NumPy array.

    Returns:
        A NumPy array with softmax normalized values.
    """

    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def minmax_normalize(x):
    min=np.nanmin(x)
    max=np.nanmax(x)
    x=(x-min)/(max-min)
    #x=np.clip(x, 0, 1)
    return x


def cum_minmax_normalize(x):
    """
    Vectorized expanding min-max normalization for NumPy arrays.
    x shape: (n_days, n_combinations)
    """
    # Calculate running extremes along the time axis (axis 0)
    running_min = np.minimum.accumulate(x, axis=0)
    running_max = np.maximum.accumulate(x, axis=0)

    # Calculate range, preventing division by zero
    rng = running_max - running_min
    rng[rng == 0] = 1e-9

    return (x - running_min) / rng

def mean_std_normalize(x,n=2):
    std=np.nanstd(x)
    mean=np.nanstd(x)
    x=(x- mean)/std #array (-3 ~ +3)
    x_min=np.nanmin(x)
    n=min(n,-x_min) #keep minimum value
    x=np.clip(x,-n,n) #array (-n ~ +n)
    x = (x +n) /(2*n) #array (0 ~ +1)
    return x

def log_normalize(x):
    min=np.nanmin(x)
    x=x-min+1
    x=np.log(x)
    return x

def print_metrics(x):
    print('np.nanmax(x)', np.nanmax(x))
    print('np.nanmin(x)', np.nanmin(x))
    print('np.nanmean(x)', np.nanmean(x))
    print('np.nanstd(x)', np.nanstd(x))

def compute_mkwtz_vectorized_local(markowitz_data_dict, metrics=True):
    """
    Get Portfolio Weights for each day that minimize Function to minimize.
    Function Sample: Volatility - CAGR

    Args in markowitz_data_dict:
        returns: dataframe timeseries (n_days, n_tickers)
        xs: array (n_combination,n_tickers) of all possible combinations of n elements within specified bounds
        opt_fun_xs: array(n_days,n_combination) of function to minimize for all xs

    Returns:
        weights: selected x array (n_days,n_tickers) from xs for each day that minimize opt_fun

    """

    # Get Data, Metrics and Function from dict
    returns,xs,volatility_xs, cagr_xs, opt_fun_xs = \
        [markowitz_data_dict[k] for k in
         ['returns','xs','volatility_xs', 'cagr_xs', 'opt_fun_xs']]


    # Localize index of xs where opt_fun is minimum
    opt_fun_min_idx_xs = np.argmin(opt_fun_xs, axis=1)

    # Get weights of this minimum
    weights = xs[opt_fun_min_idx_xs]

    # Save weights as df
    weights_df = pd.DataFrame(weights, index=returns.index, columns=returns.columns)

    # Save Metrics
    if metrics:

        # Metrics of this mimimum
        metrics_opt_df = pd.DataFrame(index=returns.index)
        metrics_opt_df['volat'] = np.take_along_axis(volatility_xs, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten()
        metrics_opt_df['ret'] = np.take_along_axis(cagr_xs, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten()
        # opt_fun_xs values for opt_fun_min_idx_xs
        opt_fun_min = np.take_along_axis(opt_fun_xs, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten()
        metrics_opt_df['opt_fun'] = opt_fun_min

        # All xs Metrics
        metrics_xs_dict = {'volatility_xs': volatility_xs, 'cagr_xs': cagr_xs}

    else:
        metrics_xs_dict = None
        metrics_opt_df = None

    return weights_df, metrics_opt_df, metrics_xs_dict






def get_returns_metrics(returns,strat_period='dayly'):

    if strat_period=='dayly': year,month,week=252,22,5
    elif strat_period=='weekly': year,month,week=53,4,1
    else: print('strat_period not defined')


    rolling_volat=returns.rolling(2*month,min_periods=month).std()*(year**0.5)
    volat=returns.std()*(year**0.5)
    volat_std=rolling_volat.std()
    volat_max = rolling_volat.max()
    rolling_cagr=returns.rolling(year).mean()*year
    cagr=returns.mean()*year
    cum_ret = (1 +returns).cumprod()
    rolling_ddn=cum_ret.rolling(18*month,min_periods=10*month).max()/cum_ret-1
    ddn_max=rolling_ddn.max()
    rolling_sharpe=rolling_cagr.clip(upper=1.0)/rolling_volat.clip(lower=0.01)
    rolling_utility=rolling_cagr.clip(upper=1.0)-rolling_volat.clip(lower=0.01)
    sharpe=cagr/volat
    sharpe_ddn=cagr/ddn_max

    #Utility Factor
    rolling_utility_mean=rolling_utility.rolling(month*3,min_periods=month).mean().fillna(0)
    rolling_utility_std = rolling_utility.rolling(month*3,min_periods=month).std()
    expected_utility_min = rolling_utility_mean-4*rolling_utility_std   #3

    expected_utility_min_mean=expected_utility_min.rolling(month*3,min_periods=month).mean()
    expected_utility_min_diff=expected_utility_min-expected_utility_min_mean
    utility_factor=(1.0 + expected_utility_min_diff*40)*1.7   #1.0/40/1.7

    utility_factor = utility_factor.clip(lower=0.3,upper=2.2)   #*1.3 #0/1.3/1.6
    utility_factor = utility_factor.rolling(week).max()
    #utility_factor = utility_factor.shift(1)

    #Save Scalars in a df
    metrics=[ volat, volat_std, volat_max, cagr, ddn_max,  sharpe,sharpe_ddn]
    metrics_df=pd.DataFrame(metrics).T
    metrics_df.columns=['volat', 'volat_std','volat_max', 'cagr', 'ddn_max',  'sharpe','sharpe_ddn']

    #Save Series in a df
    rolling_metrics=[rolling_volat, rolling_cagr, rolling_sharpe,rolling_ddn,rolling_utility,
                     #expected_ddn_max,corr_volat_ddn,expected_volat_max,expected_cagr_min,expected_utility_min,
                     utility_factor,cum_ret]

    #rolling_metrics_df=pd.DataFrame(rolling_metrics).T
    rolling_metrics_df = pd.concat(rolling_metrics,axis=1)
    rolling_metrics_df.columns = ['rolling_volat', 'rolling_cagr', 'rolling_sharpe','rolling_ddn','rolling_utility',
                                  #'expected_ddn_max','corr_volat_ddn','expected_volat_max','expected_cagr_min','expected_utility_min',
                                  'utility_factor','cum_ret']
    return metrics_df,rolling_metrics_df

def compute_utility_factor(apply_utility_factor,tickers_returns, weights_in_df, metrics_p, rolling_metrics_dict, returns_p,weight_sum_lim,strat_period='dayly'):
    key = list(rolling_metrics_dict.keys())[-1]

    if apply_utility_factor:

        # Apply Drawdown factor for Drawdawn Matched Strategy
        utility_factor = rolling_metrics_dict[key]['utility_factor'].shift(1)
        weights_uty_df = weights_in_df.multiply(utility_factor, axis='index')

    else:
        # Do nothing
        weights_uty_df = weights_in_df

    # Set Limit to Weights Sum
    weights_uty_df=set_limit_to_weights_sum(weights_uty_df,weight_sum_lim)

    # Returns of Utility Factor Startegy
    strategy_returns = get_strategy_returns(weights_uty_df, tickers_returns)


    # Metrics
    metrics, rolling_metrics = get_returns_metrics(strategy_returns,strat_period)
    metrics_p['optimized_uty'] = metrics.T
    rolling_metrics_dict['optimized_uty'] = rolling_metrics
    returns_p['optimized_uty'] = strategy_returns

    return weights_uty_df, metrics_p, rolling_metrics_dict, returns_p

def get_strategy_returns(weights, returns):
    # Start
    # Get the first index where any weight is greater than zero (using any)
    start = weights[(weights > 0).any(axis=1)].index[0]

    # Set Start at Weights and  Tickers Data
    returns = returns[start:]

    return (weights * returns).sum(axis=1)


def get_strategy_metrics(weights, tickers_returns, strat_period):
    # Strategy Returns
    strategy_returns = get_strategy_returns(weights, tickers_returns)

    # Get Metrics for this parameter
    metrics, rolling_metrics = get_returns_metrics(strategy_returns, strat_period)

    return strategy_returns, metrics, rolling_metrics

def compute_weekly_markowitz(tickers_returns,xs, settings,strat_period):

    if 'weekly' in settings['strat_periods']:

        #Get weekly data considering fridays holidays
        weekly_returns = tickers_returns.resample('W-FRI').sum()

        # Compute Weekly Markowitz Looping  over Selected Parameter
        weekly_weights_df, w_returns, weekly_returns_p , weekly_metrics_df, weekly_rolling_metrics_dict,w_markowitz_metrics_dicts= compute_markowitz_loop_over_ps(weekly_returns,xs, settings,strat_period)

        # Upsample to dayly with values of previous Friday
        weekly_weights_df = weekly_weights_df.reindex(tickers_returns.index).shift(1).fillna(method='ffill').fillna(0)

        #Get Metrics for Weekly Strategy Upsample to dayly
        w_returns , metrics, rolling_metrics = get_strategy_metrics(weekly_weights_df, tickers_returns, 'dayly')
        weekly_metrics_df['weekly']=metrics.T
        weekly_rolling_metrics_dict['weekly']=rolling_metrics

        #Set weekly multiplicator at mean weight
        w_k=1.25 #max(weekly_metrics_df.loc['sharpe', 'weekly'] - bs,0)*weekly_metrics_df.loc['sharpe_ddn', 'weekly']

    else:
        weekly_weights_df, w_returns ,weekly_metrics_df, weekly_rolling_metrics_dict,w_k=0,0,0,0,0

    return weekly_weights_df, w_returns , weekly_metrics_df, weekly_rolling_metrics_dict,w_k

def set_limit_to_weights_sum(weights_df,weight_sum_lim):
    weights_sum = weights_df.sum(axis=1)
    weights_sum_is_low = weights_sum < weight_sum_lim
    weight_sum_factor = weight_sum_lim / weights_sum
    weights_df.where(weights_sum_is_low, weights_df.multiply(weight_sum_factor, axis='index'), inplace=True)

    return weights_df






