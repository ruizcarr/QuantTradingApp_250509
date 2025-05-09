import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from datetime import date
from datetime import timedelta

from utils import weighted_mean_of_dfs_dict
#from namespace import Namespace

from Backtest_Vectorized import compute_backtest_vectorized
import Market_Data_Feed as mdf

np.random.seed(1234)

# Wider print limits
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# Silence warnings
import warnings
warnings.filterwarnings('ignore')
warnings.warn = lambda *a, **kw: False


def main():

    # Import Settings
    from config import settings
    settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

    # Update Main Settings
    #settings['start'] = '1996-01-01'  # '1996-01-01' #'2019-01-01'
    settings['end'] = (date.today() + timedelta(days=1)).isoformat()
    settings['offline'] = False
    settings['qstats'] = True
    settings['do_BT'] = True
    #settings['startcash'] = 43000  # 52000 # '2016-01-01'
    # settings['tickers_bounds'] = {'ES=F': (0,0.5), 'NQ=F': (0,0.6), 'GC=F': (0.00,0.4), 'CL=F': (0,0.15), 'EURUSD=X': (0,0.0)} # {'ES=F': (0,0.5), 'NQ=F': (-0,0.6), 'GC=F': (0.00,0.5), 'CL=F': (0,0.25), 'EURUSD=X': (-0.00,0.0)}
    #settings['apply_strategy_weights'] = True

    # Get Data

    times={}
    start_t = time.time()

    data=mdf.Data(settings['contango'],settings['tickers'],settings['start'],settings['end'],settings['add_days'])
    tickers_returns = data.tickers_returns
    print("tickers_closes\n", data.tickers_closes[:-5].tail(5))

    end_t1 = time.time()
    times['get_data, indicators, params']=round(end_t1-start_t,2)

    #Compute Markowitz for dayly, Weekly & Monthly
    weights_df, metrics_df, rolling_metrics_dict, weekly_metrics_df, monthly_metrics_df,markowitz_metrics_dicts=\
        compute_optimized_markowitz_d_w_m(tickers_returns, settings)

    end_t2 = time.time()
    times['Markowitz'] = round(end_t2 - end_t1, 2)

    #Cash Backtest
    if settings['do_BT']:
        #Compute Backtest
        bt_log_dict,log_history = compute_backtest_vectorized(weights_df,settings,data.data_dict)

        end_t3 = time.time()
        times['Backtest'] = round(end_t3 - end_t2, 2)

        #Get Backtest Results from log_dict
        pos, portfolio_value_eur, portfolio_value_usd = [bt_log_dict[key] for key in ['pos', 'portfolio_value_eur','portfolio_value']]
        backtest_returns=portfolio_value_eur.pct_change()
        #Get Backtest Results Metrics
        metrics_backtest,rolling_metrics_df=get_returns_metrics(backtest_returns,'dayly')
        metrics_df['backtest']=metrics_backtest.T

        #print('pos \n', pos)
        #print('backtest_returns \n', backtest_returns)
        print('pos\n', pos.tail(10))
        print('log_history\n', log_history.tail(40))
        print('end portfolio_value_eur', round(portfolio_value_eur.iloc[-1], 2))

    #Prints
    print('weights_df\n', weights_df.tail(10))
    print('metrics_df\n',metrics_df)

    print('weekly_metrics_df\n',weekly_metrics_df)
    #print('monthly_metrics_df\n', monthly_metrics_df)

    print(times)

    #Plots
    weights_df.plot(title='weights_df')
    pos.plot(title='pos')


    def opt_fun_min_study(markowitz_metrics_dicts):
        cagr_w='250' #str(settings['cagr_w'][0])
        opt_fun=markowitz_metrics_dicts[cagr_w]['opt_fun_xs'][-len(pos):,:]
        xs=markowitz_metrics_dicts[cagr_w]['xs'][-len(pos):,:]
        string_xs =  xs.astype(str)
        opt_fun=pd.DataFrame(opt_fun,columns=string_xs ,index=pos.index).iloc[:,[-300,-200,-100,-50,-1]]  #-10:]
        opt_fun['opt_fun_min']=pd.DataFrame(markowitz_metrics_dicts[cagr_w]['opt_fun_min']).iloc[-len(pos):]['opt_fun']

        print(opt_fun['opt_fun_min'].describe())
        plt.figure()
        opt_fun['opt_fun_min'].plot(title='opt_fun,  cagr_w=' + cagr_w)
        #opt_fun.plot(title='opt_fun,  cagr_w='+cagr_w)

        #opt_fun_factor
        opt_fun_min=opt_fun['opt_fun_min']


        opt_fun_min_factor=get_opt_fun_min_factor(opt_fun_min)

        print_metrics(opt_fun_min_factor)

    #opt_fun_min_study(markowitz_metrics_dicts)

    #Study Predictive capacity of opt_fun
    def opt_fun_predictive_capacity_study(markowitz_metrics_dicts):
        cagr_w='250'
        opt_fun_xs=markowitz_metrics_dicts[cagr_w]['opt_fun_xs'][-len(pos):,:]
        opt_fun_predition_accuracy=get_opt_fun_predition_accuracy(opt_fun_xs)


    #Opt Fun Normalization Study
    if False:
        cagr_w = '250'
        print(markowitz_metrics_dicts[cagr_w].keys())
        for key in ['risk_xs', 'cagr_xs']: #'volatility_xs_norm', 'volat_xs_std_norm','ddn_xs_std_norm'
            #key='ddn_xs_std'
            (pd.DataFrame(markowitz_metrics_dicts[cagr_w][key])).plot(title=key) #.loc[:, 0:10]

    plt.show()



#FUNCTIONS

def compute_optimized_markowitz_d_w_m(tickers_returns, settings):

    # Weights Combination xs
    tickers_bounds = {ticker: settings['tickers_bounds'][ticker] for ticker in tickers_returns.columns}

    weight_sum_lim = settings['exposition_lim']

    # Generate Fix weights Combinations to find the best
    xs = generate_xs_combinations(tickers_bounds, weight_sum_lim, step=0.10)

    # Compute Dayly Markowitz Looping  over Selected Parameter
    dayly_weights_df, d_returns, returns_p,metrics_df, rolling_metrics_dict,markowitz_metrics_dicts = compute_markowitz_loop_over_ps(tickers_returns,xs, settings,strat_period='dayly')

    #Compute Weekly Markowitz
    weekly_weights_df, w_returns, weekly_metrics_df, weekly_rolling_metrics_dict,w_k=compute_weekly_markowitz(tickers_returns,xs, settings,strat_period='weekly')

    #Compute Monthly Markowitz
    monthly_weights_df, m_returns, monthly_metrics_df, monthly_rolling_metrics_dict,m_k = compute_monthly_markowitz(tickers_returns,xs, settings,strat_period='monthly')

    #Weighed Mean Dayly / Weekly / Monthly

    mean=False

    if mean:

        #Mean Weights
        d_k,w_k,m_k = settings['mean_weights_d_w_m']
        weights_df=(dayly_weights_df*d_k+weekly_weights_df*w_k+monthly_weights_df*m_k)/(d_k+w_k+m_k)

    else:
        # Apply Markowitz to get optimal Startegy
        #Get strat_periods
        strat_periods =settings['strat_periods']
        #Create d_w_m_returns_df with n_strats columns
        d_w_m_returns_df=pd.DataFrame({period: df for period,df in zip(strat_periods,[d_returns,w_returns,m_returns])})
        # Concat d_w_m_ weights array (n_strats,n_days,n_tickers)
        d_w_m_weights = np.array([np.array(df) for period, df in zip(strat_periods, [dayly_weights_df, weekly_weights_df, weekly_weights_df])])

        volat_target=settings['volatility_target']
        weight_lim=0.8
        weight_sum_lim = 1.2
        cagr_w=250*4
        strat_period = 'dayly'

        weights_comb_array, weights_by_strategy_df = get_combined_strategy_by_markowitz(d_w_m_returns_df, d_w_m_weights, volat_target,weight_lim,weight_sum_lim,cagr_w, strat_period)

        weights_df = pd.DataFrame(weights_comb_array, index=dayly_weights_df.index, columns=dayly_weights_df.columns, dtype=float)

        #Make fast mean for smooth  curve
        w=6
        weights_df=weights_df.rolling(w).mean().fillna(0)


        #Plot debug
        #weights_by_strategy_df = weights_by_strategy_df.rolling(w).mean().fillna(0)
        #plot_df=weights_by_strategy_df.copy()
        #plot_df['sum']=plot_df.sum(axis=1)
        #plot_df.plot(title='weights_by_strategy')

    #print('weights_df',weights_df )


    #Get Metrics for Mean D/W Returns
    strategy_returns, metrics, rolling_metrics = get_strategy_metrics(weights_df, tickers_returns, 'dayly')
    metrics_df['d_w_m']=metrics.T
    rolling_metrics_dict['d_w_m']=rolling_metrics

    # Optimization by Utility Up factor
    weights_df, metrics_df, rolling_metrics_dict, returns_p = compute_utility_factor(settings['apply_utility_factor'],tickers_returns, weights_df, metrics_df, rolling_metrics_dict, returns_p,weight_sum_lim,'dayly')

    # Apply Strategy Weights
    weights_df, metrics_df, rolling_metrics_dict,_ = apply_strategy_weights(tickers_returns, settings,weights_df,metrics_df,rolling_metrics_dict,weight_sum_lim,'dayly')

    return weights_df, metrics_df, rolling_metrics_dict, weekly_metrics_df, monthly_metrics_df,markowitz_metrics_dicts

def compute_optimized_markowitz(tickers_returns, settings):
    # Compute Markowitz Looping  over Selected Parameter
    weights_df, returns_p, metrics_df, rolling_metrics_dict = compute_markowitz_loop_over_ps(tickers_returns, settings)

    # Optimization by Utility Up factor
    weights_df, metrics_df, rolling_metrics_dict, returns_p = compute_utility_factor(settings['apply_utility_factor'],tickers_returns, weights_df, metrics_df, rolling_metrics_dict, returns_p)

    # Apply Strategy Weights
    weights_df, metrics_df, rolling_metrics_dict,_ = apply_strategy_weights(tickers_returns, settings,weights_df,metrics_df,rolling_metrics_dict)

    return weights_df, metrics_df, rolling_metrics_dict,returns_p


def compute_markowitz_loop_over_ps(tickers_returns,xs,settings,strat_period='dayly'):

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

    elif strat_period=='monthly':
        cov_w = settings['cov_w_monthly']
        ps = settings[str(settings['param_to_loop']+'_monthly')]
        year=12

    #Get Markowitz Metrics

    # Substract Contango from tickers_returns
    contango_list = [settings['contango'][ticker] for ticker in tickers_returns.columns]
    tickers_returns=tickers_returns-np.array(contango_list )/100/252

    markowitz_metrics_dict=compute_markowitz_cov_metrics(tickers_returns,xs,cov_w,volat_target,strat_period)

    #Initialize df and dict to store loop resuts
    weights_p_df = pd.DataFrame(index=tickers_returns.index)
    returns_p=pd.DataFrame(index=tickers_returns.index)
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
        returns_w, m, rolling_metrics = get_strategy_metrics(weights_df,tickers_returns, strat_period)

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
    strat_returns, m, rolling_metrics = get_strategy_metrics(weights_comb_df, tickers_returns, strat_period)

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

        xs = generate_xs_combinations(p_bounds, weight_sum_lim, step=0.10)

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
    elif strat_period == 'monthly':year, week = 12, 12/53
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

    penalties_xs = high_volat_xs + low_volat_xs

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

    #Normalized Metrics Clip (0,1)
    #for key in list(dict.keys()):
    #    dict[key + "_norm"] = np.clip(dict[key], 0.0001, 1)

    return dict

def update_markowitz_cagr_metrics(cagr_w, dict,strat_period='dayly'):

    if strat_period=='dayly': year, week=252,5
    elif strat_period=='weekly': year, week =53,1
    elif strat_period == 'monthly': year, week = 12, 1
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
        dict['risk_xs'] = weighted_mean_of_dfs_dict(dfs_dict, weights_list)
        dict['risk_xs_norm'] = dict['risk_xs']

        dict['opt_fun_xs'] = dict['risk_xs_norm'] - dict['cagr_xs_norm'] + dict['penalties_xs_norm']


    else:

        dict['risk_xs'] = dict['volatility_xs'] * 0.5 + dict['ddn_xs_std']*1.5 + dict['volat_xs_std'] * 0.15
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

    # opt_fun_xs values for opt_fun_min_idx_xs
    opt_fun_min = np.take_along_axis(opt_fun_xs, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten()

    #Compare with previous values and keep if not relevant improvement
    #opt_fun_min_idx_xs, opt_fun_min=compare_min_vs_previous(opt_fun_xs,opt_fun_min_idx_xs,opt_fun_min,keep_treshold=0.01)

    # Get weights of this minimum
    weights = xs[opt_fun_min_idx_xs]

    #Apply opt_fun_min_factor inv proportional to value
    #opt_fun_factor = get_opt_fun_min_factor(opt_fun_min)
    #weights=weights*opt_fun_factor

    #Set weights to zero when opt_fun_min is high
    #weights = set_weights_to_zero_when_opt_fun_min_is_high(opt_fun_min,weights)

    # Save weights as df
    weights_df = pd.DataFrame(weights, index=returns.index, columns=returns.columns)

    # Save Metrics
    if metrics:

        # Metrics of this mimimum
        metrics_opt_df = pd.DataFrame(index=returns.index)
        metrics_opt_df['volat'] = np.take_along_axis(volatility_xs, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten()
        metrics_opt_df['ret'] = np.take_along_axis(cagr_xs, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten()
        metrics_opt_df['opt_fun'] = opt_fun_min

        # All xs Metrics
        metrics_xs_dict = {'volatility_xs': volatility_xs, 'cagr_xs': cagr_xs}

    else:
        metrics_xs_dict = None
        metrics_opt_df = None

    return weights_df, metrics_opt_df, metrics_xs_dict

def get_opt_fun_min_factor(opt_fun_min):
    opt_fun_min_factor= log_normalize(-opt_fun_min)
    #opt_fun_min_factor = minmax_normalize(-opt_fun_min)
    opt_fun_min_factor = mean_std_normalize(opt_fun_min_factor, n=2)
    #opt_fun_min_factor = mean_std_normalize(-opt_fun_min,n=1.5) #array (n_days,)
    opt_fun_min_factor = opt_fun_min_factor / np.nanmean(opt_fun_min_factor) #set mean=1
    opt_fun_min_factor =np.expand_dims(opt_fun_min_factor, axis=1) #array (n_days,1)
    opt_fun_min_factor = opt_fun_min_factor * 0.9 #0.9 #Multiply by fix factor
    opt_fun_min_factor = np.clip(opt_fun_min_factor,0,2) #Limit max, min values
    return opt_fun_min_factor

def set_weights_to_zero_when_opt_fun_min_is_high(opt_fun_min,weights):
    opt_fun_min_mean=np.nanmean(opt_fun_min)
    opt_fun_min_std = np.nanstd(opt_fun_min)
    opt_fun_upper_lim =opt_fun_min_mean+opt_fun_min_std/2
    opt_fun_min_is_high=opt_fun_min>opt_fun_upper_lim #-0.04
    weights[opt_fun_min_is_high] = weights[opt_fun_min_is_high] * 0 # / 4
    return weights

def compute_mkwtz_vectorized_local_ltd_diff(markowitz_data_dict, metrics=True):
    """
    Get Portfolio Weights for each day that minimize Function to minimize.
    Function Sample: Volatility - CAGR

    Args in markowitz_data_dict:
        returns: dataframe timeseries (n_days, n_tickers)
        xs: array (n_combination,n_tickers) of all possible combinations of n elements within specified bounds
        opt_fun_xs: array(n_days,n_combination) of function to minimize for all xs

    Returns:
        weights: selected x array (n_days,n_tickers) from xs for each day that minimize opt_fun

        Limited to max_diff betwhen current and previous weights array values (eg. max_diff=0.1

    """

    # Get Data, Metrics and Function from dict
    returns,xs,volatility_xs, cagr_xs, opt_fun_xs = \
        [markowitz_data_dict[k] for k in
         ['returns','xs','volatility_xs', 'cagr_xs', 'opt_fun_xs']]

    #Initiaize weights values
    weights=np.zeros((len(returns),len(returns.columns)))

    #Limit xs - weights to a max_diff
    max_diff=0.3

    for i in range(3):
        # Calculate the element-wise difference
        # Repeat weights along the second axis to match xs
        weights_repeated = np.repeat(weights[:, np.newaxis, :], xs.shape[0], axis=1)
        diffs = weights_repeated - xs
        diff_is_high=np.any(np.abs(diffs) > max_diff, axis=2)
        #Asign high value to opt_fun where diff is high
        opt_fun_xs_ltd=np.where(diff_is_high,1,opt_fun_xs)


        # Localize index of xs where opt_fun is minimum
        opt_fun_min_idx_xs = np.argmin(opt_fun_xs_ltd, axis=1)

        # opt_fun_xs values for opt_fun_min_idx_xs
        opt_fun_min = np.take_along_axis(opt_fun_xs_ltd, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten()

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
        metrics_opt_df['opt_fun'] = opt_fun_min

        # All xs Metrics
        metrics_xs_dict = {'volatility_xs': volatility_xs, 'cagr_xs': cagr_xs}

    else:
        metrics_xs_dict = None
        metrics_opt_df = None

    return weights_df, metrics_opt_df, metrics_xs_dict

def compare_min_vs_previous(opt_fun_xs,opt_fun_min_idx_xs,opt_fun_min,keep_treshold=0.05):
    # opt_fun keeping previous weights
    prev_opt_fun_min_idx_xs = np.roll(opt_fun_min_idx_xs, 1)
    prev_opt_fun_min_idx_xs[0] =0
    opt_fun_min_prev_x=np.take_along_axis(opt_fun_xs, np.expand_dims(prev_opt_fun_min_idx_xs, axis=-1), axis=1).flatten()

    #Compare and keep previous when difference is not relevant
    #opt_fun_min_diff=opt_fun_min/opt_fun_min_prev_x-1
    opt_fun_min_diff = opt_fun_min - opt_fun_min_prev_x
    print_metrics(opt_fun_min_diff)
    keep_treshold=np.nanstd(opt_fun_min_diff)/1000
    print('keep_treshold',keep_treshold)
    opt_fun_not_relevant_diff=opt_fun_min_diff<-keep_treshold
    print('opt_fun_not_relevant_diff pct',np.sum(opt_fun_not_relevant_diff)/opt_fun_not_relevant_diff.shape[0])

    opt_fun_sel_idx_xs = np.where(opt_fun_not_relevant_diff, prev_opt_fun_min_idx_xs, opt_fun_min_idx_xs)

    #Update metrics
    opt_fun_sel = np.take_along_axis(opt_fun_xs, np.expand_dims(opt_fun_sel_idx_xs, axis=-1), axis=1).flatten()

    return opt_fun_sel_idx_xs, opt_fun_sel

def compromise_diff_to_best_trading_cost(opt_fun_xs, opt_fun_min_idx_xs, opt_fun_min, xs):
    # Compare opt_fun values for this weights vs keeping previous weights
    # opt_fun_sel_idx_xs, opt_fun_sel = compare_min_vs_previous(opt_fun_xs,opt_fun_min_idx_xs,opt_fun_min,keep_treshold=-.05)

    # Get indexes with Top opt_fun
    # Get difference from each opt_fun_xs - opt_fun_min
    opt_fun_xs_diff_to_min = opt_fun_xs - opt_fun_min.reshape(opt_fun_xs.shape[0], 1)

    for i in range(100):
        # Get Trading Cost idem to Change of xs
        # Previous opt_fun_min_idx_xs
        prev_opt_fun_min_idx_xs = np.roll(opt_fun_min_idx_xs, 1)
        prev_opt_fun_min_idx_xs[0] = 0
        # Previous x_opt_fun_min
        prev_x_opt_fun_min = xs[prev_opt_fun_min_idx_xs]
        # Repeat `x_opt_fun_min` along the second axis to match the shape of `xs`
        x_opt_fun_min_repeated = np.repeat(prev_x_opt_fun_min[:, np.newaxis, :], xs.shape[0], axis=1)
        # Calculate the element-wise differences
        diff_xs_to_x_opt_fun_min = np.abs(np.linalg.norm(xs - x_opt_fun_min_repeated, axis=2))

        def normalized(array):
            return (array - np.nanmin(array)) / np.nanmean(array)

        # Keep the compromised indexes with best combined opt_fun + trading_cost
        opt_fun_xs_diff_to_min = np.clip(opt_fun_xs_diff_to_min, -1, 0.1)
        diff_to_best = normalized(opt_fun_xs_diff_to_min)
        trading_cost = normalized(diff_xs_to_x_opt_fun_min)
        compr_fun = 4 * diff_to_best + trading_cost
        # compr_fun = opt_fun_xs_diff_to_min + diff_xs_to_x_opt_fun_min

        # Localize index of xs where compr_fun is minimum
        opt_fun_min_idx_xs = np.argmin(compr_fun, axis=1)

        return opt_fun_min_idx_xs

def get_returns_metrics(returns,strat_period='dayly'):

    if strat_period=='dayly': year,month,week=252,22,5
    elif strat_period=='weekly': year,month,week=53,4,1
    elif strat_period == 'monthly': year, month,week = 12, 1,1
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

def apply_strategy_weights(tickers_returns, settings,weights_df,metrics_df,rolling_metrics_dict,weight_sum_lim,strat_period='dayly'):
    if settings['apply_strategy_weights']:
        ind = mdf.Indicators(tickers_returns, settings)
        indicators_dict = ind.indicators_dict
        strategy_weights = indicators_dict['comb_weights']
    else:
        strategy_weights = 1
    weights_df =weights_df * strategy_weights

    # Set Limit to Weights Sum
    weights_df=set_limit_to_weights_sum(weights_df,weight_sum_lim)


    #Update Metrics
    strategy_returns = get_strategy_returns(weights_df, tickers_returns)
    metrics,rolling_metrics = get_returns_metrics(strategy_returns,strat_period)
    metrics_df['strategy']=metrics.T
    rolling_metrics_dict['strategy']=rolling_metrics

    return weights_df,metrics_df,rolling_metrics_dict,strategy_weights


def get_log_history_NOK(bt_log_dict):
    log_history = pd.DataFrame()

def multiticker_df_to_log_df_nok(df):
    df_stack = df.stack().reset_index(name='col')  # Stack the dataframe to long format
    df_stack = df_stack.loc[~df_stack['col'].isin(['None', np.nan])].reset_index(drop=True)  # Remove 'None'
    log_df = pd.DataFrame()
    log_df[['date', 'ticker', 'col']] = df_stack[['level_0', 'level_1', 'col']]
    return log_df

def multiticker_df_to_log_df(df):
    """Converts a DataFrame with instrument columns (ES=F, NQ=F, etc.) to a long format DataFrame.

    Args:
        df: A pandas DataFrame with instrument columns as data and dates as index.

    Returns:
        A pandas DataFrame with columns 'date', 'ticker', and 'col' in long format.
    """
    # Replace 'None' by nan

    # Stack with explicit dropna for 'col' column
    df_stack = df.stack().reset_index(name='col').dropna(subset=['col']).query("col != 'None'")

    # Assign column names directly
    log_df = df_stack[['level_0', 'level_1', 'col']]

    # Rename columns for clarity (optional)
    log_df.columns = ['date', 'ticker', 'col']

    # Set index to range starting from 0
    log_df.index = range(len(log_df))

    return log_df

    log_history[['date', 'ticker', 'B_S']] = multiticker_df_to_log_df(bt_log_dict['order_dict']['B_S'])
    for key in ['exectype', 'status']:
        log_history[key] = multiticker_df_to_log_df(bt_log_dict['order_dict'][key]).iloc[:, -1]

    print(log_history)

    key = 'price'
    df = bt_log_dict['order_dict'][key]
    print(df)

    # Assuming 'date' is the common column for merging
    df_with_date = df.reset_index().rename(columns={'index': 'date'})  # Rename former index
    merged_df = log_history.merge(df_with_date, how='left', on='date')
    ticker_mask = merged_df['ticker'].isin(df.columns)  # Create a boolean mask
    merged_df['price'] = merged_df[ticker_mask][df.columns].values  # Select matching prices
    merged_df.loc[~ticker_mask, 'price'] = np.nan  # Fill non-matching with NaN (optional)
    merged_df.drop(columns=df.columns, inplace=True)  # Optional: Remove 'ticker' if not needed

    print(merged_df)

def compute_mkwtz_vectorized_local_w_trading_cost(markowitz_data_dict, metrics=True):

    #Get Metrics from dict
    returns,xs,volatility_xs,ddn_xs_std,volat_xs_std,cagr_xs,hold_cost_xs, penalties_xs=\
        [markowitz_data_dict[k] for k in
         ['returns','xs','volatility_xs','ddn_xs_std','volat_xs_std','cagr_xs','hold_cost_xs','penalties_xs']]

    # Get Function to minimize
    risk_xs=volatility_xs/2 + ddn_xs_std + volat_xs_std
    opt_fun_xs = risk_xs - cagr_xs  + penalties_xs #+ hold_cost_xs

    #Initialize trading cost
    trading_cost_xs=np.zeros(opt_fun_xs.shape)

    #Cum Returns
    cum_ret_xs=np.array((1+pd.DataFrame(returns)).cumprod())

    def get_opt_weights(opt_fun_xs,trading_cost_xs,cum_ret_xs):

        #Update opt_fun_xs with trading_cost
        opt_fun_xs=opt_fun_xs + trading_cost_xs

        # Localize minimum
        opt_fun_min_idx_xs = np.argmin(opt_fun_xs, axis=1)

        # Get weights of this minimum
        weights = xs[opt_fun_min_idx_xs]

        # Save weights as df
        weights_df = pd.DataFrame(weights, index=returns.index, columns=returns.columns)

        def get_trading_cost(weights,gamma_trade=1/500):
        #Weights difference
            #weights_diff=(weights_df-weights_df.shift(1)).abs().sum(axis=1)
            # Reshape xs using tile
            xs_res = np.tile(xs, (weights.shape[0], 1, 1))
            #Previous Weights
            prev_weights=np.roll(weights, -1, axis=1)
            # Reshape weights
            prev_weights_res = prev_weights.reshape(-1, 1, weights.shape[1])
            #get difference
            weights_diff = np.absolute(xs_res-prev_weights_res)

            #Trading Cost

            # Reshape cum_ret_xs to add a new dimension of 1 at the end
            reshaped_cum_ret_xs = cum_ret_xs[:, np.newaxis, :]
            trading_size=weights_diff*reshaped_cum_ret_xs
            trading_cost_xs=np.sum(gamma_trade*trading_size,axis=-1)

            return trading_cost_xs

        trading_cost_xs = get_trading_cost(weights,gamma_trade=1/500)

        # Save Metrics
        if metrics:

            # Metrics of this mimimum
            metrics_opt_df = pd.DataFrame(index=returns.index)
            metrics_opt_df['volat'] = np.take_along_axis(volatility_xs, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten()
            metrics_opt_df['ret'] = np.take_along_axis(cagr_xs, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten()
            metrics_opt_df['opt_fun'] = np.take_along_axis(opt_fun_xs, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten()

            # All xs Metrics
            metrics_xs_dict = {'volatility_xs': volatility_xs, 'cagr_xs': cagr_xs}

        else:
            metrics_xs_dict = None
            metrics_opt_df = None

        return weights_df,trading_cost_xs,opt_fun_min_idx_xs, metrics_opt_df, metrics_xs_dict

    #Loop to get weights df
    error_w,i=1,0
    weights_df = pd.DataFrame(index=returns.index, columns=returns.columns).fillna(0)
    #for i in range(20):
    while (error_w>0) & (i<10):
        prev_weights_df=weights_df
        weights_df, trading_cost_xs, opt_fun_min_idx_xs, metrics_opt_df, metrics_xs_dict = get_opt_weights(opt_fun_xs,trading_cost_xs,cum_ret_xs)

        error_w=abs(weights_df-prev_weights_df).sum().sum()
        i += 1

        #trading_cost_sum=np.sum(np.take_along_axis(trading_cost_xs, np.expand_dims(opt_fun_min_idx_xs, axis=-1), axis=1).flatten())
        print(i,error_w)

    return weights_df, metrics_opt_df, metrics_xs_dict

def keep_bmrk_when_volat_is_low (tickers_returns, settings,weights_df,returns_p,metrics_df,rolling_metrics_dict,volatility_tresh=0.11):
    settings['tickers_bounds']= {'ES=F': (0,1), 'NQ=F': (0,1), 'GC=F': (0.00,0.2), 'CL=F': (0,0.0), 'EURUSD=X': (0,0.0)} # {'ES=F': (0,0.5), 'NQ=F': (-0,0.5), 'GC=F': (0.00,0.5), 'CL=F': (0,0.25), 'EURUSD=X': (-0.00,0.0)}
    weights_es_df, returns_es_p, _, _ = compute_markowitz_loop_over_ps(tickers_returns, settings)
    volat_es=tickers_returns['ES=F'].rolling(44,min_periods=10).std()*16
    volat_es_is_low=volat_es.shift(1)<volatility_tresh
    k=1.25
    weights_df.where(~volat_es_is_low,weights_es_df*k,inplace=True)
    returns_p.where(~volat_es_is_low, returns_es_p*k, inplace=True)
    # Strategy Results
    strategy_returns = get_strategy_returns(weights_df, tickers_returns)
    m, rolling_metrics_dict['low_vix_filter']=get_returns_metrics(strategy_returns)
    metrics_df['low_vix_filter']=m.T

    return weights_df,returns_p, metrics_df, rolling_metrics_dict

def plot_rolling_metrics_dict(rolling_metrics_dict):
    key = list(rolling_metrics_dict.keys())[-1]
    plot_df = rolling_metrics_dict[key][['cum_ret','rolling_utility','utility_factor']].shift(1)
    plot_df['utility_factor'] *= 5
    plot_df .plot(title='utility_factor')

def end_of_week_data(df):
    """
    Get Weekly Last Data available on Friday or inmediatelly available before if Friday Holiday
    :param df:
    :return: weekly_df
    """
    # Create Calendar Dates
    start = df.index[0]
    end = df.index[-1]
    calendar_dates=pd.date_range(start, end)

    # Create date of last data available
    date_of_last_data = df.index.to_series().reindex(calendar_dates).fillna(method='ffill')

    #Keep Fridays date of last data available
    weekly_date_of_last_data = date_of_last_data.resample('W-FRI').last()

    #Get data for this dates
    weekly_df=df.loc[weekly_date_of_last_data,:]

    #Drop duplicates
    non_duplicate_indices = ~weekly_df.index.duplicated(keep='first')  # Change 'first' to 'last' if needed
    weekly_df = weekly_df.loc[non_duplicate_indices]

    #Drop rows where NaN
    weekly_df=weekly_df.dropna()

    return weekly_df

def end_of_month_data(df):
    """
    Get Month Last Data available
    :param df:
    :return: monthly_df
    """
    # Create Calendar Dates
    start = df.index[0]
    end = df.index[-1]
    calendar_dates=pd.date_range(start, end)

    # Create date of last data available
    date_of_last_data = df.index.to_series().reindex(calendar_dates).fillna(method='ffill')

    #Keep Fridays date of last data available
    monthly_date_of_last_data = date_of_last_data.resample('M').last()

    #Get data for this dates
    monthly_df=df.loc[monthly_date_of_last_data,:]

    #Drop duplicates
    non_duplicate_indices = ~monthly_df.index.duplicated(keep='first')  # Change 'first' to 'last' if needed
    monthly_df = monthly_df.loc[non_duplicate_indices]

    #Drop rows where NaN
    monthly_df=monthly_df.dropna()

    return monthly_df


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

def compute_weekly_markowitz_ok(data,xs, settings,strat_period):

    if 'weekly' in settings['strat_periods']:

        tickers_returns = data.tickers_returns

        #Get weekly data considering fridays holidays
        #weekly_returns = end_of_week_data(data.tickers_closes).pct_change().dropna()
        weekly_returns = tickers_returns.resample('W-FRI').sum()

        # Compute Weekly Markowitz Looping  over Selected Parameter
        weekly_weights_df, w_returns, weekly_returns_p , weekly_metrics_df, weekly_rolling_metrics_dict,w_markowitz_metrics_dicts= compute_markowitz_loop_over_ps(weekly_returns,xs, settings,strat_period)

        # Upsample to dayly with values of previous Friday
        weekly_weights_df = weekly_weights_df.reindex(tickers_returns.index).shift(1).fillna(method='ffill').fillna(0)

        #Get Metrics for Monthly Strategy Upsample to dayly
        w_returns , metrics, rolling_metrics = get_strategy_metrics(weekly_weights_df, tickers_returns, 'dayly')
        weekly_metrics_df['weekly']=metrics.T
        weekly_rolling_metrics_dict['weekly']=rolling_metrics

        #Set weekly multiplicator at mean weight
        w_k=1.25 #max(weekly_metrics_df.loc['sharpe', 'weekly'] - bs,0)*weekly_metrics_df.loc['sharpe_ddn', 'weekly']

    else:
        weekly_weights_df, w_returns ,weekly_metrics_df, weekly_rolling_metrics_dict,w_k=0,0,0,0,0

    return weekly_weights_df, w_returns , weekly_metrics_df, weekly_rolling_metrics_dict,w_k

def compute_weekly_markowitz(tickers_returns,xs, settings,strat_period):

    if 'weekly' in settings['strat_periods']:

        #Get weekly data considering fridays holidays
        weekly_returns = tickers_returns.resample('W-FRI').sum()

        # Compute Weekly Markowitz Looping  over Selected Parameter
        weekly_weights_df, w_returns, weekly_returns_p , weekly_metrics_df, weekly_rolling_metrics_dict,w_markowitz_metrics_dicts= compute_markowitz_loop_over_ps(weekly_returns,xs, settings,strat_period)

        # Upsample to dayly with values of previous Friday
        weekly_weights_df = weekly_weights_df.reindex(tickers_returns.index).shift(1).fillna(method='ffill').fillna(0)

        #Get Metrics for Monthly Strategy Upsample to dayly
        w_returns , metrics, rolling_metrics = get_strategy_metrics(weekly_weights_df, tickers_returns, 'dayly')
        weekly_metrics_df['weekly']=metrics.T
        weekly_rolling_metrics_dict['weekly']=rolling_metrics

        #Set weekly multiplicator at mean weight
        w_k=1.25 #max(weekly_metrics_df.loc['sharpe', 'weekly'] - bs,0)*weekly_metrics_df.loc['sharpe_ddn', 'weekly']

    else:
        weekly_weights_df, w_returns ,weekly_metrics_df, weekly_rolling_metrics_dict,w_k=0,0,0,0,0

    return weekly_weights_df, w_returns , weekly_metrics_df, weekly_rolling_metrics_dict,w_k

def compute_monthly_markowitz_ok(data,xs, settings,strat_period):

    if 'monthly' in settings['strat_periods']:

        tickers_returns = data.tickers_returns
        # Get weekly data considering fridays holidays
        monthly_returns = end_of_month_data(data.tickers_closes).pct_change().dropna()

        # Compute monthly Markowitz Looping  over Selected Parameter
        monthly_weights_df, m_returns, monthly_returns_p, monthly_metrics_df, monthly_rolling_metrics_dict, m_markowitz_metrics_dicts = compute_markowitz_loop_over_ps(monthly_returns,xs, settings,strat_period)

        # Upsample to dayly with values of previous Friday
        monthly_weights_df = monthly_weights_df.reindex(tickers_returns.index).shift(1).fillna(method='ffill').fillna(0)

        # Get Metrics for Monthly Strategy Upsample to dayly
        m_returns, metrics, rolling_metrics = get_strategy_metrics(monthly_weights_df, tickers_returns, 'dayly')
        monthly_metrics_df['monthly'] = metrics.T
        monthly_rolling_metrics_dict['monthly'] = rolling_metrics

        #Monthly multiplicator at weights mean
        m_k = 0.3 #0.66  # max(monthly_metrics_df.loc['sharpe', 'monthly'] - bs,0)*monthly_metrics_df.loc['sharpe_ddn', 'monthly']

    else:
        monthly_weights_df, m_returns,monthly_metrics_df, monthly_rolling_metrics_dict, m_k = 0, 0, 0, 0,0

    return monthly_weights_df, m_returns, monthly_metrics_df, monthly_rolling_metrics_dict,m_k

def compute_monthly_markowitz(tickers_returns,xs, settings,strat_period):

    if 'monthly' in settings['strat_periods']:

        # Get weekly data considering fridays holidays
        monthly_returns = tickers_returns.resample('M').sum()

        # Compute monthly Markowitz Looping  over Selected Parameter
        monthly_weights_df, m_returns, monthly_returns_p, monthly_metrics_df, monthly_rolling_metrics_dict, m_markowitz_metrics_dicts = compute_markowitz_loop_over_ps(monthly_returns,xs, settings,strat_period)

        # Upsample to dayly with values of previous Friday
        monthly_weights_df = monthly_weights_df.reindex(tickers_returns.index).shift(1).fillna(method='ffill').fillna(0)

        # Get Metrics for Monthly Strategy Upsample to dayly
        m_returns, metrics, rolling_metrics = get_strategy_metrics(monthly_weights_df, tickers_returns, 'dayly')
        monthly_metrics_df['monthly'] = metrics.T
        monthly_rolling_metrics_dict['monthly'] = rolling_metrics

        #Monthly multiplicator at weights mean
        m_k = 0.3 #0.66  # max(monthly_metrics_df.loc['sharpe', 'monthly'] - bs,0)*monthly_metrics_df.loc['sharpe_ddn', 'monthly']

    else:
        monthly_weights_df, m_returns,monthly_metrics_df, monthly_rolling_metrics_dict, m_k = 0, 0, 0, 0,0

    return monthly_weights_df, m_returns, monthly_metrics_df, monthly_rolling_metrics_dict,m_k

def set_limit_to_weights_sum(weights_df,weight_sum_lim):
    weights_sum = weights_df.sum(axis=1)
    weights_sum_is_low = weights_sum < weight_sum_lim
    weight_sum_factor = weight_sum_lim / weights_sum
    weights_df.where(weights_sum_is_low, weights_df.multiply(weight_sum_factor, axis='index'), inplace=True)

    return weights_df

def get_rolling_min_expected(df, w=10):
    mean = df.rolling(w, min_periods=5).mean()
    std = df.rolling(w, min_periods=5).std()
    min_expected = mean - 2 * std
    return min_expected

def get_opt_fun_predition_accuracy(opt_fun_xs):
    """
    Prediction= Yesterday Value
    Error = abs(Actual Value-Prediction)
    Accuracy=1-Error/Error_Max  (n-days,n_xs_combinations) float(0 to 1)

    Input:
        opt_fun_xs: array(n-days,n_xs_combinations)

    Output:
        accuracy_opt_fun_xs:array(n-days,n_xs_combinations) float(0 to 1)
    """

    # convert to Dataframe  for easier data manipulation
    opt_fun_xs_df = pd.DataFrame(opt_fun_xs)

    # Calculate Error = abs(Actual Value-Prediction)
    error=opt_fun_xs_df.diff().abs()

    #Normalize Error by rolling max
    error_max = error.rolling(250,min_periods=1).max().max()
    error_norm=error/error_max # float (0 to 1)

    #Get Accuray
    accuracy= 1 - error_norm

    # Get Min Accuray Expected
    accuracy_expected =get_rolling_min_expected(accuracy,w=10)
    accuracy_expected = get_rolling_min_expected(accuracy_expected,w=10)
    accuracy_expected.clip(lower=0, inplace=True)
    accuracy_expected= accuracy_expected.fillna(0)

    #Plot Debug
    (accuracy.loc[:, 0:10]).plot(title='Accuracy')
    (accuracy_expected.loc[:, 0:10]).plot(title='Accuracy Expected')

    #Convert df to np.array
    accuracy_expected_array=np.array(accuracy_expected)

    return accuracy_expected_array


if __name__ == "__main__":
    main()