# import libraries and functions
import numpy as np
import pandas as pd
#import pandas_ta as ta
import os.path

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
import datetime
from datetime import date
from datetime import timedelta
import yfinance as yf
import itertools
import quantstats as qs
# extend pandas functionality with mettickers, etc.
qs.extend_pandas()
# from arch import arch_model
#from sklearn.metrics import r2_score

from utils import limit_df_values_diff
from Strategy import Strategy

import webbrowser


# Wider print limits

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# Silence warnings
import warnings
warnings.filterwarnings('ignore')

class WalkForwardTraining:

    """"
    1.Select a window size and optimize the strategy on the first window.
    2.Train & fit the model using all the data available on or before the selected window.
    3.Once the model’s formed and parameters are established for the selected time period,
    generate the model outputs for all available data during the following time step.
    4.Save the prediction as part of a result set.
    5.Now move the window up so that all of the data through that window can be used for fitting
    and the data for the next time-step can be used for testing.
    6.Repeat steps (2) to (5) adding the new predictions to the result set when new data comes in.
    """

    def __init__(self,data_ind,settings):
        self.settings = settings
        data, indicators_dict = data_ind
        self.tickers_returns = data.tickers_returns
        self.tickers=self.tickers_returns.columns
        self.path_train = "trained_models/"
        self.path_test = self.path_train
        self.p_file = self.path_train + 'params_train.csv'
        self.tt_windows = self.TrainTestWindows(self.tickers_returns,wft_print=False)
        #self.params_train =self.get_params_train(data_ind,self.tt_windows, settings)

    def TrainTestWindows(self, tickers_returns, wft_print=True):
        train_len = self.settings['train_length']  # Lookback for training
        test_len = self.settings['test_length']  # Period 'W-FRI', 'M', 'Q', 'Y'
        train_len_min = self.settings['train_length_min']  # Minimum lookback for training

        tt_windows = pd.DataFrame()

        # Ensure datetime index
        tickers_returns.index = pd.to_datetime(tickers_returns.index)

        # Resample to test period and get end-of-train dates
        tickers_returns_res = tickers_returns.iloc[train_len_min:].resample(test_len).last()
        tt_windows['endoftrain'] = tickers_returns_res.index[:-1]

        # Compute start-of-train for each window
        tt_windows['startoftrain'] = [
            tickers_returns[:endoftrain].iloc[-train_len:].index[0]
            for endoftrain in tt_windows['endoftrain']
        ]

        # Compute start-of-test as the first date after end-of-train
        tt_windows['startoftest'] = pd.to_datetime([
            tickers_returns.index[tickers_returns.index > tt_windows['endoftrain'].iloc[i]][0]
            for i in range(len(tt_windows))
        ])

        # Ensure strictly increasing start-of-test to avoid duplicates
        tt_windows['startoftest'] = tt_windows['startoftest'].cummax()

        # End-of-test from resampled periods
        tt_windows['endoftest'] = tickers_returns_res.index[1:]

        # Keep proper column order
        tt_windows = tt_windows[['startoftrain', 'endoftrain', 'startoftest', 'endoftest']]

        if wft_print:
            print('\n WalkForwardTraining Windows')
            print('start', tickers_returns.index[0].date())
            print('end', tickers_returns.index[-1].date())
            print('train_length', train_len, 'train_len_min', train_len_min)
            print('test_length', test_len)
            print('Test Starts after:', tickers_returns.index[train_len_min].date())
            print(tt_windows)

        return tt_windows

    def get_params_train(self, data_ind, settings):

        params = settings['params']

        exist_params_to_optimize = (len(params) > 0)

        if exist_params_to_optimize:
            # Compute Parameters Training
            params_train =self.TrainingParamsOptimization(data_ind, settings)

        else:  # If not params to optimize: Initialize empty train variables
            params_train = self.tt_windows[['endoftrain']]

        if settings['verbose']:
            print('params_train\n', params_train)

        if settings['apply_post_opt']:

            # Get analytics for this optimized parameters
            raw_analytics_train = self.Traininganalytics(params_train, data_ind, do_print=settings['verbose'])

            if settings['verbose']: print('analytics_train before post_opt raw_analytics_train\n', raw_analytics_train)

            # Apply Post Optimization
            params_train =self.PostOptimize(params_train, raw_analytics_train)
            if settings['verbose']: print('params_train after post_opt\n', params_train)

        return params_train

    def TrainingParamsOptimization(self,data_ind,settings):

        #Get Optimization Parameters from Settings
        params=settings['params']
        params_bounds=settings['params_bounds']

        #Get Data & Indicators
        data, indicators_dict = data_ind
        tickers_returns=data.tickers_returns

        # Create results dataframe for params values after Training for each window
        params_train = self.tt_windows[['startoftrain', 'endoftrain']]
        p_keys=list(params.keys())
        params_train[p_keys] = list(params.values())
        bounds = np.asarray([params_bounds[p] for p in p_keys])

        # Training over Training windows
        for window in params_train.index:
            # Get Window Data
            startoftrain = params_train.startoftrain[window]
            endoftrain = params_train.endoftrain[window]
            tickers_returns_window = tickers_returns[startoftrain:endoftrain]

            #Pre Optimization of Bounds limits
            #if settings['apply_PreOpt']: PreOpt(tickers_returns_window , settings)

            # Initialize x0 with Previous results
            x0 = np.array(params_train.iloc[max(0, window - 1), 2:])

            #MyParamsTunning
            # Define params space around x0
            pct = 0.05  # pct max around x0
            num = 3  # number of values = x0 value + (num-1) values  (3 5 7 to include x0)
            #if window == 0: pct, num = 0.2, 5  # Wider space at start not so restricted to default value

            window_best_params= \
                self.MyParamsTunning(tickers_returns_window, x0, settings,bounds, pct, num,p_keys,indicators_dict)

            # Save Optimized Parameters
            params_train.iloc[window, 2:] = window_best_params
            print('params_train\n',params_train.iloc[:window])


        #make rolling mean for smooth time serie
        params_train[p_keys]=params_train[p_keys].rolling(3,min_periods=1).mean()
        print('params_train mean\n', params_train)

        #Save params_train
        params_train = params_train.drop(columns=['startoftrain'])

        # Save Optimized Params to CSV
        params_train.to_csv(self.path_train+'params_train.csv', index=False)

        #Plot
        fig, ax = plt.subplots()
        ax.scatter(params_train.endoftrain,params_train.iloc[:,0], marker=".", cmap="viridis_r")
        plt.plot(params_train.endoftrain,params_train[p_keys])

        return params_train

    def MyParamsTunning(self, data, x0, settings,bounds, pct, num,p_keys,indicators_dict):

        # Define params space around x0 avoiding zero values (x0 min acording bounds values)
        step = np.maximum(x0, np.mean(bounds, axis=1) / 5)* pct
        p_space = np.linspace(np.sum([x0, -step], axis=0), np.sum([x0, step], axis=0), num).T

        # Limited Bounds
        p_space_lim = np.asarray([p_space[p][(p_space[p] >= bounds[p, 0]) &(p_space[p] <= bounds[p, 1])]for p in range(len(x0))])

        # Combination of parameters
        p_comb = np.asarray(list(itertools.product(*p_space_lim)))

        # Drop x0 to avoid double calculation
        p_comb = p_comb[[x.tolist() != x0.tolist() for x in p_comb]]

        # Get Scores of each param combination
        scores = np.asarray([params_score(p_comb[i], p_keys, data, settings,indicators_dict)for i in range(len(p_comb))])

        # Get Best Params of Min Score
        best_params = np.asarray(p_comb[scores.argmin()])
        min_score = scores[scores.argmin()]

        # score_x0 Score of previous
        # Check if significative improvement
        imp = 0.05
        score_x0 = params_score(x0, p_keys, data, settings,indicators_dict)
        if min_score > score_x0 * (1 - imp):  # if not significative improvement
            # Keep previous params
            best_params = x0
            min_score = score_x0

        if False:
            print('p_space', p_space)
            print('p_space limited', p_space_lim)
            # print('p_comb', p_comb)
            print('score', scores)

        return best_params #, min_score

    def Traininganalytics(self,params_train,data_ind,do_print=False): #_loop

        data, indicators_dict = data_ind
        opt_keys = params_train.columns[1:]
        slice_settings = self.settings.copy()
        # Initialize df for results storage
        analytics_train = self.tt_windows[['endoftrain']].copy()

        for j, row in self.tt_windows.iterrows():

            # Get updated settings
            slice_settings = {**slice_settings, **params_train.loc[row.name, opt_keys].to_dict()}

            #Get Slice tickers
            slice_tickers_returns = data.tickers_returns.loc[row['startoftrain']:row['endoftrain']]

            # Get Returns of Slice
            st_train =  Strategy(slice_settings, slice_tickers_returns, indicators_dict)
            slice_strategy_returns = st_train.strategy_returns

            # Get Slice analytics
            slice_annalytics = compute_returns_metrics(slice_strategy_returns).T.reset_index(drop=True)

            # Store slice values
            analytics_train.loc[j, slice_annalytics.columns] = slice_annalytics.loc[0,:]

        return analytics_train

    def Traininganalytics_Vectorized(self,params_train,data_ind,do_print=False): #_Vectorized (worst performance)

        data, indicators_dict = data_ind
        opt_keys = params_train.columns[1:]
        slice_settings = self.settings.copy()

        # Assuming data, params_train, and indicators_dict are defined
        def process_slice(row,slice_settings):
            # Get updated settings
            slice_settings = {**slice_settings, **params_train.loc[row.name, opt_keys].to_dict()}

            # Get slice tickers and returns
            slice_tickers_returns = data.tickers_returns.loc[row['startoftrain']:row['endoftrain']]

            # Create and run strategy
            st_train = Strategy(slice_settings, slice_tickers_returns, indicators_dict)
            slice_strategy_returns = st_train.strategy_returns

            # Compute analytics
            slice_annalytics = compute_returns_metrics(slice_strategy_returns).T.reset_index(drop=True).loc[0,:]

            return slice_annalytics

        # Vectorized application
        results = self.tt_windows.apply(lambda row: process_slice(row, slice_settings.copy()), axis=1)

        # Update analytics_train with the results
        analytics_train=pd.concat([self.tt_windows[['endoftrain']],results],axis=1)

        if do_print:
            print('analytics_train\n', analytics_train)

        #Save to Csv & self
        analytics_train.to_csv(self.path_train+'analytics_train.csv', index=False)

        return analytics_train

    def HighDdnFilter(self,do_print=False):
        #Filter high ddn
        analytics_train_csv = pd.read_csv(self.path_train+'raw_analytics_train.csv')
        params_train= pd.read_csv(self.path_train+'params_train.csv')
        if 'drawdown' in analytics_train_csv.columns:
            ddn_df=analytics_train_csv[['drawdown']]
            ddn_lim = self.settings['ddn_target'] - 0.03
            ddn=analytics_train_csv[['drawdown']]
            ddn_f=ddn_lim / ddn
            ddn_df['ddn_factor']=np.where(ddn>ddn_lim,ddn_f**1,ddn_f**0.5)
            #ddn_df['ddn_factor'] = (ddn_lim / ddn) **1
            # Save Value with Limited Exposition
            params_train['ddn_factor'] = ddn_df['ddn_factor'].clip(upper=self.settings['ddn_factor_lim'])

        else:
            params_train['ddn_factor']  =1


        # Save Optimized Params to CSV
        params_train.to_csv(self.path_train+'params_train.csv', index=False)

        if do_print:
            print('params_train optimized \n', params_train)

    def VolatilityAdjuster(self,analytics_train ,params_train,do_print=False):

        #Apply vol_factor to adjust Volatility to target

        if 'volatility' in analytics_train.columns:
            vol_lim = self.settings['volatility_target'] + self.settings['trading_volatility_delta']
            params_train['vol_factor'] = (vol_lim / (analytics_train[['volatility']]))

        else:
            params_train['vol__factor']  = 1


        for _ in range(2):

            #Set First Value Without enough data length to 1
            #params_train['vol_factor'].iloc[0]=1

            #Limit change of values to delta
            params_train['vol_factor'] = limit_df_values_diff(params_train['vol_factor'],delta=self.settings['volatility_factor_diff_delta']) #0.35

            #Set value to 1 when CAGR is Low
            cagr_volat_is_low=(analytics_train['cagr'] + analytics_train['volatility'])<0.01
            params_train['vol_factor'][cagr_volat_is_low]=1

            #Limit Upper Limit
            params_train['vol_factor']=params_train['vol_factor'].clip(upper=self.settings['vol_factor_max'])

        # Save Optimized Params to CSV
        params_train.to_csv(self.path_train+'params_train.csv', index=False)

        if do_print:
            print('params_train optimized \n', params_train)

        return params_train

    def KellyAdjuster(self,analytics ,params,do_print=False):

        #Kelly factor
        kelly_mean=analytics['kelly'].rolling(3,min_periods=1).mean()
        params['kelly_factor'] =np.where(kelly_mean<0.0,0,1)

        # Save Optimized Params to CSV
        params.to_csv(self.path_train+'params_train.csv', index=False)
        self.params_train=params

        if do_print:
            print('params_train optimized \n', params)

        return params




    def PostOptimize(self,params_train,analytics_train):

        params_train=self.VolatilityAdjuster(analytics_train ,params_train,do_print=False)


        params_train=self.KellyAdjuster(analytics_train ,params_train,do_print=False)

        params_train['post_factor']=params_train['vol_factor']*params_train['kelly_factor']

        # Save Optimized Params to CSV
        params_train.to_csv(self.path_train+'params_train.csv', index=False)

        return params_train



    def Test_by_ttwindows_nok(self,indicators_dict,params_train,do_annalytics=True):

        #print('params_train\n', params_train)
        opt_keys = params_train.columns[1:]
        slice_settings = self.settings.copy()

        # Initialize df for results storage
        test_returns=pd.DataFrame()
        test_weights = pd.DataFrame()
        test_analytics = pd.DataFrame()
        test_positions = pd.DataFrame()
        test_fun=pd.DataFrame()
        #Check params_train lentgh

        # For each Test Slice
        for j in range(len(self.tt_windows)):
            #Get complete settings dict with updated paramenters
            try:
                for key in opt_keys: slice_settings[key]=params_train[key].iloc[j]
            except:
                print("Error of Trainning Dates Bounds. Review 'start' and 'end' at settings")
                return

            #Get Slice tickers
            startoftest = self.tt_windows.startoftest[j]
            endoftest = self.tt_windows.endoftest[j]
            pre_start_len=250*2
            slice_len=len(self.tickers_returns[startoftest:endoftest])
            tickers_returns_slice_test = self.tickers_returns[:endoftest].iloc[-slice_len-pre_start_len:]

            #Get Weights & Returns of Slice
            st = Strategy(slice_settings, tickers_returns_slice_test,indicators_dict)
            strategy_returns=st.strategy_returns
            slice_strategy_returns=strategy_returns[startoftest:endoftest]
            slice_weights = st.weights_df[startoftest:endoftest]
            slice_positions = st.positions[startoftest:endoftest]
            if st.opt_fun_df is not None:
                slice_fun = st.opt_fun_df[startoftest:endoftest]
            else:
                slice_fun = slice_strategy_returns/slice_strategy_returns #load ones df

            #Concat to save all Weights & Returns
            test_returns=pd.concat([test_returns,slice_strategy_returns],axis=0)
            test_weights = pd.concat([test_weights, slice_weights], axis=0)
            test_positions = pd.concat([test_positions, slice_positions], axis=0)
            test_fun = pd.concat([test_fun, slice_fun], axis=0)



            if do_annalytics:

                # Get Slice analytics
                slice_annalytics = compute_returns_metrics(slice_strategy_returns).T.reset_index(drop=True)
                real_endoftest=pd.DataFrame(data=[slice_strategy_returns.index[-1].date()],columns=['endoftest'])
                slice_annalytics_df=pd.concat([real_endoftest,slice_annalytics],axis=1)

                #Concat to save all Weights & Returns
                test_analytics = pd.concat([test_analytics, slice_annalytics_df], axis=0,ignore_index=True)
                test_analytics[opt_keys]=params_train[opt_keys]
                self.test_analytics=test_analytics.copy()

        #Save test_returns
        test_returns.columns = ['returns']

        #Add datetime index to weights and positions
        test_weights=test_weights.reindex(test_returns.index)
        test_positions = test_positions.reindex(test_returns.index)
        test_fun = test_fun.reindex(test_returns.index)

         #Save test_returns & weights & positions to share
        self.test_returns=test_returns
        self.test_weights=test_weights
        #self.raw_test_positions = test_positions
        #self.test_positions = 0.4*test_positions+0.6*test_weights
        # Save optimize funtion
        #self.strategy_fun = test_fun


        if do_annalytics:

            print('Test analytics\n', test_analytics)

            #Get analytics Summary
            test_analytics_summary = compute_returns_metrics(test_returns['returns'])

            #Bechmark analytics
            benchmark_returns = self.tickers_returns.loc[test_weights.index].iloc[:, 0]
            bench_analytics = compute_returns_metrics(benchmark_returns)

            #Concat Test & Benchmark analytics
            test_analytics_summary=pd.concat([ test_analytics_summary,bench_analytics],axis=1)
            test_analytics_summary.columns=['system','benchmark']
            print('\nStart Test:',test_returns.index[0].date(),'End Test:',test_returns.index[-1].date())
            print('Test analytics Summary\n',test_analytics_summary)



    def Test(self,indicators_dict,params_train,do_annalytics=True):

        """
        This Test is only valid if  settings are fix over the time
        So, if no parameters optimized diferents than post_opt: vol_factor  kelly_factor  post_factor
        """
        if len(params_train.columns) >4:
            print(" -------   This Test is only valid if  settings are fix over the time ------- ")
            print("-------   Apply Test over tt_windows with updated slice_settings ------ ")
            self.Test_by_ttwindows(indicators_dict, params_train, do_annalytics=True)
            return

        #Upsample post_factor to dayly
        post_factor = pd.DataFrame({'post_factor': np.array(params_train['post_factor'])}, index=self.tt_windows['startoftest'])
        post_factor = post_factor.reindex(self.tickers_returns.index).fillna(method='ffill').dropna()

        startoftest = post_factor.index[1]
        endoftest = post_factor.index[-1]

        #Update Settings
        st_settings=self.settings
        st_settings[ 'trading_app_only']=True

        # Get Test Weights, Positions & Returns
        st = Strategy(st_settings, self.tickers_returns, indicators_dict)
        self.st=st

        test_weights = st.weights_df[startoftest:endoftest]
        test_positions = st.positions[startoftest:endoftest]
        test_returns = st.strategy_returns[startoftest:endoftest]

        print('test_positions',test_positions)

        if self.settings['apply_post_opt']:

            test_weights = test_weights.mul(post_factor['post_factor'] ,axis=0)
            test_positions = test_positions.mul( post_factor['post_factor'],axis=0)
            test_returns = test_returns.mul(post_factor['post_factor'],axis=0)

         #Save test_returns & weights & positions to share
        self.test_returns=test_returns
        self.test_weights=test_weights
        self.test_positions = test_positions

        # Save optimize function
        #self.strategy_fun = test_fun

        if do_annalytics:

            #Get analytics Summary
            test_analytics_summary = compute_returns_metrics(test_returns)

            #Bechmark analytics
            benchmark_returns = self.tickers_returns.loc[test_weights.index].iloc[:, 0]
            bench_analytics = compute_returns_metrics(benchmark_returns)

            #Concat Test & Benchmark analytics
            test_analytics_summary=pd.concat([ test_analytics_summary,bench_analytics],axis=1)
            test_analytics_summary.columns=['system','benchmark']

            print('\nStart Test:',test_returns.index[0].date(),'End Test:',test_returns.index[-1].date())
            print('Test analytics Summary\n',test_analytics_summary.T)



def check_params_train_csv(params_train_csv,tt_windows):
    # Check params_train_csv endoftrain date not expired
    endoftrain = datetime.datetime.strptime(params_train_csv['endoftrain'].iloc[-1], "%Y-%m-%d")
    train_expired = (tt_windows['endoftrain'].iloc[-1] > endoftrain)

    # Check params_train_csv start is OK
    len_params_train_csv_ok = (len(params_train_csv) >= len(tt_windows))

    params_train_csv_is_ok = not train_expired & len_params_train_csv_ok

    return params_train_csv_is_ok

def get_params_from_csv(p_file,tt_windows, settings):
    if os.path.isfile(p_file):
        params_from_csv = pd.read_csv(p_file)
        if check_params_train_csv(params_from_csv, tt_windows):
            if settings['verbose']:
                print('Using valid parameters from CSV')
            return params_from_csv.tail(len(tt_windows))
    else:
        print('Invalid parameters in CSV. Retraining is required')
        return None

def params_score(params_x, params_keys, data, settings,indicators_dict):
    # Create params dict from params_x np
    params_x_dict = {params_keys[i]: params_x[i] for i in range(len(params_x))}
    updated_settings = {**settings, **params_x_dict}
    st = Strategy(updated_settings, data,indicators_dict)
    st_returns = st.strategy_returns
    volatility = st_returns.std() * 16
    volat_lim = settings['volatility_target'] - 0.03
    score = (abs(volatility - volat_lim) / volat_lim) ** 0.5
    return score

def compute_returns_metrics(returns):

    cum_returns = (1 + returns).cumprod()
    pnl = cum_returns.iloc[-1] - 1
    cagr = qs.stats.cagr(returns)
    volatility = returns.std() * 16
    sharpe = qs.stats.sharpe(returns)
    drawdown = -qs.stats.max_drawdown(returns)
    calmar=cagr/drawdown
    kelly = (cagr / 252) / (volatility / 16)

    return round(pd.DataFrame(
        index=['pnl', 'cagr', 'volatility', 'sharpe', 'drawdown', 'calmar','kelly'],
        data=[pnl, cagr, volatility, sharpe, drawdown,calmar, kelly],
        columns=['analytics']),3)











