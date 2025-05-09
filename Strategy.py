# import libraries and functions
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from Markowitz_Vectorized import compute_optimized_markowitz_d_w_m
from utils import weighted_mean_of_dfs_dict, create_results_by_period_df,mean_positions
from MarkowitzWeights import MarkowitzWeights


class Strategy:
    """
    This class implements a portfolio optimization strategy based on the Markowitz model.

    Attributes:
        inputs:
            settings: A dictionary containing various settings for the strategy.
            st_tickers_returns: A DataFrame containing historical returns for the assets.
            indicators_dict: A dictionary containing technical indicators.
        outputs:
            weights_df: A DataFrame containing the calculated portfolio weights.
            positions: A DataFrame containing the positions to be taken.
            strategy_returns: A Series containing the strategy returns.

    Methods:
        calculate_markowitz_weights: Calculates the optimal portfolio weights using the Markowitz model.
        apply_strategy_weights: Applies additional strategy weights (e.g., RSI-based) to the Markowitz weights.
        calculate_returns: Calculates the strategy returns based on the final positions and ticker returns.
    """
    def __init__(self,settings,st_tickers_returns,indicators_dict):

        if (not settings['mkwtz_scipy']) & (not settings['mkwtz_vectorized']) :
            print(" Error at settings: Any Optimize Strategy must be selected 'mkwtz_scipy' or/and 'mkwtz_vectorized' ")
            return

        if settings['mkwtz_scipy']:

            #Get Porfolio Allocation Weights for this slice of tickers_returns. Mean of diferent parameters like (rebalance periods, lookback,..)
            self.s_weights_df=self.PortfolioWeightsMarkowitzScipy(st_tickers_returns,indicators_dict,settings)
            self.weights_df = self.s_weights_df.copy()

        if settings['mkwtz_vectorized']:

            #Get Markowitz_Vectorized weights
            v_settings=settings.copy()
            v_settings['apply_strategy_weights']=False
            self.v_weights_df, _, _, _, _, _ = compute_optimized_markowitz_d_w_m(st_tickers_returns, v_settings)
            self.weights_df = self.v_weights_df.copy()

        if settings['mkwtz_vectorized'] and settings['mkwtz_scipy']:

            #Make mean
            self.weights_df= mean_positions(self.s_weights_df,self.v_weights_df,settings['w_upper_lim'],
                                         sf = 1.0 ,vf = 1.0 ,overall_f = 1.0)

        if False:
            from Regresion_of_Features import Regresion_of_Features
            reg_feat_list = ['rsi_high_keep']  # ,'rsi_high'
            reg_feat_dict = {key: value for key, value in indicators_dict.items() if key in reg_feat_list}
            reg_ret=self.weights_df*st_tickers_returns
            regr = Regresion_of_Features(reg_feat_dict, reg_ret, settings['volatility_target'])
            regr.results
            print('[x0,x1,...] for weight= x0 + x1 * feat1 + ...',regr.results.x)
            print('opt_fun',regr.results.fun)

        if settings['apply_strategy_weights']:
            # Apply RSI and other additional Strategy Weights on top of Markowitz weights
            positions=self.ApplyStrategyWeights(self.weights_df,indicators_dict['comb_weights'])
        else:
            positions=self.weights_df.copy()

        self.positions = positions.copy()

        self.strategy_returns=self.Returns(self.positions,st_tickers_returns)




    def PortfolioWeightsMarkowitzScipy(self,st_tickers_returns,indicators_dict,settings):

        #Compute Markowitz for diferent windows and save instance to a dict
        self.tsw_dict = {
            str(w): RollingMarkowitzWeights(w, p, settings['volatility_target'], st_tickers_returns,indicators_dict, settings)
            for w, p in zip(settings['mkwtz_ws'], settings['mkwtz_ps'])
        }

        def get_weights_df(tsw_dict, mkwtz_mean_fs, apply_opt_fun_predict_factor):

            if apply_opt_fun_predict_factor:

                # Get opt_fun_df with columns for each mktz window
                fun_df = pd.concat([tsw_dict[k].opt_fun_df for k in tsw_dict.keys()], axis=1)
                fun_df.columns = list(tsw_dict.keys())

                def get_fun_corr_factor(fun_df, w):

                    """Get Optimize Function factor when Prediction is good"""

                    # AutoCorrelation beetween current Optimize Function and previous values
                    fun_autocorrel = fun_df.rolling(window=w).corr(fun_df.shift(1))

                    #Factor calculation
                    fun_autocorrel_factor = fun_autocorrel.fillna(0).clip(upper=1, lower=-1) #limited values -1 to +1
                    fun_autocorrel_factor  = fun_autocorrel_factor.where(fun_df < 0, 1) #Keep value to 1 when portfolio fun_df is positive

                    # Factor making the mean and normalizing
                    fun_corr_factor = fun_autocorrel_factor.rolling(w).mean()
                    fun_corr_factor = fun_corr_factor / fun_corr_factor.rolling(250 * 4).mean().fillna(0.8).mean().mean()
                    fun_corr_factor = fun_corr_factor.fillna(1).clip(upper=1.2,lower=0.8) #limited values 0.8 to 1.2

                    return fun_corr_factor


                # Get Optimize Function factor when Prediction is good
                opt_fun_predict_factor = get_fun_corr_factor(fun_df, 20)

                # Apply opt_fun Predictibity factor
                weights_dict = {w: tsw_dict[w].weights_df.multiply(opt_fun_predict_factor[w], axis='index') for w in tsw_dict.keys()}

            else:

                weights_dict = {w: tsw_dict[w].weights_df for w in tsw_dict.keys()}
                fun_df = None
                opt_fun_predict_factor = None

            # Make weightged Mean
            overall_f=1.25 #
            weights_df = weighted_mean_of_dfs_dict(weights_dict, mkwtz_mean_fs) * overall_f

            return weights_df, opt_fun_predict_factor, fun_df, weights_dict

        #Get weightged Mean with opt_fun Predictibity factor
        self.weights_df, self.opt_fun_predict_factor, self.opt_fun_df,self.weights_by_period_dict= get_weights_df(self.tsw_dict,settings['mkwtz_mean_fs'],settings['apply_opt_fun_predict_factor'])

        return self.weights_df

    def ApplyStrategyWeights(self, weights_df , ind_weights):

        # Combined Weights
        w = ind_weights

        # Reindex as  weights_df index
        w = w[w.index.isin(weights_df.index)]

        # Apply Pre Optimization with Combined Weights
        positions=w * weights_df

        return positions


    def Returns(self,st_positions,st_tickers_returns):
        self.st_strategy_returns_by_ticker= st_tickers_returns * st_positions
        return self.st_strategy_returns_by_ticker.sum(axis=1)


class RollingMarkowitzWeights:
    """
    Compute MarkowitzWeigths to get Dayly Weights pd.Dataframe with columns= Tickers Names

    """
    def __init__(self,lookback,rebalance_p,volatility_target,tickers_returns,indicators_dict,settings): #Add rolling_cagr, rolling_cov_matices for this lookback
        self.rebalance_p = rebalance_p
        self.lookback=int(lookback)
        self.volatility_target=volatility_target
        self.settings=settings
        self.tickers_returns=tickers_returns
        self.tickers=tickers_returns.columns
        self.size=len(self.tickers)
        self.indicators_dict=indicators_dict



        self.compute_RollingMarkowitzWeights()



    def compute_RollingMarkowitzWeights(self):
        """
        Calculates time series of optimal weights and optimization function values.

        This function iterates through periods defined by the `rebalance_p` frequency,
        calculates optimal weights for each period using the Markowitz model, and
        upsamples the results to daily frequency.

        Attributes:
            self.rebalance_p: Rebalance frequency (e.g., 'W-FRI' for weekly Fridays,'M').
            self.lookback: Lookback window for calculating weights.  (eg. int 44,180,360 )
            self.weights_res_df: DataFrame containing weights with start, end dates,
                                 optimization function value, and weights for each asset.
            self.train_analytics_res_df: (Optional) DataFrame containing additional
                                          training period analytics (to be implemented).
            self.tickers_returns: DataFrame containing historical asset returns.

            self.volatility_target: Target volatility for the portfolio.
            self.settings: Additional settings for the Markowitz model.
            self.indicators_dict: (Optional) Dictionary of technical indicators.
            self.weights_df: DataFrame containing daily weights for each asset.
            self.opt_fun_df: DataFrame containing daily optimization function values.
            self.size: Number of assets.
        """

        #Create df to store results
        results_by_period_df = create_results_by_period_df(self.tickers_returns,self.rebalance_p,self.lookback)

        #Get data Features for this lookback: rolling_cagr, rolling_cov_matrices


        #weights_prev = np.zeros(self.size)
        #x0 = np.ones([self.size, 1]) / self.size * 0.1
        x0 = np.ones(self.size) / self.size * 0.1

        def get_results_by_loop(results_by_period_df, x0):

            for index,row in results_by_period_df.iterrows():
                slice_tickers_returns = self.tickers_returns.loc[row['start']:row['end']]

                #Calculate Slice Weights
                mw = MarkowitzWeights(slice_tickers_returns, self.volatility_target, self.settings, x0)
                results_by_period_df.loc[index,['opt_fun'] + self.tickers.tolist() ] = [mw.results.fun] + list(mw.results.x)

            return results_by_period_df

        def get_results_vectorized(results_by_period_df, x0):

            def calculate_weights_for_slice(slice_df):
                mw = MarkowitzWeights(slice_df, self.volatility_target, self.settings, x0)
                return pd.Series([mw.results.fun] + list(mw.results.x), index=['opt_fun'] + self.tickers.tolist())

            # Apply the function to each slice and assign the results to the DataFrame
            results_by_period_df.loc[:, ['opt_fun'] + self.tickers.tolist()] = results_by_period_df.apply(
                lambda row: calculate_weights_for_slice(self.tickers_returns.loc[row['start']:row['end']]), axis=1)

            return results_by_period_df


        #results_by_period_df = get_results_vectorized(results_by_period_df, x0)
        results_by_period_df = get_results_by_loop(results_by_period_df, x0)

        #Drop Duplicates
        results_by_period_df.drop_duplicates(subset='end', inplace=True)

        # Upsample to daily index and fill missing values
        results_by_period_df.set_index('end', inplace=True)
        results_dayly_df = results_by_period_df.reindex(self.tickers_returns.index).shift(1).fillna(method='ffill')

        # Separate weight and opt_fun DataFrames
        self.weights_df = results_dayly_df[self.tickers]
        self.opt_fun_df = results_dayly_df[['opt_fun']]



