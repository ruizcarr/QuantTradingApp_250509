import pandas as pd
import numpy as np

import quantstats
import webbrowser

class AfterTestOptimization:

    def __init__(self,test_positions,test_returns,mean_w,over_mean_pct,lookback,up_f,dn_f,plotting):

        cum_returns = (1+test_returns).cumprod()

        #Get after_test_factor
        #self.mean_crossed_up_factor, self.plot_df=self.get_mean_crossed_up_factor(cum_returns, mean_w, lookback, up_f, dn_f, plotting)
        #self.mean_rebound_factor, self.plot_df = self.get_mean_rebound_factor(cum_returns, mean_w,over_mean_pct, lookback, up_f, dn_f, plotting)
        self.sigmoid_ret_factor, self.plot_df = self.get_sigmoid_ret_opt_factor(cum_returns, plotting)

        self.after_test_factor = self.sigmoid_ret_factor

        #Apply after_test_factor  to positions and returns
        self.after_test_positions=test_positions.mul(self.after_test_factor,axis=0)

        if plotting:

            self.after_test_returns = test_returns * self.after_test_factor

            plot_df = self.plot_df
            plot_df['after_test_ret'] = (1 + self.after_test_returns).cumprod()
            plot_df.plot(title='Test Returns')

            # Quanstats Report
            if True:
                q_title = 'Study After Test Optimization'
                path = "results\\"
                q_filename = path + q_title + '.html'
                q_returns = self.after_test_returns
                q_benchmark = test_returns
                quantstats.reports.html(q_returns, benchmark=q_benchmark, output=True, download_filename=q_filename, title=q_title)
                webbrowser.open(q_filename)


    def get_sigmoid_ret_opt_factor(self,cum_returns, plotting=True):

        returns = cum_returns.pct_change()
        sigmoid_returns = 1 / (1 + np.exp(-1000 * returns.shift(1))) # returns values 0 to 1.
        sigmoid_returns_mean = sigmoid_returns.rolling(10).mean()
        opt_factor = (0.5 + sigmoid_returns_mean) ** 8  # returns values around 1
        opt_factor = opt_factor - opt_factor.rolling(250).mean() + 1  # Set mean == 1

        opt_factor = opt_factor.clip(upper=1.20, lower=1) # Limit Upper, Lower values

        sigmoid_ret_opt_factor = opt_factor.fillna(1)  # Set values to 1 the first year




        if plotting:
            plot_df = pd.DataFrame()
            plot_df['cum_returns'] = cum_returns
            plot_df['sigmoid_ret_opt_factor'] = sigmoid_ret_opt_factor*10

        else:
            plot_df = None

        return sigmoid_ret_opt_factor,plot_df


    def get_mean_crossed_up_factor(self,cum_returns, mean_w=22 * 3, lookback=22 * 3, up_f=1.4, dn_f=0.6, plotting=True):
        """
            Calculates a factor based on recent mean cross-up events and current trend.

            Args:
                cum_returns: A pandas Series of cumulative returns.
                mean_w: Window size for calculating the rolling mean.
                lookback: Lookback period for identifying recent mean cross-ups.
                up_f: Factor to apply when the mean is crossed up recently and the current trend is up.
                dn_f: Factor to apply when the mean is not crossed up recently or the current trend is down.
                plotting: Boolean flag to control whether to generate a plot.

            Returns:
                A tuple containing:
                    - mean_crossed_up_factor: A pandas Series of factors.
                    - plot_df: A pandas DataFrame for plotting (if plotting is True).
            """
        returns_mean = cum_returns.rolling(mean_w).mean()
        is_over_the_mean = cum_returns > returns_mean
        mean_crossed_up = (is_over_the_mean * 1).diff().eq(1)
        mean_crossed_up_recently = (mean_crossed_up * 1).rolling(lookback).max()
        mean_crossed_up_recently = (mean_crossed_up_recently * is_over_the_mean).eq(1)
        mean_crossed_up_recently = mean_crossed_up_recently.shift(1)
        mean_crossed_up_factor = np.where(mean_crossed_up_recently, up_f, dn_f)

        if plotting:
            plot_df = pd.DataFrame()
            plot_df['cum_returns'] = cum_returns
            plot_df['returns_mean'] = returns_mean
            # plot_df['mean_crossed_up_recently'] = mean_crossed_up_recently * 10
            plot_df['mean_crossed_up_factor'] = mean_crossed_up_factor * 10
            # plot_df['mean_crossed_up_ret'] = (1 + cum_returns.pct_change() * mean_crossed_up_factor).cumprod()

        else:
            plot_df = None

        return mean_crossed_up_factor, plot_df


    def get_mean_rebound_factor(self,cum_returns, mean_w=22 * 3,over_mean_pct=0.04, lookback=22 * 3, up_f=1.4, dn_f=0.6, plotting=True):
        """
            Calculates a factor based on recent mean cross-up events and current trend.

            Args:
                cum_returns: A pandas Series of cumulative returns.
                mean_w: Window size for calculating the rolling mean.
                lookback: Lookback period for identifying recent mean cross-ups.
                up_f: Factor to apply when the mean is crossed up recently and the current trend is up.
                dn_f: Factor to apply when the mean is not crossed up recently or the current trend is down.
                plotting: Boolean flag to control whether to generate a plot.

            Returns:
                A tuple containing:
                    - mean_crossed_up_factor: A pandas Series of factors.
                    - plot_df: A pandas DataFrame for plotting (if plotting is True).
            """
        returns_mean = cum_returns.rolling(mean_w).mean()
        returns_mean_pct=1 - returns_mean/cum_returns
        is_over_the_mean = returns_mean_pct>over_mean_pct
        mean_crossed_up = (is_over_the_mean * 1).diff().eq(1)
        mean_crossed_up_recently = (mean_crossed_up * 1).rolling(lookback).max()
        mean_crossed_up_recently = (mean_crossed_up_recently * is_over_the_mean).eq(1)
        mean_crossed_up_recently = mean_crossed_up_recently.shift(1)
        mean_crossed_up_factor = np.where(mean_crossed_up_recently, up_f, dn_f)

        if plotting:
            plot_df = pd.DataFrame()
            plot_df['cum_returns'] = cum_returns
            plot_df['returns_mean'] = returns_mean
            # plot_df['mean_crossed_up_recently'] = mean_crossed_up_recently * 10
            plot_df['mean_crossed_up_factor'] = mean_crossed_up_factor * 10
            # plot_df['mean_crossed_up_ret'] = (1 + cum_returns.pct_change() * mean_crossed_up_factor).cumprod()

        else:
            plot_df = None

        return mean_crossed_up_factor, plot_df