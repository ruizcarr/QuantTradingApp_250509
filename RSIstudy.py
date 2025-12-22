#IMPORT RELEVANT MODULES

#Import libraries and functions
import numpy as np,random
from quantstats.stats import volatility

from Backtest_Vectorized import plot_df

np.random.seed(42)
random.seed(42)
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

from Backtest_Vectorized_Class import compute_backtest_vectorized
#from Backtest_Vectorized import compute_backtest_vectorized
from Markowitz_Vectorized import compute_optimized_markowitz_d_w_m
from WalkForwardTraining import WalkForwardTraining
import Market_Data_Feed as mdf
from utils import mean_positions


#Get SETTINGS
from config import settings,utils
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

def run(settings):

    #Get Data & Indicators
    data_ind = mdf.Data_Ind_Feed(settings).data_ind
    data, indicators_dict = data_ind
    tickers_returns=data.tickers_returns
    cumret=(1+tickers_returns).cumprod()

    #rsi_fast=get_rsi(tickers_returns,len=14)
    #rsi_slow = get_rsi(tickers_returns, len=250)
    #rsi=rsi_slow*0.0+rsi_fast*1#
    #rsi = ((rsi_fast/100+0.50) * (rsi_slow/100+0.50) )*100-50
    #print(rsi)
    #rsi.plot()



    rsi_weight=get_rsi_curve_weight(tickers_returns)


    rsi_weight.plot()

    #Backtest
    rsi_returns=rsi_weight*tickers_returns
    rsi_cumret= (1+rsi_returns).cumprod()

    for ticker in tickers_returns.columns:
        plot_df=pd.DataFrame()
        plot_df['rsi_cumret']=rsi_cumret[ticker]
        plot_df['cumret'] = cumret[ticker]
        plot_df['rsi_weight'] = rsi_weight[ticker]
        plot_df.plot(title=ticker)

    #metrics
    bechmark_volat=tickers_returns.std()*16
    rsi_volat=rsi_returns.std()*16
    bechmark_cagr = tickers_returns.mean()*255
    rsi_cagr = rsi_returns.mean() * 255
    bechmark_sharpe=bechmark_cagr/bechmark_volat
    rsi_sharpe = rsi_cagr / rsi_volat
    print('rsi_sharpe', rsi_sharpe)
    print('bechmark_sharpe',bechmark_sharpe)

    plt.show()

def get_rsi_curve_weight(tickers_returns):

    rsi = get_rsi(tickers_returns, len=14)

    #Use yesterday values
    rsi_weight=rsi.shift(1)

    #Clip Low and High values
    low, high = 45, 55 #5 point bias uphill
    rsi_weight = rsi_weight.clip(low, high) # [low,high]

    #Normalize High | Low
    rsi_weight = (rsi_weight-low)/(high-low) # [0,1]

    # Normalize by slow rolling mean
    rsi_weight = rsi_weight / rsi_weight.rolling(6*252,min_periods=2*252).mean().clip(lower=0.0001)/2
    rsi_weight = rsi_weight.clip(upper=1.0)# [0,1]

    #Set Values bellow treshold to zero
    bottom_treshold=0.55
    rsi_weight = (rsi_weight-bottom_treshold).clip(lower=0) # [0,1-bottom_treshold]

    #Re-scale up to upper_treshold
    upper_treshold=1.6
    mult_factor=upper_treshold/(1-bottom_treshold)
    rsi_weight = rsi_weight*mult_factor # [0,upper_treshold]

    #Multiply by fix factor to enlarge lower values
    fix_factor=2.5
    rsi_weight = rsi_weight * fix_factor # [0,upper_treshold*fix_factor]

    #Make rolling mean to reduce noise
    rsi_weight = rsi_weight.rolling(10).mean()

    #Limit again to upper_treshold after Multiply by fix factor
    rsi_weight = rsi_weight.clip(0.0,upper_treshold) #value from -0 to 1.5

    #Set nan values at the begining to neutral value 1
    rsi_weight = rsi_weight.fillna(1)

    #Set ticker expceptions to 1
    tickers_exceptions=['GC=F','cash',]
    for ticker in tickers_exceptions:
        if ticker in tickers_returns.columns:
            rsi_weight[ticker] = 1

    return rsi_weight

def get_rsi(closes: pd.DataFrame, len: int = 14, returns: bool = True) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI) for multiple tickers
    using the 'ta' library.

    Args:
        closes: A pandas DataFrame where each column represents the closing
                prices or returns for a ticker. The index should be dates/times.
        len: The lookback period for the RSI calculation.
        returns: If True, assumes input 'closes' are returns and calculates
                 cumulative product to get a price series before calculating RSI.

    Returns:
        A pandas DataFrame with the RSI values for each ticker,
        with columns for the input tickers.
    """
    # Import the specific rsi function
    from ta.momentum import rsi

    # Handle the returns logic as in your original function
    if returns:
        # Calculate cumulative product assuming 'closes' are returns (e.g., daily % change + 1)
        # Note: .cumprod(axis=0) ensures product is taken down columns
        closes = (1 + closes).cumprod(axis=0)

    # Initialize an empty DataFrame to store the RSI results
    rsi_df = pd.DataFrame(index=closes.index, columns=closes.columns)

    # Calculate RSI for each ticker column
    for tick in closes.columns:
        # Use the ta.momentum.rsi function (note parameter name 'window' instead of 'length')
        rsi_df[tick] = rsi(close=closes[tick], window=len)

    return rsi_df

def get_sigmoid(df,center,k,max_val,low,high):
    z = 1 / (1 + np.exp(-k * (df - center)))
    z_min = 1 / (1 + np.exp(-k * (low - center)))
    z_max = 1 / (1 + np.exp(-k * (high - center)))
    return max_val * (z - z_min) / (z_max - z_min)

if __name__ == '__main__':
    run(settings)