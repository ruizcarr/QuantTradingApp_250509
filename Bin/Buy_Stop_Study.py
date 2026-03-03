import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Get SETTINGS
from config import settings,utils
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

#Update Settings
settings['start']='2015-01-01'

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

print(lows)

def compute_buy_volat_stop_price(closes, highs,delta=2):
    """
    Calculates the volatility-based floor for volat events.
    Uses 2-sigma of the (High-Close) 'upside wicks'.
    """
    # Upside noise: difference between intraday High and previous close
    # Use shift(1) to avoid look-ahead bias
    upside_noise = (highs-closes.shift(1)).shift(1)

    # delta-sigma buffer based on previous 20 days of downside wicks

    volat_buffer =upside_noise.rolling(20).mean() + delta * upside_noise.rolling(20).std()

    #volat_buffer = volat_buffer.clip(lower=0.0001*closes.shift(1), upper=0.12*closes.shift(1))

    # The floor level
    volat_stop_price = closes.shift(1) + volat_buffer

    #Keep lower value
    volat_stop_price = volat_stop_price.rolling(22).min().fillna(0)

    #keep real value not over open price
    #volat_stop_price =volat_stop_price.clip(upper=opens)

    return volat_stop_price

def get_buy_stop_price_as_backtest(highs):

     # Highs Downtrend --> Yesterday high < previous 5 days highest
    highs_max = highs.shift(1).rolling(5).max()

    # Get Buy Stop Price
    high_keep = highs_max.rolling(22).min()
    highs_std = highs.rolling(22).std().shift(1)
    buy_stop_price = high_keep + highs_std * 0.5

    return buy_stop_price

volat_stop_price=compute_buy_volat_stop_price(closes, highs,delta=2)

buy_stop_price=get_buy_stop_price_as_backtest(highs)


# A breach occurs when the high price is >= the stop price and previous day low is over
breach_condition = (highs >= volat_stop_price) & (highs < volat_stop_price).shift(1)

#Break Up Condition (re open position)
break_up_condition= (closes < volat_stop_price) & ((closes > volat_stop_price).shift(1) | breach_condition)


executed_stop_price=volat_stop_price[breach_condition]
re_open_price=closes[break_up_condition]

volat_stop_profit = (executed_stop_price.fillna(method='ffill') - re_open_price.fillna(method='ffill')).fillna(0)

cum_volat_stop_profit=volat_stop_profit.cumsum()
cum_volat_stop_profit.plot()


plot_df=pd.DataFrame()

import matplotlib.pyplot as plt

for ticker in lows.columns:
    # Create a fresh figure for each ticker
    fig, ax = plt.subplots(figsize=(12, 5))

    # 1. Map your data
    plot_df['close'] = closes[ticker]
    plot_df['highs'] = highs[ticker]
    plot_df['volat_stop_price'] = volat_stop_price[ticker]
    plot_df['buy_stop_price'] = buy_stop_price[ticker]

    # 2. Identify the Breach (The "Cross")
    # A Black volat breach occurs when the low price is <= the stop price
    breaches = plot_df[breach_condition[ticker]]
    breaks = plot_df[break_up_condition[ticker]]

    # 3. Plot the primary lines
    ax.plot(plot_df.index, plot_df['highs'], label='Daily High', color='royalblue', lw=1.5)
    ax.plot(plot_df.index, plot_df['close'], label='Daily Close', color='black', lw=1.5)
    ax.plot(plot_df.index, plot_df['volat_stop_price'], label='volat Stop', color='crimson', linestyle='--', alpha=0.8)
    ax.plot(plot_df.index, plot_df['buy_stop_price'], label='Buy Stop', color='red', linestyle='--', alpha=0.8)


    # 4. Highlight the "Crosses"
    if not breaches.empty:
        ax.scatter(breaches.index, breaches['volat_stop_price'],
                   color='green', marker='^', s=100, zorder=5, label='volat Breach')
        #ax.scatter(re_open_price.index, re_open_price[ticker],color='darkred', marker='v', s=100, zorder=5, label='Re-Open after Breach')

        # Optional: Shade the area where the price is below the stop
        #ax.fill_between(plot_df.index, plot_df['volat_stop_price'], plot_df['volat_stop_price'],where=breach_condition, color='red', alpha=0.2)

    # Formatting
    ax.set_title(f"Buy Stop Study: {ticker}", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()


plt.show()