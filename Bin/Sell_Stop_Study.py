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

def compute_volat_thresholds(opens,closes, lows,delta=2):
    """
    Calculates the volatility-based floor for Black volat events.
    Uses 2-sigma of the (Close - Low) 'downside wicks'.
    """
    # Downside noise: difference between close and intraday low
    downside_noise = (closes.shift(1) - lows).shift(1)

    # 2-sigma buffer based on previous 20 days of downside wicks
    # Use shift(1) to avoid look-ahead bias
    volat_buffer =downside_noise.rolling(20).mean() + delta * downside_noise.rolling(20).std()

    volat_buffer = volat_buffer.clip(lower=0.0025*closes.shift(1), upper=0.12*closes.shift(1))

    # The floor level
    volat_stop_price = closes.shift(1) - volat_buffer

    #Keep higher value
    volat_stop_price = volat_stop_price.rolling(22).max().fillna(0)

    #keep real value not over open price
    #volat_stop_price =volat_stop_price.clip(upper=opens)

    return volat_stop_price

def compute_volat_thresholds_pct(closes, lows,delta=3):
    """
    Calculates the volatility-based floor for Black volat events.
    Uses 2-sigma of the (Yesterday Close - Low) 'downside wicks'.
    """
    # Downside noise: difference between close and intraday low
    downside_noise = (closes.shift(1) - lows).shift(1)
    downside_noise_pct = (1 - lows/closes.shift(1)).shift(1)

    # 2-sigma buffer based on previous 20 days of downside wicks
    # Use shift(1) to avoid look-ahead bias
    volat_buffer =downside_noise.rolling(20).mean() + delta * downside_noise.rolling(20).std()
    volat_buffer_pct = downside_noise_pct.rolling(20).mean() + delta * downside_noise_pct.rolling(20).std()


    volat_buffer_pct = volat_buffer_pct.rolling(200).mean()

    volat_buffer_pct = volat_buffer_pct.fillna(0)

    volat_buffer = volat_buffer.clip(lower=0.005*closes.shift(1), upper=0.08*closes.shift(1))
    volat_buffer_pct = volat_buffer_pct.clip(lower=0.005, upper=0.08)


    # The floor level
    volat_stop_price = closes.shift(1) - volat_buffer
    volat_stop_price = closes.shift(1)*(1-volat_buffer_pct)

    return volat_stop_price.fillna(0)

def get_sell_stop_price_as_backtest(lows):

    # Previous 5 days lowest
    lows_min = lows.shift(1).rolling(5).min()

    # Get Sell Stop Price
    sell_stop_price = lows_min.rolling(22).max()

    return sell_stop_price

volat_stop_price=compute_volat_thresholds(opens,closes, lows,delta=9)

print(volat_stop_price)

if False:
    # Lows Uptrend
    uptrend = closes.shift(1).ge(volat_stop_price, axis=0)*1
    uptrend=uptrend+0.25
    uptrend_returns = uptrend*tickers_returns
    uptrend_cumreturns =(1+uptrend_returns).cumprod()
    for ticker in uptrend.columns:
        plot_df1=pd.DataFrame()
        plot_df1['uptrend_cumreturns']=uptrend_cumreturns[ticker]
        plot_df1['cum_rets'] = cum_rets[ticker]
        plot_df1['uptrend'] = uptrend[ticker]
        plot_df1.plot(title=ticker)

#sell_stop_price=get_sell_stop_price_as_backtest(lows)

volat_stop_pct=volat_stop_price/closes.shift(1)-1
volat_stop_pct.plot(title='Volatility Stop Price pct')

# A breach occurs when the low price is <= the stop price and previous day low is over
breach_condition = (lows <= volat_stop_price) & (lows > volat_stop_price).shift(1)

#Break Up Condition (re open position)
break_up_condition= (closes > volat_stop_price) & ((closes < volat_stop_price).shift(1) | breach_condition)

in_the_market=(break_up_condition.cumsum() -breach_condition.cumsum()).clip(0,1)
in_the_market.plot()

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
    plot_df['low'] = lows[ticker]
    plot_df['volat_stop_price'] = volat_stop_price[ticker]
    #plot_df['sell_stop_price'] = sell_stop_price[ticker]

    # 2. Identify the Breach (The "Cross")
    # A Black volat breach occurs when the low price is <= the stop price
    breaches = plot_df[breach_condition[ticker]]
    breaks = plot_df[break_up_condition[ticker]]

    # 3. Plot the primary lines
    ax.plot(plot_df.index, plot_df['low'], label='Daily Low', color='royalblue', lw=1.5)
    ax.plot(plot_df.index, plot_df['close'], label='Daily Close', color='black', lw=1.5)
    ax.plot(plot_df.index, plot_df['volat_stop_price'], label='volat Stop', color='crimson', linestyle='--', alpha=0.8)
    #ax.plot(plot_df.index, plot_df['sell_stop_price'], label='Sell Stop', color='red', linestyle='--', alpha=0.8)


    # 4. Highlight the "Crosses"
    if not breaches.empty:
        ax.scatter(breaches.index, breaches['volat_stop_price'],
                   color='darkred', marker='v', s=100, zorder=5, label='volat Breach')
        #ax.scatter(re_open_price.index, re_open_price[ticker],color='green', marker='^', s=100, zorder=5, label='Re-Open after Breach')

        # Optional: Shade the area where the price is below the stop
        #ax.fill_between(plot_df.index, plot_df['volat_stop_price'], plot_df['volat_stop_price'],where=breach_condition, color='red', alpha=0.2)

    # Formatting
    ax.set_title(f"Black volat Study: {ticker}", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()


plt.show()