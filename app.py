import streamlit as st
import pandas as pd
import numpy as np
#from datetime import datetime
import datetime
from datetime import date
from datetime import timedelta
import time
import pytz
import altair as alt
import matplotlib.pyplot as plt
import json
import random
import subprocess
import sys
import os

from check_password import check_password
from Trading_Markowitz import compute,process_log_data
from config.trading_settings import settings
import Market_Data_Feed as mdf

#For Local Run bellow in the pycharm terminal
#streamlit run app.py
#Ctrl + c to stop
#Internet url: https://quanttradingapp.streamlit.app/



#Main Code
def main(settings):

    #if check_password():
    if True:

        # Initialize session_state values
        chart_len_dict = {'Weekly': 5 + 1, 'Monthly': 22 + 1, 'Quarterly': 3 * 22 + 1}
        if 'chart_len_key' not in st.session_state:
            st.session_state.chart_len_key = 'Quarterly'
        if 'daysback' not in st.session_state:
            #get_daysback()
            st.session_state.daysback = chart_len_dict[st.session_state.chart_len_key]
        if 'data_show' not in st.session_state:
            st.session_state.data_show = 'returns'
        if 'qstats' not in st.session_state:
            st.session_state.qstats = False

        # App Settings
        # Set the page layout to wide
        st.set_page_config(layout="wide", page_title='Quant Trading App')



        #Update Settings
        # Import Trading Settings
        settings['verbose']=False
        settings['qstats']=st.session_state.qstats
        settings['do_BT'] = True

        #Get Trading results
        #log_history, _, data = compute(settings)

        #Get tickers data
        #closes=data.tickers_closes
        #returns=data.tickers_returns

        @st.cache_data(ttl=3600)  # Cache for 1 hour (adjust as needed)
        def load_and_compute_data(settings):
            """
            Fetches raw data and performs computation, with results cached.
            """
            st.write("Fetching and computing data...")
            data_ind = mdf.Data_Ind_Feed(settings).data_ind
            data, _ = data_ind

            # Assuming compute processes the data_ind output
            log_history, _ = compute(settings, data_ind)

            # Return all necessary outputs that you want to cache together
            return data, log_history

        if st.button("Refresh App"):
            #st.session_state.data = None  # Clear the data
            # Clear the cache for the load_and_process_all_data function
            load_and_compute_data.clear()
            st.rerun()  # Rerun the app to call the function again and get fresh data

        # Load data & Compute
        data, log_history = load_and_compute_data(settings)
        returns = data.tickers_returns
        closes = data.tickers_closes


        #Debug
        #st.write(st.session_state)
        #st.write(settings['end'])
        #st.write(closes)
        #st.write(returns)

        #Process Log Data
        eod_log_history,trading_history=process_log_data(log_history,settings)

        #Returns by Ticker
        ret_by_ticker = returns[settings['tickers']] * eod_log_history[settings['tickers']]

        #Get today
        today = datetime.datetime.now().date()

        #Display Title & tickers data
        display_tickers_data(closes,returns,today,settings,sidebar=False,daysback=st.session_state.daysback,data_show=st.session_state.data_show,chart=False)

        st.divider()

        #Display Portfolio Positions
        last_trade_date=trading_history.index[trading_history.index.to_series().dt.date<=today][-1]
        display_portfolio_positions(eod_log_history,trading_history,last_trade_date,settings,ret_by_ticker,returns,daysback=st.session_state.daysback)

        #Display Orders
        display_orders(log_history,settings)

        st.divider()

        #Display Next Trading Forecast
        next_trades=trading_history.index[trading_history.index.to_series().dt.date>today]
        if len(next_trades)>0:
            next_trade_date=next_trades[0]
        else:
            next_trade_date = False
        with st.expander("See Next Trading Forecast:"):
            if not next_trade_date:
                st.write(f"**Keep Current Positions. No Trading Forecast within {settings['add_days']} Days**")
            else:
                display_portfolio_positions(eod_log_history,trading_history,next_trade_date,settings,ret_by_ticker,returns,forecast=True)

        #Display Current Portfolio Value
        display_portfolio_results(eod_log_history,today,settings,daysback=st.session_state.daysback)

        #Input Display Options
        with st.expander('Display Options:'):
            cols = st.columns(4)
            #Chart length
            #cols[0].selectbox('Chart Length:', chart_len_keys, key='chart_len_key',on_change=get_daysback)
            cols[1].selectbox('Data to Show:', ['returns', 'closes'],key='data_show')
            cols[2].checkbox('Show Annalytics:',value=None,  key='qstats')

        #Display Log History
        with st.expander("See Historical Log:"):
            log_history_sort=log_history.sort_values('date', ascending=False)
            log_history_sort
            closes_sort=closes.sort_index(ascending=False)
            closes_sort
            eod_log_history
            returns
            ret_by_ticker
            trading_history
            #log_history
            closes
            #lows

#Define Functions

def get_daysback(chart_len_dict):
    st.session_state.daysback = chart_len_dict[st.session_state.chart_len_key]

def display_portfolio_positions(eod_log_history,trading_history,date,settings,ret_by_ticker,returns,daysback=3*22+1,forecast=False):

    st.write(f"**Portfolio Positions:**")
    if not forecast:
        today = datetime.datetime.now().date()

    else:
        today=None

    #Get portfolio and trading of today
    last_portfolio = eod_log_history.loc[:today].iloc[-1][settings['tickers']]
    pre_portfolio = eod_log_history.loc[:today].iloc[-2][settings['tickers']]
    last_trade = last_portfolio-pre_portfolio

    #Position & Portfolio Value at end of Today in USD and EUR

    pos_value_today=eod_log_history.loc[:today,'pos_value'].iloc[-1]
    #porfolio_value_today = eod_log_history.loc[:today, 'portfolio_value'].iloc[-1]
    porfolio_value_today_eur = eod_log_history.loc[:today, 'portfolio_value_eur'].iloc[-1]

    exchange_rate=eod_log_history.loc[:today, 'exchange_rate'].iloc[-1]
    pos_value_today_eur=pos_value_today*exchange_rate
    exposition = pos_value_today_eur / porfolio_value_today_eur * 100

    #Display Current Portfolio
    n_col=len(settings["tickers"])+1
    #col_width_list=[2]+[1]*(n_col-1)
    col_width_list = [7] + [3] * (n_col - 1)
    cols=st.columns(col_width_list)

    with cols[0]:
        st.write("Tickers:")
        st.write("Nbr of Contracts:")
        st.write(f"Last Trade date: {date}")
        st.write(f"Position Value / Exposition @: {today}")
        #st.subheader(f"{pos_value_today:,.0f} USD /  {exposition:,.0f} %")
        st.subheader(f"{pos_value_today_eur:,.0f} € /  {exposition:,.0f} %")



    for i in range(1,n_col):
        j=i-1
        ticker=trading_history.columns[j]
        label=f"**{ticker}**"
        value=int(last_portfolio[j])
        delta=int(last_trade[j])
        cols[i].metric(label=label,value=value,delta=delta)

        #Chart weights evolution

        with cols[i]:
            w=daysback #3*22
            if daysback > 6:
                chart_ts_altair(eod_log_history.iloc[-w:].loc[:today], ticker)

            if not forecast:
                cum_ret_by_ticker = (1 + ret_by_ticker.iloc[-w - settings['add_days']:-settings['add_days']]).cumprod()
                #cum_ret_by_ticker = cum_ret_by_ticker.fillna(1)
                cum_ret = (1 + returns.iloc[-w - settings['add_days']:-settings['add_days']]).cumprod()
                alt_chart1=chart_ts_altair(cum_ret_by_ticker, ticker, st_altair_chart=False)
                alt_chart2 = chart_ts_altair(cum_ret,  ticker, color="grey", st_altair_chart=False)
                st.altair_chart(alt_chart1 + alt_chart2, use_container_width=True)

def display_portfolio_results(eod_log_history,today,settings,daysback=3*22):

    st.write(f"**Portfolio Results:**")

    #col_width_list = [2] + [1] * 4
    col_width_list = [3] + [2] * 4
    cols = st.columns(col_width_list)

    # Display Portfolio Value
    #portfolio_value=eod_log_history.loc[:today,"portfolio_value"].iloc[-1]
    portfolio_value_eur = eod_log_history.loc[:today, "portfolio_value_eur"].iloc[-1]
    ret=eod_log_history.loc[:today,"portfolio_return"].iloc[-1]
    with cols[0]:
        #st.metric(label=f"**Portfolio Value {today}**",value=f"{portfolio_value:,.0f} USD", delta=f"{ret:.1%}")
        st.metric(label=f"**Portfolio Value {today}**", value=f"{portfolio_value_eur:,.0f} €", delta=f"{ret:.1%}")

        #Chart Portfolio Value
        #chart_ts_altair(eod_log_history.iloc[-daysback-settings['add_days']:-settings['add_days']], "portfolio_value")
        chart_ts_altair(eod_log_history.iloc[-daysback - settings['add_days']:-settings['add_days']], "portfolio_value_eur")

    #Display CAGR
    keys=["cagr","weekly_return","monthly_return"]
    for i,key in enumerate(keys):
        cagr=eod_log_history.loc[:today,key].iloc[-1]
        #diff=f"{cagr*portfolio_value:,.0f}"
        diff_eur = f"{cagr * portfolio_value_eur:,.0f} €"
        #cols[i+1].metric(label=f"**{key}**",value=diff, delta=f"{cagr:.1%}")
        cols[i + 1].metric(label=f"**{key}**", value=diff_eur, delta=f"{cagr:.1%}")

    #Display DDN
    #ddn=eod_log_history.loc[:today,"ddn"].iloc[-1]
    ddn = eod_log_history.loc[:today, "ddn_eur"].iloc[-1]
    cols[4].metric(label="**Drawdown YTD**", delta="", value=f"{ddn:.1%}")

def chart_ts_altair(ts,col,color="blue",st_altair_chart=True):
    df=ts[col].rename_axis('date').reset_index()
    alt_chart=alt.Chart(df,height=120).mark_line(color=color).encode(
x=alt.X('date', title=''),
y=alt.Y(col, title='', scale=alt.Scale(domain=[ts[col].min(),ts[col].max()]))
)

    if st_altair_chart:
        st.altair_chart(alt_chart,use_container_width=True )

    return alt_chart


def display_tickers_data(closes,returns,today,settings,sidebar=False,daysback=3*22,data_show='returns',chart=True):

    tickers = settings["tickers"]
    #cols = st.columns(len(tickers)+1)
    n_col = len(tickers) + 1
    #col_width_list = [2] + [1] * (n_col - 1)
    col_width_list = [7] + [3] * (n_col - 1)
    cols = st.columns(col_width_list)


    # Get the current time
    tz = pytz.timezone('Europe/Madrid')
    now =datetime.datetime.now(tz)
    # Format the time as a string
    time_string = now.strftime('%H:%M:%S')
    market_data_head_1=f"**Maket Data: {today} {time_string}**"
    market_data_head_2 = f"(data with 15min delay)"


    def get_chart_data(data,daysback=5+1,data_show='returns'):

        # Display line chart for days len back
        data_ch = data.loc[:today].iloc[-daysback:]

        #Returns
        cum_ret_ch=(1+data_ch.pct_change()).cumprod().fillna(1)

        if data_show=='returns':
            chart_data=cum_ret_ch
        else:
            chart_data = data_ch

        return chart_data


    for i,ticker in enumerate(tickers):
        #close=closes.loc[today,ticker]
        close = closes.loc[:today, ticker].iloc[-1]
        if ticker=='EURUSD=X': close_f = f"{close:,.3f}"
        elif ticker == 'CL=F':close_f = f"{close:,.2f}"
        else: close_f=f"{close:,.0f}"
        ret = returns.loc[:today, ticker].iloc[-1]
        label=f"**{ticker}**"
        value=close_f
        delta=f"{ret:.1%}"

        if ticker =='cash':
            ret=ret*255
            delta = f"@ {ret:.1%} EURIBOR"

        if not sidebar:

            # display Market Data Header
            if i==0:
                cols[0].title('Quant Trading App')
                cols[0].write(market_data_head_1+market_data_head_2)


            # display Market Data & Small Chart
            cols[i + 1].metric(label, value, delta)

            if chart:
                with cols[i + 1]:
                    chart_ts_altair(chart_data, ticker)

        else:
            st.title('Quant Trading App')
           #display Market Data Header in Sidebar
            if i == 0:
                with st.sidebar:
                    st.write(market_data_head_1)
                    st.write(market_data_head_2)

                   #Chart Options
                    chart_data = get_chart_data(data=closes,daysback=daysback,data_show=data_show)


           # display Market Data & Small Chart in Sidebar
            scol1, scol2 = st.sidebar.columns([2, 3])
            scol1.metric(label, value, delta)
            if chart:
                with scol2:
                    chart_ts_altair(chart_data, ticker)

def display_orders_ok(log_history,settings):
    def display_orders_log(df, title):
        #cols[0].write(f"{title}")
        if len(df) > 0:
            cols[0].write(f"{title} {df['date'].iloc[0]} 00:00(CET)")
            for i, row in df.iterrows():
                order_log = f"**{row['ticker']} {row['exectype']} {row['B_S']}  {row['size']}**"
                if row['exectype'] == "Stop":
                    order_log = order_log + f" @ {row['price']}"

                col=df.columns.get_loc(row['ticker'])
                cols[col].write(order_log)

        else:
            cols[0].write(f"No {title}")

            for j in range(2):
                for i in range(1,len(cols)):
                    cols[i].write(" ")

    st.write(f"**Orders to Broker:**")

    # Columns for dispaly
    n_col = len(settings["tickers"]) + 1
    col_width_list = [2] + [1] * (n_col - 1)
    cols = st.columns(col_width_list)

    # Get Today SELL Stops Log
    orders_history = log_history[log_history['event'].str.contains('Order Created')]  # [['date','event','ticker','size','price']]
    today = datetime.date.today()
    #today = datetime.datetime.strptime('2023-07-20', '%Y-%m-%d').date()
    today_orders = orders_history.loc[orders_history['date'] == today]

    display_orders_log(today_orders, 'Today Orders')

    #Space
    for i in range( len(cols)):
        cols[i].write(f"  ")

    # Get Next days SELL Stops Log
    orders_ahead = orders_history.loc[orders_history['date'] > today]
    if len(orders_ahead) > 0:
        next_day = orders_ahead['date'].iloc[0]
        next_orders = orders_history.loc[orders_history['date'] == next_day]

        display_orders_log(next_orders, 'Next Orders Forecast')

    else:
        cols[1].write("No Orders Forecast  in the next days")

def display_orders(log_history,settings):
    def display_orders_log(df, title,col=0):
        if len(df) > 0:
            cols[col].write(f"{title} **{df['date'].iloc[0]}** 00:00(CET) ")
            for i, row in df.iterrows():
                order_log = f"{row['ticker']} {row['exectype']} {row['B_S']}  {row['size']}"
                if row['exectype'] == "Stop":
                    price_log=f" @ {row['price']}"
                    order_log = order_log + price_log

                #col=df.columns.get_loc(row['ticker'])
                #cols[col].write(order_log)
                cols[col].subheader(order_log)
                #if i>0: cols[0].write("  ")

        else:
            cols[col].write(f"No {title}")
            #cols[1].write("  ")

            #for j in range(2):
            #    for i in range(1,len(cols)):
            #        cols[i].write(" ")



    # Columns for dispaly
    col_width_list = [2] +[2] + [2]
    cols = st.columns(col_width_list)

    cols[0].write(f"**Orders to Broker:**")

    # Get Today SELL Stops Log
    #orders_history = log_history[log_history['event'].str.contains('Order Created')]  # [['date','event','ticker','size','price']]
    orders_history = log_history[log_history['event'].str.contains('Created')]  # [['date','event','ticker','size','price']]
    today = datetime.date.today()
    #today = datetime.datetime.strptime('2023-07-20', '%Y-%m-%d').date()
    today_orders = orders_history.loc[orders_history['date'] == today]

    display_orders_log(today_orders, 'Today Orders',col=1)

    #Space
    #for i in range( len(cols)):
    #    cols[i].write(f"  ")

    # Get Next days SELL Stops Log
    orders_ahead = orders_history.loc[orders_history['date'] > today]
    if len(orders_ahead) > 0:
        next_day = orders_ahead['date'].iloc[0]
        next_orders = orders_history.loc[orders_history['date'] == next_day]

        display_orders_log(next_orders, 'Next Orders Forecast',col=2)

    else:
        cols[1].write("No Orders Forecast  in the next days")

if __name__ == '__main__':
    main(settings)




