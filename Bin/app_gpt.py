import streamlit as st
import datetime
import pytz
import pandas as pd
import altair as alt

from Trading_Markowitz import compute,process_log_data
from config.trading_settings import settings
import Market_Data_Feed as mdf

#For Local Run bellow in the pycharm terminal
#streamlit run app_gpt.py
#Ctrl + c to stop

# ---------------------------
# Helper Functions
# ---------------------------
def get_today(tz_name='Europe/Madrid'):
    """Return today's date in given timezone."""
    tz = pytz.timezone(tz_name)
    return datetime.datetime.now(tz).date()

def get_latest_market_data(df, target_date):
    """Return data for target_date if exists, else last available <= target_date."""
    df = df.sort_index()
    if target_date in df.index.date:
        return df.loc[df.index.date == target_date]
    return df.loc[df.index.date <= target_date].iloc[-1:]

def align_indices(df_list):
    """Align multiple DataFrames on the same datetime index, forward-fill missing."""
    all_idx = pd.concat([df.index.to_series() for df in df_list]).sort_values().drop_duplicates()
    aligned = [df.reindex(all_idx, method='ffill') for df in df_list]
    return aligned

# ---------------------------
# Cached Data Loader
# ---------------------------
@st.cache_data
def load_and_compute_data(settings):
    data_ind = mdf.Data_Ind_Feed(settings).data_ind
    data, _ = data_ind

    log_history, _ = compute(settings, data_ind)

    closes_aligned, returns_aligned = align_indices([data.tickers_closes, data.tickers_returns])
    data.tickers_closes = closes_aligned
    data.tickers_returns = returns_aligned

    return data, log_history

# ---------------------------
# Chart Function
# ---------------------------
def chart_ts_altair(ts, col, color="blue", st_altair_chart=True):
    df = ts[col].rename_axis('date').reset_index()
    alt_chart = alt.Chart(df, height=120).mark_line(color=color).encode(
        x=alt.X('date', title=''),
        y=alt.Y(col, title='', scale=alt.Scale(domain=[ts[col].min(), ts[col].max()]))
    )
    if st_altair_chart:
        st.altair_chart(alt_chart, use_container_width=True)
    return alt_chart

# ---------------------------
# Ticker Display
# ---------------------------
def display_tickers_data(closes, returns, today, settings,
                         sidebar=False, daysback=66, data_show='returns', chart=True):
    tickers = settings["tickers"]
    n_col = len(tickers) + 1
    col_width_list = [7] + [3] * (n_col - 1)
    cols = st.columns(col_width_list)

    # Header
    tz = pytz.timezone('Europe/Madrid')
    now = datetime.datetime.now(tz)
    time_string = now.strftime('%H:%M:%S')
    market_header = f"**Market Data: {today} {time_string}** (data with 15min delay)"
    cols[0].title('Quant Trading App')
    cols[0].write(market_header)

    closes_today = get_latest_market_data(closes, today)
    returns_today = get_latest_market_data(returns, today)

    for i, ticker in enumerate(tickers):
        close = closes_today[ticker].iloc[-1]
        ret = returns_today[ticker].iloc[-1]
        value = f"{close:,.2f}" if ticker in ['CL=F', 'EURUSD=X'] else f"{close:,.0f}"
        delta = f"{ret:.1%}" if ticker != 'cash' else f"@ {ret*255:.1%} EURIBOR"
        cols[i+1].metric(f"**{ticker}**", value, delta)

        if chart:
            chart_data = closes.tail(daysback) if data_show == 'closes' else returns.tail(daysback)
            with cols[i+1]:
                chart_ts_altair(chart_data, ticker)

# ---------------------------
# Portfolio Positions Display
# ---------------------------
def display_portfolio_positions(eod_log_history, trading_history, date, settings, ret_by_ticker, returns, daysback=66, forecast=False):
    st.write(f"**Portfolio Positions:**")
    today = date if forecast else get_today()
    last_portfolio = eod_log_history.loc[:today].iloc[-1][settings['tickers']]
    pre_portfolio = eod_log_history.loc[:today].iloc[-2][settings['tickers']]
    last_trade = last_portfolio - pre_portfolio

    pos_value_today = eod_log_history.loc[:today,'pos_value'].iloc[-1]
    portfolio_value_eur = eod_log_history.loc[:today, 'portfolio_value_eur'].iloc[-1]
    exchange_rate = eod_log_history.loc[:today, 'exchange_rate'].iloc[-1]
    pos_value_today_eur = pos_value_today * exchange_rate
    exposition = pos_value_today_eur / portfolio_value_eur * 100

    n_col = len(settings["tickers"]) + 1
    col_width_list = [7] + [3] * (n_col - 1)
    cols = st.columns(col_width_list)

    with cols[0]:
        st.write("Tickers:")
        st.write("Nbr of Contracts:")
        st.write(f"Last Trade date: {date}")
        st.subheader(f"{pos_value_today_eur:,.0f} € /  {exposition:,.0f} %")

    for i in range(1, n_col):
        j = i - 1
        ticker = trading_history.columns[j]
        value = int(last_portfolio[j])
        delta = int(last_trade[j])
        cols[i].metric(label=f"**{ticker}**", value=value, delta=delta)

# ---------------------------
# Portfolio Results Display
# ---------------------------
def display_portfolio_results(eod_log_history, today, settings, daysback=66):
    st.write(f"**Portfolio Results:**")
    col_width_list = [3] + [2] * 4
    cols = st.columns(col_width_list)

    portfolio_value_eur = eod_log_history.loc[:today, "portfolio_value_eur"].iloc[-1]
    ret = eod_log_history.loc[:today, "portfolio_return"].iloc[-1]

    with cols[0]:
        st.metric(label=f"**Portfolio Value {today}**", value=f"{portfolio_value_eur:,.0f} €", delta=f"{ret:.1%}")
        chart_ts_altair(eod_log_history.iloc[-daysback:], "portfolio_value_eur")

    keys = ["cagr","weekly_return","monthly_return"]
    for i, key in enumerate(keys):
        cagr = eod_log_history.loc[:today,key].iloc[-1]
        diff_eur = f"{cagr * portfolio_value_eur:,.0f} €"
        cols[i+1].metric(label=f"**{key}**", value=diff_eur, delta=f"{cagr:.1%}")

    ddn = eod_log_history.loc[:today, "ddn_eur"].iloc[-1]
    cols[4].metric(label="**Drawdown YTD**", delta="", value=f"{ddn:.1%}")

# ---------------------------
# Orders Display
# ---------------------------
def display_orders(log_history, settings):
    st.write(f"**Orders to Broker:**")
    n_col = len(settings["tickers"]) + 1
    col_width_list = [2] + [2] + [2]
    cols = st.columns(col_width_list)

    orders_history = log_history[log_history['event'].str.contains('Created')]
    today = get_today()
    today_orders = orders_history.loc[orders_history['date'] == today]

    def display_orders_log(df, title, col=0):
        if len(df) > 0:
            cols[col].write(f"{title} **{df['date'].iloc[0]}** 00:00(CET)")
            for _, row in df.iterrows():
                order_log = f"{row['ticker']} {row['exectype']} {row['B_S']}  {row['size']}"
                if row['exectype'] == "Stop":
                    order_log += f" @ {row['price']}"
                cols[col].subheader(order_log)
        else:
            cols[col].write(f"No {title}")

    display_orders_log(today_orders, 'Today Orders', col=1)

    orders_ahead = orders_history.loc[orders_history['date'] > today]
    if len(orders_ahead) > 0:
        next_day = orders_ahead['date'].iloc[0]
        next_orders = orders_ahead.loc[orders_ahead['date'] == next_day]
        display_orders_log(next_orders, 'Next Orders Forecast', col=2)
    else:
        cols[1].write("No Orders Forecast in the next days")

# ---------------------------
# Main App
# ---------------------------
def main(settings):
    today = get_today()

    if st.button("Refresh App"):
        load_and_compute_data.clear()
        st.experimental_rerun()

    data, log_history = load_and_compute_data(settings)
    closes = data.tickers_closes
    returns = data.tickers_returns

    eod_log_history, trading_history = process_log_data(log_history, settings)
    ret_by_ticker = returns[settings['tickers']] * eod_log_history[settings['tickers']]

    display_tickers_data(closes, returns, today, settings,
                         sidebar=False,
                         daysback=st.session_state.get('daysback', 66),
                         data_show=st.session_state.get('data_show', 'returns'),
                         chart=True)

    st.divider()
    last_trade_date = trading_history.index[trading_history.index.to_series().dt.date <= today][-1]
    display_portfolio_positions(eod_log_history, trading_history, last_trade_date, settings, ret_by_ticker, returns)

    display_orders(log_history, settings)

    # Future Forecast
    next_trades = trading_history.index[trading_history.index.to_series().dt.date > today]
    if len(next_trades) > 0:
        next_trade_date = next_trades[0]
        with st.expander("See Next Trading Forecast:"):
            display_portfolio_positions(eod_log_history, trading_history, next_trade_date,
                                        settings, ret_by_ticker, returns, forecast=True)
    else:
        st.write(f"**Keep Current Positions. No Trading Forecast within {settings['add_days']} Days**")

    display_portfolio_results(eod_log_history, today, settings)

# ---------------------------
# Run App
# ---------------------------
if __name__ == '__main__':
    main(settings)
