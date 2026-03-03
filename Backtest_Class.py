from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class BacktestSettings:
    """Settings for backtest execution."""
    upgrade_threshold: float = 0.20
    commision: float = 5.0
    buy_at_market: bool = False
    portfolio_window: int = 22 * 12 #22 * 12
    min_periods: int = 1


class Backtest:
    """Sequential implementation of backtest logic."""

    def __init__(self, settings: BacktestSettings):
        self.settings = settings

    def compute_backtest_sequential(
            self,
            weights_div_asset_price: pd.DataFrame,
            asset_price: pd.DataFrame,
            opens: pd.DataFrame,
            highs: pd.DataFrame,
            lows: pd.DataFrame,
            closes: pd.DataFrame,
            mults: np.ndarray,
            weights: pd.DataFrame,
            buy_trigger: pd.DataFrame,
            sell_trigger: pd.DataFrame,
            sell_stop_price: pd.DataFrame,
            buy_stop_price: pd.DataFrame,
            exchange_rate: pd.Series,
            startcash_usd: float,
            startcash: float,
            exposition_lim: float,
            max_n_contracts: int = 50,
            swan_stop_price: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Execute backtest day by day. No circular dependency possible."""

        tickers = weights.columns
        trading_dates = weights.index
        n_days = len(trading_dates)
        n_tickers = len(tickers)

        # Pre-allocate output arrays
        positions = np.zeros((n_days, n_tickers), dtype=np.int32)
        portfolio_usd = np.zeros(n_days, dtype=np.float64)
        portfolio_eur = np.zeros(n_days, dtype=np.float64)
        daily_ret_usd = np.zeros(n_days, dtype=np.float64)
        daily_ret_eur = np.zeros(n_days, dtype=np.float64)
        pos_value_arr = np.zeros(n_days, dtype=np.float64)
        exposition_arr = np.zeros(n_days, dtype=np.float64)
        pt_invest_arr = np.zeros(n_days, dtype=np.float64)

        # Order/broker log storage
        order_log = []
        broker_log = []

        # Initialise
        portfolio_usd[0] = startcash_usd
        portfolio_eur[0] = startcash

        # Fixed-window min tracking
        window_counter = 0
        current_window_min = startcash_usd
        portfolio_to_invest = startcash_usd

        # Convert everything to numpy up front for speed
        ap = asset_price.values.astype(np.float64)
        op = opens.values.astype(np.float64)
        hi = highs.values.astype(np.float64)
        lo = lows.values.astype(np.float64)
        cl = closes.values.astype(np.float64)
        wda = weights_div_asset_price.values.astype(np.float64)
        bt_arr = buy_trigger.values.astype(bool)
        bsp_arr = buy_stop_price.values.astype(np.float64)
        ssp_arr = sell_stop_price.values.astype(np.float64)
        er_arr = exchange_rate.values.astype(np.float64)

        for day in range(1, n_days):

            prev_pos = positions[day - 1].copy()
            prev_port_usd = portfolio_usd[day - 1]
            prev_port_eur = portfolio_eur[day - 1]

            # ── Fixed-window portfolio_to_invest ──────────────────────────
            window_counter += 1
            current_window_min = min(current_window_min, prev_port_usd)

            if window_counter >= self.settings.portfolio_window:
                portfolio_to_invest = current_window_min  # min of completed window
                current_window_min = prev_port_usd  # reset for next window
                window_counter = 0

            # Always use min of frozen value AND current window progress AND current portfolio
            # This catches drawdowns within the current window immediately
            effective_pti = min(portfolio_to_invest, current_window_min, prev_port_usd)
            effective_pti = max(effective_pti, 1.0)  # avoid div by zero
            pt_invest_arr[day] = effective_pti

            # ── Target size ───────────────────────────────────────────────
            target_size_raw = wda[day] * effective_pti
            mask_upgrade = target_size_raw > self.settings.upgrade_threshold
            target_size_raw = np.where(mask_upgrade,
                                       np.maximum(target_size_raw, 1.0),
                                       target_size_raw)
            target_size = np.round(target_size_raw).astype(np.int32)
            target_size = np.clip(target_size, 0, max_n_contracts)

            # ── Exposition guard ──────────────────────────────────────────
            target_pos_value = (ap[day] * target_size).sum()
            targeted_exposition = target_pos_value / effective_pti
            exposition_ok = targeted_exposition < exposition_lim

            target_trade = target_size - prev_pos

            # ── Order logic ───────────────────────────────────────────────
            is_buy = (target_trade > 0) & exposition_ok & bt_arr[day]
            is_sell = (target_trade < 0)

            if not self.settings.buy_at_market:
                buy_price = np.clip(bsp_arr[day], op[day], None)
            else:
                buy_price = op[day].copy()

            sell_price = np.clip(ssp_arr[day], None, op[day])
            order_price = np.where(is_buy, buy_price,
                                   np.where(is_sell, sell_price, 0.0))

            # ── Execution ─────────────────────────────────────────────────
            in_range = (order_price >= lo[day]) & (order_price <= hi[day])
            executed = (is_buy | is_sell) & in_range

            exec_size = np.where(executed, target_trade, 0).astype(np.int32)
            exec_price = np.where(executed, order_price, 0.0)

            # ── Update positions ──────────────────────────────────────────
            new_pos = prev_pos + exec_size
            positions[day] = new_pos

            # ── Returns ───────────────────────────────────────────────────
            price_diff = ap[day] - ap[day - 1]
            hold_ret_usd = (prev_pos * price_diff ).sum()
            trade_ret_usd = (exec_size * (cl[day] - exec_price) * mults).sum()
            cost_usd = np.abs(exec_size).sum() * self.settings.commision

            day_ret_usd = hold_ret_usd + trade_ret_usd - cost_usd
            day_ret_eur = day_ret_usd * er_arr[day]

            portfolio_usd[day] = prev_port_usd + day_ret_usd
            portfolio_eur[day] = prev_port_eur + day_ret_eur
            daily_ret_usd[day] = day_ret_usd
            daily_ret_eur[day] = day_ret_eur
            pos_value_arr[day] = (ap[day] * new_pos).sum()
            exposition_arr[day] = pos_value_arr[day] / effective_pti

            # ── Order log ─────────────────────────────────────────────────
            date = trading_dates[day]
            for t_idx, ticker in enumerate(tickers):
                if is_buy[t_idx] or is_sell[t_idx]:
                    bs = 'Buy' if is_buy[t_idx] else 'Sell'
                    etype = 'Market' if (is_buy[t_idx] and self.settings.buy_at_market) else 'Stop'
                    o_time = date + pd.Timedelta(seconds=t_idx + 1)
                    e_time = (o_time + pd.Timedelta(seconds=10)
                              if etype == 'Market'
                              else o_time + pd.Timedelta(hours=10))

                    order_log.append({
                        'date_time': o_time,
                        'event': 'Created',
                        'ticker': ticker,
                        'B_S': bs,
                        'exectype': etype,
                        'size': int(target_trade[t_idx]),
                        'price': round(float(order_price[t_idx]), 3),
                        'pos': int(prev_pos[t_idx]),
                        'commision': 0.0,
                    })

                    if executed[t_idx]:
                        broker_log.append({
                            'date_time': e_time,
                            'event': 'Executed',
                            'ticker': ticker,
                            'B_S': bs,
                            'exectype': etype,
                            'size': int(exec_size[t_idx]),
                            'price': round(float(exec_price[t_idx]), 3),
                            'pos': int(new_pos[t_idx]),
                            'commision': round(float(np.abs(exec_size[t_idx]) * self.settings.commision), 2),
                        })
                    else:
                        cancel_time = date + pd.Timedelta(hours=23, minutes=59)
                        broker_log.append({
                            'date_time': cancel_time,
                            'event': 'Canceled',
                            'ticker': ticker,
                            'B_S': bs,
                            'exectype': etype,
                            'size': int(target_trade[t_idx]),
                            'price': round(float(order_price[t_idx]), 3),
                            'pos': int(prev_pos[t_idx]),
                            'commision': 0.0,
                        })

        # ── Build output DataFrames ───────────────────────────────────────
        pos_df = pd.DataFrame(positions, index=trading_dates, columns=tickers)
        port_usd_series = pd.Series(portfolio_usd, index=trading_dates)
        port_eur_series = pd.Series(portfolio_eur, index=trading_dates)
        daily_ret_eur_s = pd.Series(daily_ret_eur, index=trading_dates)
        pos_value_series = pd.Series(pos_value_arr, index=trading_dates)
        exposition_s = pd.Series(exposition_arr, index=trading_dates)
        pt_invest_series = pd.Series(pt_invest_arr, index=trading_dates)

        def logs_to_flat_df(log_list):
            if not log_list:
                return pd.DataFrame(columns=['date_time', 'event', 'ticker', 'B_S',
                                             'exectype', 'size', 'price', 'pos', 'commision'])
            return pd.DataFrame(log_list)

        bt_log_dict = {
            'pos': pos_df,
            'portfolio_value': port_usd_series,
            'portfolio_value_eur': port_eur_series,
            'dayly_profit_eur': daily_ret_eur_s,
            'pos_value': pos_value_series,
            'exposition': exposition_s,
            'portfolio_to_invest': pt_invest_series,
            'exchange_rate': exchange_rate,
            'n_iter': 1,
            'order_dict': logs_to_flat_df(order_log),
            'broker_dict': logs_to_flat_df(broker_log),
        }


        return pos_df, port_usd_series, bt_log_dict


def compute_backtest_vectorized(
        positions: pd.DataFrame,
        settings: Dict,
        data_dict: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """Main entry point for backtest computation."""

    # Get Sanitized Index Data
    positions, data_dict, opens, highs, lows, closes = sanitize_dataset(positions, data_dict)

    # Get settings values
    mults_array, startcash, exposition_lim, commision, max_n_contracts = get_settings_values(settings, positions.columns)

    # Black Swan levels
    #swan_stop_price = compute_black_swan_thresholds(closes, lows)

    # Buy/Sell Triggers & Stop Prices
    buy_trigger, sell_trigger, sell_stop_price, buy_stop_price = compute_buy_sell_triggers(
        positions, opens, closes, lows, highs
    )

    # Exchange Rate EUR/USD (day after)
    exchange_rate = 1 / closes["EURUSD=X"].shift(1).fillna(method='bfill')

    # Set cash start
    startcash_usd = startcash / exchange_rate.iloc[0]

    # Out-of-loop calculations
    weights_div_asset_price, asset_price = compute_out_of_backtest_loop(closes, positions, mults_array)

    # Create backtest instance
    backtest_settings = BacktestSettings(
        upgrade_threshold=settings.get('upgrade_threshold', 0.20),
        commision=commision,
        buy_at_market=settings.get('buy_at_market', False)
    )
    backtest = Backtest(backtest_settings)

    # Single sequential pass — no iteration loop needed
    pos, portfolio_value_usd, bt_log_dict = backtest.compute_backtest_sequential(
        weights_div_asset_price, asset_price, opens, highs, lows, closes, mults_array,
        positions, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
        exchange_rate, startcash_usd, startcash, exposition_lim,
        max_n_contracts=max_n_contracts,
    )

    bt_log_dict['pos']             = pos
    bt_log_dict['portfolio_value'] = portfolio_value_usd

    # Get Log History
    log_history = create_log_history(bt_log_dict)

    # Quantstats Report
    if settings.get('qstats', False):
        bt_qstats_report(bt_log_dict, closes, settings['add_days'], exchange_rate)

    return bt_log_dict, log_history

def sanitize_dataset(positions,data_dict):

    # Determine common index between positions and all tickers in data_dict
    common_index = positions.index
    for df in data_dict.values():
        common_index = common_index.intersection(df.index)

    # Reindex positions and all OHLCs to this common index
    positions = positions.reindex(common_index).fillna(0)

    for tick, df in data_dict.items():
        data_dict[tick] = df.reindex(common_index).fillna(method='ffill')

    # Get Open, High, Low, Closes from data_dict - optimize by using list comprehension
    desired_order = list(data_dict.keys())

    # Optimize by creating DataFrames directly without intermediate dictionary
    opens = pd.concat([data_dict[key]['Open'] for key in desired_order], axis=1, keys=desired_order)
    highs = pd.concat([data_dict[key]['High'] for key in desired_order], axis=1, keys=desired_order)
    lows = pd.concat([data_dict[key]['Low'] for key in desired_order], axis=1, keys=desired_order)
    closes = pd.concat([data_dict[key]['Close'] for key in desired_order], axis=1, keys=desired_order)

    # Start - optimize by using boolean mask directly
    start_mask = (positions > 0).any(axis=1)
    start = positions[start_mask].index[0]

    # Set Start at positions and Tickers Data - optimize by using slicing
    positions = positions.loc[start:]
    opens = opens.loc[start:]
    highs = highs.loc[start:]
    lows = lows.loc[start:]
    closes = closes.loc[start:]

    return positions, data_dict,opens, highs, lows, closes

def get_settings_values(settings,tickers):
    # mult dict to list in the tickers order - optimize by using list comprehension
    mults_array = np.array([
        settings['mults'][tick] if tick in settings['mults']
        else (print(f"ADVERTENCIA: Falta {tick}, usando 1.0") or 1.0)
        for tick in tickers
    ])
    startcash = settings['startcash']
    exposition_lim = settings['exposition_lim']
    commision = settings['commision']
    max_n_contracts = settings['max_n_contracts']

    return mults_array,startcash, exposition_lim, commision, max_n_contracts


def compute_out_of_backtest_loop(closes, weights, mults):
    """Calculate asset prices and weights divided by asset prices."""
    asset_price = closes.multiply(mults, axis=1)  # USD
    yesterday_asset_price = asset_price.shift(1)
    yesterday_asset_price_mean = yesterday_asset_price.rolling(5, min_periods=1).mean()
    #weights_mean = weights.rolling(5, min_periods=1).mean()
    #weights_div_asset_price = weights_mean / yesterday_asset_price_mean
    weights_div_asset_price = weights / yesterday_asset_price_mean

    return weights_div_asset_price, asset_price


def compute_buy_sell_triggers(weights,opens, closes,lows, highs):
    """Calculate buy/sell triggers and stop prices."""
    # Weights Uptrend --> weight >= previous 5 days lowest
    weights_min = weights.shift(1).rolling(5).min()
    weights_up = weights.ge(weights_min, axis=0)

    # Weights Downtrend --> weight < previous 5 days highest
    weights_max = weights.shift(1).rolling(5).max()
    weights_dn = weights.lt(weights_max, axis=0)

    # Lows Uptrend --> Yesterday low > previous 5 days lowest
    #lows_min = lows.shift(1).rolling(5).min()
    #lows_up = lows.ge(lows_min, axis=0)

    #Compute Volatility Sell Stop
    sell_stop_price = compute_sell_volat_stop_price(closes, lows, delta=6) #6

    # Lows Uptrend
    lows_up = lows.ge(sell_stop_price, axis=0)

    # Highs Downtrend --> Yesterday high < previous 5 days highest
    highs_max = highs.shift(1).rolling(5).max()
    highs_dn = highs.le(highs_max, axis=0)

    # Buy Trigger
    buy_trigger =  weights_up & lows_up

    # Sell Trigger
    sell_trigger = highs_dn & weights_dn # Highs Downtrend

    # Get Sell Stop Price
    #sell_stop_price = lows_min.rolling(22).max()

    # Get Buy Stop Price
    #high_keep = highs_max.rolling(22).min()
    #highs_std = highs.rolling(22).std().shift(1)
    #buy_stop_price = high_keep + highs_std * 0.5

    # Compute Volatility Buy Stop
    buy_stop_price = compute_buy_volat_stop_price(closes, highs, delta=1)


    #Debug Plot
    debug=False
    if debug:
        plot_df=pd.DataFrame()
        ticker='CL=F'
        plot_df['high']=highs[ticker]
        plot_df['high_max']=highs_max[ticker]
        plot_df['lows_min'] = lows_min[ticker]
        #plot_df['high_keep'] = high_keep[ticker]
        plot_df['buy_stop_price'] = buy_stop_price[ticker]
        #plot_df['closes_uptrend'] = closes_uptrend[ticker]
        plot_df['weights_up'] = weights_up[ticker]
        plot_df['buy_trigger'] = buy_trigger[ticker]


        plot_df.tail(200).plot(title=ticker)

        print(plot_df[:-4].tail(5))

    return buy_trigger, sell_trigger, sell_stop_price, buy_stop_price


def create_log_history(bt_log_dict):
    """Create log history from backtest log dictionary."""
    # Get Series from dict
    portfolio_value_eur = bt_log_dict['portfolio_value_eur']
    pos = bt_log_dict['pos']

    tickers = pos.columns

    # Get dicts to create log_history
    order_dict = get_log_dict_by_ticker_dict(bt_log_dict['order_dict'], tickers)
    broker_dict = get_log_dict_by_ticker_dict(bt_log_dict['broker_dict'], tickers)

    # Reindex before concatenate
    order_dict = {ticker: order_dict[ticker].reset_index(drop=True) for ticker in tickers}
    broker_dict = {ticker: broker_dict[ticker].reset_index(drop=True) for ticker in tickers}

    # Concatenate order_dict with broker_dict
    # Use the same approach as in the original implementation
    log_history_dict = {ticker: pd.concat([order_dict[ticker], broker_dict[ticker]], axis=0).dropna().sort_values(by='date_time') for ticker in tickers}

    # Concatenate all tickers
    log_history = pd.concat(log_history_dict.values(), axis=0).sort_values(by='date_time')

    # Insert tickers as columns
    log_history[tickers] = np.nan

    # No need to check for date_time column as we've ensured it exists in the previous steps

    # Insert tickers as columns and reorder exactly as in the original implementation
    log_history = log_history[['date_time'] + list(tickers) + list(log_history.columns[1:-len(tickers)])]

    # Create End of Day df
    eod_df = pd.DataFrame(index=pos.index, columns=list(log_history.columns) + ['portfolio_value', 'portfolio_value_eur', 'pos_value', 'ddn', 'dayly_profit', 'dayly_profit_eur', 'pre_portfolio_value', 'exchange_rate', 'ddn_eur'])
    eod_df['date_time'] = pos.index + pd.Timedelta(days=0, hours=23, minutes=59, seconds=59)
    eod_df[tickers] = pos
    eod_df['event'] = 'End of Day'

    # Add Series values from log_dict
    keys = ['portfolio_value', 'portfolio_value_eur', 'pos_value', 'dayly_profit_eur', 'exchange_rate']
    for key in keys:
        eod_df[key] = bt_log_dict[key]

    # Add calculated series
    eod_df['dayly_profit'] = eod_df['dayly_profit_eur'] / eod_df['exchange_rate']
    eod_df['pre_portfolio_value'] = eod_df['portfolio_value'].shift(1)
    eod_df = round(eod_df, 2)

    eod_df['ddn'] = eod_df['portfolio_value'].rolling(252 * 3, min_periods=250).max() / eod_df['portfolio_value'] - 1
    eod_df['ddn_eur'] = eod_df['portfolio_value_eur'].rolling(252 * 3, min_periods=250).max() / eod_df['portfolio_value_eur'] - 1
    eod_df[['ddn', 'ddn_eur']] = round(eod_df[['ddn', 'ddn_eur']], 4)

    eod_df = eod_df.reset_index(drop=True)

    # Concatenate log_history with eod_df
    log_history = pd.concat([log_history, eod_df], axis=0).sort_values(by='date_time')

    # Update tickers positions - do this before renaming events
    for ticker in tickers:
        is_executed = log_history['event'] == 'Executed'
        is_ticker = log_history['ticker'] == ticker
        log_history.loc[is_executed & is_ticker, ticker] = log_history.loc[is_executed & is_ticker, 'pos']

    log_history[tickers] = log_history[tickers].fillna(method='ffill')
    log_history[tickers] = log_history[tickers].fillna(0).astype(int)

    # Rename event
    event_values = ['Sell Order Created', 'Sell Order Canceled', 'Sell Order Executed', 'Buy Order Created', 'Buy Order Canceled', 'Buy Order Executed', 'End of Day']

    # Note: In the original implementation, there's a comment about renaming events,
    # but no actual code that does it. We're keeping the same behavior here.

    # Keep only date at date_time
    log_history['date_time'] = pd.to_datetime(log_history['date_time'], errors='coerce')
    log_history['date'] = log_history['date_time'].dt.date

    return log_history


def get_log_dict_by_ticker_dict(bt_log_dict_entry, tickers):
    """Convert log entry to dictionary of DataFrames by ticker.
    Handles both flat DataFrame (sequential) and wide-dict (old vectorized) formats.
    """
    bt_log_by_ticker_dict = {}

    # ── Flat DataFrame format (new sequential) ──
    if isinstance(bt_log_dict_entry, pd.DataFrame):
        for ticker in tickers:
            ticker_df = bt_log_dict_entry[bt_log_dict_entry['ticker'] == ticker].copy()
            bt_log_by_ticker_dict[ticker] = ticker_df.reset_index(drop=True)
        return bt_log_by_ticker_dict

    # ── Wide dict format (old vectorized) ──
    for ticker in tickers:
        df = pd.DataFrame()
        for key, value in bt_log_dict_entry.items():
            try:
                if isinstance(value, pd.DataFrame) and ticker in value.columns:
                    df[key] = value[ticker]
                elif isinstance(value, pd.Series):
                    df[key] = value
            except Exception as e:
                print(f"Warning: Error processing {key} for ticker {ticker}: {str(e)}")
                continue
        bt_log_by_ticker_dict[ticker] = df

    return bt_log_by_ticker_dict


def bt_qstats_report(bt_log_dict, closes,add_days,exchange_rate):
    q_title = 'Cash Backtest Markowitz Vectorized'
    path = "results\\"
    q_filename = os.path.abspath(path + q_title + '.html')
    q_returns = bt_log_dict['portfolio_value_eur'].pct_change().iloc[:-add_days]
    q_returns.index = pd.to_datetime(q_returns.index, utc=True).tz_convert(None)

    q_benchmark_ticker = 'ES=F'
    q_benchmark = (closes[q_benchmark_ticker] * exchange_rate).pct_change().iloc[:-add_days]
    q_benchmark.index = pd.to_datetime(q_benchmark.index, utc=True).tz_convert(None)

    import quantstats_lumi as quantstats
    quantstats.reports.html(q_returns, title=q_title, benchmark=q_benchmark, benchmark_title=q_benchmark_ticker, output=q_filename)
    import webbrowser
    webbrowser.open(q_filename)

    return q_returns, q_title, q_benchmark, q_benchmark_ticker,q_filename



def compute_sell_volat_stop_price(closes, lows, delta=6):
    downside_noise = (closes.shift(1) - lows).shift(1)

    roll_mean = downside_noise.rolling(20, min_periods=1).mean()
    roll_std  = downside_noise.rolling(20, min_periods=1).std().fillna(0)

    volat_buffer = roll_mean + delta * roll_std
    volat_buffer = volat_buffer.clip(
        lower=0.0001 * closes.shift(1),
        upper=0.12   * closes.shift(1)
    )

    volat_stop_price = closes.shift(1) - volat_buffer
    volat_stop_price = volat_stop_price.rolling(22, min_periods=1).max()

    # Fill early zeros with first valid value (backfill)
    volat_stop_price = volat_stop_price.replace(0, np.nan).bfill().fillna(0)

    return volat_stop_price


def compute_buy_volat_stop_price(closes, highs, delta=1):
    upside_noise = (highs - closes.shift(1)).shift(1)

    roll_mean = upside_noise.rolling(20, min_periods=1).mean()
    roll_std  = upside_noise.rolling(20, min_periods=1).std().fillna(0)

    volat_buffer     = roll_mean + delta * roll_std
    volat_stop_price = closes.shift(1) + volat_buffer
    volat_stop_price = volat_stop_price.rolling(22, min_periods=1).min()

    # Fill early zeros with first valid value (backfill)
    volat_stop_price = volat_stop_price.replace(0, np.nan).bfill().fillna(0)

    return volat_stop_price

