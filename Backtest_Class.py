from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

# ── Margin Rules ──────────────────────────────────────────────────────────────
ETF_TICKERS = ['NQ=F', 'ES=F', 'GC=F']
ETF_CONTRACTS_LIMIT = 2 #2
MARGIN_RATE = 0.10
MIN_EURIBOR = 0.001


@dataclass
class BacktestSettings:
    """Settings for backtest execution."""
    upgrade_threshold: float = 0.20
    commision: float = 5.0
    buy_at_market: bool = False
    portfolio_window: int = 22 * 12
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
            euribor: pd.Series,
            max_n_contracts: int = 50,
            swan_stop_price: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Execute backtest day by day. No circular dependency possible."""

        tickers = weights.columns
        tickers_list = list(tickers)
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
        cash_eur_arr = np.zeros(n_days, dtype=np.float64)

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
        st_arr = sell_trigger.values.astype(bool)
        bsp_arr = buy_stop_price.values.astype(np.float64)
        ssp_arr = sell_stop_price.values.astype(np.float64)
        er_arr = exchange_rate.values.astype(np.float64)
        euribor_arr = euribor.values.astype(np.float64)

        upgrade_eligible = np.array([t in ETF_TICKERS for t in tickers_list])

        for day in range(1, n_days):

            prev_pos = positions[day - 1].copy()
            prev_port_usd = portfolio_usd[day - 1]
            prev_port_eur = portfolio_eur[day - 1]

            # ── Fixed-window portfolio_to_invest ──────────────────────────
            window_counter += 1
            current_window_min = min(current_window_min, prev_port_usd)

            if window_counter >= self.settings.portfolio_window:
                portfolio_to_invest = current_window_min
                current_window_min = prev_port_usd
                window_counter = 0

            effective_pti = min(portfolio_to_invest, current_window_min, prev_port_usd)
            effective_pti = max(effective_pti, 1.0)
            pt_invest_arr[day] = effective_pti

            # ── Target size ───────────────────────────────────────────────
            target_size_raw = wda[day] * effective_pti
            mask_upgrade = (target_size_raw > self.settings.upgrade_threshold) & (target_size_raw < 2.0) #& upgrade_eligible
            target_size_raw = np.where(mask_upgrade, np.maximum(target_size_raw, 1.0), target_size_raw)
            target_size = np.round(target_size_raw).astype(np.int32)
            target_size = np.clip(target_size, 0, max_n_contracts)

            # ── Exposition guard ──────────────────────────────────────────
            target_pos_value = (ap[day] * target_size).sum()
            targeted_exposition = target_pos_value / effective_pti
            exposition_ok = targeted_exposition < exposition_lim

            target_trade = target_size - prev_pos

            # ── Order logic ───────────────────────────────────────────────
            is_buy = (target_trade > 0) & exposition_ok & bt_arr[day]
            is_sell = (target_trade < 0) & st_arr[day]

            #Order Sent to Broker with Stop prices
            sent_order_price=np.where(is_buy, bsp_arr[day],
                                   np.where(is_sell, ssp_arr[day], 0.0))

            # ── Execution ─────────────────────────────────────────────────

            #Open price filtered
            buy_price = np.clip(bsp_arr[day], op[day], None)
            sell_price = np.clip(ssp_arr[day], None, op[day])
            order_price = np.where(is_buy, buy_price,
                                   np.where(is_sell, sell_price, 0.0))


            in_range = (order_price >= lo[day]) & (order_price <= hi[day])
            executed = (is_buy | is_sell) & in_range

            exec_size = np.where(executed, target_trade, 0).astype(np.int32)
            exec_price = np.where(executed, order_price, 0.0)

            # ── Update positions ──────────────────────────────────────────
            new_pos = prev_pos + exec_size
            positions[day] = new_pos

            # ── Returns ───────────────────────────────────────────────────
            price_diff = ap[day] - ap[day - 1]
            hold_ret_usd = (prev_pos * price_diff).sum()
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

            # ── Cash Euribor Return ───────────────────────────────────────
            blocked_eur = 0.0
            for t_idx, ticker in enumerate(tickers_list):
                n_contracts = int(new_pos[t_idx])
                if n_contracts == 0:
                    continue
                notional_eur = ap[day][t_idx] * n_contracts * er_arr[day]
                if ticker == 'BTC-USD':
                    blocked_eur += notional_eur
                elif ticker in ETF_TICKERS:
                    etf_c = min(n_contracts, ETF_CONTRACTS_LIMIT)
                    margin_c = max(n_contracts - ETF_CONTRACTS_LIMIT, 0)
                    etf_notional = etf_c * ap[day][t_idx] * er_arr[day]
                    margin_notional = margin_c * ap[day][t_idx] * er_arr[day] * MARGIN_RATE
                    blocked_eur += etf_notional + margin_notional
                else:
                    blocked_eur += notional_eur * MARGIN_RATE


            free_cash_eur = max(portfolio_eur[day] - blocked_eur, 0.0)  if euribor_arr[day] > MIN_EURIBOR else 0.0
            cash_return_eur = free_cash_eur * euribor_arr[day] / 255

            portfolio_eur[day] += cash_return_eur
            daily_ret_eur[day] += cash_return_eur
            cash_eur_arr[day] = free_cash_eur

            # ── Order log ─────────────────────────────────────────────────
            date = trading_dates[day]
            for t_idx, ticker in enumerate(tickers_list):
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
                        #'price': round(float(order_price[t_idx]), 3),
                        'price': round(float(sent_order_price[t_idx]), 3),
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
        cash_eur_series = pd.Series(cash_eur_arr, index=trading_dates)

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
            'cash_eur': cash_eur_series,
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

    # ── Separate cash from futures ────────────────────────────────────────
    futures_tickers = [t for t in positions.columns if t != 'cash']
    positions_futures = positions[futures_tickers]
    opens_futures     = opens[futures_tickers]
    highs_futures     = highs[futures_tickers]
    lows_futures      = lows[futures_tickers]
    closes_futures    = closes[futures_tickers]
    mults_array_futures = np.array([
        settings['mults'][t] if t in settings['mults'] else 1.0
        for t in futures_tickers
    ])

    # Buy/Sell Triggers & Stop Prices (futures only)
    buy_trigger, sell_trigger, sell_stop_price, buy_stop_price = compute_buy_sell_triggers(
        positions_futures, opens_futures, closes_futures, lows_futures, highs_futures
    )

    # Exchange Rate EUR/USD (day after)
    exchange_rate = 1 / closes["EURUSD=X"].shift(1).fillna(method='bfill')

    # Set cash start
    startcash_usd = startcash / exchange_rate.iloc[0]

    # Euribor series aligned to backtest dates
    from Market_Data_Feed import get_euribor_1y_daily
    euribor_df = get_euribor_1y_daily().reindex(closes.index, method="ffill")
    euribor_series = euribor_df['Euribor'].fillna(0)

    # Out-of-loop calculations (futures only)
    weights_div_asset_price, asset_price = compute_out_of_backtest_loop(
        closes_futures, positions_futures, mults_array_futures
    )

    # Create backtest instance
    backtest_settings = BacktestSettings(
        upgrade_threshold=settings.get('upgrade_threshold', 0.20),
        commision=commision,
        buy_at_market=settings.get('buy_at_market', False)
    )
    backtest = Backtest(backtest_settings)

    # Single sequential pass — futures only
    pos, portfolio_value_usd, bt_log_dict = backtest.compute_backtest_sequential(
        weights_div_asset_price, asset_price,
        opens_futures, highs_futures, lows_futures, closes_futures,
        mults_array_futures, positions_futures,
        buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
        exchange_rate, startcash_usd, startcash, exposition_lim,
        euribor=euribor_series,
        max_n_contracts=max_n_contracts,
    )

    bt_log_dict['pos']             = pos
    bt_log_dict['portfolio_value'] = portfolio_value_usd

    # Add cash positions back for log history if cash exists
    if 'cash' in positions.columns:
        bt_log_dict['pos']['cash'] = positions['cash'].reindex(pos.index).fillna(0).astype(int)

    # Get Log History
    log_history = create_log_history(bt_log_dict)

    # Quantstats Report
    if settings.get('qstats', False):
        bt_qstats_report(bt_log_dict, closes, settings['add_days'], exchange_rate)

    plot_portfolio_composition(bt_log_dict, closes, exchange_rate, settings)

    return bt_log_dict, log_history


def sanitize_dataset(positions, data_dict):

    common_index = positions.index
    for df in data_dict.values():
        common_index = common_index.intersection(df.index)

    positions = positions.reindex(common_index).fillna(0)

    for tick, df in data_dict.items():
        data_dict[tick] = df.reindex(common_index).fillna(method='ffill')

    desired_order = list(data_dict.keys())

    opens  = pd.concat([data_dict[key]['Open']  for key in desired_order], axis=1, keys=desired_order)
    highs  = pd.concat([data_dict[key]['High']  for key in desired_order], axis=1, keys=desired_order)
    lows   = pd.concat([data_dict[key]['Low']   for key in desired_order], axis=1, keys=desired_order)
    closes = pd.concat([data_dict[key]['Close'] for key in desired_order], axis=1, keys=desired_order)

    start_mask = (positions > 0).any(axis=1)
    start = positions[start_mask].index[0]

    positions = positions.loc[start:]
    opens     = opens.loc[start:]
    highs     = highs.loc[start:]
    lows      = lows.loc[start:]
    closes    = closes.loc[start:]

    return positions, data_dict, opens, highs, lows, closes


def get_settings_values(settings, tickers):
    mults_array = np.array([
        settings['mults'][tick] if tick in settings['mults']
        else (print(f"ADVERTENCIA: Falta {tick}, usando 1.0") or 1.0)
        for tick in tickers
    ])
    startcash       = settings['startcash']
    exposition_lim  = settings['exposition_lim']
    commision       = settings['commision']
    max_n_contracts = settings['max_n_contracts']

    return mults_array, startcash, exposition_lim, commision, max_n_contracts


def compute_out_of_backtest_loop(closes, weights, mults):
    """Calculate asset prices and weights divided by asset prices."""
    asset_price = closes.multiply(mults, axis=1)
    yesterday_asset_price = asset_price.shift(1)
    yesterday_asset_price_mean = yesterday_asset_price.rolling(5, min_periods=1).mean()
    weights_div_asset_price = weights / yesterday_asset_price_mean

    return weights_div_asset_price, asset_price


def compute_buy_sell_triggers(weights, opens, closes, lows, highs):
    """Calculate buy/sell triggers and stop prices."""
    weights_min = weights.shift(1).rolling(5).min()
    weights_up  = weights.ge(weights_min, axis=0)

    weights_max = weights.shift(1).rolling(5).max()
    weights_dn  = weights.le(weights_max, axis=0)


    lows_up_threshold=compute_sell_volat_stop_price(closes, lows, delta=7) #6
    lows_up = lows.shift(1).ge(lows_up_threshold, axis=0)


    highs_max = highs.shift(2).rolling(5).max()*1.005
    highs_dn  = highs.shift(1).le(highs_max, axis=0)

    buy_trigger  = weights_up & lows_up
    sell_trigger = highs_dn & weights_dn

    sell_stop_price = compute_sell_volat_stop_price(closes, lows, delta=3)  # 6
    buy_stop_price = compute_buy_volat_stop_price(closes, highs, delta=3.5) #3.5

    debug = False
    if debug:
        plot_df = pd.DataFrame()
        ticker = 'NQ=F'
        plot_df['closes'] = closes[ticker]
        plot_df['high'] = highs[ticker]
        plot_df['buy_stop_price'] = buy_stop_price[ticker]
        #plot_df['lows'] = lows[ticker]
        #plot_df['sell_stop_price'] = sell_stop_price[ticker]
        #plot_df['weights_up']     = weights_up[ticker]
        #plot_df['buy_trigger']    = buy_trigger[ticker]
        plot_df.tail(400).plot(title=ticker)
        print(plot_df[:-4].tail(5))

    return buy_trigger, sell_trigger, sell_stop_price, buy_stop_price


def create_log_history(bt_log_dict):
    """Create log history from backtest log dictionary."""
    portfolio_value_eur = bt_log_dict['portfolio_value_eur']
    pos = bt_log_dict['pos']
    tickers = pos.columns

    order_dict  = get_log_dict_by_ticker_dict(bt_log_dict['order_dict'], tickers)
    broker_dict = get_log_dict_by_ticker_dict(bt_log_dict['broker_dict'], tickers)

    order_dict  = {ticker: order_dict[ticker].reset_index(drop=True)  for ticker in tickers}
    broker_dict = {ticker: broker_dict[ticker].reset_index(drop=True) for ticker in tickers}

    log_history_dict = {
        ticker: pd.concat([order_dict[ticker], broker_dict[ticker]], axis=0)
                  .dropna().sort_values(by='date_time')
        for ticker in tickers
    }

    log_history = pd.concat(log_history_dict.values(), axis=0).sort_values(by='date_time')
    log_history[tickers] = np.nan
    log_history = log_history[['date_time'] + list(tickers) + list(log_history.columns[1:-len(tickers)])]

    # Create End of Day df
    eod_df = pd.DataFrame(
        index=pos.index,
        columns=list(log_history.columns) + [
            'portfolio_value', 'portfolio_value_eur', 'pos_value',
            'ddn', 'dayly_profit', 'dayly_profit_eur',
            'pre_portfolio_value', 'exchange_rate', 'ddn_eur', 'cash_eur'
        ]
    )
    eod_df['date_time'] = pos.index + pd.Timedelta(days=0, hours=23, minutes=59, seconds=59)
    eod_df[tickers]     = pos
    eod_df['event']     = 'End of Day'

    keys = ['portfolio_value', 'portfolio_value_eur', 'pos_value',
            'dayly_profit_eur', 'exchange_rate', 'cash_eur']
    for key in keys:
        eod_df[key] = bt_log_dict[key]

    eod_df['dayly_profit']        = eod_df['dayly_profit_eur'] / eod_df['exchange_rate']
    eod_df['pre_portfolio_value'] = eod_df['portfolio_value'].shift(1)
    eod_df = round(eod_df, 2)

    eod_df['ddn']     = eod_df['portfolio_value'].rolling(252 * 3, min_periods=250).max() / eod_df['portfolio_value'] - 1
    eod_df['ddn_eur'] = eod_df['portfolio_value_eur'].rolling(252 * 3, min_periods=250).max() / eod_df['portfolio_value_eur'] - 1
    eod_df[['ddn', 'ddn_eur']] = round(eod_df[['ddn', 'ddn_eur']], 4)
    eod_df = eod_df.reset_index(drop=True)

    log_history = pd.concat([log_history, eod_df], axis=0).sort_values(by='date_time')

    for ticker in tickers:
        is_executed = log_history['event'] == 'Executed'
        is_ticker   = log_history['ticker'] == ticker
        log_history.loc[is_executed & is_ticker, ticker] = log_history.loc[is_executed & is_ticker, 'pos']

    log_history[tickers] = log_history[tickers].fillna(method='ffill')
    log_history[tickers] = log_history[tickers].fillna(0).astype(int)

    log_history['date_time'] = pd.to_datetime(log_history['date_time'], errors='coerce')
    log_history['date']      = log_history['date_time'].dt.date

    return log_history


def get_log_dict_by_ticker_dict(bt_log_dict_entry, tickers):
    bt_log_by_ticker_dict = {}

    if isinstance(bt_log_dict_entry, pd.DataFrame):
        for ticker in tickers:
            ticker_df = bt_log_dict_entry[bt_log_dict_entry['ticker'] == ticker].copy()
            bt_log_by_ticker_dict[ticker] = ticker_df.reset_index(drop=True)
        return bt_log_by_ticker_dict

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


def bt_qstats_report(bt_log_dict, closes, add_days, exchange_rate):
    q_title    = 'Cash Backtest Markowitz Vectorized'
    path       = "results\\"
    q_filename = os.path.abspath(path + q_title + '.html')
    q_returns  = bt_log_dict['portfolio_value_eur'].pct_change().iloc[:-add_days]
    q_returns.index = pd.to_datetime(q_returns.index, utc=True).tz_convert(None)

    q_benchmark_ticker = 'ES=F'
    q_benchmark = (closes[q_benchmark_ticker] * exchange_rate).pct_change().iloc[:-add_days]
    q_benchmark.index = pd.to_datetime(q_benchmark.index, utc=True).tz_convert(None)

    import quantstats_lumi as quantstats
    quantstats.reports.html(
        q_returns, title=q_title,
        benchmark=q_benchmark, benchmark_title=q_benchmark_ticker,
        output=q_filename
    )
    import webbrowser
    webbrowser.open(q_filename)

    return q_returns, q_title, q_benchmark, q_benchmark_ticker, q_filename


def compute_sell_volat_stop_price(closes, lows, delta=6):
    downside_noise = (closes.shift(1) - lows).shift(1)

    roll_mean = downside_noise.rolling(20, min_periods=1).mean()
    roll_std  = downside_noise.rolling(20, min_periods=1).std().fillna(0)

    volat_buffer = roll_mean + delta * roll_std
    volat_buffer = volat_buffer.clip(lower=0.0001 * closes.shift(1),upper=0.12   * closes.shift(1))
    #(volat_buffer / closes.shift(1)).plot(title='sell volat_buffer pct')

    volat_stop_price = closes.shift(1) - volat_buffer
    volat_stop_price = volat_stop_price.rolling(22, min_periods=1).max()
    volat_stop_price = volat_stop_price.replace(0, np.nan).bfill().fillna(0)


    return volat_stop_price


def compute_buy_volat_stop_price(closes, highs, delta=1):
    upside_noise = (highs - closes.shift(1)).shift(1)

    roll_mean = upside_noise.rolling(20, min_periods=1).mean()
    roll_std  = upside_noise.rolling(20, min_periods=1).std().fillna(0)

    volat_buffer     = roll_mean + delta * roll_std
    volat_buffer = volat_buffer.clip(lower=0.0001 * closes.shift(1), upper=0.12 * closes.shift(1))
    #(volat_buffer/closes.shift(1)).plot(title='buy volat_buffer pct')
    volat_stop_price = closes.shift(1) + volat_buffer
    volat_stop_price = volat_stop_price.rolling(22, min_periods=1).min()
    volat_stop_price = volat_stop_price.replace(0, np.nan).bfill().fillna(0)

    return volat_stop_price

def plot_portfolio_composition(bt_log_dict, closes, exchange_rate, settings):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Get EOD positions and portfolio value
    pos = bt_log_dict['pos']
    portfolio_eur = bt_log_dict['portfolio_value_eur']
    cash_eur = bt_log_dict['cash_eur']
    er = bt_log_dict['exchange_rate']

    futures_tickers = [t for t in pos.columns if t != 'cash']

    # Compute EUR value per ticker over time
    ticker_eur = pd.DataFrame(index=pos.index)
    for ticker in futures_tickers:
        if ticker in closes.columns and ticker in settings['mults']:
            mult = settings['mults'][ticker]
            ticker_eur[ticker] = pos[ticker] * closes[ticker] * mult * er

    ticker_eur = ticker_eur.clip(lower=0)
    ticker_eur['cash'] = cash_eur.reindex(ticker_eur.index).fillna(0)
    ticker_eur = ticker_eur.reindex(portfolio_eur.index).fillna(0)

    # Compute blocked_eur over time
    blocked_eur = pd.Series(0.0, index=pos.index)
    for ticker in futures_tickers:
        if ticker not in closes.columns or ticker not in settings['mults']:
            continue
        mult = settings['mults'][ticker]
        n_contracts = pos[ticker]
        notional_eur = n_contracts * closes[ticker] * mult * er

        if ticker == 'BTC-USD':
            blocked_eur += notional_eur
        elif ticker in ETF_TICKERS:
            etf_c = n_contracts.clip(upper=ETF_CONTRACTS_LIMIT)
            margin_c = (n_contracts - ETF_CONTRACTS_LIMIT).clip(lower=0)
            blocked_eur += (etf_c * closes[ticker] * mult * er) + \
                           (margin_c * closes[ticker] * mult * er * MARGIN_RATE)
        else:
            blocked_eur += notional_eur * MARGIN_RATE

    # Euribor
    from Market_Data_Feed import get_euribor_1y_daily
    euribor_df = get_euribor_1y_daily().reindex(pos.index, method="ffill")
    euribor = euribor_df['Euribor'].fillna(0)

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    # ── Chart 1: Stacked area - portfolio composition ──
    ax1 = axes[0]
    colors = plt.cm.tab10.colors
    labels = list(ticker_eur.columns)
    ax1.stackplot(
        ticker_eur.index,
        [ticker_eur[col] for col in labels],
        labels=labels,
        colors=colors[:len(labels)],
        alpha=0.7
    )
    ax1.plot(portfolio_eur.index, portfolio_eur.values, 'k-', linewidth=1.5, label='Portfolio Total')
    ax1.set_title('Portfolio Composition (EUR)', fontsize=12)
    ax1.set_ylabel('EUR')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}€'))

    # ── Chart 2: Blocked EUR vs Portfolio EUR ──
    ax2 = axes[1]
    ax2.plot(portfolio_eur.index, portfolio_eur.values, 'b-', linewidth=1.5, label='Portfolio EUR')
    ax2.plot(blocked_eur.index, blocked_eur.values, 'r-', linewidth=1.5, label='Blocked EUR')
    ax2.fill_between(blocked_eur.index, blocked_eur.values, alpha=0.2, color='red')
    ax2.set_title('Blocked EUR vs Portfolio EUR', fontsize=12)
    ax2.set_ylabel('EUR')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}€'))

    # ── Chart 3: Cash % of portfolio ──
    ax3 = axes[2]
    cash_pct = (bt_log_dict['cash_eur'].reindex(portfolio_eur.index).fillna(0) / portfolio_eur * 100).fillna(0)
    ax3.fill_between(cash_pct.index, cash_pct.values, alpha=0.5, color='green', label='Cash %')
    ax3.plot(cash_pct.index, cash_pct.values, 'g-', linewidth=1)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
    ax3.set_title('Cash % of Portfolio', fontsize=12)
    ax3.set_ylabel('%')
    ax3.legend(loc='upper left', fontsize=8)

    # ── Chart 4: Euribor ──
    ax4 = axes[3]
    ax4.plot(euribor.index, euribor.values * 100, 'purple', linewidth=1.5, label='Euribor 1Y')
    ax4.fill_between(euribor.index, euribor.values * 100, alpha=0.2, color='purple')
    ax4.axhline(y=0.1, color='red', linestyle='--', linewidth=0.8, label='Min threshold')
    ax4.set_title('Euribor 1Y (%)', fontsize=12)
    ax4.set_ylabel('%')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax4.xaxis.set_major_locator(mdates.YearLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()






