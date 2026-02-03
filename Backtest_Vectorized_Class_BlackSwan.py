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
    portfolio_window: int = 22 * 12
    min_periods: int = 1


class BacktestVectorized:
    """Vectorized implementation of backtest logic."""

    def __init__(self, settings: BacktestSettings):
        self.settings = settings

    def compute_backtest_until_convergence(
            self,
            weights_div_asset_price: pd.DataFrame,
            asset_price: pd.DataFrame,
            opens: pd.DataFrame,
            highs: pd.DataFrame,
            lows: pd.DataFrame,
            closes: pd.DataFrame,
            mults: np.ndarray,
            portfolio_value_usd: pd.Series,
            weights: pd.DataFrame,
            buy_trigger: pd.DataFrame,
            sell_trigger: pd.DataFrame,
            sell_stop_price: pd.DataFrame,
            buy_stop_price: pd.DataFrame,
            exchange_rate: pd.Series,
            startcash_usd: float,
            startcash: float,
            exposition_lim: float,
            pos: pd.DataFrame,
            max_iterations: int = 200,
            max_n_contracts: int = 50,
            swan_stop_price: pd.DataFrame = None,  # Added this argument
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Execute backtest computation until positions converge or max iterations reached."""

        # Pre-allocate arrays for all iterations
        n_rows, n_cols = pos.shape
        all_positions = np.zeros((max_iterations + 1, n_rows, n_cols), dtype=np.int32)

        # Initialize with starting positions
        all_positions[0] = pos.values

        # Pre-allocate array for portfolio values
        all_portfolio_values = np.zeros((max_iterations + 1, len(portfolio_value_usd)), dtype=np.float64)
        all_portfolio_values[0] = portfolio_value_usd.values

        # Store the log dictionaries
        all_bt_log_dicts = [{} for _ in range(max_iterations + 1)]

        # Vectorized computation of all iterations
        for i in range(max_iterations):
            # Create DataFrame from the current position values
            current_pos = pd.DataFrame(
                all_positions[i],
                index=pos.index,
                columns=pos.columns
            )

            # Create Series from the current portfolio values
            current_portfolio = pd.Series(
                all_portfolio_values[i],
                index=portfolio_value_usd.index
            )

            # Call compute_backtest - now passing swan_stop_price forward
            new_pos, new_portfolio, bt_log_dict = self.compute_backtest(
                weights_div_asset_price, asset_price, opens, highs, lows, closes, mults,
                current_portfolio, weights, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
                exchange_rate, startcash_usd, startcash, exposition_lim, current_pos,
                max_n_contracts, swan_stop_price
            )

            # Store results for this iteration
            all_positions[i + 1] = new_pos.values
            all_portfolio_values[i + 1] = new_portfolio.values
            all_bt_log_dicts[i + 1] = bt_log_dict

            # Check for convergence
            if np.array_equal(all_positions[i + 1], all_positions[i]):
                break

        # Get the final results
        final_pos = pd.DataFrame(
            all_positions[i + 1],
            index=pos.index,
            columns=pos.columns
        )

        final_portfolio = pd.Series(
            all_portfolio_values[i + 1],
            index=portfolio_value_usd.index
        )

        final_bt_log_dict = all_bt_log_dicts[i + 1]
        final_bt_log_dict['n_iter'] = i + 1

        return final_pos, final_portfolio, final_bt_log_dict

    def compute_backtest(
            self,
            weights_div_asset_price: pd.DataFrame,
            asset_price: pd.DataFrame,
            opens: pd.DataFrame,
            highs: pd.DataFrame,
            lows: pd.DataFrame,
            closes: pd.DataFrame,
            mults: np.ndarray,
            portfolio_value_usd: pd.Series,
            weights: pd.DataFrame,
            buy_trigger: pd.DataFrame,
            sell_trigger: pd.DataFrame,
            sell_stop_price: pd.DataFrame,
            buy_stop_price: pd.DataFrame,
            exchange_rate: pd.Series,
            startcash_usd: float,
            startcash: float,
            exposition_lim: float,
            pos: pd.DataFrame,
            max_n_contracts: int,
            swan_stop_price: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        bt_log_dict = {}
        tickers = weights.columns
        trading_dates = weights.index

        # 1. Prep Previous Positions and Target Sizes
        prev_pos = pos.shift(1).fillna(0).astype(int)
        portfolio_to_invest = portfolio_value_usd.shift(1).rolling(
            self.settings.portfolio_window, min_periods=self.settings.min_periods
        ).min()

        target_size_raw = weights_div_asset_price.multiply(portfolio_to_invest, axis=0).fillna(0)
        target_size_raw[target_size_raw > self.settings.upgrade_threshold] = target_size_raw.clip(lower=1)
        target_size = round(target_size_raw, 0).astype(int).clip(lower=0, upper=max_n_contracts)
        target_trade_size = target_size - prev_pos

        # 2. Define the "Broker" Stop Price
        # We take the best available protection (Max of Technical vs Swan)
        # This ensures the order is 'placed' at this price
        effective_sell_stop = np.maximum(sell_stop_price, swan_stop_price)
        sell_stop_price_adj = effective_sell_stop.clip(lower=None, upper=opens)

        # 3. Order Creation Logic
        is_buy = (target_trade_size > 0) & buy_trigger
        is_sell = (target_trade_size < 0)

        # Identify if the execution was driven by the Swan logic for logging
        # (Swan is the active trigger if it's higher than the technical stop)
        is_swan_active = (swan_stop_price > sell_stop_price)

        # Initialize Order Data
        order_prices = pd.DataFrame(index=trading_dates, columns=tickers, dtype=float)
        order_type = pd.DataFrame('None', index=trading_dates, columns=tickers)
        order_side = pd.DataFrame('None', index=trading_dates, columns=tickers)

        # Fill Buy Orders
        buy_stop_adj = buy_stop_price.clip(lower=opens, upper=None)
        order_prices.where(~is_buy, buy_stop_adj, inplace=True)
        order_side.where(~is_buy, 'Buy', inplace=True)
        order_type.where(~is_buy, 'Stop', inplace=True)

        # Fill Sell Orders (Using the effective_sell_stop)
        order_prices.where(~is_sell, sell_stop_price_adj, inplace=True)
        order_side.where(~is_sell, 'Sell', inplace=True)
        order_type.where(~is_sell, 'Stop', inplace=True)

        # 4. Execution Engine
        # Check if price touched the orders
        in_market = (order_prices >= lows) & (order_prices <= highs)
        broker_event = pd.DataFrame('None', index=trading_dates, columns=tickers)
        broker_event = np.where(in_market & (order_side != 'None'), 'Executed', 'Canceled')
        # Refine Swan Labeling: If it was a sell execution and swan was the tighter stop
        swan_fill = (broker_event == 'Executed') & (order_side == 'Sell') & is_swan_active
        broker_event = np.where(swan_fill, 'Swan_Executed', broker_event)
        broker_event = pd.DataFrame(broker_event, index=trading_dates, columns=tickers)
        broker_event.where(order_side != 'None', 'None', inplace=True)

        # 5. Determine Execution Size
        # If Swan_Executed, we force the half-reduction
        half_reduction = -np.ceil(prev_pos * 0.5).astype(int)

        exec_size = pd.DataFrame(0, index=trading_dates, columns=tickers)
        # Standard fills (Technical Buys / Technical Sells)
        standard_fill = (broker_event == 'Executed')
        exec_size = exec_size.where(~standard_fill, target_trade_size)
        # Swan fills (Override with half reduction)
        exec_size = exec_size.where(~swan_fill, half_reduction)

        # Final Cleaning
        exec_price = order_prices.where(broker_event != 'None', 0)
        trading_cost = exec_size.abs() * self.settings.commision
        updated_pos = prev_pos + exec_size

        # 6. Returns Calculation
        hold_returns = (prev_pos * asset_price.diff().fillna(0)).sum(axis=1)
        trade_returns = (exec_size * (closes - exec_price).multiply(mults, axis=1)).sum(axis=1)
        daily_returns_usd = hold_returns + trade_returns - trading_cost.sum(axis=1)

        portfolio_value_usd_new = startcash_usd + daily_returns_usd.cumsum()

        # 7. Update bt_log_dict (Maintaining your log structure)
        bt_log_dict['order_dict'] = {
            'date_time': pd.DataFrame({t: trading_dates for t in tickers}, index=trading_dates),
            'event': pd.DataFrame(np.where(order_side != 'None', 'Created', 'None'), index=trading_dates, columns=tickers),
            'pos': prev_pos,
            'ticker': pd.DataFrame({t: [t] * len(trading_dates) for t in tickers}, index=trading_dates),
            'B_S': order_side,
            'size': target_trade_size,
            'price': order_prices,
        }

        bt_log_dict['broker_dict'] = {
            'date_time': bt_log_dict['order_dict']['date_time'] + pd.Timedelta(seconds=10),
            'event': broker_event,
            'pos': updated_pos,
            'ticker': bt_log_dict['order_dict']['ticker'],
            'B_S': order_side,
            'size': exec_size,
            'price': exec_price,
            'commision': trading_cost,
        }

        bt_log_dict.update({
            'pos': updated_pos,
            'pos_value': (asset_price * updated_pos).sum(axis=1),
            'portfolio_value': portfolio_value_usd_new,
            'portfolio_value_eur': startcash + (daily_returns_usd * exchange_rate).cumsum(),
            'dayly_profit_eur': daily_returns_usd * exchange_rate,
            'exchange_rate': exchange_rate,
            'exposition': (asset_price * updated_pos).sum(axis=1) / portfolio_to_invest
        })

        return updated_pos, portfolio_value_usd_new, bt_log_dict

def compute_backtest_vectorized(
        positions: pd.DataFrame,
        settings: Dict,
        data_dict: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """Main entry point for vectorized backtest computation."""

    #Get Sanitized Index Data
    positions, data_dict, opens, highs, lows, closes = sanitize_dataset(positions, data_dict)

    # Get settings values from settings
    mults_array,startcash, exposition_lim, commision, max_n_contracts=get_settings_values(settings,positions.columns)

    # NEW: Calculate the Black Swan levels
    swan_stop_price = compute_black_swan_thresholds(closes, lows)

    # Get Buy/Sell Triggers & Stop Prices
    buy_trigger, sell_trigger, sell_stop_price, buy_stop_price = compute_buy_sell_triggers(positions,closes, lows, highs)

    # Get historical of Exchange Rate EUR/USD (day after)
    exchange_rate = 1 / closes["EURUSD=X"].shift(1).fillna(method='bfill')

    # Set cash start
    startcash_usd = startcash / exchange_rate.iloc[0]  # USD

    # Initialize Portfolio Value, Positions, orders
    portfolio_value_usd = pd.Series(startcash_usd, index=positions.index)
    pos = pd.DataFrame(0, index=positions.index, columns=positions.columns)

    # Out of loop calculations
    weights_div_asset_price, asset_price = compute_out_of_backtest_loop(closes, positions, mults_array)

    # Create backtest instance with settings - reuse the same instance
    backtest_settings = BacktestSettings(
        upgrade_threshold=settings.get('upgrade_threshold', 0.20),
        commision=commision,
        buy_at_market=settings.get('buy_at_market', False)
    )

    # Create backtest instance only once
    backtest = BacktestVectorized(backtest_settings)

    # Use the compute_backtest_until_convergence method to handle the loop internally
    pos, portfolio_value_usd, bt_log_dict = backtest.compute_backtest_until_convergence(
        weights_div_asset_price, asset_price, opens, highs, lows, closes, mults_array,
        portfolio_value_usd, positions, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
        exchange_rate, startcash_usd, startcash, exposition_lim, pos,
        max_n_contracts=max_n_contracts,
        swan_stop_price=swan_stop_price  # <--- Pass it here
    )

    # Add Series to dict - optimize by updating directly
    bt_log_dict['pos'] = pos
    bt_log_dict['portfolio_value'] = portfolio_value_usd

    # Get Log History
    log_history = create_log_history(bt_log_dict)

    # Quantstats Report
    if settings.get('qstats', False):

        bt_qstats_report(bt_log_dict, closes,settings['add_days'],exchange_rate)

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
    weights_mean = weights.rolling(5, min_periods=1).mean()
    weights_div_asset_price = weights_mean / yesterday_asset_price_mean

    return weights_div_asset_price, asset_price


def compute_buy_sell_triggers(weights, closes,lows, highs):
    """Calculate buy/sell triggers and stop prices."""
    # Weights Uptrend --> weight > previous 5 days lowest
    weights_min = weights.shift(1).rolling(5).min()
    weights_up = weights.gt(weights_min, axis=0)

    # Weights Downtrend --> weight < previous 5 days highest
    weights_max = weights.shift(1).rolling(5).max()
    weights_dn = weights.lt(weights_max, axis=0)

    # Lows Uptrend --> Yesterday low > previous 5 days lowest
    lows_min = lows.shift(1).rolling(5).min()
    lows_up = lows.ge(lows_min, axis=0)

    # Closes Uptrend
    if False:
        closes_mean_fast=closes.shift(1).rolling(5).mean()
        closes_mean_slow=closes.shift(1).rolling(22).mean()
        closes_up_fast=closes_mean_fast.shift(1).gt(closes_mean_slow.shift(4))
        closes_up_slow=closes_mean_slow.shift(1).gt(closes_mean_slow.shift(4))
        closes_mean_crosed_up=closes_mean_fast.mean().ge(closes_mean_slow)
        closes_uptrend=  closes_up_fast & closes_up_slow #& closes_mean_crosed_up #

    # Highs Downtrend --> Yesterday high < previous 5 days highest
    highs_max = highs.shift(1).rolling(5).max()
    highs_dn = highs.le(highs_max, axis=0)

    # Buy Trigger
    buy_trigger = lows_up & weights_up

    # Sell Trigger
    sell_trigger = highs_dn & weights_dn # Highs Downtrend

    # Get Sell Stop Price
    low_keep = lows_min.rolling(22).max()
    sell_stop_price = low_keep.fillna(0)

    # Get Buy Stop Price
    high_keep = highs_max.rolling(22).min()
    highs_std = highs.rolling(22).std().shift(1)
    buy_stop_price = high_keep + highs_std * 0.5

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
    """
    Final optimized version: Flattens trade data, merges daily risk metrics,
    and ensures all columns required by process_log_data are present.
    """

    def flatten_dict(d):
        ticker_df = d['ticker']
        flattened_rows = []
        for col in ticker_df.columns:
            temp_df = pd.DataFrame({
                'date_time': d['date_time'][col],
                'ticker': d['ticker'][col],
                'event': d['event'][col],
                'B_S': d['B_S'][col],
                'size': d['size'][col],
                'price': d['price'][col],
                'pos': d['pos'][col],
                'commision': d.get('commision', pd.DataFrame(0, index=ticker_df.index, columns=ticker_df.columns))[col]
            })
            flattened_rows.append(temp_df)
        return pd.concat(flattened_rows).reset_index(drop=True)

    # 1. Flatten Trades
    order_df = flatten_dict(bt_log_dict['order_dict'])
    broker_df = flatten_dict(bt_log_dict['broker_dict'])

    # 2. Build Daily Metrics (EOD)
    # This includes the metrics that were causing the KeyError
    eod_df = pd.DataFrame({
        'date_time': bt_log_dict['portfolio_value'].index + pd.Timedelta(hours=23, minutes=59, seconds=59),
        'event': 'End of Day',
        'portfolio_value': bt_log_dict['portfolio_value'].values,
        'portfolio_value_eur': bt_log_dict['portfolio_value_eur'].values,
        'dayly_profit_eur': bt_log_dict['dayly_profit_eur'].values,
        'exchange_rate': bt_log_dict['exchange_rate'].values,
        'pos_value': bt_log_dict.get('pos_value', pd.Series(0, index=bt_log_dict['portfolio_value'].index)).values,
        'ddn': bt_log_dict.get('ddn', pd.Series(0, index=bt_log_dict['portfolio_value'].index)).values,
        'dayly_profit': bt_log_dict.get('dayly_profit', pd.Series(0, index=bt_log_dict['portfolio_value'].index)).values
    })

    # 3. Combine and Clean
    log_history = pd.concat([order_df, broker_df, eod_df], ignore_index=True)
    log_history = log_history.sort_values(by=['date_time', 'ticker']).reset_index(drop=True)

    # Filter significant events and remove 'cash' ticker noise
    mask = (log_history['event'].isin(['Executed', 'Swan_Executed', 'End of Day', 'Created'])) & (log_history['ticker'] != 'cash')
    log_history = log_history[mask].copy()

    # 4. Fill Gaps & Add Missing Columns
    # Forward fill financial totals so trade rows show current portfolio context
    fill_cols = ['portfolio_value', 'portfolio_value_eur', 'exchange_rate', 'pos_value', 'ddn', 'dayly_profit', 'dayly_profit_eur']
    log_history[fill_cols] = log_history[fill_cols].ffill().bfill()

    # Add 'date' column for drop_duplicates logic
    log_history['date'] = pd.to_datetime(log_history['date_time']).dt.date

    # 5. Merge Ticker Positions (ES=F, NQ=F, etc.)
    tickers = bt_log_dict['pos'].columns.tolist()
    log_history = log_history.merge(bt_log_dict['pos'], left_on='date_time', right_index=True, how='left', suffixes=('', '_drop'))
    log_history[tickers] = log_history[tickers].ffill().fillna(0).astype(int)

    return log_history


def get_log_dict_by_ticker_dict(bt_log_dict, tickers):
    """Convert log dictionary to dictionary of DataFrames by ticker."""
    bt_log_by_ticker_dict = {}
    for ticker in tickers:
        # Save dict values
        df = pd.DataFrame()
        for key, value in bt_log_dict.items():
            try:
                if isinstance(value, pd.DataFrame):
                    if ticker in value.columns:
                        df[key] = value[ticker]
                elif isinstance(value, dict):
                    for dict_key, dict_value in value.items():
                        if isinstance(dict_value, pd.DataFrame) and ticker in dict_value.columns:
                            df[dict_key] = dict_value[ticker]
                        elif isinstance(dict_value, pd.Series):
                            df[dict_key] = dict_value
            except Exception as e:
                # Skip this key if there's an error
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


def compute_black_swan_thresholds(closes, lows):
    """
    Calculates the volatility-based floor for Black Swan events.
    Uses 2-sigma of the (Close - Low) 'downside wicks'.
    """
    # Downside noise: difference between close and intraday low
    downside_noise = (closes - lows)

    # 2-sigma buffer based on previous 20 days of downside wicks
    # Use shift(1) to avoid look-ahead bias
    swan_buffer = 2 * downside_noise.shift(1).rolling(20).std()

    # The floor level
    swan_stop_price = closes.shift(1) - swan_buffer

    return swan_stop_price.fillna(0)