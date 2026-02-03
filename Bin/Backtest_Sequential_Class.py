from dataclasses import dataclass
from typing import Dict, Tuple, List
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
    commission: float = 5.0
    buy_at_market: bool = False
    portfolio_window: int = 22 * 12
    min_periods: int = 1


class BacktestSequential:
    """Sequential implementation of backtest logic, processing one day at a time."""

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
            max_iterations: int = 200
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Execute backtest computation until positions converge or max iterations reached."""
        i = 0
        bt_log_dict = {}

        # Use numpy arrays for faster comparison
        prev_pos_values = None

        while i < max_iterations:
            # Call compute_backtest with all required arguments
            pos, portfolio_value_usd, bt_log_dict = self.compute_backtest(
                weights_div_asset_price, asset_price, opens, highs, lows, closes, mults,
                portfolio_value_usd, weights, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
                exchange_rate, startcash_usd, startcash, exposition_lim, pos
            )

            # Check if positions have changed using numpy for faster comparison
            pos_values = pos.values
            if prev_pos_values is not None and np.array_equal(pos_values, prev_pos_values):
                break

            prev_pos_values = pos_values.copy()
            i += 1

        bt_log_dict['n_iter'] = i
        return pos, portfolio_value_usd, bt_log_dict

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
            pos: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Execute backtest computation sequentially, day by day."""
        # Initialize log dictionary
        bt_log_dict = {}

        # Get dates and tickers
        dates = pos.index
        tickers = pos.columns

        # Create empty DataFrames for results with the same shape as input
        updated_pos = pos.copy()
        updated_portfolio_value_usd = portfolio_value_usd.copy()

        # Initialize DataFrames for tracking daily returns
        daily_returns_usd = pd.Series(0.0, index=dates)
        daily_returns_eur = pd.Series(0.0, index=dates)
        portfolio_value_eur = pd.Series(0.0, index=dates)

        # Initialize DataFrames for order and execution tracking
        order_data = {
            'prices': pd.DataFrame(0.0, index=dates, columns=tickers),
            'B_S': pd.DataFrame('None', index=dates, columns=tickers),
            'exectype': pd.DataFrame('None', index=dates, columns=tickers),
            'event': pd.DataFrame('None', index=dates, columns=tickers),
            'tickers_df': pd.DataFrame(index=dates, columns=tickers),
            'size': pd.DataFrame(0, index=dates, columns=tickers),
            'pos': pd.DataFrame(0, index=dates, columns=tickers)
        }

        # Fill tickers_df with ticker names
        for ticker in tickers:
            order_data['tickers_df'][ticker] = ticker

        execution_data = {
            'broker_event': pd.DataFrame('None', index=dates, columns=tickers),
            'exec_size': pd.DataFrame(0, index=dates, columns=tickers),
            'exec_price': pd.DataFrame(0.0, index=dates, columns=tickers),
            'trading_cost': pd.DataFrame(0.0, index=dates, columns=tickers),
            'updated_pos': pd.DataFrame(0, index=dates, columns=tickers)
        }

        # Process each day sequentially
        for i, date in enumerate(dates):
            if i == 0:
                # First day initialization
                prev_pos = pd.Series(0, index=tickers)
                prev_portfolio_value_usd = startcash_usd
            else:
                prev_date = dates[i-1]
                prev_pos = updated_pos.loc[prev_date]
                prev_portfolio_value_usd = updated_portfolio_value_usd.loc[prev_date]

            # Calculate portfolio to invest (min of previous portfolio values)
            # Use the same approach as in the vectorized implementation
            if i == 0:
                portfolio_to_invest = prev_portfolio_value_usd
            else:
                # Create a Series of previous portfolio values
                prev_values = pd.Series(updated_portfolio_value_usd.iloc[:i].values, index=dates[:i])
                # Shift by 1 and calculate rolling min
                portfolio_to_invest = prev_values.shift(1).rolling(
                    self.settings.portfolio_window,
                    min_periods=self.settings.min_periods
                ).min().iloc[-1]

            # Get current day's data
            current_weights_div_asset_price = weights_div_asset_price.loc[date]
            current_asset_price = asset_price.loc[date]
            current_open = opens.loc[date]
            current_high = highs.loc[date]
            current_low = lows.loc[date]
            current_close = closes.loc[date]
            current_exchange_rate = exchange_rate.loc[date]
            current_weight = weights.loc[date]
            current_buy_trigger = buy_trigger.loc[date]
            current_sell_trigger = sell_trigger.loc[date]
            current_sell_stop_price = sell_stop_price.loc[date]
            current_buy_stop_price = buy_stop_price.loc[date]

            # Compute target size
            target_size_raw = current_weights_div_asset_price * portfolio_to_invest
            target_size_raw = target_size_raw.fillna(0)  # Handle NA values
            target_size_raw[target_size_raw > self.settings.upgrade_threshold] = target_size_raw.clip(lower=1)
            target_size = round(target_size_raw).astype(int)

            # Calculate target trade size
            target_trade_size = target_size - prev_pos

            # Calculate position value and exposition
            target_pos_value = (current_asset_price * target_size).sum()
            targeted_exposition = target_pos_value / portfolio_to_invest if portfolio_to_invest > 0 else 0
            exposition_is_low = pd.Series({col: (targeted_exposition < exposition_lim) for col in tickers})

            # Generate orders
            for ticker in tickers:
                # Initialize order data
                order_data['pos'].loc[date, ticker] = prev_pos[ticker]
                order_data['size'].loc[date, ticker] = target_trade_size[ticker]

                # Process buy orders
                if target_trade_size[ticker] > 0 and exposition_is_low[ticker]:
                    order_data['B_S'].loc[date, ticker] = 'Buy'
                    if not self.settings.buy_at_market:
                        order_data['exectype'].loc[date, ticker] = 'Stop'
                        buy_price = max(current_open[ticker], current_buy_stop_price[ticker])
                        order_data['prices'].loc[date, ticker] = buy_price
                    else:
                        order_data['exectype'].loc[date, ticker] = 'Market'
                        order_data['prices'].loc[date, ticker] = current_open[ticker]
                    order_data['event'].loc[date, ticker] = 'Created'

                # Process sell orders
                elif target_trade_size[ticker] < 0:
                    order_data['B_S'].loc[date, ticker] = 'Sell'
                    order_data['exectype'].loc[date, ticker] = 'Stop'
                    sell_price = min(current_open[ticker], current_sell_stop_price[ticker])
                    order_data['prices'].loc[date, ticker] = sell_price
                    order_data['event'].loc[date, ticker] = 'Created'

            # Execute orders
            for ticker in tickers:
                if order_data['event'].loc[date, ticker] == 'Created':
                    price = order_data['prices'].loc[date, ticker]
                    # Check if price is in market range
                    if current_low[ticker] <= price <= current_high[ticker]:
                        execution_data['broker_event'].loc[date, ticker] = 'Executed'
                        execution_data['exec_size'].loc[date, ticker] = target_trade_size[ticker]
                        execution_data['exec_price'].loc[date, ticker] = price
                        execution_data['trading_cost'].loc[date, ticker] = abs(target_trade_size[ticker]) * self.settings.commission
                    else:
                        execution_data['broker_event'].loc[date, ticker] = 'Canceled'

            # Update positions
            current_pos = prev_pos.copy()
            for ticker in tickers:
                if execution_data['broker_event'].loc[date, ticker] == 'Executed':
                    current_pos[ticker] += execution_data['exec_size'].loc[date, ticker]

            # Store updated positions
            for ticker in tickers:
                execution_data['updated_pos'].loc[date, ticker] = current_pos[ticker]
                updated_pos.loc[date, ticker] = current_pos[ticker]

            # Calculate returns
            # Holding returns
            hold_returns_usd = 0
            if i > 0:
                for ticker in tickers:
                    price_diff = current_asset_price[ticker] - asset_price.loc[dates[i-1], ticker]
                    hold_returns_usd += prev_pos[ticker] * price_diff

            # Trading returns
            trading_returns_usd = 0
            trading_costs_usd = 0
            for ticker in tickers:
                if execution_data['broker_event'].loc[date, ticker] == 'Executed':
                    price_diff = (current_close[ticker] - execution_data['exec_price'].loc[date, ticker]) * mults[tickers.get_loc(ticker)]
                    trading_returns_usd += execution_data['exec_size'].loc[date, ticker] * price_diff
                    trading_costs_usd += execution_data['trading_cost'].loc[date, ticker]

            # Total daily returns
            daily_returns_usd.loc[date] = hold_returns_usd + trading_returns_usd - trading_costs_usd
            daily_returns_eur.loc[date] = daily_returns_usd.loc[date] * current_exchange_rate

            # Update portfolio values
            if i == 0:
                updated_portfolio_value_usd.loc[date] = startcash_usd + daily_returns_usd.loc[date]
                portfolio_value_eur.loc[date] = startcash + daily_returns_eur.loc[date]
            else:
                updated_portfolio_value_usd.loc[date] = updated_portfolio_value_usd.loc[dates[i-1]] + daily_returns_usd.loc[date]
                portfolio_value_eur.loc[date] = portfolio_value_eur.loc[dates[i-1]] + daily_returns_eur.loc[date]

        # Create log dictionary
        bt_log_dict['order_dict'] = order_data
        bt_log_dict['broker_dict'] = {
            'date_time': pd.DataFrame(index=dates, columns=tickers),  # This would need proper datetime values
            'event': execution_data['broker_event'],
            'pos': execution_data['updated_pos'],
            'ticker': order_data['tickers_df'],
            'B_S': order_data['B_S'],
            'exectype': order_data['exectype'],
            'size': execution_data['exec_size'],
            'price': execution_data['exec_price'].astype(float).round(3),
            'commission': execution_data['trading_cost'],
        }

        # Add position and portfolio values
        bt_log_dict['pos'] = updated_pos
        bt_log_dict['pos_value'] = pd.Series(index=dates)
        for date in dates:
            bt_log_dict['pos_value'].loc[date] = (asset_price.loc[date] * updated_pos.loc[date]).sum()

        bt_log_dict['portfolio_value'] = updated_portfolio_value_usd
        bt_log_dict['portfolio_value_eur'] = portfolio_value_eur
        bt_log_dict['dayly_profit_eur'] = daily_returns_eur
        bt_log_dict['exchange_rate'] = exchange_rate

        # Calculate exposition
        bt_log_dict['exposition'] = bt_log_dict['pos_value'] / portfolio_value_usd.shift(1).rolling(
            self.settings.portfolio_window,
            min_periods=self.settings.min_periods
        ).min()

        return updated_pos, updated_portfolio_value_usd, bt_log_dict


def compute_backtest_sequential(
        positions: pd.DataFrame,
        settings: Dict,
        data_dict: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """Main entry point for sequential backtest computation."""
    # Data from settings
    mults = settings['mults']
    startcash = settings['startcash']
    exposition_lim = settings['exposition_lim']
    commission = settings.get('commission', 5.0)

    # Get Open, High, Low, Closes from data_dict
    desired_order = list(data_dict.keys())
    opens = pd.concat([data_dict[key]['Open'] for key in desired_order], axis=1, keys=desired_order)
    highs = pd.concat([data_dict[key]['High'] for key in desired_order], axis=1, keys=desired_order)
    lows = pd.concat([data_dict[key]['Low'] for key in desired_order], axis=1, keys=desired_order)
    closes = pd.concat([data_dict[key]['Close'] for key in desired_order], axis=1, keys=desired_order)

    # Start - find the first date with positions
    start_mask = (positions > 0).any(axis=1)
    start = positions[start_mask].index[0]

    # Set Start at positions and Tickers Data
    positions = positions.loc[start:]
    opens = opens.loc[start:]
    highs = highs.loc[start:]
    lows = lows.loc[start:]
    closes = closes.loc[start:]

    # Get Buy/Sell Triggers & Stop Prices
    buy_trigger, sell_trigger, sell_stop_price, buy_stop_price = compute_buy_sell_triggers(positions, lows, highs)

    # mult dict to list in the tickers order
    mults_array = np.array([mults[tick] for tick in closes.columns])

    # Get historical of Exchange Rate EUR/USD (day after)
    exchange_rate = 1 / closes["EURUSD=X"].shift(1).fillna(method='bfill')

    # Set cash start
    startcash_usd = startcash / exchange_rate[start]  # USD

    # Initialize Portfolio Value, Positions
    portfolio_value_usd = pd.Series(startcash_usd, index=positions.index)
    pos = pd.DataFrame(0, index=positions.index, columns=positions.columns)

    # Out of loop calculations
    weights_div_asset_price, asset_price = compute_out_of_backtest_loop(closes, positions, mults_array)

    # Create backtest instance with settings
    backtest_settings = BacktestSettings(
        upgrade_threshold=settings.get('upgrade_threshold', 0.20),
        commission=commission,
        buy_at_market=settings.get('buy_at_market', False)
    )

    # Create backtest instance
    backtest = BacktestSequential(backtest_settings)

    # Use the compute_backtest_until_convergence method to handle the loop internally
    pos, portfolio_value_usd, bt_log_dict = backtest.compute_backtest_until_convergence(
        weights_div_asset_price, asset_price, opens, highs, lows, closes, mults_array,
        portfolio_value_usd, positions, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
        exchange_rate, startcash_usd, startcash, exposition_lim, pos
    )

    # Add Series to dict
    bt_log_dict['pos'] = pos
    bt_log_dict['portfolio_value'] = portfolio_value_usd

    # Get Log History
    log_history = create_log_history(bt_log_dict)

    # Quantstats Report
    if settings.get('qstats', False):
        q_title = 'Cash Backtest Markowitz Sequential'
        path = "../results\\"
        q_filename = os.path.abspath(path + q_title + '.html')
        q_returns = bt_log_dict['portfolio_value_eur'].pct_change().iloc[:-settings['add_days']]
        q_benchmark_ticker = 'ES=F'
        q_benchmark = (closes[q_benchmark_ticker] * exchange_rate).pct_change().iloc[:-settings['add_days']]

        import quantstats_lumi as quantstats
        quantstats.reports.html(q_returns, title=q_title, benchmark=q_benchmark, benchmark_title=q_benchmark_ticker, output=q_filename)
        import webbrowser
        webbrowser.open(q_filename)

    return bt_log_dict, log_history


# Reuse these functions from Backtest_Vectorized_Class.py
def compute_out_of_backtest_loop(closes, weights, mults):
    """Calculate asset prices and weights divided by asset prices."""
    asset_price = closes.multiply(mults, axis=1)  # USD
    yesterday_asset_price = asset_price.shift(1)
    yesterday_asset_price_mean = yesterday_asset_price.rolling(5, min_periods=1).mean()
    weights_mean = weights.rolling(5, min_periods=1).mean()
    weights_div_asset_price = weights_mean / yesterday_asset_price_mean

    return weights_div_asset_price, asset_price


def compute_buy_sell_triggers(weights, lows, highs):
    """Calculate buy/sell triggers and stop prices."""
    # Weights Uptrend --> Yesterday low > previous 3 days lowest
    weights_min = weights.shift(2).rolling(5).min()
    weights_up = weights.shift(1).gt(weights_min, axis=0)

    # Lows Uptrend --> Yesterday low > previous 5 days lowest
    lows_min = lows.shift(1).rolling(5).min()
    lows_up = lows.shift(1).ge(lows_min, axis=0)

    # Highs Downtrend --> Yesterday high < previous 5 days highest
    highs_max = highs.shift(2).rolling(5).max()
    highs_dn = (highs.shift(1)).le(highs_max, axis=0)

    # Buy Trigger
    buy_trigger = lows_up & weights_up

    # Sell Trigger
    sell_trigger = highs_dn  # Highs Downtrend

    # Get Sell Stop Price
    low_keep = lows_min.rolling(22).max()
    sell_stop_price = low_keep

    # Get Buy Stop Price
    high_keep = highs_max.rolling(22).min()
    highs_std = highs.rolling(22).std().shift(1)
    buy_stop_price = high_keep + highs_std * 0.5

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
    log_history_dict = {}
    for ticker in tickers:
        try:
            # Check if both DataFrames have data
            if not order_dict[ticker].empty and not broker_dict[ticker].empty:
                # Concatenate and sort if 'date_time' column exists
                combined_df = pd.concat([order_dict[ticker], broker_dict[ticker]], axis=0).dropna()
                if 'date_time' in combined_df.columns:
                    log_history_dict[ticker] = combined_df.sort_values(by='date_time')
                else:
                    # If 'date_time' doesn't exist, just use the concatenated DataFrame
                    log_history_dict[ticker] = combined_df
            elif not order_dict[ticker].empty:
                log_history_dict[ticker] = order_dict[ticker]
            elif not broker_dict[ticker].empty:
                log_history_dict[ticker] = broker_dict[ticker]
            else:
                # Create an empty DataFrame with the same columns as would be expected
                log_history_dict[ticker] = pd.DataFrame(columns=['date_time', 'event', 'ticker', 'B_S', 'exectype', 'size', 'price', 'commission'])
        except Exception as e:
            print(f"Warning: Error processing log history for ticker {ticker}: {str(e)}")
            # Create an empty DataFrame with the same columns as would be expected
            log_history_dict[ticker] = pd.DataFrame(columns=['date_time', 'event', 'ticker', 'B_S', 'exectype', 'size', 'price', 'commission'])

    # Concatenate all tickers
    try:
        log_history = pd.concat(log_history_dict.values(), axis=0)
        if 'date_time' in log_history.columns:
            log_history = log_history.sort_values(by='date_time')
    except Exception as e:
        print(f"Warning: Error concatenating log history: {str(e)}")
        # Create an empty DataFrame with the expected columns
        log_history = pd.DataFrame(columns=['date_time', 'event', 'ticker', 'B_S', 'exectype', 'size', 'price', 'commission'])

    # Insert tickers as columns
    try:
        # Add ticker columns if they don't exist
        for ticker in tickers:
            if ticker not in log_history.columns:
                log_history[ticker] = np.nan

        # Reorder columns if date_time exists
        if 'date_time' in log_history.columns:
            # Get columns that are not tickers and not date_time
            other_cols = [col for col in log_history.columns if col != 'date_time' and col not in tickers]
            log_history = log_history[['date_time'] + list(tickers) + other_cols]
    except Exception as e:
        print(f"Warning: Error reordering log history columns: {str(e)}")

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

    # Update tickers positions
    for ticker in tickers:
        is_executed = log_history['event'] == 'Executed'
        is_ticker = log_history['ticker'] == ticker
        log_history.loc[is_executed & is_ticker, ticker] = log_history.loc[is_executed & is_ticker, 'pos']

    log_history[tickers] = log_history[tickers].fillna(method='ffill')
    log_history[tickers] = log_history[tickers].astype(int)

    # Keep only date at date_time
    log_history['date'] = log_history['date_time'].dt.date

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


# Alias for compatibility with Trading_Markowitz.py
def compute_backtest_vectorized(positions, settings, data_dict):
    """Alias for compute_backtest_sequential for compatibility with Trading_Markowitz.py."""
    return compute_backtest_sequential(positions, settings, data_dict)
