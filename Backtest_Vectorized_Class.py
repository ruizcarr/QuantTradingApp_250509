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
    commission: float = 5.0
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
            max_iterations: int = 200
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
        all_bt_log_dicts = [{}] * (max_iterations + 1)

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

            # Call compute_backtest with current positions and portfolio values
            new_pos, new_portfolio, bt_log_dict = self.compute_backtest(
                weights_div_asset_price, asset_price, opens, highs, lows, closes, mults,
                current_portfolio, weights, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
                exchange_rate, startcash_usd, startcash, exposition_lim, current_pos
            )

            # Store results for this iteration
            all_positions[i+1] = new_pos.values
            all_portfolio_values[i+1] = new_portfolio.values
            all_bt_log_dicts[i+1] = bt_log_dict

            # Check for convergence using vectorized comparison
            if np.array_equal(all_positions[i+1], all_positions[i]):
                break

        # Get the final results
        final_pos = pd.DataFrame(
            all_positions[i+1], 
            index=pos.index, 
            columns=pos.columns
        )

        final_portfolio = pd.Series(
            all_portfolio_values[i+1],
            index=portfolio_value_usd.index
        )

        final_bt_log_dict = all_bt_log_dicts[i+1]
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
            pos: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Execute backtest computation."""
        bt_log_dict = {}

        # Calculate positions and sizes
        prev_pos = pos.shift(1).fillna(0).astype(int)

        # Set Size of Portfolio to Invest
        portfolio_to_invest = portfolio_value_usd.shift(1).rolling(
            self.settings.portfolio_window,
            min_periods=self.settings.min_periods
        ).min()

        # Compute Target Size of number of contracts
        target_size_raw = weights_div_asset_price.multiply(portfolio_to_invest, axis=0).fillna(0)
        target_size_raw[target_size_raw > self.settings.upgrade_threshold] = target_size_raw.clip(lower=1)
        target_size = round(target_size_raw, 0).astype(int)
        target_trade_size = target_size - prev_pos

        # Position Value & Exposition with Target Size
        target_pos_value = (asset_price * target_size).sum(axis=1)
        targeted_exposition = target_pos_value / portfolio_to_invest
        exposition_is_low = pd.DataFrame({col: (targeted_exposition < exposition_lim) for col in weights.columns})

        # Create/Reset Orders dfs
        tickers = weights.columns
        trading_dates = weights.index
        base_df = pd.DataFrame(columns=tickers, index=trading_dates)

        # Initialize order DataFrames
        prices = pd.DataFrame(columns=tickers, index=trading_dates, dtype=float)
        B_S = base_df.copy().fillna('None')
        exectype = base_df.copy().fillna('None')
        event = base_df.copy().fillna('None')
        tickers_df = base_df.copy()
        tickers_df.loc[:, :] = tickers

        # Process buy orders
        is_buy = (target_trade_size > 0) & exposition_is_low
        B_S.where(~is_buy, 'Buy', inplace=True)

        if not self.settings.buy_at_market:
            exectype.where(~is_buy, 'Stop', inplace=True)
            buy_stop_price_adj = buy_stop_price.clip(lower=opens, upper=None)
            prices.where(~is_buy, buy_stop_price_adj, inplace=True)
        else:
            exectype.where(~is_buy, 'Market', inplace=True)

        # Process sell orders
        is_sell = (target_trade_size < 0)
        B_S.where(~is_sell, 'Sell', inplace=True)
        exectype.where(~is_sell, 'Stop', inplace=True)

        sell_stop_price_adj = sell_stop_price.clip(lower=None, upper=opens)
        prices.where(~is_sell, sell_stop_price_adj, inplace=True)

        # Set market order prices
        prices.where(
            ~(exectype == 'Market'),
            opens,
            inplace=True
        )

        # Set events
        event.where(
            ~(B_S != 'None'),
            'Created',
            inplace=True
        )

        # Create order data dictionary
        order_data = {
            'prices': prices,
            'B_S': B_S,
            'exectype': exectype,
            'event': event,
            'tickers_df': tickers_df,
            'size': target_trade_size,
            'pos': prev_pos
        }

        # Execute orders
        prices_in_market = (
            (order_data['event'] == 'Created') &
            (order_data['prices'] >= np.array(lows)) &
            (order_data['prices'] <= np.array(highs))
        )

        broker_event = order_data['event'].where(~prices_in_market, 'Executed').copy()
        broker_event.where(~(broker_event == 'Created'), 'Canceled', inplace=True)

        exec_size = target_trade_size.where((broker_event == 'Executed'), 0).copy()
        exec_price = order_data['prices'].where((broker_event == 'Executed'), 0).copy()

        # Calculate trading costs
        trading_cost = exec_size.abs() * self.settings.commission

        # Update positions
        is_trading_day = (broker_event == 'Executed')
        updated_pos = prev_pos.where(~is_trading_day, prev_pos + exec_size).copy()

        execution_data = {
            'broker_event': broker_event,
            'exec_size': exec_size,
            'exec_price': exec_price,
            'trading_cost': trading_cost,
            'updated_pos': updated_pos
        }

        # Calculate returns
        # Calculate holding returns
        hold_price_diff = asset_price.diff().fillna(0)
        hold_returns_raw_usd = (prev_pos * hold_price_diff).sum(axis=1)

        # Calculate trading returns
        trading_price_diff = (closes - execution_data['exec_price']).multiply(mults, axis=1)
        trading_returns = execution_data['exec_size'] * trading_price_diff

        # Calculate total returns
        daily_returns_usd = (
            hold_returns_raw_usd +
            trading_returns.sum(axis=1) -
            execution_data['trading_cost'].sum(axis=1)
        )
        daily_returns_eur = daily_returns_usd * exchange_rate

        # Calculate portfolio values
        portfolio_value_usd_new = startcash_usd + daily_returns_usd.cumsum()
        portfolio_value_eur = startcash + daily_returns_eur.cumsum()

        returns_data = {
            'daily_returns_usd': daily_returns_usd,
            'daily_returns_eur': daily_returns_eur,
            'portfolio_value_usd': portfolio_value_usd_new,
            'portfolio_value_eur': portfolio_value_eur
        }

        # Create log dictionary
        bt_log_dict['order_dict'] = order_data
        bt_log_dict['broker_dict'] = {
            'date_time': execution_data.get('exec_time', pd.DataFrame()),
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
        bt_log_dict['pos'] = execution_data['updated_pos']
        bt_log_dict['pos_value'] = (asset_price * execution_data['updated_pos']).sum(axis=1)
        bt_log_dict['portfolio_value'] = portfolio_value_usd_new
        bt_log_dict['portfolio_value_eur'] = returns_data['portfolio_value_eur']
        bt_log_dict['dayly_profit_eur'] = returns_data['daily_returns_eur']
        bt_log_dict['exchange_rate'] = exchange_rate
        bt_log_dict['exposition'] = bt_log_dict['pos_value'] / portfolio_to_invest

        return (
            execution_data['updated_pos'],
            returns_data['portfolio_value_usd'],
            bt_log_dict
        )

    # The following methods have been inlined into compute_backtest for performance reasons:
    # _calculate_positions, _generate_orders, _execute_orders, _calculate_returns, _create_log_dict


def compute_backtest_vectorized(
        positions: pd.DataFrame,
        settings: Dict,
        data_dict: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """Main entry point for vectorized backtest computation."""
    # Data from settings
    mults = settings['mults']
    startcash = settings['startcash']
    exposition_lim = settings['exposition_lim']
    commission = settings.get('commission', 5.0)

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

    # Get Buy/Sell Triggers & Stop Prices
    buy_trigger, sell_trigger, sell_stop_price, buy_stop_price = compute_buy_sell_triggers(positions, lows, highs)

    # mult dict to list in the tickers order - optimize by using list comprehension
    mults_array = np.array([mults[tick] for tick in closes.columns])

    # Get historical of Exchange Rate EUR/USD (day after)
    exchange_rate = 1 / closes["EURUSD=X"].shift(1).fillna(method='bfill')

    # Set cash start
    startcash_usd = startcash / exchange_rate[start]  # USD

    # Initialize Portfolio Value, Positions, orders
    portfolio_value_usd = pd.Series(startcash_usd, index=positions.index)
    pos = pd.DataFrame(0, index=positions.index, columns=positions.columns)

    # Out of loop calculations
    weights_div_asset_price, asset_price = compute_out_of_backtest_loop(closes, positions, mults_array)

    # Create backtest instance with settings - reuse the same instance
    backtest_settings = BacktestSettings(
        upgrade_threshold=settings.get('upgrade_threshold', 0.20),
        commission=commission,
        buy_at_market=settings.get('buy_at_market', False)
    )

    # Create backtest instance only once
    backtest = BacktestVectorized(backtest_settings)

    # Use the compute_backtest_until_convergence method to handle the loop internally
    pos, portfolio_value_usd, bt_log_dict = backtest.compute_backtest_until_convergence(
        weights_div_asset_price, asset_price, opens, highs, lows, closes, mults_array,
        portfolio_value_usd, positions, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
        exchange_rate, startcash_usd, startcash, exposition_lim, pos
    )

    # Add Series to dict - optimize by updating directly
    bt_log_dict['pos'] = pos
    bt_log_dict['portfolio_value'] = portfolio_value_usd

    # Get Log History
    log_history = create_log_history(bt_log_dict)

    # Quantstats Report
    if settings.get('qstats', False):
        q_title = 'Cash Backtest Markowitz Vectorized'
        path = "results\\"
        q_filename = os.path.abspath(path + q_title + '.html')
        q_returns = bt_log_dict['portfolio_value_eur'].pct_change().iloc[:-settings['add_days']]
        q_benchmark_ticker = 'ES=F'
        q_benchmark = (closes[q_benchmark_ticker] * exchange_rate).pct_change().iloc[:-settings['add_days']]

        import quantstats_lumi as quantstats
        quantstats.reports.html(q_returns, title=q_title, benchmark=q_benchmark, benchmark_title=q_benchmark_ticker, output=q_filename)
        import webbrowser
        webbrowser.open(q_filename)

    return bt_log_dict, log_history


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
