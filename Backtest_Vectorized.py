import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#import quantstats
import quantstats_lumi as quantstats
import webbrowser

from sympy.core.expr import unchanged


#Cash Backtest with backtest Vectorized

def compute_backtest_vectorized(weights, settings, data_dict):

        #Data from settings
        mults = settings['mults']
        startcash = settings['startcash']
        exposition_lim = settings['exposition_lim']
        commision = 5

        # Get Open,High ,Low,Closes from data_dict
        #opens,highs,lows,closes =[ pd.concat([df[col] for df in data_dict.values()], axis=1, keys=data_dict.keys()) for col in ['Open','High','Low','Close']]

        desired_order = list(data_dict.keys())  # Crucial: Store the desired order

        ohlc = {}
        for col in ['Open', 'High', 'Low', 'Close']:
            ohlc[col] = pd.concat([data_dict[key][col] for key in desired_order], axis=1, keys=desired_order)

        opens, highs, lows, closes = ohlc['Open'], ohlc['High'], ohlc['Low'], ohlc['Close']

        #Start
        # Get the first index where any weight is greater than zero (using any)
        start = weights[(weights > 0).any(axis=1)].index[0]

        #Set Start at Weights and  Tickers Data
        weights,opens,highs,lows,closes=[df[start:] for df in [weights,opens,highs,lows,closes]]

        if False: #Debug print
            print('start',start)
            print('weights',weights)
            print('opens', opens)

        # Get Buy/Sell Trigers & Stop Prices
        buy_trigger, sell_trigger, sell_stop_price,buy_stop_price = compute_buy_sell_triggers(weights, lows, highs)

        # mult dict to list in the tickers order
        mults =np.array([mults[tick] for tick in closes.columns])

        # Get historical of Exchange Rate EUR/USD (day after)
        exchange_rate = 1 / closes["EURUSD=X"].shift(1).fillna(method='bfill')

        # Set cash start
        startcash_usd = startcash / exchange_rate[start] # USD

        #Initialize Portfolio Value, Positions,orders
        portfolio_value_usd = pd.Series(startcash_usd, index=weights.index)
        pos=weights*0

        # Out of loop calculations
        weights_div_asset_price, asset_price = compute_out_of_backtest_loop(closes, weights, mults)

        #Backtest Loop for Positions/Portfolio Value Update
        pos, portfolio_value_usd, bt_log_dict=compute_backtest_loop(weights_div_asset_price, asset_price,opens,highs,lows,closes,mults,
                                 portfolio_value_usd,commision,weights,
                                 buy_trigger, sell_trigger, sell_stop_price,buy_stop_price,
                                 exchange_rate,startcash_usd,startcash,exposition_lim,pos)

        #Add Series to dict
        bt_log_series_dict={'pos': pos, 'portfolio_value': portfolio_value_usd}
        bt_log_dict.update(bt_log_series_dict)

        # Get Log History
        log_history = create_log_history(bt_log_dict)

        # Quanstats Report
        if settings['qstats']:
            q_title = 'Cash Backtest Markowitz Vectorized'
            path = "results\\"
            q_filename = os.path.abspath(path+ q_title + '.html')
            q_returns = bt_log_dict['portfolio_value_eur'].pct_change().iloc[:-settings['add_days']]
            q_benchmark_ticker='ES=F'
            q_benchmark = (closes[q_benchmark_ticker]*exchange_rate).pct_change().iloc[:-settings['add_days']]

            quantstats.reports.html(q_returns, title=q_title,benchmark=q_benchmark,benchmark_title=q_benchmark_ticker,output=q_filename) #
            webbrowser.open(q_filename)


        return bt_log_dict,log_history

def compute_backtest_loop_OK(weights_div_asset_price, asset_price,opens,highs,lows,closes,mults,
                         portfolio_value_usd,commision,weights,
                         buy_trigger, sell_trigger, sell_stop_price,
                         exchange_rate,startcash_usd,startcash,exposition_lim,pos):

    error,error_pos,i=1,1,0
    while ((error>0.001) |(error_pos>0)) & (i<200) :
        prev_end_value=portfolio_value_usd.iloc[-1]
        prev_pos=pos
        pos, portfolio_value_usd,bt_log_dict=\
            compute_backtest(weights_div_asset_price, asset_price,opens,highs,lows,closes,mults,
                             portfolio_value_usd,commision,weights,
                             buy_trigger, sell_trigger, sell_stop_price,
                             exchange_rate,startcash_usd,startcash,exposition_lim,pos)

        #Portfolio value Iteration Error
        error=abs(prev_end_value/portfolio_value_usd.iloc[-1]-1)

        # Position value Iteration Error
        zeros_df=closes*0
        error_pos = zeros_df.where(~bt_log_dict['iter_error'], 1).sum().sum()
        pos_equal=pos.equals(prev_pos)

        i+=1

        #print(i,error,error_pos,portfolio_value_usd[-1],pos_equal)

    return pos, portfolio_value_usd,bt_log_dict,i,error,error_pos

def compute_backtest_loop(weights_div_asset_price, asset_price,opens,highs,lows,closes,mults,
                         portfolio_value_usd,commision,weights,
                         buy_trigger, sell_trigger, sell_stop_price,buy_stop_price,
                         exchange_rate,startcash_usd,startcash,exposition_lim,pos):

    i=0
    max_i=200

    while True :

        prev_loop_pos=pos.copy()

        pos, portfolio_value_usd,bt_log_dict=\
            compute_backtest(weights_div_asset_price, asset_price,opens,highs,lows,closes,mults,
                             portfolio_value_usd,commision,weights,
                             buy_trigger, sell_trigger, sell_stop_price,buy_stop_price,
                             exchange_rate,startcash_usd,startcash,exposition_lim,pos)

        #Position do not change
        pos_equal=pos.equals(prev_loop_pos)

        if pos_equal | (i > max_i):
            bt_log_dict['n_iter']=i
            break

        i += 1

    return pos, portfolio_value_usd,bt_log_dict

def   compute_backtest(weights_div_asset_price,asset_price,opens,highs,lows,closes,mults,portfolio_value_usd,commision,weights,buy_trigger, sell_trigger, sell_stop_price,buy_stop_price,exchange_rate,startcash_usd,startcash,exposition_lim,pos):

    # Set Buy Execution Type (True -> Market, False -> Stop)
    buy_at_market = False

    # Upgrade target_size to 1 when target_size is over treshold
    upgrade_treshold = 0.20

    #Log Dict
    bt_log_dict = {}

    #Save Previous Position for further use
    prev_pos=pos.shift(1).fillna(0).astype(int)

    # target_size & trade_size Calculations for this data

    #Set Size of Portfolio to Invest
    portfolio_to_invest=portfolio_value_usd.shift(1).rolling(22*12,min_periods=1).min() #YTD Minimum

    #Compute Target Size of number of contracts
    target_size_raw = weights_div_asset_price.multiply(portfolio_to_invest, axis=0).fillna(0)

    #Upgrade target_size to 1 when target_size is over treshold
    mask = target_size_raw > upgrade_treshold

    # Exclude CL=F if present
    if "CL=F" in target_size_raw.columns:
        mask["CL=F"] = False

    # Apply upgrade
    target_size_raw[mask] = target_size_raw[mask].clip(lower=1)

    #Round to get Target Number of Contracts
    target_size = round(target_size_raw,0).astype(int)

    #Target Trade Size
    target_trade_size=target_size-prev_pos

    #Save Target Size data to dict for debug only
    #bt_log_dict['sod_pos']=prev_pos #Start of Day Position
    #bt_log_dict['target_size_raw'] = target_size_raw
    #bt_log_dict['target_size'] = target_size
    #bt_log_dict['target_trade_size'] = target_trade_size

    #Position Value & Exposition with Traget Size
    target_pos_value = (asset_price * target_size).sum(axis=1)  #USD
    targeted_exposition=target_pos_value/portfolio_to_invest
    exposition_is_low = pd.DataFrame({(col): (targeted_exposition < exposition_lim) for col in weights.columns})

    #Create/Reset Orders dfs
    tickers = weights.columns
    trading_dates=weights.index
    df=pd.DataFrame(columns=tickers, index=trading_dates)
    prices = pd.DataFrame(columns=tickers, index=trading_dates,dtype=float)
    B_S= df.copy().fillna('None')
    exectype = df.copy().fillna('None')
    event = df.copy().fillna('None')
    tickers_df=df.copy()
    tickers_df.loc[:,:]=tickers

    #Buy Orders
    is_buy=(target_trade_size>0) & exposition_is_low # & buy_trigger
    B_S.where(~is_buy, 'Buy', inplace=True)

    if buy_at_market:
        # Set Buy Orders as Market Orders
        exectype.where(~is_buy, 'Market', inplace=True) # Buy Order allways Market at Open Price

    else:
        # Set Buy Orders as Stop Orders
        exectype.where(~is_buy, 'Stop', inplace=True)
        # Ammend Buy Stop Prices bellow the Market and set minimum to Open
        buy_stop_price.clip(lower=opens, upper=None, inplace=True)
        #Set Buy Price as buy_stop_price
        prices.where(~is_buy, buy_stop_price, inplace=True)

        #Exceptions
        #Set cash always to market
        if 'cash' in is_buy.columns:
            exectype['cash'].where(~is_buy['cash'], 'Market', inplace=True)
        #Set to Market when target_trade_size==1
        #exectype.where(~(target_trade_size==1), 'Market', inplace=True)

    #Sell Orders
    is_sell=(target_trade_size < 0) #& sell_trigger
    B_S.where(~is_sell, 'Sell', inplace=True)
    #is_sell_stop =is_sell  #&  (weights!=0)
    exectype.where(~is_sell, 'Stop', inplace=True)
    # Ammend Sell Stop Prices over the Market and set maximum to Open
    sell_stop_price.clip(lower=None, upper=opens, inplace=True)
    prices.where(~is_sell, sell_stop_price, inplace=True)

    #Set cash sell orders as Market
    #if 'cash' in exectype.columns:
    #    exectype['cash'].where(~is_sell['cash'], 'Market', inplace=True)

    #Sell to Market when weights==0
    #is_sell_market = (is_sell & (weights==0))
    #exectype.where(~is_sell_market, 'Market', inplace=True)

    #Set Market Order Prices as Open
    prices.where(~(exectype=='Market'), opens, inplace=True)

    #Set event to 'created' in case
    event.where(~(B_S != 'None'), 'Created', inplace=True)

    #Set order_time at start of the day consecutives by ticker
    secs=np.arange(1, len(df.columns) + 1)  # Start from 1, end at l (inclusive)
    order_time_dict = {ticker: (event.index + pd.Timedelta(days=0, hours=0, minutes=0, seconds=sec)) for sec, ticker in zip(secs, tickers)}
    order_time= pd.DataFrame(order_time_dict,index=weights.index)
    order_time.where((event== 'Created'),np.nan,inplace=True)

    #Create Order Dict data
    bt_log_dict['order_dict'] = {
        'date_time':order_time,
        'event': event,
        'pos':prev_pos,  #Start of Day Position
        'ticker':tickers_df,
        'B_S':B_S,
        'exectype':exectype,
        'size':target_trade_size,
        'price': round(prices, 3),
        'commission': pos*0,
    }

    #Broker
    #Get Order Execution Confirmation

    #prices.where(~((exectype!='Sell')&(prices >np.array(opens))),opens, inplace=True)
    #prices.where(~((exectype != 'Buy') & (prices < np.array(highs))), highs, inplace=True)
    prices_in_the_market=(event=='Created') & (prices >=np.array(lows)) & (prices <=np.array(highs))
    broker_event=event.where(~prices_in_the_market, 'Executed').copy()
    broker_event.where(~(broker_event=='Created'),'Canceled',inplace=True)
    exec_size=target_trade_size.where((broker_event=='Executed'), 0).copy()
    exec_price = prices.where((broker_event == 'Executed'), 0).copy()
    exec_time=order_time+pd.Timedelta(days=0, hours=0, minutes=0, seconds=10)
    exec_time = exec_time.where(broker_event == 'Executed',np.nan)
    exec_time_stop=order_time+pd.Timedelta(days=0, hours=10, minutes=0, seconds=0)
    exec_time_stop = exec_time_stop.where(broker_event == 'Executed', np.nan)
    exec_time = exec_time.where(~(exectype == 'Stop'),exec_time_stop)
    exec_time_cancel = order_time + pd.Timedelta(days=0, hours=23, minutes=59, seconds=0)
    exec_time.where(~(broker_event  == 'Canceled'), exec_time_cancel,inplace=True)

    #Update End of Day Positions and B/S Price
    is_trading_day=(broker_event=='Executed')
    pos = prev_pos.where(~is_trading_day, prev_pos+exec_size).copy()

    #Trading Cost
    trading_cost_usd_by_ticker=exec_size.abs()*commision
    trading_cost_usd = trading_cost_usd_by_ticker.sum(axis=1)

    #Generate Broker Dict
    bt_log_dict['broker_dict'] = {
        'date_time': exec_time,
        'event': broker_event,
        'pos': pos,
        'ticker': tickers_df,
        'B_S':B_S,
        'exectype':exectype,
        'size':exec_size,
        'price': round(exec_price.astype(float), 3),
        'commission':trading_cost_usd_by_ticker,
    }

    #Get Gross Dayly Returns

    #Holding Days Returns
    #hold_pos=prev_pos
    hold_price_diff=asset_price.diff().fillna(0)
    hold_returns_raw_usd = (prev_pos * hold_price_diff).sum(axis=1)  # USD

    #Add Trading Days Returns
    trading_price_diff=(closes-exec_price).multiply(mults, axis=1)
    trading_day_returns=exec_size*trading_price_diff # USD
    trading_day_returns_sum = trading_day_returns.sum(axis=1)
    dayly_returns_raw_usd = hold_returns_raw_usd + trading_day_returns_sum

    #Add Trading Cost



    #Net Dayly Returns
    dayly_returns_usd = dayly_returns_raw_usd - trading_cost_usd
    dayly_returns_eur = dayly_returns_usd *exchange_rate

    #Update Portfolio Value
    portfolio_value_usd = startcash_usd + dayly_returns_usd.cumsum()
    portfolio_value_eur=startcash + dayly_returns_eur.cumsum()

    #Iteration Error
    #pos_diff=abs(pos - not_updated_pos)
    #true_df=pd.DataFrame(index=pos.index,columns=pos.columns).fillna(True)
    #iter_error=true_df.where(pos_diff>0,False)

    #More Data to log_dict
    #bt_log_dict['trading_day_returns']=trading_day_returns
    #bt_log_dict['iter_error'] =iter_error
    bt_log_dict['pos_value']  = (asset_price * pos).sum(axis=1)  #USD
    bt_log_dict['exposition']=bt_log_dict['pos_value'] /portfolio_to_invest
    bt_log_dict['commission'] = trading_cost_usd
    bt_log_dict['dayly_profit_eur'] = dayly_returns_eur
    bt_log_dict['exchange_rate'] = exchange_rate
    bt_log_dict['portfolio_value_eur'] = portfolio_value_eur

    return pos,portfolio_value_usd,bt_log_dict


def compute_backtest_rev(weights_div_asset_price, asset_price, opens, highs, lows, closes, mults,
                     portfolio_value_usd, commission, weights, buy_trigger, sell_trigger,
                     sell_stop_price, exchange_rate, startcash_usd, startcash, exposition_lim, pos):
    """
    Computes the backtest for a trading strategy using the provided parameters.

    Parameters:
    - weights_div_asset_price: DataFrame of weights divided by asset prices.
    - asset_price: DataFrame of asset prices.
    - opens, highs, lows, closes: DataFrames of market data.
    - mults: Multiplier for trading price differences.
    - portfolio_value_usd: Series of portfolio values in USD.
    - commission: Transaction cost per trade.
    - weights: DataFrame of asset weights.
    - buy_trigger, sell_trigger: Conditions for executing trades (not used in this code).
    - sell_stop_price: Price at which to sell assets.
    - exchange_rate: Current exchange rate.
    - startcash_usd: Initial cash in USD.
    - startcash: Initial cash in another currency.
    - exposition_lim: Limit for portfolio exposition.
    - pos: DataFrame of current positions.

    Returns:
    - pos: Updated positions DataFrame.
    - portfolio_value_usd: Updated portfolio value in USD.
    - bt_log_dict: Dictionary containing logs for the backtest.
    """

    # Log Dictionary
    bt_log_dict = {}

    # Save Previous Position for further use
    prev_pos = pos.shift(1).fillna(0).astype(int)

    # Set Size of Portfolio to Invest
    portfolio_to_invest = portfolio_value_usd.shift(1).rolling(22 * 12, min_periods=1).min()

    # Compute Target Size of number of contracts
    target_size = (weights_div_asset_price.multiply(portfolio_to_invest, axis=0).fillna(0)).round().astype(int)

    # Target Trade Size
    target_trade_size = target_size - prev_pos

    # Position Value & Exposition with Target Size
    target_pos_value = (asset_price * target_size).sum(axis=1)  # USD
    targeted_exposition = target_pos_value / portfolio_to_invest
    exposition_is_low = pd.DataFrame({col: targeted_exposition < exposition_lim for col in weights.columns})

    # Create/Reset Orders DataFrames
    tickers = weights.columns
    trading_dates = weights.index
    orders_df = pd.DataFrame(columns=tickers, index=trading_dates).fillna('None')
    prices = pd.DataFrame(columns=tickers, index=trading_dates, dtype=float)
    exec_type = orders_df.copy()
    event = orders_df.copy()

    # Buy Orders
    is_buy = (target_trade_size > 0) & exposition_is_low
    orders_df.where(~is_buy, 'Buy', inplace=True)
    exec_type.where(~is_buy, 'Market', inplace=True)  # Buy Order always Market at Open Price

    # Sell Orders
    is_sell = (target_trade_size < 0)
    orders_df.where(~is_sell, 'Sell', inplace=True)
    exec_type.where(~is_sell, 'Stop', inplace=True)
    prices.where(~is_sell, sell_stop_price, inplace=True)

    # Set Market Order Prices as Open
    prices.where(~(exec_type == 'Market'), opens, inplace=True)

    # Set event to 'created' in case
    event.where(~(orders_df != 'None'), 'Created', inplace=True)

    # Set order_time at start of the day consecutively by ticker
    order_time = create_order_time(event, trading_dates, tickers)

    # Create Order Log
    bt_log_dict['order_dict'] = create_order_dict(order_time, event, prev_pos, orders_df, exec_type, target_trade_size, prices)

    # Broker Execution Confirmation
    prices = amend_prices(prices, opens)
    broker_event, exec_size, exec_price, exec_time = execute_orders(event, prices, target_trade_size, order_time, highs, lows)

    # Update Positions
    pos = update_positions(prev_pos, exec_size, broker_event)

    # Calculate Trading Costs
    trading_costs = calculate_trading_cost(exec_size, commission)

    # Generate Broker Log
    bt_log_dict['broker_dict'] = create_broker_dict(exec_time, broker_event, pos, orders_df, exec_type, exec_size, exec_price, trading_costs)

    # Calculate Daily Returns
    dayly_returns_usd, dayly_returns_eur, portfolio_value_usd, portfolio_value_eur = calculate_daily_returns(
        exec_size,asset_price, prev_pos, closes, exec_price, mults, trading_costs, startcash_usd, startcash, exchange_rate
    )

    # Update Backtest DataFrame with Portfolio Values
    bt_log_dict['pos_value'] = (asset_price * pos).sum(axis=1)  # USD
    bt_log_dict['exposition'] = bt_log_dict['pos_value'] / portfolio_to_invest
    bt_log_dict['commission'] = trading_costs['total']
    bt_log_dict['dayly_profit_eur'] = dayly_returns_eur
    bt_log_dict['exchange_rate'] = exchange_rate
    bt_log_dict['portfolio_value_eur'] = portfolio_value_eur

    return pos, portfolio_value_usd, bt_log_dict


def create_order_time(event, trading_dates, tickers):
    """
    Create a DataFrame for order timing based on tickers.
    """
    secs = np.arange(1, len(tickers) + 1)  # Start from 1, end at l (inclusive)
    order_time_dict = {ticker: (event.index + pd.Timedelta(seconds=sec)) for sec, ticker in zip(secs, tickers)}
    order_time = pd.DataFrame(order_time_dict, index=trading_dates)
    order_time.where(event == 'Created', np.nan, inplace=True)
    return order_time


def create_order_dict(order_time, event, prev_pos, orders_df, exec_type, target_trade_size, prices):
    """
    Create a dictionary for logging order details.
    """
    return {
        'date_time': order_time,
        'event': event,
        'pos': prev_pos,  # Start of Day Position
        'ticker': orders_df,
        'B_S': orders_df,
        'exectype': exec_type,
        'size': target_trade_size,
        'price': round(prices, 3),
        'commission': prices * 0,  # Placeholder for commission calculation
    }


def amend_prices(prices, opens):
    """
    Amend prices over the market and set to open if necessary.
    """
    prices.where(~(prices > opens), opens, inplace=True)
    return prices


def execute_orders(event, prices, target_trade_size, order_time, highs, lows):
    """
    Execute orders based on the event status and market conditions.
    """
    prices_in_the_market = (event == 'Created') & (prices >= lows) & (prices <= highs)
    broker_event = event.where(~prices_in_the_market, 'Executed').copy()
    broker_event.where(broker_event != 'Created', 'Canceled', inplace=True)

    exec_size = target_trade_size.where(broker_event == 'Executed', 0).copy()
    exec_price = prices.where(broker_event == 'Executed', 0).copy()
    exec_time = order_time + pd.Timedelta(seconds=10)
    exec_time = exec_time.where(broker_event == 'Executed', np.nan)

    return broker_event, exec_size, exec_price, exec_time


def update_positions(prev_pos, exec_size, broker_event):
    """
    Update positions based on executed sizes.
    """
    is_trading_day = (broker_event == 'Executed')
    return prev_pos.where(~is_trading_day, prev_pos + exec_size).copy()


def calculate_trading_cost(exec_size, commission):
    """
    Calculate the total trading cost based on executed sizes and commission.
    """
    trading_cost_usd_by_ticker = exec_size.abs() * commission
    trading_cost_usd = trading_cost_usd_by_ticker.sum(axis=1)
    return {'by_ticker': trading_cost_usd_by_ticker, 'total': trading_cost_usd}


def create_broker_dict(exec_time, broker_event, pos, orders_df, exec_type, exec_size, exec_price, trading_costs):
    """
    Create a dictionary for logging broker details.
    """
    return {
        'date_time': exec_time,
        'event': broker_event,
        'pos': pos,
        'ticker': orders_df,
        'B_S': orders_df,
        'exectype': exec_type,
        'size': exec_size,
        'price': round(exec_price.astype(float), 3),
        'commission': trading_costs['by_ticker'],
    }


def calculate_daily_returns(exec_size,asset_price, prev_pos, closes, exec_price, mults, trading_costs, startcash_usd, startcash, exchange_rate):
    """
    Calculate daily returns based on asset prices, execution prices, and trading costs.
    """
    hold_price_diff = asset_price.diff().fillna(0)
    hold_returns_raw_usd = (prev_pos * hold_price_diff).sum(axis=1)

    trading_price_diff = (closes - exec_price).multiply(mults, axis=1)
    trading_day_returns = exec_size * trading_price_diff  # USD
    trading_day_returns_sum = trading_day_returns.sum(axis=1)

    dayly_returns_raw_usd = hold_returns_raw_usd + trading_day_returns_sum
    dayly_returns_usd = dayly_returns_raw_usd - trading_costs['total']
    dayly_returns_eur = dayly_returns_usd * exchange_rate

    portfolio_value_usd = startcash_usd + dayly_returns_usd.cumsum()
    portfolio_value_eur = startcash + dayly_returns_eur.cumsum()

    return dayly_returns_usd, dayly_returns_eur, portfolio_value_usd, portfolio_value_eur


def create_log_history(bt_log_dict):
    """
    :param bt_log_dict:
    :return: log_history
    log_history.columns
    ['date', 'ES=F', 'NQ=F', 'GC=F', 'CL=F', 'EURUSD=X', 'event', 'ticker','B_S', 'exectype', 'size', 'price', 'commission',
    'portfolio_value','portfolio_value_eur', 'pos_value', 'ddn', 'dayly_profit','dayly_profit_eur', 'pre_portfolio_value', 'exchange_rate', 'ddn_eur']
    """
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
    log_history_dict = {ticker: pd.concat([order_dict[ticker], broker_dict[ticker]], axis=0).dropna().sort_values(by='date_time') for ticker in tickers}

    # Concatenate all tickers
    log_history = pd.concat(log_history_dict.values(), axis=0).sort_values(by='date_time')

    # Insert tickers as columns
    log_history[tickers] = np.nan
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

    # Update tickers positions
    for ticker in tickers:
        is_executed = log_history['event'] == 'Executed'
        is_ticker = log_history['ticker'] == ticker
        log_history.loc[is_executed & is_ticker, ticker] = log_history.loc[is_executed & is_ticker, 'pos']

    log_history[tickers] = log_history[tickers].fillna(method='ffill')
    log_history[tickers] = log_history[tickers].astype(int)

    # Rename event
    event_values = ['Sell Order Created', 'Sell Order Canceled', 'Sell Order Executed', 'Buy Order Created', 'Buy Order Canceled', 'Buy Order Executed', 'End of Day']

    # Keep only date at date_time
    log_history['date'] = log_history['date_time'].dt.date

    return log_history

def compute_out_of_backtest_loop(closes,weights,mults):
    asset_price = closes.multiply(mults,axis=1) # USD
    yesterday_asset_price=asset_price.shift(1)
    yesterday_asset_price_mean = yesterday_asset_price.rolling(5,min_periods=1).mean() #YTD Minimum
    weights_mean = weights.rolling(5, min_periods=1).mean()
    weights_div_asset_price = weights_mean / yesterday_asset_price_mean

    return  weights_div_asset_price,asset_price

def compute_buy_sell_triggers(weights,lows, highs):

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
    # Yesterday low > previous 5 days lowest
    # & Yesterday weight > previous 5 days lowest weight
    # Next five days Positive Weight evolution
    buy_trigger = lows_up & weights_up

    # Sell Trigger
    sell_trigger = highs_dn  # Highs Downtrend

    #Get Sell Stop Price

    # Max of last month  lows_min
    low_keep = lows_min.rolling(22).max()
    #lows_std = lows.rolling(22).std().shift(1)
    sell_stop_price = low_keep


    #Stop Volatility
    #Fast Drawdown previous 5d
    #ddn=highs_max- lows_min
    #ddn_std = ddn.rolling(5).std()
    #ddn_lim = 5 * ddn_std
    #stop_volat=highs_max-ddn_lim
    #stop_volat=stop_volat.rolling(22).max()

    # Save sell_stop_price: Keep Minimum of low_keep & stop_volat
    #sell_stop_price = low_keep.where(low_keep < stop_volat, stop_volat)

    #Min of last month highs_min
    high_keep = highs_max.rolling(22).min()
    highs_std=highs.rolling(22).std().shift(1)
    buy_stop_price = high_keep + highs_std*0.5

    return buy_trigger,sell_trigger,sell_stop_price,buy_stop_price

def plot_df():
    #Plot
    col='ES=F'
    plot_df=pd.concat([df[col] for df in [lows,sell_stop_price,is_sell*5000]], axis=1, keys=['lows','sell_stop_price','is_sell'])
    plot_df.plot()

def get_end_of_year_values(ds):
    year=pd.Series(ds.index.quarter,index=ds.index)
    year_change=year.diff()
    eoy=ds.shift(1).where(year_change==1,np.nan).fillna(method='ffill').fillna(ds[0])
    return eoy


def get_log_dict_by_ticker_dict(bt_log_dict,tickers):
    bt_log_by_ticker_dict = {}
    #tickers = list(bt_log_dict.values())[0].columns
    for ticker in tickers:
        # Save dict values
        df = pd.DataFrame()
        for key, value in bt_log_dict.items():
            if isinstance(value, pd.DataFrame):
                df[key] = value[ticker]
            elif isinstance(value, dict):
                for dict_key, dict_value in value.items():
                    df[dict_key] = dict_value[ticker]

        bt_log_by_ticker_dict[ticker] = df

    return bt_log_by_ticker_dict