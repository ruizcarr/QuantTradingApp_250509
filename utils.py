import numpy as np

import pandas as pd

def weighted_mean_of_dfs_dict(dfs_dict, weights_list):
  """Calculates the weighted mean of multiple DataFrames.

  Args:
    dfs_dict: A dictionary of DataFrames.
    weights_list: A list of weights corresponding to each DataFrame.

  Returns:
    A DataFrame containing the weighted mean of the input DataFrames.
  """

  # Ensure all DataFrames have the same index
  #common_index = dfs_dict[list(dfs_dict.keys())[0]].index
  #for df in dfs_dict.values():
  #  df.index = common_index

  # Calculate the weighted sum using list comprehension
  weighted_sum_df = sum([df * weight for df, weight in zip(dfs_dict.values(), weights_list)])

  # Calculate the sum of weights
  sum_of_weights = sum(weights_list)

  # Calculate the weighted mean
  weighted_mean_df = weighted_sum_df / sum_of_weights

  return weighted_mean_df

def sigmoid(x,center=0, width=1,max=1):
   '''
   Returns array of a horizontal mirrored normalized sigmoid function
   output between 0 and 1
   x should be > 1
   '''
   s = (1 / (1 + np.exp((x - center)/width/2 ))) * max
   return s

def get_end_of_period(df, p_freq='W-FRI'):
    """Gets the end of the specified period for a given period in the DataFrame.

    Args:
        df: The DataFrame.
        p_freq: The Period frequency string, such as 'W-FRI' for Friday of the week, 'M' for the end of the month, etc.

    Returns:
        A list of end-of-period dates.
    """

    # Create a DatetimeIndex with the desired frequency and range
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=p_freq)

    # Find the nearest indices in the DataFrame's index for each end-of-period date
    nearest_indices = df.index.searchsorted(idx, side='right') - 1
    nearest_indices = nearest_indices.clip(0, len(df.index) - 1)  # Ensure indices are within bounds

    # Extract the nearest dates from the DataFrame's index
    end_of_period_dates = df.index[nearest_indices]

    return end_of_period_dates.tolist()

def get_start_of_period(df, end_dates_df, lookback_period):
    """Gets the start of the period for a given end date and lookback period.

    Args:
        df: The DataFrame.
        end_dates_df: The Series or df column  with end dates of the period.
        lookback_period: The number of periods to look back.

    Returns:
        The start date of the period.
    """

    start_date = end_dates_df - pd.Timedelta(days=lookback_period)

    # Find the nearest indices in the DataFrame's index for each end-of-period date
    nearest_index = df.index.searchsorted(start_date, side='right') - 1
    nearest_index = nearest_index.clip(0, len(df.index) - 1)  # Ensure indices are within bounds


    start_date = df.index[nearest_index]

    return start_date


def create_results_by_period_df(tickers_returns, rebalance_p, lookback):
    """
    Create df to store results create_results_by_period_df with columns,
        'end': end date of rebalance period (nearets included in tickers_returns.index)
        'start': start date of period = end - lookback (nearets included in tickers_returns.index)
    """

    results_by_period_df = pd.DataFrame()

    # Store List with End of Rebalance Period Date
    results_by_period_df['end'] = get_end_of_period(tickers_returns, rebalance_p)

    # Calculate the Start dates of lookback period
    results_by_period_df['start'] = get_start_of_period(tickers_returns, results_by_period_df['end'], lookback)

    # Keep rows where period len is greater than loockback
    results_by_period_df = results_by_period_df[(results_by_period_df['end'] - results_by_period_df['start']).dt.days >= lookback]

    return results_by_period_df


def mean_positions(positions, vector_positions, upper_pos_lim,
                   sf = 1,vf = 1,overall_f = 1.20):
    vector_positions = vector_positions.reindex(positions.index)
    positions = overall_f * (positions * sf + vector_positions * vf) / (sf + vf)

    # Limit Upper individual position
    positions=positions.clip(upper=upper_pos_lim)

    return positions




def limit_df_values_diff(df, delta, max_iter=10):
    """
    Limits the changes in consecutive values of a DataFrame column to a specified delta.

    Args:
        df: The DataFrame containing the column to be limited.
        delta: The maximum allowed change between consecutive values.
        max_iter: The maximum number of iterations.

    Returns:
        The modified DataFrame.
    """

    for _ in range(max_iter):
        # Calculate the upper and lower limits
        upper_limit = df.shift(1) + delta['up']
        lower_limit = df.shift(1) - delta['dn']

        # Clip the values to the specified range
        df = df.clip(lower=lower_limit, upper=upper_limit)

        # Re-calculate the difference after clipping and breack if bellow or equal delta
        if df.diff().abs().max() <= delta['up']:
            break

    return df


def strategies_markowitz_mean_weights(strategies_weights_dict, tickers_returns, settings):
    # Reference Index
    ref_idx = list(strategies_weights_dict.values())[0].index

    # Make sure weights and tickers returns has same index
    tickers_returns = tickers_returns.reindex(ref_idx)

    # Create Dataframe with Returns of each Strategie
    strategies_ret = pd.DataFrame()

    for key, w_df in strategies_weights_dict.items():
        # Make sure weights and tickers returns has same index
        w_df = w_df.reindex(ref_idx)

        strategies_weights_dict[key] = w_df

        # Create Dataframe with Returns of each Strategie
        strategies_ret[key] = (w_df * tickers_returns).sum(axis=1)

    # Compute Vectorized markowitz and Get Weights for Strategies Mean
    from Markowitz_Vectorized import get_combined_strategy_by_markowitz
    volat_target = settings['volatility_target']
    weight_lim = 1.0  # 0.8
    weight_sum_lim = 1.0  # 1.2
    cagr_w = 250 * 1
    strat_period = 'weekly'  # 'dayly'

    # Concat d_w_m_ weights array (n_strats,n_days,n_tickers)
    strategies_weights_dfs = np.array([np.array(df) for key, df in strategies_weights_dict.items()])

    weights_comb_array, weights_by_strategy_df = get_combined_strategy_by_markowitz(strategies_ret, strategies_weights_dfs, volat_target, weight_lim, weight_sum_lim, cagr_w, strat_period)

    weights_by_strategy_mean = weights_by_strategy_df.mean()

    # Apply weights_by_strategy_mean
    weights_comb_df = pd.DataFrame(columns=tickers_returns.columns, index=ref_idx).fillna(0)

    for key, w_df in strategies_weights_dict.items():
        # Create Dataframe with Returns of each Strategie
        weights_comb_df = weights_comb_df + w_df * weights_by_strategy_df[key].mean()

    return weights_comb_df, weights_by_strategy_mean


def equi_volatility_strategies_weights_sum(strategies_weights_dict, tickers_returns, settings):
    # Reference Index
    ref_idx = list(strategies_weights_dict.values())[0].index

    # Make sure weights and tickers returns has same index
    tickers_returns = tickers_returns.reindex(ref_idx)

    # Create Dataframe with Returns of each Strategie
    strategies_ret = pd.DataFrame()

    for key, w_df in strategies_weights_dict.items():
        # Make sure weights and tickers returns has same index
        w_df = w_df.reindex(ref_idx)

        strategies_weights_dict[key] = w_df

        # Create Dataframe with Returns of each Strategie
        strategies_ret[key] = (w_df * tickers_returns).sum(axis=1)

    strategies_volat = strategies_ret.std() * 16
    strategies_equi_volat_factor = strategies_volat.max() / strategies_volat / len(strategies_weights_dict)

    # Apply weights_by_strategy_mean
    weights_comb_df = pd.DataFrame(columns=tickers_returns.columns, index=ref_idx).fillna(0)

    for key, w_df in strategies_weights_dict.items():
        # Create Dataframe with Returns of each Strategie
        weights_comb_df = weights_comb_df + w_df * strategies_equi_volat_factor[key]

    return weights_comb_df, strategies_equi_volat_factor