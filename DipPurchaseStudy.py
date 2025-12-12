# Get Data
from config.trading_settings import settings
#Update Settings
settings['start']='1996-01-01'

import Market_Data_Feed as mdf

data_ind = mdf.Data_Ind_Feed(settings).data_ind
data, _ = data_ind
data_dict = data.data_dict

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# ===========================
# 1️⃣ Compute drawdown
# ===========================
def compute_drawdown(prices: pd.Series, lookback=252) -> pd.Series:
    """
    Drawdown relative to rolling max of last `lookback` days
    """
    roll_max = prices.rolling(lookback, min_periods=100).max()
    dd = (prices - roll_max) / roll_max
    return dd  # negative numbers

# ===========================
# 2️⃣ Cumulative weight function
# ===========================
def weight_from_dd_cumulative(prev_weight, ddn, ddn_prev, add_on_recovery, sub_on_drawdown, max_weight, dd_max_stop, dd_small_threshold):
    """
    Cumulative weight update based on drawdown.
    - ddn: current day's drawdown (negative)
    - ddn_prev: previous day's drawdown (negative)
    - prev_weight: previous day's weight
    - add_on_recovery: amount to add when ddn improves
    - sub_on_drawdown: amount to subtract when ddn worsens
    - max_weight: upper limit for weight
    """
    # Hard stop
    if ddn <= dd_max_stop:  # dd_max_stop is negative
        return 0.0

    # Neutral zone
    if abs(ddn) < dd_small_threshold:
        return max(prev_weight * 0.995,1.4)

    # Correct ddn change: positive = recovery, negative = deeper drawdown
    ddn_change = ddn_prev - ddn
    ddn_change = -25 * ddn_change # flip sign so recovery is positive

    if ddn_change > 0:
        # Recovery -> increase weight
        w = prev_weight+ ddn_change * add_on_recovery
    elif ddn_change < 0:
        # Deeper drawdown -> decrease weight
        w = prev_weight+ ddn_change  * sub_on_drawdown
    else:
        # No change
        w = prev_weight
    # Clip weight
    w = max(0.0, min(max_weight, w))

    return w

# ===========================
# 3️⃣ Compute weights DataFrame
# ===========================
def compute_weights(data_dict, params_per_asset, span=5):
    """
    Compute cumulative weights per asset using previous-day smoothed drawdown (EWMA).

    Returns:
        weights_df : DataFrame of weights (0 to max_weight)
        ddn_df     : DataFrame of raw smoothed drawdown for plotting
    """
    import pandas as pd
    import numpy as np

    weights_df = pd.DataFrame(index=list(data_dict.values())[0].index)
    ddn_df = pd.DataFrame(index=list(data_dict.values())[0].index)

    for ticker, df in data_dict.items():
        if ticker == 'cash':
            weights_df[ticker] = params_per_asset[ticker]['max_weight']
            ddn_df[ticker] = 0.0
            continue

        # Raw drawdown (negative values)
        dd_raw = compute_drawdown(df['Close'],252*2)

        # Smoothed drawdown for weight calculation, shifted by 1 to avoid lookahead
        dd_smooth = dd_raw.shift(1).fillna(0) #.ewm(span=span, adjust=False).mean().shift(1).fillna(0)

        # Store smoothed negative drawdown for plotting
        ddn_df[ticker] = dd_smooth. copy()

        p = params_per_asset[ticker]

        # Initialize weight array
        weights = []
        prev_weight = p['max_weight']  # start from max_weight

        for i in range(len(dd_smooth)):
            ddn_today = dd_smooth.iloc[i]
            ddn_prev = dd_smooth.iloc[i - 1] if i > 0 else ddn_today

            w = weight_from_dd_cumulative(
                prev_weight=prev_weight,
                ddn=ddn_today,
                ddn_prev=ddn_prev,
                add_on_recovery=p['aggr_increase'],
                sub_on_drawdown=p['aggr_decrease'],
                max_weight=p['max_weight'],
                dd_max_stop=p['dd_max_stop'],
                dd_small_threshold=p['dd_small_threshold']
            )
            weights.append(w)
            prev_weight = w  # update for next day

        weights_df[ticker] = pd.Series(weights, index=df.index)

    return weights_df, ddn_df


# ===========================
# 4️⃣ Compute per-asset equity
# ===========================
def compute_per_asset_equity(data_dict, weights_df):
    equity_dict = {}
    for ticker, df in data_dict.items():
        price = df['Close']
        ret = price.pct_change().fillna(0)
        weighted_ret = ret * weights_df[ticker]
        equity = (1 + weighted_ret).cumprod()
        buy_and_hold = (1 + ret).cumprod()
        equity_dict[ticker] = pd.DataFrame({
            'Weighted': equity,
            'Buy & Hold': buy_and_hold
        })
    return equity_dict

# ===========================
# 5️⃣ Metrics computation
# ===========================
def compute_metrics(equity_dict):
    metrics = {}
    for ticker, df in equity_dict.items():
        weighted = df['Weighted']
        bh = df['Buy & Hold']
        cagr_weigheted = np.log(weighted.iloc[-1])/len(weighted) * 252
        cagr_bh = np.log(bh.iloc[-1])/len(bh) * 252
        vol_weighted = weighted.pct_change().std() * np.sqrt(252)
        vol_bh = bh.pct_change().std() * np.sqrt(252)
        sharpe_weighted=cagr_weigheted/vol_weighted
        sharpe_bh = cagr_bh / vol_bh
        max_dd_weighted = (weighted / weighted.cummax() - 1).min()
        max_dd_bh = (bh / bh.cummax() - 1).min()
        metrics[ticker] = {
            'CAGR Weighted': cagr_weigheted,
            'CAGR BH': cagr_bh,
            'Vol Weighted': vol_weighted,
            'Vol Buy & Hold': vol_bh,
            'Sharpe Weighted': sharpe_weighted,
            'Sharpe BH': sharpe_bh,
            'Max DD Weighted': max_dd_weighted,
            'Max DD Buy & Hold': max_dd_bh,
            'Final Equity Weighted': weighted.iloc[-1],
            'Final Equity Buy & Hold': bh.iloc[-1],

        }
    return metrics

def print_metrics(metrics):
    df = pd.DataFrame(metrics).T
    print(df)

# ===========================
# 6️⃣ Plot equity + weights
# ===========================
def plot_equity_weights_ddn(equity_dict, weights_df, ddn_df):
    """
    Plot per asset:
        - Weighted strategy equity
        - Buy & Hold equity
        - Strategy weights
        - Real drawdown (ddn)
    """
    for ticker in equity_dict.keys():
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Plot equity curves on primary y-axis
        ax1.plot(equity_dict[ticker]['Weighted'], label='Weighted Strategy', color='blue')
        ax1.plot(equity_dict[ticker]['Buy & Hold'], label='Buy & Hold', linestyle='--', color='orange')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity')
        ax1.legend(loc='upper left')

        # Secondary axis for weights
        ax2 = ax1.twinx()
        ax2.plot(weights_df[ticker], label='Weight', color='green', alpha=0.6)
        ax2.set_ylabel('Weight')
        ax2.set_ylim(0, weights_df[ticker].max()*1.1)
        ax2.legend(loc='upper right')

        # Third axis for drawdown
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.1))  # offset third axis
        ax3.plot(ddn_df[ticker], label='Drawdown', color='red', alpha=0.5)
        ax3.set_ylabel('Drawdown')
        ax3.legend(loc='lower right')

        plt.title(f'{ticker} - Equity, Weights & Drawdown')



# ===========================
# 7️⃣ Manual parameters per asset
# ===========================
manual_params = {
    'ES=F': {'dd_small_threshold': 0.02, 'dd_max_stop': -0.12, 'aggr_decrease': 1.0, 'aggr_increase': 5.0, 'max_weight': 3.0},
    'NQ=F': {'dd_small_threshold': 0.04, 'dd_max_stop': -0.16, 'aggr_decrease': 1.0, 'aggr_increase': 3.5, 'max_weight': 3.0},
    'GC=F': {'dd_small_threshold': 0.02, 'dd_max_stop': -0.12, 'aggr_decrease': 1.0, 'aggr_increase': 2.0, 'max_weight': 3.0},
    'CL=F': {'dd_small_threshold': 0.02, 'dd_max_stop': -0.16, 'aggr_decrease': 1.0, 'aggr_increase': 3.5, 'max_weight': 3.0},
    'EURUSD=X': {'dd_small_threshold': 0.02, 'dd_max_stop': -0.12, 'aggr_decrease': 1.0, 'aggr_increase': 5.0, 'max_weight': 3.0},
    'cash': {'dd_small_threshold': 0.01, 'dd_max_stop': 0.0, 'aggr_decrease': 0.0, 'aggr_increase': 0.0, 'max_weight': 1.0}
}

# ===========================
# 8️⃣ Run pipeline
# ===========================
if __name__ == "__main__":
    # data_dict must be your historical data
    weights_df,ddn_df = compute_weights(data_dict, manual_params)

    weights_df= weights_df.rolling(3).mean().fillna(1)
    weights_df = weights_df * 0.4

    equity_dict = compute_per_asset_equity(data_dict, weights_df)
    metrics = compute_metrics(equity_dict)
    print_metrics(metrics)
    plot_equity_weights_ddn(equity_dict, weights_df, ddn_df)


    plt.show()