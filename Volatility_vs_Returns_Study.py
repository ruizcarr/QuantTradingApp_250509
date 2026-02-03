import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Wider print limits
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
# Silence warnings
import warnings
warnings.filterwarnings('ignore')

from config.trading_settings import settings
import Market_Data_Feed as mdf

#Main Code
def main(settings):

    # -------------------------------
    # SETTINGS
    # -------------------------------

    # Update Settings
    settings['start'] = '1996-01-01'

    #Study Settings
    window_vol= 22  # rolling window for vol
    future_horizon = 22  # how far ahead to measure returns
    corr_window = 22*12 # rolling window for correlation
    vol_change_window = 5 # Short-term volatility change

    # -------------------------------
    # DATA
    # -------------------------------
    data_ind = mdf.Data_Ind_Feed(settings).data_ind
    data, _ = data_ind

    returns = data.tickers_returns

    # -------------------------------
    # STEP 1: compute rolling volatility
    # -------------------------------
    vol = returns.rolling(window_vol).std()

    # -------------------------------
    # STEP 2: compute short-term vol change (slope)
    # -------------------------------
    vol_change = vol.pct_change(vol_change_window)

    # -------------------------------
    # STEP 3: compute quantile-based regimes
    # -------------------------------
    def compute_regimes(df, col_name):
        lower_q=0.33           #0.33
        upper_q =0.66        #0.66
        q = df[col_name].quantile([lower_q, upper_q])
        regimes = np.where(df[col_name] < q.iloc[0], "Bullish",
                           np.where(df[col_name] > q.iloc[1], "Bearish", "Neutral"))
        return pd.DataFrame(regimes, index=df.index, columns=df.columns)

    # Volatility level regimes
    vol_level_regimes = compute_regimes(vol, col_name=None) if vol.columns is None else \
        pd.DataFrame({c: compute_regimes(vol[[c]], c)[c] for c in vol.columns})
    regime_rename_level = {"Bearish": "HighVol", "Neutral": "MediumVol", "Bullish": "LowVol"}
    vol_level_regimes = vol_level_regimes.replace(regime_rename_level)

    # Volatility change regimes
    vol_change_regimes = compute_regimes(vol_change, col_name=None) if vol_change.columns is None else \
        pd.DataFrame({c: compute_regimes(vol_change[[c]], c)[c] for c in vol_change.columns})
    regime_rename_change = {"Bearish": "RisingVol", "Neutral": "StableVol", "Bullish": "FallingVol"}
    vol_change_regimes = vol_change_regimes.replace(regime_rename_change)

    print('vol_level_regimes',vol_level_regimes)

    # -------------------------------
    # STEP 4: compute forward returns
    # -------------------------------
    future_returns = returns.shift(-1).rolling(future_horizon).sum()

    # -------------------------------
    # STEP 5: aggregate mean future returns by regime
    # -------------------------------
    def regime_returns_summary(future_returns, regimes):
        rows = []

        for asset in future_returns.columns:
            # group future returns by the regime for this asset
            grouped = future_returns[asset].groupby(regimes[asset])

            mean_vals = grouped.mean()
            std_vals = grouped.std()  # <-- added standard deviation

            # build rows
            for regime in mean_vals.index:
                rows.append([
                    asset,
                    regime,
                    mean_vals.loc[regime],
                    std_vals.loc[regime]
                ])

        summary = pd.DataFrame(
            rows,
            columns=["Asset", "Regime", "FutureReturn", "FutureReturnStd"]
        )

        # Significance metrics
        summary["SharpeLike"] = summary["FutureReturn"] / summary["FutureReturnStd"]

        summary = summary.sort_values(["Asset", "Regime"]).reset_index(drop=True)

        return summary

    # Future returns by vol level
    summary_vol_level = regime_returns_summary(future_returns, vol_level_regimes)

    # Future returns by vol change
    summary_vol_change = regime_returns_summary(future_returns, vol_change_regimes)


    # -------------------------------
    # STEP 7: print summaries
    # -------------------------------
    print("=== FUTURE RETURNS BY VOLATILITY LEVEL ===")
    print(summary_vol_level)

    print("\n=== FUTURE RETURNS BY VOLATILITY CHANGE ===")
    print(summary_vol_change)

    # -------------------------------
    # Regime Combinations
    # -------------------------------
    def regime_combination_summary(future_returns, combined_regimes):
        rows = []

        for asset in future_returns.columns:
            grp = future_returns[asset].groupby(combined_regimes[asset])

            means = grp.mean()
            stds = grp.std()
            counts = grp.count()

            for regime in means.index:
                rows.append([
                    asset,
                    regime,
                    means.loc[regime],
                    stds.loc[regime],
                    counts.loc[regime]
                ])

        summary = pd.DataFrame(
            rows,
            columns=["Asset", "Regime", "FutureReturn", "FutureReturnStd", "Count"]
        )

        summary["SharpeLike"] = summary["FutureReturn"] / summary["FutureReturnStd"]

        return summary

    combined_regimes = vol_level_regimes + "|" + vol_change_regimes
    combined_summary = regime_combination_summary(future_returns, combined_regimes)

    print("\n=== COMBINED BY VOLATILITY CHANGE  | VOLATILITY CHANGE  ===")
    print(combined_summary)

    #Add regime_weights for booth summary_vol_level and summary_vol_change
    def add_regime_weight_OK(df):
        max=df['FutureReturn'].max()
        min=df['FutureReturn'].min()
        df['result_norm']=np.where(df['FutureReturn']>0,df['FutureReturn']/max,-df['FutureReturn']/min)
        df['regime_weight'] = (1 + df['result_norm']/2).round(1)
        return df

    def add_regime_weight(df):
        # Compute max of absolute value of FutureReturn per Asset
        #grouped = df.groupby("Asset")["FutureReturn"]
        grouped = df.groupby("Asset")["SharpeLike"]
        max_vals = grouped.transform(lambda x: x.abs().max())

        # Normalized value per asset
        #df["result_norm"] =df["FutureReturn"] / max_vals
        df["result_norm"] = df["SharpeLike"] / max_vals

        # Convert normalized value → weight 0.5–1.5 (center 1.0)
        df["regime_weight"] = (1 + df["result_norm"] / 1.0).round(1)

        return df

    add_regime_weight(summary_vol_level)
    print("summary_vol_level\n",summary_vol_level)

    add_regime_weight(summary_vol_change)
    print("summary_vol_change\n", summary_vol_change)

    add_regime_weight(combined_summary)
    print("combined_summary\n", combined_summary)

    # Build mapping table indexed by (Asset, Regime)
    map_weights = (
        summary_vol_level
        .set_index(["Asset", "Regime"])["regime_weight"]
    )

    # Copy template
    weights_level_df = vol_level_regimes.copy()

    # Replace regime labels by numbers for each column/asset
    for asset in weights_level_df.columns:
        # Create a mapping for this specific asset
        asset_map = map_weights.loc[asset].to_dict()

        # Replace regime names → numeric weights
        weights_level_df[asset] = weights_level_df[asset].map(asset_map)

    print("weights_level_df\n",weights_level_df)

    # Build mapping for vol change
    map_change = (
        summary_vol_change
        .set_index(["Asset", "Regime"])["regime_weight"]
    )

    weights_change_df = vol_change_regimes.copy()

    for asset in weights_change_df.columns:
        asset_map = map_change.loc[asset].to_dict()
        weights_change_df[asset] = weights_change_df[asset].map(asset_map)

    # Build mapping for combined vol change | vol levels

    map_weights_combined = (
        combined_summary
        .set_index(["Asset", "Regime"])["regime_weight"]
    )

    weights_combined_df = combined_regimes.copy()

    for asset in weights_combined_df.columns:
        asset_map = map_weights_combined.loc[asset].to_dict()
        weights_combined_df[asset] = weights_combined_df[asset].map(asset_map)

    k_level=0
    k_change=0
    k_comb=1
    weights_combined = (k_level*weights_level_df + k_change*weights_change_df+k_comb*weights_combined_df) / (k_level+k_change+k_comb)

    #weights_combined = weights_combined.div(weights_combined.mean(axis=1), axis=0)
    #weights_combined = weights_combined*0.85

    print(weights_combined.tail(10))

    weights_combined.plot()

    #Back test
    if True:

        # Fix scale_factor
        scale_factor = np.array([0.9, 0.68, 0.60, 2.8, 0.2, 0.6])

        weights_combined_scaled = weights_combined.shift(1).fillna(1) * scale_factor

        weighted_returns= returns * weights_combined_scaled
        cum_weighted_returns = (1 + weighted_returns).cumprod()

        cum_ret= (1 + returns).cumprod()

        for ticker in returns.columns:
            plot_df=pd.DataFrame()
            plot_df['cum_ret']=cum_ret[ticker]
            plot_df['cum_weighted_returns']=cum_weighted_returns[ticker]
            #plt.figure()
            plot_df.plot(title=ticker)

    #cum_weighed_returns.plot()

    # Metrics
    weighed_cagr = weighted_returns.mean() * 252
    weighed_vol = weighted_returns.std() * 16
    weighed_sharpe = weighed_cagr / weighed_vol

    print("weighed_sharpe", weighed_sharpe)



    plt.show()


def compute_rolling_corr_series(returns, window_vol, future_horizon, corr_window):
    df = pd.DataFrame()
    df["ret"] = returns

    df["vol"] = df["ret"].rolling(window_vol).std()
    df["vol"] = df["vol"].pct_change(5)
    df["future_cumret"] = df["ret"].shift(-1).rolling(future_horizon).sum()

    q = df["vol"].quantile([0.33, 0.66])
    regimes_np = np.where(df["vol"] < q.iloc[0], "Bullish",
                            np.where(df["vol"] > q.iloc[1], "Bearish", "Neutral"))

    # rolling correlation
    rolling_corr = (
        df[["vol", "future_cumret"]]
        .rolling(corr_window)
        .corr()
        .groupby(level=0)
        .apply(lambda x: x.iat[0, 1])
    )

    return rolling_corr,regimes_np

def back_study(returns,wondow_vol):
    #Rolling Volatility
    volat=returns.rolling(wondow_vol).std().dropna()*16

    cumret=(1+returns.reindex(volat.index)).cumprod()

    for ticker in volat.columns:
        df_plot = pd.DataFrame()
        df_plot['volat']=volat[ticker]*10
        df_plot['cumret']=cumret[ticker]

        df_plot.plot(title=ticker)

def define_regimes(rolling_corr_smooth):
    regimes = pd.DataFrame(index=rolling_corr_smooth.index)

    for asset in rolling_corr_smooth.columns:
        rc = rolling_corr_smooth[asset]

        regimes[asset] = np.where(
            rc < -0.2, "Bullish",
            np.where(rc > 0.2, "Bearish", "Neutral")
        )

    return regimes

if __name__ == '__main__':
    main(settings)