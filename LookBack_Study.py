#Import libraries and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import os.path
import os
import joblib

from quantstats_lumi.stats import volatility

# Wider print limits
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
# Silence warnings
import warnings
warnings.filterwarnings('ignore')

from Backtest_Vectorized_Class import compute_backtest_vectorized
#from Backtest_Vectorized import compute_backtest_vectorized
from Markowitz_Vectorized import compute_optimized_markowitz_d_w_m
from WalkForwardTraining import WalkForwardTraining
import Market_Data_Feed as mdf
from utils import mean_positions


#Get SETTINGS
from config import settings,utils
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



# MAIN CODE
def run(settings):

    settings["start"]='2001-01-01'

    #DATA & INDICATORS
    data_ind=mdf.Data_Ind_Feed(settings).data_ind
    data, indicators_dict = data_ind
    tickers_returns=data.tickers_returns

    lookbacks = {
        #"1W": 5,  # 1 week
        #"1M": 21,  # 1 month
        #"3M": 63,  # 3 months
        #"6M": 126,  # 6 months
        "1Y": 252,  # 1 year
        #"2Y": 252*2  # 2 year
    }

    # Train models and save in 'trained_models'
    features, target_week = train_models(tickers_returns, lookbacks, model_path="trained_models")

    # Load models from 'trained_models' and predict next-week returns
    pred_next_week = predict_next_week(tickers_returns, lookbacks, model_path="trained_models")

    print(pred_next_week)

    pred_next_week.plot()

    weights = predictions_to_minmax_weights_global_scalar(pred_next_week)

    print(weights)

    weights.plot()

    portfolio_ret= backtest_portfolio(weights, tickers_returns)

    print("portfolio_ret",portfolio_ret)


    # Portfolio cumulative return
    cum_portfolio = (1 + portfolio_ret).cumprod()

    # Individual assets cumulative return
    cum_assets = (1 + tickers_returns.reindex(portfolio_ret.index)).cumprod()

    plot_df =pd.DataFrame()
    for col in weights.columns:
        plot_df["cum_portfolio"]=cum_portfolio[col]
        plot_df["cum_assets"] = cum_assets[col]

        plot_df.plot(title=col)

    plt.show()


def backtest_portfolio(weights, tickers_returns):
    """
    Compute portfolio returns per asset (weighted) from weights and tickers_returns.

    Returns a DataFrame with assets as columns and dates as index.
    """
    tickers = weights.columns
    returns = tickers_returns[tickers].reindex(weights.index)

    # Weighted returns per asset
    portfolio_ret = weights * returns

    # Ensure numeric
    portfolio_ret = portfolio_ret.apply(pd.to_numeric, errors='coerce')

    # Drop rows where all assets are NaN
    portfolio_ret = portfolio_ret.dropna(how='all')

    return portfolio_ret


def predictions_to_minmax_weights_global_scalar(predictions, clip_lower=None, eps=1e-8):
    """
    Convert predicted next-week returns into portfolio weights using global min-max scaling,
    safely handling rows where sum=0.
    """
    # Drop 'cash'
    tickers = [t for t in predictions.columns if t != "cash"]
    weights = predictions[tickers].copy()

    # Drop rows where all tickers are NaN
    weights = weights.dropna(how="all")

    if weights.empty:
        print("No valid predictions available to create weights.")
        return weights

    # Global min and max
    min_val = weights.min().min()
    max_val = weights.max().max()

    # Min-max scaling
    weights = (weights - min_val) / (max_val - min_val + eps)  # avoid division by zero

    # Optional lower bound
    if clip_lower is not None:
        weights = weights.clip(lower=clip_lower)

    # Normalize each row safely
    row_sums = weights.sum(axis=1)
    weights = weights.div(row_sums.replace(0, np.nan), axis=0)  # skip rows with sum=0

    weights = weights/weights.mean()#.mean()

    weights= weights.clip(lower=0.5, upper=1.5)*1.1

    return weights


def train_models(tickers_returns, lookbacks, model_path="trained_models"):
    """
    Train linear regression models to predict next-week risk-adjusted returns
    and save them in 'trained_models' folder.

    tickers_returns: DataFrame of daily returns (tickers x dates)
    lookbacks: dict, e.g. {"1W":5, "1M":21, "3M":63, "6M":126, "1Y":252}
    model_path: folder to save models
    """
    os.makedirs(model_path, exist_ok=True)  # create folder if it doesn't exist

    # Drop cash
    tickers = [t for t in tickers_returns.columns if t != "cash"]
    tickers_returns = tickers_returns[tickers]

    # Step 1: Compute volatility and risk-adjusted returns
    volatility = tickers_returns.rolling(22).std() * 16
    tickers_returns_volat = tickers_returns #/ volatility

    # Step 2: Compute next-week risk-adjusted returns
    target_week = tickers_returns_volat.rolling(5).sum().shift(-5)
    #target_week = tickers_returns_volat.rolling(22).sum().shift(-22)

    # Step 3: Construct features: past cumulative returns at each lookback
    features = pd.DataFrame(index=tickers_returns.index,
                            columns=pd.MultiIndex.from_product([tickers_returns.columns, lookbacks.keys()]))

    for ticker in tickers_returns.columns:
        for name, lb in lookbacks.items():
            features[ticker, name] = tickers_returns_volat[ticker].rolling(lb).sum()

    # Step 4: Train and save model for each ticker
    for ticker in tickers_returns.columns:
        # Align features and target to avoid KeyError
        df = pd.concat([features[ticker], target_week[ticker]], axis=1)
        df = df.dropna()  # remove rows with NaN in feature or target

        if len(df) == 0:
            continue

        X = df.iloc[:, :-1].values  # all lookback columns
        y = df.iloc[:, -1].values  # next-week risk-adjusted return

        # Train linear regression
        model = LinearRegression().fit(X, y)

        # Save model
        joblib.dump(model, f"{model_path}/model_{ticker}.pkl")

    print(f"Training complete. Models saved in '{model_path}'.")
    return features, target_week  # optional return for inspection


def predict_next_week(tickers_returns, lookbacks, model_path="trained_models"):
    """
    Predict next-week risk-adjusted returns as a time series for each ticker
    using saved linear regression models.

    Returns: DataFrame indexed by date, columns = tickers
    """
    # Drop cash
    tickers = [t for t in tickers_returns.columns if t != "cash"]
    tickers_returns = tickers_returns[tickers]

    # Compute volatility and risk-adjusted returns
    volatility = tickers_returns.rolling(22).std() * 16
    tickers_returns_volat = tickers_returns / volatility

    # Construct features
    features = pd.DataFrame(index=tickers_returns.index,
                            columns=pd.MultiIndex.from_product([tickers_returns.columns, lookbacks.keys()]))

    for ticker in tickers_returns.columns:
        for name, lb in lookbacks.items():
            features[ticker, name] = tickers_returns_volat[ticker].rolling(lb).sum()

    # Initialize predictions DataFrame
    predictions = pd.DataFrame(index=tickers_returns.index, columns=tickers_returns.columns)

    # Loop over each ticker
    for ticker in tickers_returns.columns:
        try:
            model_file = os.path.join(model_path, f"model_{ticker}.pkl")
            if not os.path.exists(model_file):
                continue
            model = joblib.load(model_file)

            # Loop over each date where features are valid
            for i in range(len(features)):
                X_row = features[ticker].iloc[i].values
                if pd.isna(X_row).any():
                    continue
                predictions.iloc[i, predictions.columns.get_loc(ticker)] = model.predict(X_row.reshape(1, -1))[0]
        except:
            predictions[ticker] = None

    return predictions

def predict_next_week_simple(tickers_returns, lookbacks):
    """
    Predict next-week risk-adjusted returns using a single linear regression per lookback.
    Returns regression coefficients (slope) per ticker x lookback.
    """
    # Weekly volatility
    volatility = tickers_returns.rolling(22).std() * 16
    tickers_returns_volat = tickers_returns / volatility

    # Target: next week risk-adjusted return
    target_week = tickers_returns_volat.rolling(5).sum().shift(-5)

    results = pd.DataFrame(index=tickers_returns.columns, columns=lookbacks.keys())

    for ticker in tickers_returns.columns:
        y = target_week[ticker]

        for name, lb in lookbacks.items():
            X = tickers_returns_volat[ticker].rolling(lb).sum()

            df = pd.concat([X, y], axis=1).dropna()
            if len(df) == 0:
                continue

            model = LinearRegression().fit(df.iloc[:, 0].values.reshape(-1, 1), df.iloc[:, 1].values)
            # Use coefficient as measure of predictive power
            results.loc[ticker, name] = model.coef_[0]

    return results

def evaluate_hybrid_direction(tickers_returns, lookbacks, period="day"):
    """
    Hybrid directional evaluation:
    - Feature: real cumulative return over lookback
    - Target: directional next-period return (1 if positive, 0 if negative)

    period: "day", "week", "month"
    """
    results = pd.DataFrame(index=tickers_returns.columns, columns=lookbacks.keys())

    # Define target based on horizon
    if period == "week":
        target = tickers_returns.rolling(5).sum().shift(-5)
    elif period == "month":
        target = tickers_returns.rolling(21).sum().shift(-21)
    else:
        target = tickers_returns.shift(-1)

    target_dir = (target > 0).astype(int)

    for ticker in tickers_returns.columns:
        for name, lb in lookbacks.items():
            # Feature: real past return over lookback
            signal = tickers_returns[ticker].rolling(lb).sum()

            valid = signal.notna() & target_dir[ticker].notna()

            # Use sign of feature to predict direction
            signal_dir = (signal[valid] > 0).astype(int)

            # Accuracy = fraction of correct directional predictions
            accuracy = (signal_dir == target_dir[ticker][valid]).mean()
            results.loc[ticker, name] = accuracy

    return results


def evaluate_direction(tickers_returns, lookbacks, period="week"):
    """
    Evaluate directional accuracy of lookback returns predicting next-period returns.

    period: "day"=next day, "week"=5d, "month"=21d
    """
    results = pd.DataFrame(index=tickers_returns.columns, columns=lookbacks.keys())

    if period == "week":
        target = tickers_returns.rolling(5).sum().shift(-5)
    elif period == "month":
        target = tickers_returns.rolling(21).sum().shift(-21)
    else:
        target = tickers_returns.shift(-1)

    target_dir = (target > 0).astype(int)

    for ticker in tickers_returns.columns:
        for name, lb in lookbacks.items():
            signal = tickers_returns[ticker].rolling(lb).sum()

            valid = signal.notna() & target_dir[ticker].notna()
            signal_dir = (signal[valid] > 0).astype(int)

            # Accuracy = fraction of times past direction matches future direction
            accuracy = (signal_dir == target_dir[ticker][valid]).mean()
            results.loc[ticker, name] = accuracy

    return results


def evaluate_correlation_week(tickers_returns, lookbacks):
    results = pd.DataFrame(index=tickers_returns.columns, columns=lookbacks.keys())

    next_week = tickers_returns.rolling(21).sum().shift(-21)  # target

    for ticker in tickers_returns.columns:
        next_ret = next_week[ticker]

        for name, lb in lookbacks.items():
            past_ret = tickers_returns[ticker].rolling(lb).sum()
            valid = past_ret.notna() & next_ret.notna()
            corr = past_ret[valid].corr(next_ret[valid])
            results.loc[ticker, name] = corr

    return results


def evaluate_regression_week(tickers_returns, lookbacks, window=252):
    results = pd.DataFrame(index=tickers_returns.columns, columns=lookbacks.keys())

    next_week = tickers_returns.rolling(21).sum().shift(-21)

    for ticker in tickers_returns.columns:
        y_full = next_week[ticker]

        for name, lb in lookbacks.items():
            X_full = tickers_returns[ticker].rolling(lb).sum().shift(1)  # predictor
            df = pd.concat([X_full, y_full], axis=1).dropna()

            if len(df) < window:
                continue

            preds, truth = [], []
            for t in range(window, len(df)):
                X_train = df.iloc[t - window:t, 0].values.reshape(-1, 1)
                y_train = df.iloc[t - window:t, 1].values
                X_test = df.iloc[t, 0].reshape(1, -1)
                y_test = df.iloc[t, 1]

                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_test)[0]
                preds.append(y_pred)
                truth.append(y_test)

            if len(preds) > 0:
                results.loc[ticker, name] = r2_score(truth, preds)

    return results


def summarize_results(corr_results, reg_results):
    """
    Summarize best lookback per ticker for correlation and regression.

    Returns a DataFrame with per-ticker best lookback and value.
    """
    corr_abs = corr_results.astype(float).abs()
    reg_vals = reg_results.astype(float)

    summary = pd.DataFrame(index=corr_results.index)

    # Best correlation
    summary["BestCorrLookback"] = corr_abs.idxmax(axis=1)

    # Get actual correlation values (with numpy indexing)
    corr_vals = corr_results.to_numpy()
    corr_idx = [list(corr_results.columns).index(col) for col in summary["BestCorrLookback"]]
    summary["BestCorrValue"] = corr_vals[np.arange(len(corr_results)), corr_idx]

    # Best regression
    summary["BestRegLookback"] = reg_vals.idxmax(axis=1)

    reg_vals_array = reg_vals.to_numpy()
    reg_idx = [list(reg_vals.columns).index(col) for col in summary["BestRegLookback"]]
    summary["BestRegR2"] = reg_vals_array[np.arange(len(reg_vals)), reg_idx]

    return summary


def evaluate_correlation(tickers_returns, lookbacks):
    results = pd.DataFrame(index=tickers_returns.columns, columns=lookbacks.keys())

    for ticker in tickers_returns.columns:
        rets = tickers_returns[ticker]
        next_day = rets.shift(-1)

        for name, lb in lookbacks.items():
            past_ret = rets.rolling(lb).sum()
            valid = past_ret.notna() & next_day.notna()
            corr = past_ret[valid].corr(next_day[valid])
            results.loc[ticker, name] = corr

    return results


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def evaluate_regression(tickers_returns, lookbacks, window=252):
    results = pd.DataFrame(index=tickers_returns.columns, columns=lookbacks.keys())

    for ticker in tickers_returns.columns:
        rets = tickers_returns[ticker]
        next_day = rets.shift(-1)

        for name, lb in lookbacks.items():
            X = rets.rolling(lb).sum().shift(1)  # predictor must be lagged
            y = next_day

            # Align data
            df = pd.concat([X, y], axis=1).dropna()
            if len(df) < window:
                continue

            # Rolling regression evaluation
            preds, truth = [], []
            for t in range(window, len(df)):
                X_train = df.iloc[t - window:t, 0].values.reshape(-1, 1)
                y_train = df.iloc[t - window:t, 1].values
                X_test = df.iloc[t, 0].reshape(1, -1)
                y_test = df.iloc[t, 1]

                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_test)[0]
                preds.append(y_pred)
                truth.append(y_test)

            if len(preds) > 0:
                results.loc[ticker, name] = r2_score(truth, preds)

    return results




# -----------------------------
# Investment System
# -----------------------------
def build_investment_system(tickers_returns, method="regression"):
    """
    Generate daily portfolio weights based on correlation or regression forecasts
    from different lookback periods.

    Parameters
    ----------
    tickers_returns : pd.DataFrame
        Daily returns of tickers, index=Date, columns=tickers.
    method : str, "correlation" or "regression"
        Signal generation method.

    Returns
    -------
    weights : pd.DataFrame
        Daily normalized portfolio weights for each ticker.
    """

    lookbacks = {"1W": 5, "1M": 21, "3M": 63, "1Y": 252}
    weights = pd.DataFrame(index=tickers_returns.index, columns=tickers_returns.columns)

    for ticker in tickers_returns.columns:
        rets = tickers_returns[ticker]
        next_day = rets.shift(-1)

        # Build features: past lookback returns
        X = pd.concat(
            {name: rets.rolling(lb).sum() for name, lb in lookbacks.items()},
            axis=1
        )

        if method == "correlation":
            signals = pd.DataFrame(index=rets.index, columns=lookbacks.keys())
            for name in lookbacks:
                signals[name] = X[name].rolling(60).corr(next_day)  # 60-day corr window
            forecast = signals.mean(axis=1)  # average correlation signal

        elif method == "regression":
            forecast = pd.Series(index=rets.index, dtype=float)
            window = 252  # 1 year rolling regression training
            for t in range(window, len(X) - 1):
                X_train = X.iloc[t - window:t].fillna(0)
                y_train = next_day.iloc[t - window:t].fillna(0)
                if y_train.notna().sum() > 0:
                    model = LinearRegression().fit(X_train, y_train)
                    forecast.iloc[t] = model.predict(
                        X.iloc[t].fillna(0).values.reshape(1, -1)
                    )[0]
        else:
            raise ValueError("method must be 'correlation' or 'regression'")

        weights[ticker] = forecast

    # Normalize weights (sum to 1 each day)
    weights = weights.div(weights.abs().sum(axis=1), axis=0)
    return weights



if __name__ == '__main__':
    run(settings)