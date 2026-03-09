import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ignore SettingWithCopyWarning from Pandas, as operations are intentional
#warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)

# Wider print limits
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)

import Market_Data_Feed as mdf

#Get SETTINGS
from config import settings,utils
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py


def run(settings):

    # DATA & INDICATORS
    data_ind = mdf.Data_Ind_Feed(settings).data_ind
    data, indicators_dict = data_ind
    tickers_returns = data.tickers_returns
    data_dict=data.data_dict

    #print('tickers_returns',tickers_returns)
    #print('data_dict',data_dict)

    positions=get_positions().fillna(0)
    #print(positions)

    #Pull back Study

    #Settings
    positions_treshold=0.1
    future_len=20

    #All tickers calculations
    tickers_returns=tickers_returns.reindex(positions.index)
    cumret=(1+tickers_returns).cumprod()
    ddn=cumret/cumret.rolling(5,min_periods=1).max() -1
    uptrend=(positions.shift(1)>positions_treshold)*1
    cagr_20d=tickers_returns.shift(-future_len).rolling(future_len).sum()#*250/future_len
    min_20d=cumret.shift(-future_len).rolling(future_len).min()
    risk_20d=min_20d/cumret-1#*250/future_len
    #risk_20d=(cumret.shift(-future_len).rolling(future_len).min()/cumret-1)*250/future_len
    reward_20d=(cagr_20d+risk_20d)#.clip(lower=-1)

    pos_returns=tickers_returns*positions
    pos_cumret=(1+pos_returns).cumprod()

    ddn_mean=ddn.shift(1).mean()
    ddn_normalized=-ddn/ddn_mean


    #Ticker by ticker calculations
    for ticker in settings['tickers']:
        df=pd.DataFrame()
        df['pos_cumret'] = pos_cumret[ticker].copy()
        df['cum_ret'] = cumret[ticker].copy()
        df['uptrend']=uptrend[ticker].copy()
        df['ddn'] = ddn[ticker].copy()

        df['cagr_20d'] = cagr_20d[ticker].copy()
        df['risk_20d'] = risk_20d[ticker].copy()
        df['reward_20d'] = reward_20d[ticker].copy()

        df['ddn_normalized'] = ddn_normalized[ticker].copy()

        #df['ddn_index'] = ddn_index[ticker].copy()
        #df['strategy_cumret'] = strategy_cumret[ticker].copy()

        df.dropna(inplace=True)

        print(ticker, df)
        #print(df['ddn'].describe())
        df.plot(title=ticker)

        #Keep only values when uptrend
        df=df[df['uptrend']>0]



        #Keep only low ddn values
        df=df[df['ddn']>-0.14]

        #Keep only values not very low
        df = df[df['ddn'] < -0.01] #-0.13

        print(df)

        x_y_regresion_plot(df,
                           'ddn' ,#'ddn_normalized',
                           'cagr_20d', #'reward_20d',#'risk_20d', #
                           ticker,
                           degree=2)


    plt.show()

def get_positions():
    # Creates the correct path for your OS
    folder_name = 'results'
    csv_filename = 'training_positions.csv' #'trading_log_history.csv'
    full_path = os.path.join(folder_name, csv_filename) # Creates the correct path for your OS

    # Load the DataFrame back from the CSV file within the "Datasets" folder
    #positions = pd.read_csv(full_path)

    positions = pd.read_csv(
        full_path,
        index_col='Unnamed: 0',  # Specify the column to use as index
        parse_dates=True,  # Tell pandas to parse this column as dates
        dayfirst=False  # Set to True if your dates are DD-MM-YYYY, otherwise False (MM-DD-YYYY) or omit for auto-detection
    )
    # Rename the index for consistency and clarity
    positions.index.name = 'Date'

    return positions

def get_ddn_index(ddn_normalized):
    uptrend_pullback_noise_treshold = -0.25  # 0.5 * ddn_mean
    ddn_index = ddn_normalized.clip(upper=uptrend_pullback_noise_treshold)
    uptrend_pullback_treshold = -0.5  # 0.5 * ddn_mean
    ddn_index[ddn_index > uptrend_pullback_treshold] = -ddn_index
    ddn_index = (ddn_index + uptrend_pullback_treshold - uptrend_pullback_noise_treshold + 1.6).clip(lower=0)

    return ddn_index

def get_ddn_index_cumret(ddn_normalized,pos_returns):
    ddn_index = get_ddn_index(ddn_normalized)
    strategy_cumret = (1 + pos_returns * ddn_index.shift(1)).cumprod()

    return strategy_cumret

def x_y_regresion_plot(df, x_col, y_col, sub_title, degree=1):

        """
        Generates a scatter plot with a polynomial regression curve.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            x_col (str): The name of the column to use for the x-axis (independent variable).
            y_col (str): The name of the column to use for the y-axis (dependent variable).
            sub_title (str): A subtitle for the plot.
            degree (int): The degree of the polynomial regression (1 for linear, 2 for quadratic, 3 for cubic).
                          Defaults to 1.
        """

        # df non empty check:
        if len(df) == 0:
            print("Error: df is an empty df. Cannot perform polyfit.")
            return

        # Extract x and y data
        x = df[x_col]
        y = df[y_col]

        # Handle NaN values: Drop rows where either x or y is NaN
        # Create a mask for non-NaN values in both x and y
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid_mask]
        y = y[valid_mask]

        # Crucial check: Ensure x and y are not empty after NaN removal
        if len(x) == 0 or len(y) == 0:
            print(f"Error: x or y vector is empty after dropping NaNs. Cannot perform polyfit.")
            return

        # Ensure x and y have the same length
        if len(x) != len(y):
            print(f"Error: x and y have different lengths ({len(x)} vs {len(y)}). Cannot perform polyfit.")
            return

        # --- Regression Steps ---

        # Calculate the coefficients of the polynomial regression
        # The 'degree' parameter controls the polynomial order
        coefficients = np.polyfit(x, y, degree)

        # Create a polynomial function from the coefficients
        # This allows us to easily calculate y values for the regression curve
        polynomial_function = np.poly1d(coefficients)

        # Print the calculated coefficients
        print(f"Calculated polynomial coefficients (degree {degree}): {coefficients}")

        # --- Plotting Steps ---
        plt.figure(figsize=(10, 7))
        plt.scatter(x, y, label='Data Points', alpha=0.6)  # Added alpha for better visualization of dense points

        # Generate x values for the smooth regression curve
        # It's important to sort the x values for plotting a smooth curve
        x_for_plot = np.linspace(x.min(), x.max(), 500)  # Create a range of x values for a smooth curve
        y_for_plot = polynomial_function(x_for_plot)

        # Add the polynomial regression curve to the plot
        curve_label = f'{degree}-Degree Polynomial Regression'
        plt.plot(x_for_plot, y_for_plot, color='red', label=curve_label, zorder=2, linewidth=2)

        # Add titles and labels for clarity
        plot_title = f'Scatter Plot with {degree}-Degree Polynomial Regression'
        plt.title(plot_title)
        plt.suptitle(sub_title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)  # Add a grid for better readability


# --- 1. Data Preparation (Column Renaming and Cleaning) ---
def prepare_ohlcv_data(df_ohlcv,positions):
    """
    Prepares the OHLCV DataFrame by ensuring consistent column names.
    Prioritizes 'Close' over 'Adj Close' if both exist.
    Reindex as positions df

    Args:
        df_ohlcv (pd.DataFrame): DataFrame with OHLCV data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with expected column names.
    """
    # --- ADDED CHECK HERE ---
    print(df_ohlcv)
    if not isinstance(df_ohlcv, pd.DataFrame):
        raise TypeError("Input to prepare_ohlcv_data must be a pandas DataFrame.")
    # --- END ADDED CHECK ---



    # Standardize column names
    df_ohlcv.columns = [col.replace(' ', '_').replace('.', '_') for col in df_ohlcv.columns]

    # Ensure essential columns exist and are correctly named
    # Use 'Close' as the primary closing price
    if 'Close' not in df_ohlcv.columns and 'Adj_Close' in df_ohlcv.columns:
        df_ohlcv['Close'] = df_ohlcv['Adj_Close']
    elif 'Close' in df_ohlcv.columns and 'Adj_Close' in df_ohlcv.columns:
        # If both exist, and Adj_Close has NaNs where Close doesn't, use Close.
        # Otherwise, if Adj_Close is fully populated and adjusted, it might be preferred.
        # For this analysis, we'll stick to 'Close' as it's typically what TA-Lib expects
        # and is present throughout your example.
        pass

    # Check for required columns
    required_cols = ['High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df_ohlcv.columns:
            print(f"Warning: Missing required column '{col}'. Some analysis steps might be skipped or inaccurate.")
            # If a critical column like 'Close' is missing, return None to signal an issue.
            if col == 'Close':
                return None
            # For others, fill with NaN or dummy values if absolutely necessary,
            # but it's better to have real data.
            df_ohlcv[col] = np.nan  # Add as NaN if missing to prevent errors

    # Drop rows with any NaN in critical OHLC columns (High, Low, Close)
    # Volume can be NaN if not available for some assets, but we'll handle it during analysis.
    df_ohlcv = df_ohlcv.dropna(subset=['High', 'Low', 'Close'])

    # Convert Volume to numeric, handling potential non-numeric entries
    if 'Volume' in df_ohlcv.columns:
        df_ohlcv['Volume'] = pd.to_numeric(df_ohlcv['Volume'], errors='coerce')
        # Fill NaN volumes with 0 or a small number if you want to keep rows,
        # but for volume-based analysis, NaN will naturally exclude them.
        df_ohlcv['Volume'] = df_ohlcv['Volume'].fillna(0)  # Fill NaN volumes with 0
    else:
        df_ohlcv['Volume'] = 0  # Add a volume column if completely missing

        # Reindex as positions index
        df_ohlcv = df_ohlcv.reindex(positions.index)

    return df_ohlcv




if __name__ == '__main__':
    run(settings)