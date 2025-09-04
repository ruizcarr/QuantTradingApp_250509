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

    positions=get_positions()
    print(positions)

    # Run the analysis for all tickers in the data_dict
    all_generated_signals = run_analysis_for_tickers(data_dict,positions)

    if False:
        # Run the analysis only for NQ=F
        #data_dict={'NQ=F':data_dict['NQ=F']}
        #all_generated_signals = run_analysis_for_tickers(data_dict,positions)

        # Run the NQ=F specific trading system
        portfolio_performance_df = run_nq_trading_system(data_dict['NQ=F'],positions, initial_account_size=100000)

        print(f"\n{'=' * 50}\nNQ=F Trading System Simulation Complete.\n{'=' * 50}")
        print("\nDaily Portfolio Performance DataFrame (first 10 rows):")
        print(portfolio_performance_df.head(10))
        print("\nDaily Portfolio Performance DataFrame (last 10 rows):")
        print(portfolio_performance_df.tail(10))

        # Optional: Plotting portfolio value over time
        if not portfolio_performance_df.empty:
            plt.figure(figsize=(15, 7))
            plt.plot(portfolio_performance_df.index, portfolio_performance_df['Portfolio_Value'], label='Portfolio Value', color='purple')
            plt.title('NQ=F Simulated Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            plt.tight_layout()

            plt.figure(figsize=(15, 5))
            plt.plot(portfolio_performance_df.index, portfolio_performance_df['Daily_Weight_Invested_Pct'], label='Daily Weight Invested (%)', color='teal', alpha=0.7)
            plt.title('NQ=F Daily Percentage of Portfolio Invested')
            plt.xlabel('Date')
            plt.ylabel('Weight (%)')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            plt.tight_layout()



    #print(f"\n{'=' * 50}\nAnalysis Complete for All Tickers.\n{'=' * 50}")
    # You can now further process all_generated_signals if needed
    # For example, combine all signals into a single DataFrame or analyze them collectively.
    # print("\nAll Generated Signals:")
    # for ticker, signals in all_generated_signals.items():
    #     print(f"\nSignals for {ticker}:")
    #     for signal in signals:
    #         print(signal)

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

# --- Custom Indicator Implementations (Replacing TA-Lib) ---

def calculate_sma(series, period):
    """Calculates Simple Moving Average (SMA)."""
    return series.rolling(window=period).mean()


def calculate_atr(df, period=14):
    """
    Calculates Average True Range (ATR).
    TR = max(High - Low, abs(High - Prev. Close), abs(Low - Prev. Close))
    ATR is the SMA of TR.
    """
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        print("Error: DataFrame must contain 'High', 'Low', and 'Close' columns for ATR calculation.")
        return pd.Series(np.nan, index=df.index)

    high_low = df['High'] - df['Low']
    high_prev_close = np.abs(df['High'] - df['Close'].shift(1))
    low_prev_close = np.abs(df['Low'] - df['Close'].shift(1))

    true_range = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)

    # ATR is typically an SMA of True Range
    atr = calculate_sma(true_range, period)
    return atr


def calculate_adx(df, period=14):
    """
    Calculates Average Directional Index (ADX), Plus Directional Indicator (+DI),
    and Minus Directional Indicator (-DI).
    Based on Wilder's DMI.
    """
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        print("Error: DataFrame must contain 'High', 'Low', and 'Close' columns for ADX calculation.")
        return pd.DataFrame({'ADX': np.nan, 'PLUS_DI': np.nan, 'MINUS_DI': np.nan}, index=df.index)

    # Calculate Directional Movement (DM)
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Ensure movements are only considered if they are greater than the opposing movement
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    # Calculate True Range (TR)
    high_low = df['High'] - df['Low']
    high_prev_close = np.abs(df['High'] - df['Close'].shift(1))
    low_prev_close = np.abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)

    # Smooth DM and TR using Wilder's smoothing (similar to EMA)
    # Wilder's smoothing is essentially an EMA with alpha = 1/period
    def wilder_smoothing(series, period):
        return series.ewm(alpha=1 / period, adjust=False).mean()

    smoothed_plus_dm = wilder_smoothing(plus_dm, period)
    smoothed_minus_dm = wilder_smoothing(minus_dm, period)
    smoothed_tr = wilder_smoothing(true_range, period)

    # Calculate DI
    plus_di = (smoothed_plus_dm / smoothed_tr) * 100
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100

    # Calculate DX
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    dx = dx.replace([np.inf, -np.inf], np.nan)  # Handle division by zero if DI sum is 0

    # Calculate ADX (smoothed DX)
    adx = wilder_smoothing(dx, period)

    return pd.DataFrame({'ADX': adx, 'PLUS_DI': plus_di, 'MINUS_DI': minus_di}, index=df.index)


# --- 1. Data Preparation (Column Renaming and Cleaning) ---
def prepare_ohlcv_data(df_ohlcv,positions):
    """
    Prepares the OHLCV DataFrame by ensuring consistent column names.
    Prioritizes 'Close' over 'Adj Close' if both exist.

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

    #Reindex as positions index
    df_ohlcv=df_ohlcv.reindex(positions.index)

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

    return df_ohlcv


# --- 2. Identify Sustained Uptrends ---
def identify_uptrend(df,positions, short_ma_period=50, long_ma_period=200, adx_period=14, adx_threshold=25):
    """
    Identifies periods of sustained uptrend in the DataFrame.

    A sustained uptrend is defined by:
    - The short-term Moving Average (SMA_Short) is above the long-term Moving Average (SMA_Long).
    - Both Moving Averages have an upward slope.
    - The ADX is greater than a threshold (indicating strong trend).
    - The +DI line is above the -DI line (indicating bullish direction).

    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns.
        short_ma_period (int): Period for the short Moving Average (e.g., 50 days).
        long_ma_period (int): Period for the long Moving Average (e.g., 200 days).
        adx_period (int): Period for ADX calculation (e.g., 14 days).
        adx_threshold (int): Threshold for ADX strength (e.g., 25).

    Returns:
        pd.DataFrame: The original DataFrame with additional columns for indicators
                      and a boolean column 'Sustained_Uptrend'.
    """
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        print("Error: DataFrame must contain 'High', 'Low', and 'Close' columns for uptrend identification.")
        df['Sustained_Uptrend'] = False
        return df

    # Calculate Moving Averages using custom function
    df[f'SMA_{short_ma_period}'] = calculate_sma(df['Close'], short_ma_period)
    df[f'SMA_{long_ma_period}'] = calculate_sma(df['Close'], long_ma_period)

    # Calculate the slope of Moving Averages (using diff for simplicity)
    df[f'SMA_{short_ma_period}_Slope'] = df[f'SMA_{short_ma_period}'].diff()
    df[f'SMA_{long_ma_period}_Slope'] = df[f'SMA_{long_ma_period}'].diff()

    # Calculate ADX, +DI, -DI using custom function
    if len(df) > adx_period * 2:  # ADX needs more data than just the period
        adx_data = calculate_adx(df, adx_period)
        df['ADX'] = adx_data['ADX']
        df['PLUS_DI'] = adx_data['PLUS_DI']
        df['MINUS_DI'] = adx_data['MINUS_DI']
    else:
        df['ADX'] = np.nan
        df['PLUS_DI'] = np.nan
        df['MINUS_DI'] = np.nan
        print(f"Warning: Not enough data points ({len(df)}) for ADX calculation (requires more than {adx_period} days). ADX will be NaN.")

    # Define MA uptrend conditions
    ma_uptrend_condition = (df[f'SMA_{short_ma_period}'] > df[f'SMA_{long_ma_period}']) & \
                           (df[f'SMA_{short_ma_period}_Slope'] > 0) & \
                           (df[f'SMA_{long_ma_period}_Slope'] > 0)

    # Define ADX uptrend conditions (only if ADX was calculated and not all NaNs)
    if 'ADX' in df.columns and not df['ADX'].isnull().all():
        adx_uptrend_condition = (df['ADX'] > adx_threshold) & \
                                (df['PLUS_DI'] > df['MINUS_DI'])
        # Combine all conditions for sustained uptrend
        df['Sustained_Uptrend'] = ma_uptrend_condition & adx_uptrend_condition
    else:
        df['Sustained_Uptrend'] = ma_uptrend_condition  # Fallback if ADX is not available/valid
        print("ADX conditions skipped due to insufficient data or NaN values.")

    # Drop initial rows with NaN values due to indicator calculations
    # We drop based on the longest MA period and ADX if it was calculated
    subset_for_dropna = [f'SMA_{long_ma_period}']
    if 'ADX' in df.columns and not df['ADX'].isnull().all():
        subset_for_dropna.append('ADX')

    df = df.dropna(subset=subset_for_dropna)

    print(f"Uptrend identified for the period {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}.")
    return df


# --- 3. Detect and Quantify Pullbacks ---
def detect_and_quantify_pullbacks(df, uptrend_column='Sustained_Uptrend', min_pullback_pct=0.03, max_pullback_pct=0.20, min_pullback_duration=3, pre_pullback_volume_days=10):
    """
    Detects and quantifies significant pullbacks within sustained uptrend periods.

    A pullback is defined as a price drop from a peak, followed by a trough,
    and then a resumption of the climb, all within a confirmed uptrend.
    Includes volume analysis.

    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close', 'Volume', and 'Sustained_Uptrend' columns.
        uptrend_column (str): Name of the boolean column indicating uptrend.
        min_pullback_pct (float): Minimum percentage drop from peak to consider a pullback (e.g., 0.03 for 3%).
        max_pullback_pct (float): Maximum percentage drop from peak to consider a valid pullback (e.g., 0.20 for 20%).
                                  Drops larger than this might indicate a trend change.
        min_pullback_duration (int): Minimum duration in days for a pullback.
        pre_pullback_volume_days (int): Number of days to calculate average volume before pullback.

    Returns:
        pd.DataFrame: DataFrame with details of each detected pullback.
    """
    pullbacks = []
    in_pullback = False
    pullback_start_idx = None
    pullback_peak_price = None
    pullback_peak_date = None
    current_trough_price = np.inf
    current_trough_idx = None

    # Check if 'Volume' column is all zeros or mostly NaNs, then skip volume analysis
    perform_volume_analysis = 'Volume' in df.columns and not df['Volume'].isnull().all() and (df['Volume'] != 0).any()
    if not perform_volume_analysis:
        print("Warning: Volume data is not available or is all zeros. Volume analysis for pullbacks will be skipped.")

    for i in range(1, len(df)):
        # Only look for pullbacks if we are in a sustained uptrend
        if df[uptrend_column].iloc[i]:
            current_price = df['Close'].iloc[i]
            previous_price = df['Close'].iloc[i - 1]
            current_date = df.index[i]

            if not in_pullback:
                # Identify a peak: current price is lower than previous AND previous was the highest in a small lookback window
                # Using 'High' for peak detection is more accurate than 'Close'
                if current_price < previous_price and \
                        (df['High'].iloc[i - 1] == df['High'].iloc[max(0, i - 5):i].max()):  # Peak in 5-day window
                    pullback_start_idx = i - 1
                    pullback_peak_price = df['High'].iloc[pullback_start_idx]  # Use High for peak price
                    pullback_peak_date = df.index[pullback_start_idx]
                    in_pullback = True
                    current_trough_price = pullback_peak_price  # Initialize trough with peak price
                    current_trough_idx = pullback_start_idx
            else:
                # If we are in a pullback, update the trough
                # Using 'Low' for trough detection is more accurate
                if df['Low'].iloc[i] < current_trough_price:
                    current_trough_price = df['Low'].iloc[i]
                    current_trough_idx = i

                # If price starts to rise significantly from the trough or stabilizes, pullback might have ended
                # End of pullback criteria: current price (Close) has risen from the trough
                # And we are past the actual trough day
                if current_price > df['Close'].iloc[i - 1] and i > current_trough_idx:
                    # A rebound from the trough has been detected
                    pullback_trough_price = current_trough_price
                    pullback_trough_date = df.index[current_trough_idx]

                    magnitude = ((pullback_peak_price - pullback_trough_price) / pullback_peak_price) * 100
                    duration = (pullback_trough_date - pullback_peak_date).days

                    # Calculate volume metrics if volume data is valid
                    volume_pullback = np.nan
                    pre_pullback_volume = np.nan
                    if perform_volume_analysis:
                        # Ensure date range is valid for volume slice
                        volume_pullback_slice = df['Volume'].loc[pullback_peak_date:pullback_trough_date]
                        volume_pullback = volume_pullback_slice.mean() if not volume_pullback_slice.empty else np.nan

                        pre_pullback_start_idx = max(0, pullback_start_idx - pre_pullback_volume_days)
                        pre_pullback_volume_slice = df['Volume'].iloc[pre_pullback_start_idx:pullback_start_idx]
                        pre_pullback_volume = pre_pullback_volume_slice.mean() if not pre_pullback_volume_slice.empty else np.nan

                    # Apply min and max pullback percentage criteria
                    if magnitude >= min_pullback_pct * 100 and magnitude <= max_pullback_pct * 100 and duration >= min_pullback_duration:
                        pullbacks.append({
                            'Pullback_Start_Date': pullback_peak_date,
                            'Pullback_End_Date': pullback_trough_date,
                            'Peak_Price': round(pullback_peak_price, 2),
                            'Trough_Price': round(pullback_trough_price, 2),
                            'Magnitude_Drop_Pct': round(magnitude, 2),
                            'Duration_Days': duration,
                            'Volume_Avg_Pullback': round(volume_pullback, 2) if not pd.isna(volume_pullback) else np.nan,
                            'Volume_Avg_Pre_Pullback': round(pre_pullback_volume, 2) if not pd.isna(pre_pullback_volume) else np.nan
                        })
                    in_pullback = False
                    pullback_start_idx = None
                    pullback_peak_price = None
                    pullback_peak_date = None
                    current_trough_price = np.inf
                    current_trough_idx = None
        else:  # If not in uptrend, reset pullback state
            in_pullback = False
            pullback_start_idx = None
            pullback_peak_price = None
            pullback_peak_date = None
            current_trough_price = np.inf
            current_trough_idx = None

    pullbacks_df = pd.DataFrame(pullbacks)
    if not pullbacks_df.empty:
        # Calculate volume ratio only if volume analysis was performed and pre_pullback_volume is not zero
        if perform_volume_analysis:
            pullbacks_df['Volume_Ratio_Pullback_PrePullback'] = pullbacks_df.apply(
                lambda row: row['Volume_Avg_Pullback'] / row['Volume_Avg_Pre_Pullback'] if row['Volume_Avg_Pre_Pullback'] > 0 else np.nan, axis=1
            )
        print(f"Detected {len(pullbacks_df)} pullbacks.")
    else:
        print("No significant pullbacks detected.")
    return pullbacks_df


# --- 4. Calculate Post-Pullback Returns ---
def calculate_post_pullback_returns(df, pullbacks_df, days_after_pullback=20):
    """
    Calculates the price return N days after the end of each pullback.

    Args:
        df (pd.DataFrame): Original DataFrame with price data.
        pullbacks_df (pd.DataFrame): DataFrame with detected pullbacks.
        days_after_pullback (int): Number of days to calculate the subsequent return.

    Returns:
        pd.DataFrame: pullbacks_df with additional columns 'Close_Price_Post_Pullback_N_Days'
                      and 'Post_Pullback_Return_Pct'.
    """
    if pullbacks_df.empty:
        return pullbacks_df

    returns = []
    post_pullback_prices = []

    for index, row in pullbacks_df.iterrows():
        end_date = row['Pullback_End_Date']

        # Find the actual date in the DataFrame's index that is >= end_date
        # This replaces df.index.get_loc(end_date, method='bfill') for older Pandas versions
        valid_dates_after_pullback = df.index[df.index >= end_date]

        if valid_dates_after_pullback.empty:
            returns.append(np.nan)
            post_pullback_prices.append(np.nan)
            continue

        # Get the first valid date on or after the pullback end date
        actual_pullback_end_date = valid_dates_after_pullback.min()

        # Now get the exact location of this actual date
        try:
            end_date_loc = df.index.get_loc(actual_pullback_end_date)
        except KeyError:
            # This should ideally not happen if actual_pullback_end_date came from df.index,
            # but as a safeguard.
            returns.append(np.nan)
            post_pullback_prices.append(np.nan)
            continue

        # Calculate the date N days later
        target_idx = end_date_loc + days_after_pullback
        if target_idx < len(df):
            price_at_pullback_end = row['Trough_Price']  # Use the trough price as the entry point
            price_n_days_later = df['Close'].iloc[target_idx]
            ret = ((price_n_days_later - price_at_pullback_end) / price_at_pullback_end) * 100
            returns.append(round(ret, 2))
            post_pullback_prices.append(round(price_n_days_later, 2))
        else:
            returns.append(np.nan)  # Not enough data to calculate return
            post_pullback_prices.append(np.nan)

    pullbacks_df[f'Close_Price_Post_Pullback_{days_after_pullback}_Days'] = post_pullback_prices
    pullbacks_df[f'Post_Pullback_Return_Pct_{days_after_pullback}_Days'] = returns
    print(f"Post-pullback returns calculated for {days_after_pullback} days.")
    return pullbacks_df


# --- 5. Perform Statistical Analysis ---
def perform_statistical_analysis(pullbacks_df, return_column):
    """
    Performs Pearson correlation and linear regression analysis.

    Args:
        pullbacks_df (pd.DataFrame): DataFrame with pullbacks and their returns.
        return_column (str): Name of the column containing post-pullback returns.
    """
    if pullbacks_df.empty or return_column not in pullbacks_df.columns:
        print("Insufficient data for statistical analysis or return column does not exist.")
        return

    # Clean NaN values in relevant columns for the main analysis
    analysis_df = pullbacks_df.dropna(subset=['Magnitude_Drop_Pct', 'Duration_Days', return_column])
    if analysis_df.empty:
        print("No pullbacks with complete data for statistical analysis after dropping NaNs.")
        return

    print("\n--- Statistical Analysis Results ---")

    # Correlate Magnitude of Drop vs. Subsequent Return
    print("\nPearson Correlation: Magnitude of Drop (%) vs. Post-Pullback Return (%)")
    corr_mag_ret, p_value_mag_ret = pearsonr(analysis_df['Magnitude_Drop_Pct'], analysis_df[return_column])
    print(f"  Correlation Coefficient (r): {corr_mag_ret:.4f}")
    print(f"  P-value: {p_value_mag_ret:.4f}")
    if p_value_mag_ret < 0.05:
        print("  Correlation is statistically significant (p < 0.05).")
    else:
        print("  Correlation is NOT statistically significant (p >= 0.05).")

    # Correlate Duration of Days vs. Subsequent Return
    print("\nPearson Correlation: Duration of Days vs. Post-Pullback Return (%)")
    corr_dur_ret, p_value_dur_ret = pearsonr(analysis_df['Duration_Days'], analysis_df[return_column])
    print(f"  Correlation Coefficient (r): {corr_dur_ret:.4f}")
    print(f"  P-value: {p_value_dur_ret:.4f}")
    if p_value_dur_ret < 0.05:
        print("  Correlation is statistically significant (p < 0.05).")
    else:
        print("  Correlation is NOT statistically significant (p >= 0.05).")

    # Correlate Volume Ratio vs. Subsequent Return (if volume data was available)
    if 'Volume_Ratio_Pullback_PrePullback' in pullbacks_df.columns:
        analysis_df_vol = pullbacks_df.dropna(subset=['Volume_Ratio_Pullback_PrePullback', return_column])
        if not analysis_df_vol.empty:
            print("\nPearson Correlation: Volume Ratio (Pullback/Pre-Pullback) vs. Post-Pullback Return (%)")
            corr_vol_ret, p_value_vol_ret = pearsonr(analysis_df_vol['Volume_Ratio_Pullback_PrePullback'], analysis_df_vol[return_column])
            print(f"  Correlation Coefficient (r): {corr_vol_ret:.4f}")
            print(f"  P-value: {p_value_vol_ret:.4f}")
            if p_value_vol_ret < 0.05:
                print("  Correlation is statistically significant (p < 0.05).")
            else:
                print("  Correlation is NOT statistically significant (p >= 0.05).")
        else:
            print("\nNo sufficient volume ratio data for correlation analysis.")

    # Linear Regression: Magnitude of Drop as predictor of Subsequent Return
    print("\nLinear Regression: Post-Pullback Return (%) ~ Magnitude of Drop (%)")
    X = sm.add_constant(analysis_df['Magnitude_Drop_Pct'])  # Independent variable with constant for intercept
    Y = analysis_df[return_column]  # Dependent variable
    model = sm.OLS(Y, X)  # Fit Ordinary Least Squares (OLS) model
    results = model.fit()
    print(results.summary())  # Print regression summary

    # Note: Consider non-linear models or data segmentation for more complex relationships.


# --- 6. Dynamic Position Sizing ---
# --- 6. Dynamic Position Sizing ---
def calculate_dynamic_position_size(current_price, current_atr, account_size, risk_per_trade_pct, atr_multiplier_for_stop_loss):
    """
    Calculates the position size dynamically based on volatility (ATR).

    Args:
        current_price (float): Current asset price.
        current_atr (float): Current Average True Range (ATR) value.
        account_size (float): Total trading account capital.
        risk_per_trade_pct (float): Percentage of capital to risk per trade (e.g., 0.01 for 1%).
        atr_multiplier_for_stop_loss (float): Multiplier to define stop-loss distance in ATRs.

    Returns:
        tuple: (int: number of units to buy, float: dollar risk per unit)
    """
    if current_atr <= 0:  # Avoid division by zero or invalid ATR
        # print("  Warning: ATR is zero or negative. Position size set to 0.")
        return 0, 0.0

    # Risk in dollars per unit based on ATR and stop-loss multiplier
    dollar_risk_per_unit = current_atr * atr_multiplier_for_stop_loss

    # Maximum allowed dollar risk per trade
    max_dollar_risk = account_size * risk_per_trade_pct

    # Number of units to buy
    num_units = max_dollar_risk / dollar_risk_per_unit

    # print(f"  Position Size Calc: Price={current_price:.2f}, ATR={current_atr:.2f}, Risk/Unit=${dollar_risk_per_unit:.2f}, Max Risk=${max_dollar_risk:.2f}")
    return int(num_units), dollar_risk_per_unit  # Return integer number of units and dollar risk per unit


# --- 7. Trading Signal Generation (Conceptual) ---
def generate_trading_signals(df, pullbacks_df, days_after_pullback_for_entry_confirmation=1, atr_period=14, base_risk_per_trade_pct=0.01, max_risk_scaling_factor=0.5, min_pullback_pct_for_scaling=0.03, max_pullback_pct_for_scaling=0.20):
    """
    Generates conceptual BUY and SELL signals based on the pullback strategy,
    with dynamic position sizing based on pullback magnitude.

    Args:
        df (pd.DataFrame): Original DataFrame with price data and indicators.
        pullbacks_df (pd.DataFrame): DataFrame with detected pullbacks.
        days_after_pullback_for_entry_confirmation (int): Days after pullback end to look for entry confirmation.
        atr_period (int): Period for ATR calculation.
        base_risk_per_trade_pct (float): The base percentage of capital to risk per trade.
        max_risk_scaling_factor (float): Max factor to scale risk based on pullback magnitude (e.g., 0.5 means risk can increase by 50%).
        min_pullback_pct_for_scaling (float): The minimum pullback magnitude (as a fraction) to start scaling risk.
        max_pullback_pct_for_scaling (float): The maximum pullback magnitude (as a fraction) for full risk scaling.

    Returns:
        list: List of dictionaries with generated trading signals.
    """
    signals = []

    # Calculate ATR for position management, if OHLC data is available and sufficient data points
    if all(col in df.columns for col in ['High', 'Low', 'Close']) and len(df) > atr_period:
        df['ATR'] = calculate_atr(df, atr_period)
    else:
        df['ATR'] = np.nan
        print("Warning: ATR cannot be calculated due to missing OHLC data or insufficient data points. Position sizing will be 0.")

    account_size = 100000  # Example trading capital
    atr_multiplier_for_stop_loss = 2  # Stop-loss at 2 times ATR

    for index, pullback_row in pullbacks_df.iterrows():
        pullback_end_date = pullback_row['Pullback_End_Date']

        # Find the actual date in the main DataFrame that is >= pullback_end_date
        valid_dates_after_pullback = df.index[df.index >= pullback_end_date]
        if valid_dates_after_pullback.empty:
            print(f"Warning: No valid dates found in main DataFrame on or after {pullback_end_date}. Skipping signal generation for this pullback.")
            continue
        actual_pullback_end_date = valid_dates_after_pullback.min()

        # Now get the exact location of this actual date
        try:
            pullback_end_loc = df.index.get_loc(actual_pullback_end_date)
        except KeyError:
            print(f"Warning: Exact date {actual_pullback_end_date} not found in main DataFrame index. Skipping signal generation for this pullback.")
            continue

        # --- Entry Signal ---
        # Look for bounce confirmation a few days after the pullback trough
        entry_date_loc = pullback_end_loc + days_after_pullback_for_entry_confirmation
        if entry_date_loc < len(df):
            entry_date = df.index[entry_date_loc]
            entry_price = df['Close'].iloc[entry_date_loc]
            current_atr = df['ATR'].iloc[entry_date_loc] if not pd.isna(df['ATR'].iloc[entry_date_loc]) else 0  # Use 0 if ATR is NaN

            # Dynamic Position Sizing based on Pullback Magnitude
            pullback_magnitude_pct = pullback_row['Magnitude_Drop_Pct']  # This is in percentage (e.g., 5.0 for 5%)

            # Normalize magnitude between min and max for scaling
            # Scale only if within the defined range, otherwise use base risk
            if min_pullback_pct_for_scaling * 100 <= pullback_magnitude_pct <= max_pullback_pct_for_scaling * 100:
                normalized_magnitude = (pullback_magnitude_pct - (min_pullback_pct_for_scaling * 100)) / \
                                       ((max_pullback_pct_for_scaling * 100) - (min_pullback_pct_for_scaling * 100))
                # Ensure normalized_magnitude is between 0 and 1
                normalized_magnitude = max(0, min(1, normalized_magnitude))

                # Apply scaling factor
                # Example: if normalized_magnitude is 0, factor is 1. If 1, factor is 1 + max_risk_scaling_factor
                scaling_factor = 1 + (normalized_magnitude * max_risk_scaling_factor)
                adjusted_risk_per_trade_pct = base_risk_per_trade_pct * scaling_factor
                print(f"  Pullback Magnitude: {pullback_magnitude_pct:.2f}%, Normalized: {normalized_magnitude:.2f}, Scaling Factor: {scaling_factor:.2f}")
                print(f"  Adjusted Risk Per Trade: {adjusted_risk_per_trade_pct * 100:.2f}%")
            else:
                adjusted_risk_per_trade_pct = base_risk_per_trade_pct
                print(f"  Pullback Magnitude {pullback_magnitude_pct:.2f}% outside scaling range. Using base risk {base_risk_per_trade_pct * 100:.2f}%.")

            # Simplified bounce condition: price has risen from the trough
            if entry_price > pullback_row['Trough_Price']:
                position_size = calculate_dynamic_position_size(
                    entry_price, current_atr, account_size, adjusted_risk_per_trade_pct, atr_multiplier_for_stop_loss
                )
                if position_size > 0:
                    signals.append({
                        'Type': 'BUY',
                        'Date': entry_date,
                        'Price': round(entry_price, 2),
                        'Position_Size': position_size,
                        'Pullback_Magnitude_Pct': pullback_row['Magnitude_Drop_Pct'],
                        'Pullback_Duration_Days': pullback_row['Duration_Days']
                    })
                    print(f"BUY Signal generated: Date={entry_date.strftime('%Y-%m-%d')}, Price={entry_price:.2f}, Size={position_size}")

                    # --- Exit Signal (conceptual for the trade) ---
                    # For a real simulation, we'd need a loop that follows the trade
                    # Here, we just define theoretical stop-loss and take-profit levels
                    if current_atr > 0:
                        stop_loss_price = entry_price - (current_atr * atr_multiplier_for_stop_loss)
                    else:
                        stop_loss_price = entry_price * 0.98  # Fallback fixed %
                    take_profit_price = entry_price * (1 + 0.05)  # Example: 5% fixed take-profit

                    print(f"  Theoretical Stop-Loss: {stop_loss_price:.2f}, Theoretical Take-Profit: {take_profit_price:.2f}")
        else:
            print(f"Warning: Not enough data for entry confirmation after pullback on {pullback_end_date.strftime('%Y-%m-%d')}.")

    return signals
# --- Plotting Functions ---
def plot_price_and_pullbacks(df, pullbacks_df, ticker_symbol, short_ma_period=50, long_ma_period=200):
    """
    Plots the asset's closing price, uptrend periods, and detected pullbacks.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)
    plt.plot(df.index, df[f'SMA_{short_ma_period}'], label=f'SMA {short_ma_period}', color='orange', linestyle='--')
    plt.plot(df.index, df[f'SMA_{long_ma_period}'], label=f'SMA {long_ma_period}', color='red', linestyle='--')

    # Highlight uptrend periods
    uptrend_periods = df[df['Sustained_Uptrend']].index
    if not uptrend_periods.empty:
        # Find continuous blocks of uptrend dates
        start_uptrend = None
        for i in range(len(uptrend_periods)):
            if start_uptrend is None:
                start_uptrend = uptrend_periods[i]
            if i + 1 < len(uptrend_periods) and (uptrend_periods[i + 1] - uptrend_periods[i]).days == 1:
                continue
            else:
                plt.axvspan(start_uptrend, uptrend_periods[i], color='green', alpha=0.1, label='_nolegend_')
                start_uptrend = None
        if start_uptrend is not None:  # Handle case where uptrend extends to end of data
            plt.axvspan(start_uptrend, uptrend_periods[-1], color='green', alpha=0.1, label='Sustained Uptrend')
        else:  # Add legend entry if there was at least one uptrend period
            plt.axvspan(uptrend_periods[0], uptrend_periods[0], color='green', alpha=0.1, label='Sustained Uptrend')

    # Mark pullbacks
    if not pullbacks_df.empty:
        plt.scatter(pullbacks_df['Pullback_Start_Date'], pullbacks_df['Peak_Price'],
                    marker='v', color='red', s=100, label='Pullback Peak', zorder=5)
        plt.scatter(pullbacks_df['Pullback_End_Date'], pullbacks_df['Trough_Price'],
                    marker='^', color='green', s=100, label='Pullback Trough', zorder=5)

        # Draw lines for pullbacks
        for _, row in pullbacks_df.iterrows():
            plt.plot([row['Pullback_Start_Date'], row['Pullback_End_Date']],
                     [row['Peak_Price'], row['Trough_Price']],
                     color='purple', linestyle='-', linewidth=1.5, alpha=0.7, label='_nolegend_')

    plt.title(f'{ticker_symbol} Price Chart with Uptrends and Pullbacks')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()


def plot_pullback_vs_return(pullbacks_df, return_column, ticker_symbol):
    """
    Plots a scatter plot of pullback magnitude vs. subsequent return with regression line.
    """
    if pullbacks_df.empty or return_column not in pullbacks_df.columns:
        print(f"No data for scatter plot for {ticker_symbol}.")
        return

    plot_df = pullbacks_df.dropna(subset=['Magnitude_Drop_Pct', return_column])

    if plot_df.empty:
        print(f"No complete data points for scatter plot after dropping NaNs for {ticker_symbol}.")
        return

    plt.figure(figsize=(10, 7))
    sns.regplot(x='Magnitude_Drop_Pct', y=return_column, data=plot_df, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})

    # Add correlation and p-value to title
    corr, p_value = pearsonr(plot_df['Magnitude_Drop_Pct'], plot_df[return_column])
    plt.title(f'{ticker_symbol}: Pullback Magnitude (%) vs. {return_column.replace("_Pct", "")} (%) \n (Corr: {corr:.2f}, p-value: {p_value:.3f})')
    plt.xlabel('Pullback Magnitude (%)')
    plt.ylabel(f'{return_column.replace("_Pct", "")} (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()


def plot_distributions(pullbacks_df, ticker_symbol):
    """
    Plots histograms for pullback magnitude, duration, and subsequent return.
    """
    if pullbacks_df.empty:
        print(f"No data for distributions plots for {ticker_symbol}.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{ticker_symbol}: Distribution of Pullback Metrics', fontsize=16)

    # Magnitude Drop
    sns.histplot(pullbacks_df['Magnitude_Drop_Pct'].dropna(), kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Pullback Magnitude (%)')
    axes[0].set_xlabel('Magnitude (%)')
    axes[0].set_ylabel('Frequency')

    # Duration Days
    sns.histplot(pullbacks_df['Duration_Days'].dropna(), kde=True, ax=axes[1], color='lightcoral')
    axes[1].set_title('Pullback Duration (Days)')
    axes[1].set_xlabel('Duration (Days)')
    axes[1].set_ylabel('Frequency')

    # Post-Pullback Return
    return_col = [col for col in pullbacks_df.columns if 'Post_Pullback_Return_Pct' in col]
    if return_col:
        sns.histplot(pullbacks_df[return_col[0]].dropna(), kde=True, ax=axes[2], color='lightgreen')
        axes[2].set_title(f'{return_col[0].replace("_Pct", "")} (%)')
        axes[2].set_xlabel('Return (%)')
        axes[2].set_ylabel('Frequency')
    else:
        axes[2].set_title('Post-Pullback Return (N/A)')
        axes[2].text(0.5, 0.5, 'Return column not found', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap


# --- Main Execution Block ---
def run_analysis_for_tickers(data_dict,positions, std_dev_multiplier=1.0, std_dev_period=22*12):
    """
    Runs the full pullback analysis and trading system simulation for a dictionary of tickers.

    Args:
        data_dict (dict): A dictionary where keys are ticker symbols (str)
                          and values are pandas DataFrames with OHLCV data.
        std_dev_multiplier (float): Multiplier for standard deviation to determine max_pullback_pct.
        std_dev_period (int): Period over which to calculate the standard deviation of daily returns.
    """
    all_signals = {}
    for ticker, df_ohlcv_raw in data_dict.items():
        print(f"\n{'=' * 50}\nAnalyzing Ticker: {ticker}\n{'=' * 50}")

        # Reindex as positions index
        df_ohlcv_raw = df_ohlcv_raw.reindex(positions.index)

        # 1. Prepare and clean OHLCV data
        df = prepare_ohlcv_data(df_ohlcv_raw.copy(),positions)  # Use .copy() to avoid modifying original dict dfs

        if df is None or df.empty:
            print(f"Skipping {ticker} due to data preparation issues or empty DataFrame.")
            continue

        # Calculate daily returns for standard deviation
        df['Daily_Return'] = df['Close'].pct_change() * 100  # Percentage change

        # Determine specific parameters for NQ=F or use general ones
        current_min_pullback_pct = 0.03  # Default min pullback for detection
        current_max_pullback_pct_floor = 10.0  # Default minimum for dynamic max pullback

        # Parameters for dynamic position sizing (passed to generate_trading_signals)
        base_risk_per_trade_pct = 0.01  # Default base risk
        max_risk_scaling_factor = 0.5  # Default max risk increase (e.g., 50% more than base risk)

        if ticker == 'NQ=F':
            current_min_pullback_pct = 0.025  # NQ=F specific min pullback for detection
            current_max_pullback_pct_floor = 10.0  # NQ=F specific min floor for dynamic max pullback
            base_risk_per_trade_pct = 0.01  # Can also be adjusted for NQ=F if desired
            max_risk_scaling_factor = 0.75  # Potentially higher scaling for NQ=F due to stronger correlation
            print(f"  Applying NQ=F specific parameters: min_pullback_pct={current_min_pullback_pct * 100}%, max_pullback_pct_floor={current_max_pullback_pct_floor}%, max_risk_scaling_factor={max_risk_scaling_factor * 100}%.")

        # Calculate rolling standard deviation of daily returns
        if len(df) > std_dev_period:
            rolling_std_dev = df['Daily_Return'].rolling(window=std_dev_period).std().iloc[-1]
            # Convert standard deviation (which is in percentage points) to a max pullback percentage
            dynamic_max_pullback_pct = rolling_std_dev * std_dev_multiplier
            # Ensure a reasonable minimum, e.g., 10% if std dev is very low
            dynamic_max_pullback_pct = max(dynamic_max_pullback_pct, 10) #10
            print(f"  Dynamic Max Pullback Pct for {ticker}: {dynamic_max_pullback_pct:.2f}% (based on {std_dev_period}-day std dev of returns * {std_dev_multiplier})")
        else:
            dynamic_max_pullback_pct = 20.0  # Fallback to a fixed value if not enough data for std dev
            print(f"  Not enough data for dynamic max pullback. Using default {dynamic_max_pullback_pct:.2f}%.")

        # 2. Identify uptrend
        df_with_uptrend = identify_uptrend(df.copy(),positions)

        #Use positions to identify Uptrend
        df_with_uptrend['positions'] =positions[ticker]

        # Filter for uptrend periods before detecting pullbacks
        #df_uptrend_only = df_with_uptrend[df_with_uptrend['Sustained_Uptrend']].copy()
        df_uptrend_only = df_with_uptrend[df_with_uptrend['positions']>0.005].copy()

        if not df_uptrend_only.empty:
            # 3. Detect and quantify pullbacks
            # Pass the dynamic_max_pullback_pct here
            pullbacks_df = detect_and_quantify_pullbacks(df_uptrend_only, min_pullback_pct=0.025, max_pullback_pct=dynamic_max_pullback_pct / 100, min_pullback_duration=3)
        else:
            print(f"No sustained uptrend periods found for {ticker} to detect pullbacks.")
            pullbacks_df = pd.DataFrame()  # Ensure it's an empty DataFrame

        if not pullbacks_df.empty:
            # 4. Calculate post-pullback returns
            days_for_post_pullback_return = 20  # Calculate return 20 days after pullback
            pullbacks_df = calculate_post_pullback_returns(df, pullbacks_df, days_for_post_pullback_return)

            # Display first and last 5 detected pullbacks with returns
            print(f"\n--- First 5 Pullbacks Detected for {ticker} with Post-Returns ---")
            print(pullbacks_df.head())
            print(f"\n--- Last 5 Pullbacks Detected for {ticker} with Post-Returns ---")
            print(pullbacks_df.tail())

            # 5. Perform statistical analysis
            return_col_name = f'Post_Pullback_Return_Pct_{days_for_post_pullback_return}_Days'
            perform_statistical_analysis(pullbacks_df, return_col_name)

            # --- PLOTTING ---
            print(f"\n--- Generating Plots for {ticker} ---")
            plot_price_and_pullbacks(df_with_uptrend, pullbacks_df, ticker, short_ma_period=50, long_ma_period=200)
            plot_pullback_vs_return(pullbacks_df, return_col_name, ticker)
            plot_distributions(pullbacks_df, ticker)
            # --- END PLOTTING ---

            # 6. & 7. Generate trading signals and conceptual position sizing
            #trading_signals = generate_trading_signals(df_with_uptrend, pullbacks_df)

            if False:
                trading_signals = generate_trading_signals(
                    df_with_uptrend, pullbacks_df,
                    min_pullback_pct_for_scaling=current_min_pullback_pct,
                    max_pullback_pct_for_scaling=dynamic_max_pullback_pct / 100,
                    base_risk_per_trade_pct=base_risk_per_trade_pct,
                    max_risk_scaling_factor=max_risk_scaling_factor
                )

                if trading_signals:
                    print(f"\nSummary of Generated Trading Signals for {ticker}:")
                    for signal in trading_signals:
                        print(signal)
                    all_signals[ticker] = trading_signals
                else:
                    print(f"No trading signals generated for {ticker}.")
            else:
                print(f"No pullbacks found for {ticker}. Skipping statistical analysis and signal generation.")

    return all_signals


# --- 7. Trading Signal Generation and Portfolio Simulation (Main Logic) ---
def run_nq_trading_system(nq_df_raw,positions, initial_account_size=100000):
    """
    Runs the pullback trading system specifically for NQ=F, simulating daily portfolio performance.

    Args:
        nq_df_raw (pd.DataFrame): DataFrame with OHLCV data for NQ=F.
        initial_account_size (float): Starting capital for the portfolio.

    Returns:
        pd.DataFrame: A daily indexed DataFrame with portfolio value, daily returns,
                      and percentage of portfolio invested.
    """
    print(f"\n{'=' * 50}\nRunning NQ=F Trading System Simulation\n{'=' * 50}")

    # 1. Prepare and clean OHLCV data
    df = prepare_ohlcv_data(nq_df_raw.copy(),positions)

    if df is None or df.empty:
        print(f"Skipping NQ=F due to data preparation issues or empty DataFrame.")
        return pd.DataFrame()

    # Calculate daily returns for standard deviation
    df['Daily_Return_Price'] = df['Close'].pct_change() * 100  # Percentage change of price

    # NQ=F specific parameters
    std_dev_period = 60
    std_dev_multiplier = 2.0
    min_pullback_pct_detection = 0.015  # 1.5% minimum pullback for NQ=F (for detection)
    max_pullback_pct_floor_dynamic = 7.0  # 7% minimum floor for dynamic max pullback (for detection)
    base_risk_per_trade_pct = 0.01  # 1% base risk per trade (for position sizing)
    max_risk_scaling_factor = 0.75  # Max 75% increase in risk for deepest pullbacks (for position sizing)
    atr_period = 14  # Period for ATR calculation
    atr_multiplier_for_stop_loss = 2  # Stop-loss at 2 times ATR
    days_after_pullback_for_entry_confirmation = 1  # Days after pullback end to look for entry confirmation

    print(f"  Using NQ=F specific parameters:")
    print(f"    Pullback Detection: min_pullback_pct={min_pullback_pct_detection * 100}%, max_pullback_pct_floor={max_pullback_pct_floor_dynamic}%.")
    print(f"    Position Sizing: base_risk_per_trade_pct={base_risk_per_trade_pct * 100}%, max_risk_scaling_factor={max_risk_scaling_factor * 100}%.")

    # Calculate dynamic max_pullback_pct for detection
    if len(df) > std_dev_period:
        rolling_std_dev = df['Daily_Return_Price'].rolling(window=std_dev_period).std().iloc[-1]
        dynamic_max_pullback_pct_value = rolling_std_dev * std_dev_multiplier
        dynamic_max_pullback_pct_value = max(dynamic_max_pullback_pct_value, max_pullback_pct_floor_dynamic)
        print(f"  Calculated Dynamic Max Pullback Pct for NQ=F: {dynamic_max_pullback_pct_value:.2f}%")
    else:
        dynamic_max_pullback_pct_value = 20.0  # Fallback to a fixed value if not enough data for std dev
        print(f"  Not enough data for dynamic max pullback calculation. Using default {dynamic_max_pullback_pct_value:.2f}%.")

    # 2. Identify uptrend and calculate indicators
    df_with_indicators = identify_uptrend(df.copy())
    if all(col in df_with_indicators.columns for col in ['High', 'Low', 'Close']) and len(df_with_indicators) > atr_period:
        df_with_indicators['ATR'] = calculate_atr(df_with_indicators, atr_period)
    else:
        df_with_indicators['ATR'] = np.nan
        print("Warning: ATR cannot be calculated due to missing OHLC data or insufficient data points. Position sizing will be affected.")

    # Filter for uptrend periods before detecting pullbacks
    df_uptrend_only = df_with_indicators[df_with_indicators['Sustained_Uptrend']].copy()

    # 3. Detect and quantify pullbacks
    pullbacks_df = pd.DataFrame()
    if not df_uptrend_only.empty:
        pullbacks_df = detect_and_quantify_pullbacks(
            df_uptrend_only,
            min_pullback_pct=min_pullback_pct_detection,
            max_pullback_pct=dynamic_max_pullback_pct_value / 100,  # Convert to fraction for function
            min_pullback_duration=3
        )
    else:
        print(f"No sustained uptrend periods found for NQ=F to detect pullbacks.")

    # Prepare for daily portfolio tracking
    portfolio_data = []
    current_portfolio_value = initial_account_size
    open_trade = None  # Stores information about an active trade: {'entry_price', 'units', 'stop_loss', 'take_profit'}

    # Iterate through the main DataFrame day by day for simulation
    for i in range(len(df_with_indicators)):
        current_date = df_with_indicators.index[i]
        current_close = df_with_indicators['Close'].iloc[i]
        current_high = df_with_indicators['High'].iloc[i]
        current_low = df_with_indicators['Low'].iloc[i]
        current_atr = df_with_indicators['ATR'].iloc[i] if 'ATR' in df_with_indicators.columns and not pd.isna(df_with_indicators['ATR'].iloc[i]) else 0

        daily_portfolio_return_pct = 0.0
        daily_weight_invested = 0.0  # Percentage of portfolio invested in NQ=F

        # --- Manage Open Trade ---
        if open_trade:
            # Calculate daily PnL of the open position
            # This is a simplified PnL calculation based on close-to-close
            # For more realism, consider high/low for stop/profit hits within the day
            pnl_per_unit = current_close - open_trade['entry_price']
            trade_pnl = pnl_per_unit * open_trade['units']

            # Check for stop-loss or take-profit hit within the day's range
            trade_closed_today = False
            exit_reason = None
            exit_price = None

            # Stop Loss Check (if low goes below SL)
            if current_low <= open_trade['stop_loss']:
                exit_price = open_trade['stop_loss']
                trade_pnl = (exit_price - open_trade['entry_price']) * open_trade['units']
                current_portfolio_value += trade_pnl
                daily_portfolio_return_pct = (trade_pnl / (current_portfolio_value - trade_pnl)) * 100  # Return on the capital used for trade
                exit_reason = "Stop Loss"
                trade_closed_today = True
                print(f"  Trade closed on {current_date.strftime('%Y-%m-%d')} by {exit_reason} at {exit_price:.2f}. PnL: {trade_pnl:.2f}")

            # Take Profit Check (if high goes above TP, and not already stopped out)
            elif current_high >= open_trade['take_profit']:
                exit_price = open_trade['take_profit']
                trade_pnl = (exit_price - open_trade['entry_price']) * open_trade['units']
                current_portfolio_value += trade_pnl
                daily_portfolio_return_pct = (trade_pnl / (current_portfolio_value - trade_pnl)) * 100
                exit_reason = "Take Profit"
                trade_closed_today = True
                print(f"  Trade closed on {current_date.strftime('%Y-%m-%d')} by {exit_reason} at {exit_price:.2f}. PnL: {trade_pnl:.2f}")

            else:  # Trade is still open
                # Calculate the current market value of the position
                current_position_value = open_trade['units'] * current_close
                # Calculate the percentage of the current portfolio value that the position represents
                daily_weight_invested = (current_position_value / current_portfolio_value) * 100
                # Calculate daily return from the previous day's close
                if i > 0:
                    prev_close = df_with_indicators['Close'].iloc[i - 1]
                    # Portfolio value update based on the change in position value
                    # This is a simplified update, assuming only the position contributes to daily change
                    # A more robust backtester would track cash and invested capital separately.
                    if open_trade['units'] > 0:  # Long position
                        daily_pnl_from_position = (current_close - prev_close) * open_trade['units']
                        current_portfolio_value += daily_pnl_from_position
                        daily_portfolio_return_pct = (daily_pnl_from_position / (current_portfolio_value - daily_pnl_from_position)) * 100  # Return on previous day's value

                # print(f"  Trade open on {current_date.strftime('%Y-%m-%d')}. Current PnL: {trade_pnl:.2f}. Portfolio Value: {current_portfolio_value:.2f}")

            if trade_closed_today:
                open_trade = None  # Close the trade
                daily_weight_invested = 0.0  # No longer invested

        # --- Check for New Trade Signal ---
        if not open_trade:  # Only look for new trades if no position is currently open
            # Check if current_date is a pullback end date + confirmation days
            potential_pullbacks = pullbacks_df[
                (pullbacks_df['Pullback_End_Date'] == current_date - pd.Timedelta(days=days_after_pullback_for_entry_confirmation))
            ]

            if not potential_pullbacks.empty:
                # Take the first potential pullback if multiple are found for simplicity
                pullback_row = potential_pullbacks.iloc[0]
                entry_price = current_close  # Entry at current day's close
                pullback_trough_price = pullback_row['Trough_Price']
                pullback_magnitude_pct = pullback_row['Magnitude_Drop_Pct']

                # Only enter if price has risen from the trough (confirmation)
                if entry_price > pullback_trough_price:
                    # Dynamic Position Sizing based on Pullback Magnitude
                    # Normalize magnitude between min and max for scaling
                    range_magnitude = (dynamic_max_pullback_pct_value) - (min_pullback_pct_detection * 100)
                    if range_magnitude > 0:
                        normalized_magnitude = (pullback_magnitude_pct - (min_pullback_pct_detection * 100)) / range_magnitude
                    else:
                        normalized_magnitude = 0
                    normalized_magnitude = max(0, min(1, normalized_magnitude))

                    scaling_factor = 1 + (normalized_magnitude * max_risk_scaling_factor)
                    adjusted_risk_per_trade_pct = base_risk_per_trade_pct * scaling_factor

                    units_to_buy, dollar_risk_per_unit = calculate_dynamic_position_size(
                        entry_price, current_atr, current_portfolio_value, adjusted_risk_per_trade_pct, atr_multiplier_for_stop_loss
                    )

                    if units_to_buy > 0:
                        # Define conceptual stop-loss and take-profit
                        stop_loss_price = entry_price - dollar_risk_per_unit
                        take_profit_price = entry_price * (1 + 0.05)  # Example: 5% fixed take-profit target

                        open_trade = {
                            'entry_price': entry_price,
                            'units': units_to_buy,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'pullback_magnitude_pct': pullback_magnitude_pct  # Store for logging
                        }
                        # Calculate initial position value and update portfolio value (deduct from cash or assume immediate investment)
                        cost_of_position = units_to_buy * entry_price
                        # For simplicity, we assume portfolio value reflects the market value of assets
                        # A more complex model would track cash separately.
                        # Here, we'll just track the overall portfolio value.
                        daily_weight_invested = (cost_of_position / current_portfolio_value) * 100
                        print(f"BUY Signal: {current_date.strftime('%Y-%m-%d')}, Price={entry_price:.2f}, Units={units_to_buy}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, Weight: {daily_weight_invested:.2f}%")
                        print(f"  Initial Portfolio Value: {current_portfolio_value:.2f}")

        # Record daily portfolio status
        portfolio_data.append({
            'Date': current_date,
            'Close_Price': current_close,
            'Portfolio_Value': current_portfolio_value,
            'Daily_Weight_Invested_Pct': daily_weight_invested,
            'Daily_Portfolio_Return_Pct': daily_portfolio_return_pct,  # This will be updated for trade days
            'Trade_Status': 'Open' if open_trade else 'Closed'
        })

    portfolio_df = pd.DataFrame(portfolio_data).set_index('Date')

    # Calculate overall portfolio returns and volatility
    portfolio_df['Daily_Portfolio_Return_Pct'] = portfolio_df['Portfolio_Value'].pct_change() * 100
    portfolio_df['Daily_Portfolio_Return_Pct'] = portfolio_df['Daily_Portfolio_Return_Pct'].fillna(0)  # First day return is 0

    total_return = ((portfolio_df['Portfolio_Value'].iloc[-1] - initial_account_size) / initial_account_size) * 100
    annualized_volatility = portfolio_df['Daily_Portfolio_Return_Pct'].std() * np.sqrt(252)  # Assuming 252 trading days

    print(f"\n--- NQ=F Portfolio Performance Summary ---")
    print(f"Initial Account Size: ${initial_account_size:,.2f}")
    print(f"Final Portfolio Value: ${portfolio_df['Portfolio_Value'].iloc[-1]:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Volatility (Daily Returns): {annualized_volatility:.2f}%")

    return portfolio_df

if __name__ == '__main__':
    run(settings)