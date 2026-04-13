# import libraries and functions
import numpy as np
import pandas as pd
#import pandas_ta as ta
import os.path
import pandas_market_calendars as mcal


import pytz

import ta  # Make sure you have installed the 'ta' library (pip install ta)
from ta.volatility import BollingerBands
from ta.momentum import rsi # Import the specific rsi function

from scipy import stats
from datetime import date
from datetime import datetime
from datetime import timedelta
import yfinance as yf
#import quantstats as qs
# extend pandas functionality with mettickers, etc.
#qs.extend_pandas()
#from arch import arch_model
from sklearn.metrics import r2_score
import time
import random

from Interest_Rates_Download import get_euribor_1y_daily
from utils import sigmoid

# Wider print limits
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# Silence warnings
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import logging
logger = logging.getLogger(__name__)

CACHE_PATH = Path('data/price_cache.parquet')
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)  # creates 'data/' folder if missing



class Data_Ind_Feed:
    def __init__(self,settings):

         # Get Data Instance
        self.data=Data(settings,settings['tickers'],settings['start'],settings['end'],settings['add_days'],settings['offline'])

        #Get Indicators Instance
        self.ind=Indicators(self.data,settings)

        #Get Data , Indicators Dict tuple
        self.data_ind=(self.data,self.ind.indicators_dict)

class Data:
    def __init__(self, settings, tickers=['ES=F'], start='2003-12-01',
                 end=(date.today() + timedelta(days=1)).isoformat(),
                 add_days=0, offline=False):

        self.path = "datasets/"
        self.db_file = os.path.join(self.path, 'data_bundle.csv')


        # -----------------------------
        # 1️⃣ Load or download data_bundle
        # -----------------------------
        self.settings = settings
        self.yf_data_bundle(tickers, start, end, add_days)

        self.data_bundle_yf_raw=self.data_bundle.copy()

        #print("last data_bundle date after Load",self.data_bundle.index[-1])

        # -----------------------------
        # 2️⃣ Sanitize data_bundle
        # -----------------------------
        # Remove duplicate indices
        self.data_bundle = self.data_bundle[~self.data_bundle.index.duplicated(keep='first')]

        # Remove tz if tz-aware
        if self.data_bundle.index.tz is not None:
            self.data_bundle.index = self.data_bundle.index.tz_convert(None)

        # Forward/backward fill missing data
        self.data_bundle = self.data_bundle.sort_index()
        self.data_bundle = self.data_bundle.ffill().bfill()

        # Ensure numeric
        self.data_bundle = self.data_bundle.apply(pd.to_numeric, errors='coerce')
        self.data_bundle = self.data_bundle.ffill().bfill()

        #print("last data_bundle date after Sanitize", self.data_bundle.index[-1])

        # -----------------------------
        # 3️⃣ Add next days if requested
        # -----------------------------
        if add_days > 0:
            self.add_next_days_same_value(add_days)

        #print("last data_bundle date after add_next_days", self.data_bundle.index[-1])

        # -----------------------------
        # 4️⃣ Pack into data_dict
        # -----------------------------
        self.data_dict = {}
        for tick in settings['tickers']:
            if tick in self.data_bundle.columns:
                self.data_dict[tick] = self.data_bundle[tick].copy()

        #Sanitize tz
        for k, df in self.data_dict.items():
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
            self.data_dict[k] = df

        # -----------------------------
        # 4b️⃣ Extend with historical CSVs if needed
        # -----------------------------
        self.extended_data(self.data_dict, settings['start'])


        # -----------------------------
        # 5️⃣ Create tickers_closes
        # -----------------------------
        closes = {tick: df['Close'] for tick, df in self.data_dict.items()}
        self.tickers_closes = pd.DataFrame(closes)



        # Ensure tz-naive
        #self.tickers_closes.index = pd.to_datetime(self.tickers_closes.index, utc=True).tz_convert(None)
        #

        # -----------------------------
        # 6️⃣ Add Cash (EURIBOR)
        # -----------------------------
        self.euribor_df = get_euribor_1y_daily().reindex(self.tickers_closes.index, method="ffill")
        self.cash_closes=pd.DataFrame()
        self.cash_closes['cash'] = 1000 * (1 + self.euribor_df['Euribor'] / 255).cumprod()
        if settings.get('add_cash', False):
            self.tickers_closes['cash'] = self.cash_closes.copy()

            # Add cash to data_dict
            df_cash = list(self.data_dict.values())[0].copy()
            for col in df_cash.columns:
                df_cash[col] = self.tickers_closes['cash']
            self.data_dict['cash'] = df_cash

        # -----------------------------
        # 6️⃣.1 Create tickers_opens
        # -----------------------------

        opens = {tick: df['Open'] for tick, df in self.data_dict.items()}
        self.tickers_opens = pd.DataFrame(opens)


        # -----------------------------
        # 7️⃣ Compute returns
        # -----------------------------
        self.tickers_returns = self.tickers_closes.pct_change().fillna(0)

        self.intraday_tickers_returns=(self.tickers_closes/self.tickers_opens-1).fillna(0)


        # -----------------------------
        # 8️⃣ Sanitize Open, High, Low
        # -----------------------------
        self.data_dict_sanitize_OHL()


        # -----------------------------
        # 9️⃣ Exchange rate and EUR returns
        # -----------------------------
        if "EURUSD=X" in self.tickers_closes.columns:
            exchange_rate = 1 / self.tickers_closes["EURUSD=X"].shift(1).fillna(method='bfill')
        else:
            exchange_rate = 1.0
            print("No EURUSD=X available")
        self.exchange_rate = exchange_rate

        tickers_closes_eur = self.tickers_closes.multiply(self.exchange_rate, axis='index')
        self.tickers_returns_eur = tickers_closes_eur.pct_change().fillna(0)

        # -----------------------------
        # 10 Final Sanity Check
        # -----------------------------
        self.final_sanity_check(verbose=False)


    def yf_data_bundle_ok(self, tickers, start, add_days=0, offline=False):
        tz = pytz.timezone("Europe/Madrid")
        today = datetime.now(tz).date()
        yf_end = today + timedelta(days=1)  # end exclusive

        data_bundle = yf.download(
            " ".join(tickers),
            start=start,
            end=yf_end,
            group_by="ticker",
            progress=False,
            timeout=30  # Increase from default to 30 seconds
        ).dropna()

        #print("data_bundle before change index",data_bundle.index)

        data_bundle.index = pd.to_datetime(data_bundle.index, utc=True).tz_convert(None)
        data_bundle = data_bundle.drop_duplicates()

        self.data_bundle = data_bundle
        return data_bundle

    def yf_data_bundle(self, tickers, start, add_days=0, offline=False):
        tz = pytz.timezone("Europe/Madrid")
        today = datetime.now(tz).date()
        yf_end = today + timedelta(days=1)

        if not self.settings.get('trading_app_only', False):
            # ── TRAINING mode — full historical refresh ───────────────────
            logger.info("TRAINING mode: full historical refresh from Yahoo")
            data_bundle = yf.download(
                " ".join(tickers),
                start=start,
                end=yf_end,
                group_by="ticker",
                progress=False,
                timeout=30
            ).dropna()

            data_bundle.index = pd.to_datetime(data_bundle.index, utc=True).tz_convert(None)
            data_bundle = data_bundle.drop_duplicates()
            data_bundle.to_parquet(CACHE_PATH)

        else:
            # ── TRADING mode — frozen history + append from last cached day ─
            if not CACHE_PATH.exists():
                raise FileNotFoundError("No cache found. Run Training first.")

            cached = pd.read_parquet(CACHE_PATH)
            last_cached_date = cached.index[-1]
            freeze_date = pd.Timestamp(today - timedelta(days=5))

            # Warn if cache is stale
            days_stale = (pd.Timestamp(today) - last_cached_date).days
            if days_stale > 30:
                print(f"WARNING: Cache is {days_stale} days old. Consider running Training mode to refresh full history.")

            # Fetch from last cached date — no arbitrary 5 day limit
            fresh_raw = yf.download(
                " ".join(tickers),
                start=last_cached_date,
                end=yf_end,
                group_by="ticker",
                progress=False,
                timeout=30
            ).dropna()

            fresh_raw.index = pd.to_datetime(fresh_raw.index, utc=True).tz_convert(None)
            fresh_raw = fresh_raw.drop_duplicates()

            # ── Sanity check on last 5 days only ─────────────────────────
            overlap = cached.index.intersection(fresh_raw.index)
            recent_overlap = overlap[(overlap >= freeze_date) & (overlap < pd.Timestamp(today))]
            if not recent_overlap.empty:
                diff = (cached.loc[recent_overlap] - fresh_raw.loc[recent_overlap]).abs()
                suspicious = diff[diff > 0.01].dropna(how='all')
                if not suspicious.empty:
                    print(f"WARNING: Yahoo revised recent data on: {suspicious.index.tolist()}")

            # Combine frozen history + fresh recent data
            data_bundle = pd.concat([cached, fresh_raw])
            data_bundle = data_bundle[~data_bundle.index.duplicated(keep='last')]
            data_bundle.to_parquet(CACHE_PATH)

        self.data_bundle = data_bundle
        return data_bundle


    def extended_data(self, data_dict, start):
        """
        Extend with historical data if requested.
        Only extends if 'start' is before available Yahoo data.
        """
        aka_dict = {'ES=F': '^GSPC', 'NQ=F': '^NDX', 'GC=F': 'GOLD',
                    'EURUSD=X': 'EURUSD', 'CL=F': 'OIL','BTC-USD': 'BTCUSD'}
        tickers = list(data_dict.keys())

        # Identify historical tickers
        h_tickers = []
        for tick in tickers:
            if tick in aka_dict.keys():
                h_tickers.append(aka_dict[tick])
            elif tick in aka_dict.values():
                h_tickers.append(tick)
            else:
                print(f"{tick} has no historical data file available!")

        # Load historical CSV data
        self.data_from_csv(h_tickers)
        h_data_dict = self.data_dict_csv

        # Determine earliest date from Yahoo data
        yahoo_start = max(min(df.index) for df in data_dict.values())

        # Only extend if 'start' is before Yahoo start
        if pd.to_datetime(start) >= yahoo_start:
            # No extension needed
            return

        # Determine earliest date to take from historical CSVs
        h_date_0 = max(pd.to_datetime(start), min(min(df.index) for df in h_data_dict.values()))

        e_data_dict = {}
        #e_closes = pd.DataFrame()

        for i, tick in enumerate(tickers):
            h_tick = h_tickers[i]
            h_data_tick = h_data_dict[h_tick].copy()
            data_tick = data_dict[tick].copy()

            # Slice historical data between h_date_0 and yahoo_start
            h_data_tick = h_data_tick.loc[(h_data_tick.index >= h_date_0) & (h_data_tick.index < yahoo_start)]

            # Ensure indices align
            if i == 0:
                idx_0 = h_data_tick.index
            else:
                h_data_tick = h_data_tick.reindex(idx_0).ffill()

            # Concatenate historical + Yahoo data
            e_data_tick = pd.concat([h_data_tick, data_tick])
            e_data_dict[tick] = e_data_tick

            # Build closes DataFrame
            #s = e_data_tick['Close'].rename(tick)
            #e_closes = pd.concat([e_closes, s], axis=1)

        #e_closes.index = pd.to_datetime(e_closes.index)
        self.data_dict = e_data_dict
        #self.tickers_closes = e_closes



    def repair_data(self, data, tick):

        #Add missing columns
        col=data.columns
        print(tick, col)
        yf_col=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_col=[c for c in yf_col if c not in col]
        print('missing_col',missing_col)
        data[missing_col]=np.nan


        # Check for duplicates
        duplicated = data[data.index.duplicated(keep=False)]
        print('duplicated', duplicated)
        if len(duplicated) == 0:
            data = data[~data.index.duplicated(keep='first')]

        # Check for NaN at Close
        nans_close = data[data.Close.isnull().values]
        print('nans_close', len(nans_close))
        if len(nans_close) > 1:
            # Close nan Fill Close nan with Open day after
            data.Close = data.Close.fillna(data.Open.shift(-1)).fillna(method='ffill')

        # Check for NaN at Open
        nans_open = data[data.Open.isnull().values]
        print('nans_open', len(nans_open))
        if len(nans_open) > 1:
            # Fill Open nan with Close day before
            data.Open = data.Open.fillna(data.Close.shift(1)).fillna(method='ffill')

        #Fill nan at [ 'High', 'Low', 'Adj Close'] with close
        data.High=data.High.fillna(data.Close)
        data.Low = data.Low.fillna(data.Close)
        data['Adj Close']= data['Adj Close'].fillna(data.Close)

        data.Volume=data.Volume.fillna(0)

        #Reorder columns as yf style
        data=data[yf_col]

        # Check for NaN
        nans_close = data[data.Close.isnull().values]
        nans_open = data[data.Open.isnull().values]
        print('nans_close', len(nans_close), 'nans_open', len(nans_open))

        # Save repaired data to csv
        data.to_csv(tick + '.csv')

    def data_bundle_to_tick_csv(self, data_bundle):
        """Tickers must be at  level 0
            Save individual OHLC to tick.csv file"""
        tickers = list(data_bundle.columns.levels[0])
        for tick in tickers: data_bundle[tick].to_csv(self.path+tick + '.csv')

    def data_from_csv(self, tickers):
        closes_csv = pd.DataFrame()
        data_dict_csv = {}
        for tick in tickers:
            if os.path.isfile(self.path+tick + '.csv'):
                data = pd.read_csv(self.path+tick + '.csv', index_col=0)
                # Convert the index to datetime with naive timestamps (no timestamps)
                data.index = pd.to_datetime(data.index).tz_localize(None)
                data.sort_index(inplace=True)
                data_dict_csv[tick] = data
                closes_csv[tick] = data['Close']
            else:
                print(self.path+tick + '.csv', 'do not exist !')
        closes_csv.index = pd.DatetimeIndex(closes_csv.index)
        #Reset tz
        closes_csv.index = closes_csv.index.tz_localize(None)

        self.returns_csv = closes_csv.pct_change().fillna(0)
        self.closes_csv = closes_csv
        self.data_dict_csv = data_dict_csv

    def yf_data_to_csv(self, tickers, start, end=(date.today() + timedelta(days=1)).isoformat()):
        for tick in tickers:
            yf_data = yf.download(tick, start, end, progress=False)
            yf_data.to_csv(self.path+tick + '.csv')

    def add_next_days(self,add_days):

        # Create and concat df copy of last days of data_bundle with index next business days range
        last_day=self.data_bundle.index[-1]
        next_days_range = pd.bdate_range(start=last_day, periods=add_days+1 , inclusive='right')

        next_days_data=self.data_bundle.tail(add_days).copy()
        next_days_data.index=next_days_range
        next_days_data.loc[:,] = [np.asarray(self.data_bundle.loc[last_day,])]
        self.data_bundle=pd.concat([self.data_bundle,next_days_data],axis=0)

    def add_next_days_random(self, num_future_days, seed=40):
        """
        Extends self.data_bundle with random values based on the mean and standard deviation of the last 20 calendar days,
        generating random values for the *added* business days. The future DataFrame has the same columns as self.data_bundle.

        Args:
            self: The class instance with a data_bundle attribute (Pandas DataFrame with DatetimeIndex).
            num_future_days (int, optional): The number of future business days to add. Defaults to 5.
            seed (int, optional): The seed for the random number generator. Defaults to 42.

        Raises:
            ValueError: If self.data_bundle is not a DataFrame with a DatetimeIndex or has fewer than 20 rows.
        """
        if not isinstance(self.data_bundle.index, pd.DatetimeIndex):
            raise ValueError("self.data_bundle index must be a DatetimeIndex.")

        lookback=10
        if len(self.data_bundle) < lookback:
            raise ValueError("self.data_bundle must contain at least 20 rows to calculate mean and std.")

        last_days = self.data_bundle.iloc[-lookback:]
        last_days_mean = last_days.mean()
        last_days_std = last_days.std()

        last_business_day = self.data_bundle.index[-1]
        future_dates = pd.bdate_range(start=last_business_day, periods=num_future_days + 1, inclusive='right')

        np.random.seed(seed)
        random_values = np.random.normal(last_days_mean.values, last_days_std.values, size=(num_future_days, len(self.data_bundle.columns)))

        future_df = pd.DataFrame(random_values, index=future_dates, columns=self.data_bundle.columns)

        self.data_bundle = pd.concat([self.data_bundle, future_df])

    def add_next_days_random_pct(self, num_future_days, seed=40):
        """
        Extends self.data_bundle with random percentage changes using cumprod.

        Args:
            ... (same as before)
        """
        if not isinstance(self.data_bundle.index, pd.DatetimeIndex):
            raise ValueError("self.data_bundle index must be a DatetimeIndex.")

        lookback = 20
        if len(self.data_bundle) < lookback:
            raise ValueError(f"self.data_bundle must contain at least {lookback} rows to calculate mean and std.")

        last_days = self.data_bundle.iloc[-lookback:].copy()
        last_days_pct = last_days.pct_change()
        last_days_pct = last_days_pct.iloc[1:]  # remove first row.
        last_days_pct_mean = last_days_pct.mean()
        last_days_pct_std = last_days_pct.std()

        last_business_day = self.data_bundle.index[-1]
        future_dates = pd.bdate_range(start=last_business_day, periods=num_future_days + 1, inclusive='right')

        np.random.seed(seed)
        random_values_pct = np.random.normal(last_days_pct_mean.values, last_days_pct_std.values, size=(num_future_days, len(self.data_bundle.columns)))

        last_values = self.data_bundle.iloc[-1].values

        cum_prod = (1 + random_values_pct).cumprod(axis=0)
        future_values = last_values * cum_prod

        future_df = pd.DataFrame(future_values, index=future_dates, columns=self.data_bundle.columns)

        # Replace Non Close columns with close
        future_df = future_df.apply(pd.to_numeric, errors='coerce')  # convert to numeric.

        for ticker in future_df.columns.get_level_values(0).unique():
            close_col = (ticker, 'Close')
            if close_col in future_df.columns:
                close_values = future_df[close_col].values

                # Fill any NaN values in close_values
                nan_mask = np.isnan(close_values)
                if np.any(nan_mask):
                    mean_val = np.nanmean(close_values)
                    close_values[nan_mask] = mean_val

                other_cols = [col for col in future_df.columns if col[0] == ticker and col[1] != 'Close']

                # Explicit Loop-Based Assignment
                for row_index, row in future_df.iterrows():
                    close_val = close_values[future_df.index.get_loc(row_index)]
                    for col in other_cols:
                        try:
                            future_df.loc[row_index, col] = close_val
                        except Exception as e:
                            print(f"Error assigning {close_val} to {row_index}, {col}: {e}")

        self.data_bundle = pd.concat([self.data_bundle, future_df])

    def add_next_days_same_value(self, num_future_days, exchange="XNYS"):
        if num_future_days <= 0:
            return

        if not isinstance(self.data_bundle.index, pd.DatetimeIndex):
            self.data_bundle.index = pd.to_datetime(self.data_bundle.index, errors='coerce')
            if self.data_bundle.index.isnull().any():
                raise ValueError("self.data_bundle index must be convertible to DatetimeIndex")

        if len(self.data_bundle) == 0:
            raise ValueError("self.data_bundle must contain at least 1 row.")

        # Force tz-naive for comparison
        last_ts = self.data_bundle.index[-1]
        if last_ts.tz is not None:
            last_ts = last_ts.tz_convert("UTC").tz_localize(None)

        # Get exchange calendar
        cal = mcal.get_calendar(exchange)
        schedule = cal.schedule(
            start_date=last_ts.normalize(),
            end_date=last_ts + pd.Timedelta(days=60)
        )

        # Extract next valid trading days
        valid_days = schedule.index[schedule.index > last_ts.normalize()][:num_future_days]

        # Repeat last row
        last_row = self.data_bundle.iloc[-1]
        future_df = pd.DataFrame(
            [last_row.values] * len(valid_days),
            index=valid_days,  # tz-naive like schedule.index
            columns=self.data_bundle.columns
        )

        self.data_bundle = pd.concat([self.data_bundle, future_df]).sort_index()

    def data_dict_sanitize_OHL(self,verbose=False):
        """
        Sanitize OHLC dataframes in self.data_dict.
        Instead of truncating at earliest common period, keep full range and just fix NaNs.
        """
        new_dict = {}
        for t, df in self.data_dict.items():
            df = df.copy()

            # Ensure datetime index
            df.index = pd.to_datetime(df.index, utc=False)
            df = df[~df.index.duplicated(keep='last')].sort_index()

            if verbose:
                # ⚠ Don't drop post-2003 rows just because Adj Close has NaNs.
                if "Adj Close" in df.columns:
                    if df["Adj Close"].isna().any():
                        print(f"⚠ {t}: found NaNs in Adj Close → forward/backward filling instead of dropping")
                        df["Adj Close"] = df["Adj Close"].fillna(method="ffill").fillna(method="bfill")

            # If OHLC missing, patch with Close
            for col in ["Open", "High", "Low"]:
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(df["Close"])

            new_dict[t] = df

        self.data_dict = new_dict

    def sanity_check_data(self):
        """
        Sanity check for data_dict, tickers_closes, tickers_returns, tickers_returns_eur, and exchange_rate.
        Prints duplicate index warnings and checks if all indexes are aligned.
        """
        print("\n✅ Data Sanity Check:")

        # Helper to check duplicates
        def check_duplicates(df, name):
            if df.index.duplicated().any():
                print(f"⚠️ Duplicate index found in {name}")
            else:
                print(f"✔ {name} index OK")

        # Check data_dict
        for tick, df in self.data_dict.items():
            check_duplicates(df, f"data_dict[{tick}]")

        # Check main dataframes/series
        check_duplicates(self.tickers_closes, "tickers_closes")
        check_duplicates(self.tickers_returns, "tickers_returns")
        check_duplicates(self.tickers_returns_eur, "tickers_returns_eur")
        if isinstance(self.exchange_rate, pd.Series):
            check_duplicates(self.exchange_rate, "exchange_rate")

        # Check alignment of all indexes
        all_indexes = [df.index for df in self.data_dict.values()] + [
            self.tickers_closes.index,
            self.tickers_returns.index,
            self.tickers_returns_eur.index,
            self.exchange_rate.index if isinstance(self.exchange_rate, pd.Series) else pd.Index([])
        ]

        if all(all_indexes[0].equals(idx) for idx in all_indexes[1:]):
            print("✔ All indexes are aligned")
        else:
            print("⚠️ Index mismatch detected between data_dict, tickers_returns, tickers_returns_eur, exchange_rate")

    def align_all_indexes(self):
        """
        Aligns all DataFrames/Series to the same index (intersection of all indexes)
        and removes duplicates.
        """
        print("\n🔧 Aligning all data indexes...")

        # Remove duplicates first
        for tick, df in self.data_dict.items():
            df = df[~df.index.duplicated(keep='first')]
            self.data_dict[tick] = df

        # Determine common index (intersection of all)
        common_index = self.tickers_closes.index
        for df in self.data_dict.values():
            common_index = common_index.intersection(df.index)
        common_index = common_index.intersection(self.tickers_returns.index)
        common_index = common_index.intersection(self.tickers_returns_eur.index)
        if isinstance(self.exchange_rate, pd.Series):
            common_index = common_index.intersection(self.exchange_rate.index)

        # Reindex everything to common_index
        for tick, df in self.data_dict.items():
            self.data_dict[tick] = df.reindex(common_index)

        self.tickers_closes = self.tickers_closes.reindex(common_index)
        self.tickers_returns = self.tickers_returns.reindex(common_index)
        self.tickers_returns_eur = self.tickers_returns_eur.reindex(common_index)
        if isinstance(self.exchange_rate, pd.Series):
            self.exchange_rate = self.exchange_rate.reindex(common_index)

        print(f"✔ All data aligned to {len(common_index)} rows")

    def final_sanity_check(self, verbose=True):
        """
        Sanitize and align all internal data:
        - Remove duplicate dates
        - Convert all indexes to tz-naive
        - Remove invalid dates (NaT)
        - Sort indexes
        - Forward/backward fill missing values
        - Align all DataFrames/Series to a common index
        """
        common_index = None
        for tick, df in self.data_dict.items():
            df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
            df.index = df.index.tz_convert(None)
            df = df[~df.index.duplicated(keep='first')]
            df = df[~df.index.isna()]
            df = df.sort_index()
            df = df.apply(pd.to_numeric, errors='coerce').ffill().bfill()
            self.data_dict[tick] = df

            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        common_index = common_index[~common_index.isna()]

        # Tickers closes
        self.tickers_closes.index = pd.to_datetime(self.tickers_closes.index, errors='coerce', utc=True).tz_convert(None)
        self.tickers_closes = self.tickers_closes[~self.tickers_closes.index.duplicated(keep='first')]
        self.tickers_closes = self.tickers_closes.loc[self.tickers_closes.index.notna()]
        self.tickers_closes = self.tickers_closes.loc[self.tickers_closes.index.intersection(common_index)]

        # Returns
        self.tickers_returns = self.tickers_closes.pct_change().fillna(0)
        self.tickers_returns = self.tickers_returns[~self.tickers_returns.index.duplicated(keep='first')]

        # EUR returns
        tickers_closes_eur = self.tickers_closes.multiply(self.exchange_rate, axis='index')
        self.tickers_returns_eur = tickers_closes_eur.pct_change().fillna(0)
        self.tickers_returns_eur = self.tickers_returns_eur[~self.tickers_returns_eur.index.duplicated(keep='first')]

        # Exchange rate
        if isinstance(self.exchange_rate, pd.Series):
            self.exchange_rate.index = pd.to_datetime(self.exchange_rate.index, errors='coerce', utc=True).tz_convert(None)
            self.exchange_rate = self.exchange_rate[~self.exchange_rate.index.duplicated(keep='first')]
            self.exchange_rate = self.exchange_rate.loc[self.exchange_rate.index.notna()]
            self.exchange_rate = self.exchange_rate.loc[self.exchange_rate.index.intersection(self.tickers_closes.index)]
        elif isinstance(self.exchange_rate, pd.DataFrame):
            self.exchange_rate.index = pd.to_datetime(self.exchange_rate.index, errors='coerce', utc=True).tz_convert(None)
            self.exchange_rate = self.exchange_rate[~self.exchange_rate.index.duplicated(keep='first')]
            self.exchange_rate = self.exchange_rate.loc[self.exchange_rate.index.notna()]
            self.exchange_rate = self.exchange_rate.loc[self.exchange_rate.index.intersection(self.tickers_closes.index)]

        if verbose:
            print("🔧 Running final data sanity check...")
            for tick, df in self.data_dict.items():
                print(f"✔ data_dict[{tick}] index OK, {len(df)} rows")
            print(f"✔ tickers_closes index OK, {len(self.tickers_closes)} rows")
            print(f"✔ tickers_returns index OK, {len(self.tickers_returns)} rows")
            print(f"✔ tickers_returns_eur index OK, {len(self.tickers_returns_eur)} rows")
            print(f"✔ exchange_rate index OK, {len(self.exchange_rate)} rows")
            print(f"✔ All data aligned to {len(self.tickers_closes)} rows")
            print("✅ Final data sanity check passed.")


def replace_fut_by_cash_returns_at_q_exp_or_after_dates(tickers_returns, fut_cash_tickers_dict,calendar,offline=False):

    path= "datasets/" # "datasets\\"
    # Get Expiration Dates
    exp_or_dayafter_dates = calendar.loc[calendar['is_expire'] | (calendar['days_to_exp'] == '1')].dropna().index
    end_is_before_expire=tickers_returns.index[-1]<=exp_or_dayafter_dates[-2]

    # Replace returns values of futures at expiration_dates by cash returns where available
    # Get Cash Returns
    cash_tickers = list(fut_cash_tickers_dict.values())

    if not offline & end_is_before_expire:
        cash_data_bundle=yf.download(cash_tickers, progress=False)
        cash_data_bundle.index=cash_data_bundle.index.tz_localize(None)
        cash_returns = cash_data_bundle['Close'].pct_change().dropna()

        #Save data to csv for further use
        for ticker in cash_tickers:
            cash_data_ticker=cash_data_bundle.xs(ticker, axis=1, level=1)
            cash_data_ticker.to_csv(path+ticker+'.csv')

    else:
        cash_returns=pd.DataFrame()
        for ticker in cash_tickers:
            if os.path.isfile(path+ticker+'.csv'):
                data = pd.read_csv(path+ticker+'.csv', index_col=0)
                data.index = pd.to_datetime(data.index)
            else:
                print(path+ticker + '.csv file not available')

            cash_returns[ticker]=data['Close'].pct_change()

        cash_returns=cash_returns.dropna()

        if end_is_before_expire:
            print('Offline Warning. Data at expiration not updated')


    # Replace returns values of Future at expiration_dates by Cash return
    start=max(cash_returns.index[0],tickers_returns.index[0])
    end=min(cash_returns.index[-1],tickers_returns.index[-1])

    for fut, cash in fut_cash_tickers_dict.items():
        tickers_returns[fut][start:end].loc[tickers_returns[start:end].index.isin(exp_or_dayafter_dates)] = cash_returns[cash].loc[cash_returns.index.isin(exp_or_dayafter_dates)].copy()

    return tickers_returns


class Indicators:

    def __init__(self,data,settings):

        tickers_returns = data.tickers_returns
        cash_returns=data.cash_closes.pct_change().fillna(0)
        cum_ret = (1 + tickers_returns).cumprod()

        #Get RSI  for further use
        self.rsi = self.get_rsi(closes=tickers_returns,len=settings['rsi_w'],returns=True)

        #Get RSI Weights
        self.rsi_reverse_keep_weights=self.rsi_reverse_keep(self.rsi, upp=settings['rsi_upp'],window=settings['rsi_window'])

        # Get Data Normalized Weights
        data_norm=self.get_data_norm(cum_ret, window=252*2,min_periods=252)
        r_opt_fun = self.get_rolling_opt_fun(tickers_returns, 22) #22
        self.norm_sharpe_weights = self.get_corr_idx(data_norm, r_opt_fun, settings,fw=22, center=1.2, width=2.0) #center=1.2, width=2.0)

        #Euribor Indicator Weights
        from EuriborCorrStudy import get_Euribor_ind
        # Retrieve Training model and get Euribor Ind
        Euribor_series = cash_returns['cash'] * 255 * 100
        self.Euribor_ind = get_Euribor_ind(Euribor_series)

        # Combined Weights
        self.comb_weights =   self.rsi_reverse_keep_weights *self.Euribor_ind *self.norm_sharpe_weights

        if 'cash' in tickers_returns.columns:
            self.comb_weights =self.comb_weights.copy()

        # Softed Factor
        raw_weight_pct =settings['raw_weight_pct']
        self.comb_weights = raw_weight_pct + (1 - raw_weight_pct) * self.comb_weights

        #Exceptions by Asset
        if 'cash' in self.comb_weights.columns:
            self.comb_weights['cash'] = self.Euribor_ind['cash']

        tickers_exceptions=['EURUSD=X'] #,'BTC-USD'
        for ticker in tickers_exceptions:
            if ticker in self.comb_weights.columns:
                self.comb_weights[ticker] = 1

        #Final Clip
        self.comb_weights =self.comb_weights.clip(upper=2.5,lower=0).fillna(1)

        #Store indicators in a dict
        self.indicators_dict={
            'Euribor_ind': self.Euribor_ind,
            'norm_sharpe_weights': self.norm_sharpe_weights,
            'rsi_reverse_keep_weights': self.rsi_reverse_keep_weights,
            'comb_weights': self.comb_weights,
        }



    def get_rsi(self,closes: pd.DataFrame, len: int = 14, returns: bool = False) -> pd.DataFrame:
        """
        Calculates the Relative Strength Index (RSI) for multiple tickers
        using the 'ta' library.

        Args:
            closes: A pandas DataFrame where each column represents the closing
                    prices or returns for a ticker. The index should be dates/times.
            len: The lookback period for the RSI calculation.
            returns: If True, assumes input 'closes' are returns and calculates
                     cumulative product to get a price series before calculating RSI.

        Returns:
            A pandas DataFrame with the RSI values for each ticker,
            with columns for the input tickers.
        """
        # Handle the returns logic as in your original function
        if returns:
            # Calculate cumulative product assuming 'closes' are returns (e.g., daily % change + 1)
            # Note: .cumprod(axis=0) ensures product is taken down columns
            closes = (1 + closes).cumprod(axis=0)

        # Initialize an empty DataFrame to store the RSI results
        rsi_df = pd.DataFrame(index=closes.index, columns=closes.columns)

        # Calculate RSI for each ticker column
        for tick in closes.columns:
            # Use the ta.momentum.rsi function (note parameter name 'window' instead of 'length')
            rsi_df[tick] = rsi(close=closes[tick], window=len)

        return rsi_df

    def rsi_reverse_keep(self,rsi, upp=70,  window=22,width=.5): #_sigmoid rsi_not_low# upp_v=0.25, low_v=1.25, k=1.50,

        # Set rsi position with sigmoid function
        rsi_high =1 - sigmoid(x=rsi, center=upp,width=width)
        self.rsi_high = pd.DataFrame(rsi_high, columns=rsi.columns, index=rsi.index)

        # Keep high / low position for a while=window
        self.rsi_high_keep = self.rsi_high.rolling(window).max().fillna(0)

        # Set value around 1
        mid=1.33
        amp= 0.66
        self.rsi_weights = (mid - amp*self.rsi_high_keep).clip(lower=0)

        # Shift for yesterday close data
        return self.rsi_weights.shift(1)


    def get_data_norm(self,data, window=252*8,min_periods=252):
        mid = data.rolling(window,min_periods=min_periods).mean()
        std = data.rolling(window,min_periods=min_periods).std()
        data_norm = (data - mid) / std
        return data_norm

    def get_rolling_opt_fun(self,ret, fw):
        # Rolling CAGR
        r_cagr = ret.rolling(fw).mean() * 252
        # Rolling VolatilityAnnualized
        r_volat = ret.rolling(fw).std() * (252) ** 0.5
        # Sharpe=CAGR/Volat
        r_opt_fun = r_cagr / r_volat
        # Optimize Function =  Monthly CAGR - Monthly average volatility
        # opt_fun=r_cagr-r_volat

        return r_opt_fun

    def get_corr_idx(self,indicator, r_opt_fun, settings,fw=22, center=1.0, width=2.0):
        """Correlation opt_fun vs indicator fw days ago"""
        corr = r_opt_fun.rolling(252).corr(indicator.shift(fw))

        #Keep only Significative Correlation over critical value
        corr_citical = 0.25 #Bellow 0.30 means negligible correlation
        corr_signif = np.where(np.abs(corr) > corr_citical, corr, 0)
        corr_signif = pd.DataFrame(corr_signif, columns=corr.columns, index=corr.index)

        # Raw Correlation Index
        corr_idx_raw = corr_signif * indicator

        # idx cut upper, lower parameters
        nd = 1.5
        std = corr_idx_raw.rolling(252*8,min_periods=252).std()
        mean = corr_idx_raw.rolling(252*8,min_periods=252).mean()

        u = mean + nd * std
        l = mean - nd * std

        # idx oxcilation params: center, width
        width = min(center * 2, width)  # to avoid negative values

        # corr_idx adjustement
        # Make idx oscilate from zero to width
        corr_idx = (corr_idx_raw.clip(upper=u, lower=l, axis=1) - l) / (u - l) * width
        # Raise mean to the center value
        rolling_mean = corr_idx.rolling(252 * 8, min_periods=252).mean()
        corr_idx = corr_idx + (center - rolling_mean)

        # Resample with value of friday
        if settings['weekly_trading_only']:
            corr_idx = corr_idx.resample('W-FRI').mean().reindex(corr_idx.index).fillna(method='ffill')

        # Use yesterday close value
        corr_idx = corr_idx.shift(1)

        return corr_idx



# Expiration Functions

def get_es_trading_calendar(returns,expiration_freq = 'Q'):
    """
    https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.contractSpecs.html
    MICRO E-MINI S&P 500 INDEX FUTURES, Also Valid fo Nasdaq NQ=F
    Quarterly contracts (Mar, Jun, Sep, Dec): q_months = [3, 6, 9, 12]
    TERMINATION OF TRADING: Trading terminates at 9:30 a.m. ET on the 3rd Friday of the contract month.

    :param returns:
    :return: calendar
    """

    #Create Calendar Dates to update
    start=returns.index[0]
    end=returns.index[-1]

    #Create a Calendar with all calendar data from Start to end of available tickers_returns
    calendar=pd.DataFrame(index=pd.date_range(start,end),columns=['is_expire']).fillna(False)

    #Add date of last data available
    calendar['date_of_last_data']=returns.index.to_series().reindex(calendar.index).fillna(method='ffill')

    #Add Month Expiration third friday of month
    third_friday_index=pd.date_range(start=start, end=end, freq='WOM-3FRI')
    #calendar['is_third_friday'].loc[third_friday_index]=True
    calendar['is_third_friday']=calendar.index.isin(third_friday_index)

    #Add quarterly Expiration
    # Define quarterly months
    q_months = [3, 6, 9, 12]
    is_contract_month = calendar.index.month.isin(q_months)
    calendar['is_q_third_friday']= calendar['is_third_friday'] & is_contract_month

    #Bollean for Expiration as Calendar dates as per selected expiration frequencey
    # Monhtly: 'M' or Quarterly: 'Q'
    if expiration_freq=='Q':
        is_calendar_exp=calendar['is_q_third_friday']
    elif expiration_freq=='M':
        is_calendar_exp = calendar['is_third_friday']

    #Add Quarterly Expiration Dates with last date of available data
    calendar['exp_date']=calendar['date_of_last_data'].loc[is_calendar_exp]

    #Add is effective/real expiration date (previous date in case of market close)
    calendar['is_expire'].loc[calendar['exp_date'].dropna()] = True

    #Drop Rows not in returns df
    calendar=calendar.reindex(returns.index)

    # Keep only 'is_expire'
    calendar = calendar[['is_expire']]  # 'day_exp_q',

    # Add days to Expiration
    calendar['days_to_exp'] = np.nan
    i_range = [-4, -3, -2, -1, 0, 1, 2]
    #i_range = [ -3, -2  ]
    for i in i_range:
        is_quarter_expire_i = calendar['is_expire'].shift(i).fillna(False)
        index_quarter_expire_i = is_quarter_expire_i.loc[is_quarter_expire_i].index
        calendar['days_to_exp'].loc[index_quarter_expire_i] = str(i)

    return calendar

def get_gc_trading_calendar(returns):
    """
    https://www.cmegroup.com/markets/metals/precious/e-micro-gold.contractSpecs.html
    Micro Gold Futures and Options
    TERMINATION OF TRADING:Trading terminates on the third last business day of the contract month.
    contract_months = [2, 4, 6, 8, 10, 12]
    :param tickers_returns:
    :return:
    df with expiration dates and more
    """

    #Create Calendar Dates
    start=returns.index[0]
    end=returns.index[-1]

    #Create a Calendar with all calendar data from Start to end of available returns
    calendar=pd.DataFrame(index=pd.date_range(start,end))

    #Get Calendar End of Month
    calendar['is_cal_end_of_month']=calendar.index.isin(calendar.resample('M').last().index)

    #Add contract months
    contract_months = [2, 4, 6, 8, 10, 12]
    is_contract_month = calendar.index.month.isin(contract_months)
    calendar['is_cal_end_of_contract_month'] = (calendar['is_cal_end_of_month'] & is_contract_month)

    # Add date of last data available in returns df
    calendar['date_of_last_data'] = returns.index.to_series().reindex(calendar.index).fillna(method='ffill')

    #Add End of Month Contract in returns index
    ret_end_of_contract_month =calendar['date_of_last_data'][calendar['is_cal_end_of_contract_month']]

    #Get calendar index End of Month Contract in returns index
    calendar['is_ret_end_of_contract_month']=calendar.index.isin(ret_end_of_contract_month)

    #Drop Rows not in returns df
    calendar=calendar.reindex(returns.index)

    #Get Third previous available day
    calendar['is_expire'] = calendar['is_ret_end_of_contract_month'].shift(-2)

    #Drop auxiliar columns
    calendar=calendar[['is_expire']]

    # Add days to Expiration
    calendar['days_to_exp'] = np.nan
    i_range = [-4, -3, -2, -1, 0, 1, 2]
    for i in i_range:
        is_quarter_expire_i = calendar['is_expire'].shift(i).fillna(False)
        index_quarter_expire_i = is_quarter_expire_i.loc[is_quarter_expire_i].index
        calendar['days_to_exp'].loc[index_quarter_expire_i] = str(i)


    return calendar


def get_yearly_dict_rolling_bull_prob_around_expiration_dates(tickers_returns,calendar,tickers):

    # Returns around expiration dates
    dates_around_expiration_bool = ~calendar['days_to_exp'].isna()
    ret_around_expiration = tickers_returns.loc[dates_around_expiration_bool]
    #Add days to expiration for further use
    ret_around_expiration['days_to_exp']=calendar['days_to_exp']
    # Returns out of around expiration dates
    ret_out_dates_around_exp = tickers_returns[~dates_around_expiration_bool]

    #Create a dict with bull_prob_dict for all tickers and year end as key

    #Ceate Years intervals for rolling
    years = list(ret_around_expiration.index.year.unique())
    y=10 #10
    ends=[str(year) for year in years[y:]]
    starts=[str(year) for year in years[:-y]]
    ends_0=[str(year) for year in years[2:y]]
    starts_0=[str(years[0]) for i in range(len(ends_0))]
    starts=starts_0+starts
    ends=ends_0+ends

    #Rolling loop
    bull_prob_dict={}
    for start, end in zip(starts,ends):
        slice_in=ret_around_expiration.loc[start:end]
        slice_out = ret_out_dates_around_exp.loc[start:end]
        slice_bull_prob_ret_around_expiration_by_day_exp = get_bull_prob_around_expiration_dates(slice_in, slice_out)
        bull_prob_dict[end]=slice_bull_prob_ret_around_expiration_by_day_exp

    #Extract rolling_bull_prob_weight_dict for each selected ticker trough the years from bull_prob_dict for all tickers with years as keys
    rolling_bull_prob_weight_dict={}


    for ticker in tickers:

        ticker_rolling = pd.DataFrame()

        for year,bull_prob_df in bull_prob_dict.items():
            ticker_rolling[year] = bull_prob_df[ticker]

        #p_value df for the ticker
        ticker_p_value= ticker_rolling.T

        # Compute Times Series Consistency-> easy to predict

        # Create df for weight
        ticker_rolling_bull_prob_weight = pd.DataFrame(index=ticker_p_value.index, columns=ticker_p_value.columns)

        for m in ticker_p_value.columns:
            ts=ticker_p_value[m].dropna()#.shift(1) #Avoid last year to keep out of sample
            #Get Time Series Consistency
            ts_is_consistent, r2_value, std, model = time_series_consistency(ts, std_lim=0.10, r2_lim=0.65)

            if not ts_is_consistent:
                ticker_rolling_bull_prob_weight[m]=1

            else:

                # Create Weight from predicted p_value

                #Predicted p_value
                predicted_p_value=model(range(len(ticker_p_value)))

                if True:

                    # Continous weight
                    ticker_rolling_bull_prob_weight[m] = (predicted_p_value + 1) / 2 + 0.5

                else:

                    ##Discrete cut of p_value (weight is 1 when p_value > p_value_lim else 0)
                    p_value_lim = 0.80  # 0.85
                    ticker_rolling_bull_prob_weight[m]  = np.where(predicted_p_value > p_value_lim, 1.5,
                                                np.where(predicted_p_value < -p_value_lim, 0.0001, 1))  # 0.0001 Avoid zero for further calc

        #Clip Weights beetween 0.5-1.5
        ticker_rolling_bull_prob_weight=ticker_rolling_bull_prob_weight.clip(upper=1.5,lower=0.5)

        #print(ticker, ' ', ' ticker_rolling_bull_prob_weight\n', ticker_rolling_bull_prob_weight)

        #Plot p-value & Weight
        #ticker_p_value.plot(title=ticker + '  p_value')
        #ticker_rolling_bull_prob_weight.plot(title=ticker + '  weight')

        #Save weight to dict
        rolling_bull_prob_weight_dict[ticker]=ticker_rolling_bull_prob_weight

    return rolling_bull_prob_weight_dict

def time_series_consistency(ts, std_lim=0.07, r2_lim=0.85): #
    # R2 from Regresion
    n_dimension = 3  # 1 for linear regresion, 2 for quadratic regresions,...
    y_observed = ts
    x_observed = range(len(y_observed))
    model = np.poly1d(np.polyfit(x_observed, y_observed, n_dimension))
    y_model = model(x_observed)
    r2_value = r2_score(y_observed, y_model)

    # Standard Deviation
    std = ts.pct_change().fillna(0).std().item()

    # Values with Low std bellow std_lim
    low_std = std < std_lim

    # Values with high R2 over r2_lim
    high_r2 = r2_value > r2_lim

    # Time-series is consistent when Std is Low or R2 is High
    ts_is_consistent = (high_r2 | low_std)

    #Plot
    if False & ts_is_consistent:
        plot_df = pd.DataFrame(index=ts.index)
        plot_df['y_observed'] = ts
        plot_df['y_model'] = model(range(len(ts)))
        plot_df.plot(title=' R2=' + str(r2_value)[:4] + ' std=' + str(std)[:4] +
                           'ts_is_consistent = ' + str(ts_is_consistent))

    return ts_is_consistent, r2_value, std, model




def get_np_cov_matrices(tickers_returns,len):

    # Calculate the rolling covariance matrix
    cov_matrices_df = tickers_returns.rolling(window=len).cov()

    #Get numpy array
    return get_np_cov_matrices_from_df(cov_matrices_df)

def get_np_cov_matrices_from_df(cov_matrices_df):
    #Get numpy array
    np_cov_matrices=np.array(cov_matrices_df)
    l,n=np.shape(np_cov_matrices)

    #Reshape as Matrix (n_days,n_assets,n_assets)
    np_cov_matrices = np_cov_matrices.reshape(int(l / n), n , n)

    return np_cov_matrices



def get_garch_var(returns):
    returns=returns.fillna(0.0001)
    garch_var=pd.DataFrame(index=returns.index)
    np.random.seed(10)
    for ticker in returns.columns:
        ret=returns[ticker]
        model = arch_model(ret, vol='ARCH', p=1)
        #model = arch_model(ret)  # GARCH (with a Constant Mean)
        model_fit = model.fit(disp=False)
        forecast= model_fit.forecast(start=0)
        garch_var[ticker]=forecast.variance*252

    return garch_var

def get_np_cov_matrices_replaced_diagonal_with_garch_var(tickers_returns, np_cov_matrices):
    garch_var = get_garch_var(tickers_returns)
    m, n, _ = np_cov_matrices.shape
    np_cov_matrices[np.arange(m)[:, None], np.arange(n), np.arange(n)] = np.array(garch_var)

    return np_cov_matrices

def get_cash_values(df_index,rate,cash_init=1000):
    cash_rate_series = pd.Series(index=df_index, dtype='float64')
    cash_rate_series.iloc[:] = rate
    cash_values = cash_init * (1 + cash_rate_series / 255).cumprod()

    return cash_values

def add_cash_to_data_bundle(data_bundle, cash_rate):

    cash_values = get_cash_values(data_bundle.index,cash_rate,cash_init=1000)

    # Get the existing sub-column names
    sub_columns = data_bundle.columns.get_level_values(1).unique()

    # Assign the cash values to the new columns
    for sub_column in sub_columns:
        data_bundle[('cash', sub_column)] = cash_values.values  # Use .values to avoid alignment issues

    data_bundle_with_cash = data_bundle.copy()

    return data_bundle_with_cash


def get_pure_event_returns(tickers_returns, sigma_multiplier=2.5, rolling_years=4):
    """
    Algoritmo de Salto Vectorizado con Umbrales Dinámicos (Rolling Std).
    """
    # 1. Limpieza y Preparación
    clean_rets = tickers_returns.clip(lower=-0.5, upper=0.5)
    relevant_cols = [c for c in clean_rets.columns if 'cash' not in c.lower()]

    # Indices para NumPy
    rel_indices = [clean_rets.columns.get_loc(c) for c in relevant_cols]
    data_values = clean_rets.values
    dates = clean_rets.index

    # --- 2. CÁLCULO DE UMBRALES DINÁMICOS (Rolling) ---
    # Convertimos años a días de trading (aprox 252 por año)
    window_size = int(252 * rolling_years)

    # Pre-calculamos la std móvil para todo el histórico de una vez (Vectorizado por Pandas)
    # min_periods=30 asegura que empiece a funcionar al mes, expandiéndose hasta llegar a 4 años
    rolling_std = clean_rets.iloc[:, rel_indices].rolling(window=window_size, min_periods=30).std()

    # Rellenamos los primeros 30 días con la primera volatilidad disponible para no tener NaNs
    rolling_std = rolling_std.bfill().fillna(0.01)

    # Matriz de Umbrales Logarítmicos [N_Rows, N_Assets]
    # Cada día tiene su propio umbral basado en los 4 años anteriores
    dynamic_thresholds = (rolling_std * sigma_multiplier).values
    log_dynamic_thresholds = np.log1p(dynamic_thresholds)

    # 3. Pre-cálculo Global de Precios (Log-Space)
    log_data = np.log1p(data_values)
    global_cum_log = np.cumsum(log_data, axis=0)

    # Variables de estado
    n_rows = len(dates)
    current_idx = 0
    event_returns = []
    event_dates_idx = []

    # 4. Bucle de Saltos con Umbral Adaptativo
    while current_idx < n_rows - 1:
        # a. Ancla actual
        current_anchor = global_cum_log[current_idx, rel_indices]

        # b. Curva futura relativa
        future_curve = global_cum_log[current_idx + 1:, rel_indices] - current_anchor

        # c. UMBRAL ADAPTATIVO
        # Usamos el umbral calculado en la fecha del 'current_idx' (conocido al momento de la decisión)
        # Broadcasting: future_curve (N, Assets) vs current_thresh (Assets)
        current_thresh_vector = log_dynamic_thresholds[current_idx]

        # d. Detección Vectorizada
        breach_mask = np.abs(future_curve) >= current_thresh_vector
        days_triggers = breach_mask.any(axis=1)

        if not days_triggers.any():
            break

        # e. Salto al siguiente evento
        next_jump = np.argmax(days_triggers)
        real_event_idx = current_idx + 1 + next_jump

        # f. Calcular y Guardar
        pct_change = np.expm1(global_cum_log[real_event_idx] - global_cum_log[current_idx])
        event_returns.append(pct_change)
        event_dates_idx.append(real_event_idx)

        # g. Actualizar puntero
        current_idx = real_event_idx

    # 5. Resultado
    if not event_returns:
        return pd.DataFrame()

    df_events = pd.DataFrame(event_returns, index=dates[event_dates_idx], columns=clean_rets.columns)
    return df_events