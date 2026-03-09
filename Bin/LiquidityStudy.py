API_KEY = 'c61c263839a87c7e828a53d80de05215'

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime


class PredictiveLiquidityLab:
    def __init__(self, start="2015-01-01"):
        self.start = start
        self.fred_tickers = {'WALCL': 'Fed', 'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP'}
        self.mkt_tickers = {'^GSPC': 'SP500', 'GC=F': 'Gold'}

    def get_data(self):
        print("--- Step 1: Data Acquisition ---")
        try:
            # 1. Fetch FRED Balance Sheet Data
            liq = web.DataReader(list(self.fred_tickers.keys()), 'fred', self.start)
            liq.rename(columns=self.fred_tickers, inplace=True)

            # 2. Fetch Prices (Robust for yfinance multi-index)
            prices = yf.download(list(self.mkt_tickers.keys()), start=self.start, progress=False)
            if isinstance(prices.columns, pd.MultiIndex):
                price_df = prices['Adj Close'] if 'Adj Close' in prices.columns.levels[0] else prices['Close']
            else:
                price_df = prices['Adj Close'] if 'Adj Close' in prices.columns else prices['Close']

            price_df = price_df.rename(columns=self.mkt_tickers)

            # 3. Combine & Calculate Net Liquidity
            df = pd.concat([liq, price_df], axis=1).ffill().dropna()
            df['Net_Liq'] = (df['Fed'] / 1000.0) - df['TGA'] - df['RRP']
            return df
        except Exception as e:
            print(f"Data Acquisition Error: {e}")
            return None

    def run_analysis(self, df, lookback=20, lead_days=10):
        print(f"--- Step 2: Predictive Lead-Lag Study ({lead_days} Day Lead) ---")
        df = df.copy()

        # 1. Current Liquidity Signal
        df['Liq_ROC'] = df['Net_Liq'].pct_change(lookback)
        df['Liq_Z'] = (df['Liq_ROC'] - df['Liq_ROC'].rolling(252).mean()) / df['Liq_ROC'].rolling(252).std()

        for asset in ['SP500', 'Gold']:
            # 2. PREDICTIVE CORRELATION
            # Does Liquidity from 10 days ago correlate with Price moves today?
            # This is your suggested logic: shifting the signal to see its leading impact.
            corr_col = f'Corr_{asset}'
            df[corr_col] = df['Liq_ROC'].shift(lead_days).rolling(252).corr(df[asset].pct_change(lookback))

            # 3. Volatility Scaler
            daily_ret = df[asset].pct_change()
            vol_ratio = daily_ret.rolling(20).std() / daily_ret.rolling(252).std()
            df[f'Vol_Scale_{asset}'] = np.where(vol_ratio > 1.2, 0.7, 1.0)

            # 4. Adaptive Weight
            # Weight = 1.0 + (Leverage * Today's Liq Signal * Past Predictive Success)
            weight_col = f'Weight_{asset}'
            df[weight_col] = (1.0 + (0.4 * df['Liq_Z'] * np.sign(df[corr_col]))).clip(0, 2.0)
            #df[weight_col] = (1.0 + (2 * df['Liq_Z'] *  np.sign(df[corr_col]))).clip(0.5, 1.5)
            #df[weight_col] = df[weight_col] * df[f'Vol_Scale_{asset}']

            # 5. Trading Returns (Shift weight by 1 to execute 'tomorrow')
            df[f'Strat_{asset}_Ret'] = df[weight_col].shift(1) * daily_ret
            df[f'Strat_{asset}_Cum'] = (1 + df[f'Strat_{asset}_Ret']).cumprod()
            df[f'Bench_{asset}_Cum'] = (1 + daily_ret).cumprod()

        return df.dropna()

    def print_performance(self, df):
        print("\n" + "=" * 50)
        print(f"STRATEGY REPORT: {df.index[0].date()} to {df.index[-1].date()}")
        print("=" * 50)
        results = []
        for asset in ['SP500', 'Gold']:
            for t in ['Bench', 'Strat']:
                col = f'{t}_{asset}_Cum'
                ret_col = f'{t}_{asset}_Ret' if f'{t}_{asset}_Ret' in df else f'Daily_{asset}_Ret'  # Placeholder

                # Simple Annualized Return
                total_ret = (df[col].iloc[-1] - 1) * 100
                years = (df.index[-1] - df.index[0]).days / 365.25
                cagr = ((df[col].iloc[-1]) ** (1 / years) - 1) * 100
                results.append([f"{asset} {t}", f"{total_ret:.1f}%", f"{cagr:.2f}%"])

        print(pd.DataFrame(results, columns=["Asset Name", "Total Return", "Annualized (CAGR)"]))

    def plot_asset_page(self, df, asset_name, color_theme):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f"Predictive Liquidity Study: {asset_name}", fontsize=18, fontweight='bold', y=0.96)

        # Equity Curve
        ax1.plot(df.index, df[f'Bench_{asset_name}_Cum'], label='Buy & Hold', color='black', alpha=0.3)
        ax1.plot(df.index, df[f'Strat_{asset_name}_Cum'], label='Predictive Adaptive Strat', color=color_theme, linewidth=2)
        ax1.set_title("Performance: $1 Investment", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.1)

        # Predictive Correlation

        ax2.plot(df.index, df[f'Corr_{asset_name}'], color=color_theme)
        ax2.axhline(0, color='black', linewidth=1.5, linestyle='--')
        ax2.fill_between(df.index, 0, df[f'Corr_{asset_name}'], where=(df[f'Corr_{asset_name}'] >= 0), color='green', alpha=0.1)
        ax2.fill_between(df.index, 0, df[f'Corr_{asset_name}'], where=(df[f'Corr_{asset_name}'] < 0), color='red', alpha=0.1)
        ax2.set_title(f"Predictive Correlation (Liq Lead {asset_name} by 10 Days)", fontsize=12)

        # Exposure Levels
        ax3.fill_between(df.index, 1, df[f'Weight_{asset_name}'], color=color_theme, alpha=0.2)
        ax3.plot(df.index, df[f'Weight_{asset_name}'], color='black', linewidth=0.5, alpha=0.5)
        ax3.axhline(1.0, color='blue', linestyle=':', alpha=0.5)
        ax3.set_title("Strategy Multiplier (Position Sizing)", fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])



if __name__ == "__main__":
    lab = PredictiveLiquidityLab(start="2016-01-01")
    data = lab.get_data()
    if data is not None:
        results = lab.run_analysis(data)
        lab.print_performance(results)
        lab.plot_asset_page(results, "SP500", "royalblue")
        lab.plot_asset_page(results, "Gold", "darkgoldenrod")

    plt.show()