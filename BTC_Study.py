# Get Data
from config.trading_settings import settings
#Update Settings
settings['start']='1996-01-01'
settings['tickers']= [ 'ES=F','NQ=F', 'GC=F','CL=F', 'EURUSD=X','BTC-USD']

import Market_Data_Feed as mdf

data_ind = mdf.Data_Ind_Feed(settings).data_ind
data, _ = data_ind
data_dict = data.data_dict

tickers_returns= data.tickers_returns
print(tickers_returns)

cum_ret=(1+tickers_returns).cumprod()
cum_ret.plot()



import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import yfinance as yf

if __name__ == "__main__":


    #Get previous data
    import requests
    import time


    def get_btc_cryptocompare():
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            "fsym": "BTC",
            "tsym": "USD",
            "limit": 2000,  # max per request
            "toTs": None  # timestamp of last day
        }

        all_data = []

        while True:
            response = requests.get(url, params=params).json()
            batch = response['Data']['Data']
            if not batch:
                break
            all_data.extend(batch)
            # get earliest timestamp for next batch
            earliest = batch[0]['time'] - 1
            if earliest <= 1262304000:  # Jan 1 2010
                break
            params['toTs'] = earliest

        df = pd.DataFrame(all_data)
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df = df[['Date', 'open', 'high', 'low', 'close', 'volumeto']]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Adj Close'] = df['Close']
        # Remove rows where OHLC are all zero
        df = df[~((df['Open'] == 0) & (df['High'] == 0) & (df['Low'] == 0) & (df['Close'] == 0))]
        df = df.sort_values('Date').reset_index(drop=True)
        return df


    #df = get_btc_cryptocompare()
    #print(df)

    # Save to CSV
    def save_to_CSV():
        import os
        csv_path = os.path.join("datasets", "BTCUSD.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved BTC data to {csv_path}")

    # Plot Close prices
    def plot():
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], label='BTC Close', color='orange')
        plt.title('Bitcoin Close Price USD')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)



    plt.show()