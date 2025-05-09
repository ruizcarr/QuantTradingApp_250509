import requests
import pandas as pd
import io
import matplotlib.pyplot as plt
import json

def get_euribor_1y_daily():
    """
    Retrieves EURIBOR 1-year data from the ECB API, processes it,
    and returns a Pandas DataFrame with daily reindexing and forward fill.

    Returns:
        pandas.DataFrame: DataFrame with 'date' as index and 'Euribor' as values,
                          or None if an error occurs.
    """
    dataflow_id = "FM/M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA"
    api_url = f"https://data-api.ecb.europa.eu/service/data/{dataflow_id}?format=csvdata"

    try:
        response = requests.get(api_url)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))

        df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        df.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "Euribor"}, inplace=True)

        # Convert 'Euribor' to float
        df["Euribor"] = pd.to_numeric(df["Euribor"], errors='coerce')/100  # coerce will turn non number values into NaN.

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        #df = df.reindex(daily_index, method="ffill")
        df = df.reindex(daily_index)
        df['Euribor'] = df['Euribor'].interpolate(method='linear')

        #Add Cash Values applying to starting 1.000 USD
        #df['cash_values']=1000*(1+df['Euribor']/255).cumprod()
        #df['dayly_Euribor']=df['cash_values'].pct_change() #*255

        df.to_csv("datasets/EURIBOR.csv")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        if response is not None:
            print(response.text)
        return None  # Return None to indicate an error
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return None # Return None to indicate an error

def get_fed_1year_treasury_yield_daily():
    """
    Retrieves 1-year Treasury constant maturity rate (TCM) data from the FRED API,
    processes it, and returns a Pandas DataFrame with daily reindexing and linear interpolation.

    Args:
        api_key (str): Your FRED API key.

    Returns:
        pandas.DataFrame: DataFrame with 'DATE' as index and 'FED' as values,
                          or None if an error occurs.
    """
    api_key = "82b5eddf94843e6d65a1633fca41b973"  # Replace with your API key
    series_id = "DFF"  # FRED series ID for Effective Federal Funds Rate
    api_url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"

    try:
        response = requests.get(api_url)
        response.raise_for_status()

        data = response.json()
        observations = data['observations']
        df = pd.DataFrame(observations)
        df = df[["date", "value"]].copy()
        df.rename(columns={"date": "date", "value": "FED"}, inplace=True)

        # Convert 'FED' to float
        df["FED"] = pd.to_numeric(df["FED"], errors='coerce')/100 # coerce will turn non number values into NaN.

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        #Keep Monthly values only to avoid trading dips
        df = df.resample('M').mean()

        daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        df = df.reindex(daily_index)
        df['FED'] = df['FED'].interpolate(method='linear')


        df.to_csv("datasets/FED.csv")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        if response is not None:
            print(response.text)
        return None  # Return None to indicate an error
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON: {e}")
        return None  # Return None to indicate an error



if __name__ == "__main__": #Example of how to use the function.
    euribor_df = get_euribor_1y_daily()
    if euribor_df is not None:
        print(euribor_df)
        euribor_df.plot()
    else:
        print("Failed to retrieve EURIBOR data.")


    fed_df= get_fed_1year_treasury_yield_daily()
    if euribor_df is not None:
        print(fed_df)
        fed_df.plot()
    else:
        print("Failed to retrieve FED data.")

    plot_df = pd.concat([euribor_df, fed_df], axis=1, join='outer')  # outer join handles missing values
    plot_df.columns = ["Euribor", "Fed"]

    rates_df = euribor_df.copy()
    rates_df["Fed"] = fed_df

    rates_df.plot()


plt.show()