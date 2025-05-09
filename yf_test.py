import numpy as np
import pandas as pd

import yfinance as yf

tickers=['ES=F', 'GC=F', 'CL=F', 'EURUSD=X', 'NQ=F']
start='1996-01-01'
end='2025-01-01'
tickers_space_sep = " ".join(tickers)
data_bundle = yf.download(tickers_space_sep, start, end, group_by='ticker', progress=False)


print('data_bundle',data_bundle)