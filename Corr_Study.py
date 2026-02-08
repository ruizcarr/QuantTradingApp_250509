import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Get SETTINGS
from config import settings,utils
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

#Update Settings
settings['start']='1996-01-01'

import Market_Data_Feed as mdf

#Get Data
data_ind = mdf.Data_Ind_Feed(settings).data_ind
data, ind = data_ind
data_dict = data.data_dict

tickers_returns=data.tickers_returns
tickers_returns['NQ+GC']=(tickers_returns['NQ=F']+tickers_returns['GC=F'])/2
cum_rets=(1+tickers_returns).cumprod()
#print('tickers_returns.describe()',tickers_returns.describe())


corr=ind['corr_df']

print(corr)

corr_tickers=['ES_NQ','NQ_GC'] #,'NQ_BT','GC_BT'
sel_corr=corr[corr_tickers]
sel_corr.plot(title='Correlation')
for ticker in corr_tickers:
    corr[[ticker]].plot(title=ticker)


corr_factor=corr['NQ_GC'] #.abs()
corr_factor=(1-corr_factor)
corr_factor=np.exp(1+corr_factor)-1.55-4.0
#corr_factor=(corr_factor**2)*1.1
corr_factor=corr_factor.clip(upper=1.75,lower=0.25).shift(1)


corr_rets=tickers_returns['NQ+GC']*corr_factor
corr_cumret=(1+corr_rets).cumprod()

plot_df=cum_rets[['NQ=F','GC=F','NQ+GC']]
plot_df['corr_NQ_GC']=corr['NQ_GC']
plot_df['corr_factor']=corr_factor
plot_df['corr_rets']=corr_cumret

plot_df.plot()

plt.show()