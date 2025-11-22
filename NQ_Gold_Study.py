import pandas as pd
import datetime


import numpy as np, random
np.random.seed(42)
random.seed(42)

from Trading_Markowitz import compute, get_trading_positions,process_log_data
from config.trading_settings import settings
import Market_Data_Feed as mdf
from Backtest_Vectorized_Class import compute_backtest_vectorized

#Main Code
def main(settings):

    # Update Settings
    settings['start'] = "2015-01-01"

    #Load data
    data_ind = mdf.Data_Ind_Feed(settings).data_ind
    data, indicators_dict = data_ind

    #Get Positions Markowitz
    positions_mktw = get_trading_positions(data_ind, settings, indicators_dict)

    #Get Positions NQ_Gold
    pos_nq_gc=data.tickers_returns
    pos_nq_gc = pos_nq_gc.reindex(positions_mktw.index)
    pos_nq_gc =pos_nq_gc *0
    pos_nq_gc["NQ=F"]=0.5#np.where(positions_mktw["NQ=F"]>0.01,0.5,0)
    pos_nq_gc["GC=F"] =0.5# np.where(positions_mktw["GC=F"]>0.01,0.5,0)

    #Positions to Back Test
    positions=pos_nq_gc*0.5+positions_mktw*0.5

    #Back Test
    _, log_history = compute_backtest_vectorized(positions, settings, data.data_dict)

    weekago = pd.Timestamp.today().normalize() - pd.Timedelta(days=7)
    print("log_history\n", log_history[log_history["date_time"] >= weekago])


if __name__ == '__main__':
    main(settings)