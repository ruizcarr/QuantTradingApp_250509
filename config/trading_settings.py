from datetime import date
from datetime import timedelta

# Import Settings
from config.utils import retrieve_training_settings
settings =retrieve_training_settings()

#from config.settings import settings


# Update Settings to Trading Settings
settings['trading_app_only']=True
settings['start'] = '2017-01-01' #'2020-01-01' # '2019-01-01'
settings['end'] =(date.today() + timedelta(days=1)).isoformat()
settings['startcash'] = 110000 #95000 2017#200000#240000 #47000 #56500#210000#35000 #30650 #27800 #54300 #52300 # #EUR
settings['max_n_contracts'] = 6
