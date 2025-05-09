from datetime import date
from datetime import timedelta

# Import Settings
from config.utils import retrieve_training_settings
settings =retrieve_training_settings()

#from config.settings import settings


# Update Settings to Trading Settings
settings['trading_app_only']=True
settings['start'] = '2019-01-01' # '2018-01-01'
settings['end'] =(date.today() + timedelta(days=1)).isoformat()
settings['startcash'] = 250000 #47000 #56500#210000#35000 #30650 #27800 #54300 #52300 # #EUR
settings['verbose'] = True
settings['qstats'] = True
settings['do_BT']= True
settings['offline'] = False
settings['add_cash'] = True
