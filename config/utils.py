"""
To retrieve settings
with open('training_settings.json', 'r') as f:
    training_settings = json.load(f)
"""
import json

def get_path():
    return "config/" #"config/json_files/" #config\\json_files\\


def settings_to_JASON (settings):

    path = get_path()

    #from config.settings import settings
    #settings=settings.get_settings()

    # Save Settings to training_settings.json file
    #with open(path+'training_settings.json', 'w') as f:
    with open('config/training_settings.json', 'w') as f:
        json.dump(settings, f)


def retrieve_training_settings():

    path = get_path()

    # Import Settings
    #with open(path+'training_settings.json', 'r') as f:
    with open('config/training_settings.json', 'r') as f:
        settings = json.load(f)


    return settings
