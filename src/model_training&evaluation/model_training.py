import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#logging configuration for model building
logger = logging.getLogger(name='model_building')
logger.setLevel(logging.DEBUG)

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.DEBUG)

