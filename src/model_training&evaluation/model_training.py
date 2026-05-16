import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os


#logging configuration for model building
logger = logging.getLogger(name='model_building')
logger.setLevel(logging.DEBUG)

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def load_train_data(train_path):
    try:
        data = pd.read_csv(train_path)
        logger.debug('Train data loaded successfully')
        return data
    except pd.errors.ParserError as e:
        logger.errro('Error in parsing the data')
    except FileNotFoundError as e:
        logger.error('An error occured in locating the error')
    except Exception as e:
        logger.error('An error occured',e)


def main(X_train:pd.DataFrame,y_train:pd.DataFrame,ngram_range:int,max_features:int,max_df:int, random_state:int,k_neigbors):
    try:
        vectorizer =
    

def model_building():
    pass

def pickle(pkl_path:str='./pkl'):
    try:
        os.makedirs(pkl_path,exist_ok=True)
        return os.path.abspath(pkl_path)
    except Exception as e:
        logger.error('Failed to create the model path')

def save_model():
    pass

def train_model():
    pass


