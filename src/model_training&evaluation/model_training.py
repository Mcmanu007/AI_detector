import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import joblib
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
        logger.error('Error in parsing the data')
    except FileNotFoundError as e:
        logger.error('An error occured in locating the error')
    except Exception as e:
        logger.error('An error occured',e)


def main(X_train:pd.DataFrame,y_train:pd.DataFrame,ngram_range:int,max_features:int,max_df:int, random_state:int,k_neigbors):
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            max_df=max_df
         )
        X_train_vec = vectorizer.fit_transform(X_train['text_preprocessed'])
        #saving the vectorizer for later use in the inference pipeline
        joblib.dump(vectorizer,os.path.join(pickle(),'vectorizer.pkl'))
        smote = SMOTE(random_state=random_state, k_neighbors=k_neigbors)
        X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

        logger.debug('The model has been trained successfully')
        return X_train_res, y_train_res
    except Exception as e:
        logger.error('An error occured',e)
    

def model_building(X_train:np.ndarray,y_train:np.ndarray):
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        logger.debug('The model has been trained and saved successfully')
    except Exception as e:
        logger.error('An error occured',e)
    


def pickle(pkl_path:str='./pkl'):
    try:
        os.makedirs(pkl_path,exist_ok=True)
        return os.path.abspath(pkl_path)
    except Exception as e:
        logger.error('Failed to create the model path')

def save_model(model):
    try:
        joblib.dump(model,os.path.join(pickle(),'model.pkl'))
        logger.debug('The model has been saved successfully')
    except Exception as e:
        logger.error('An error occured in saving the model',e)

def train_model():
    pass


