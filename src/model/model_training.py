import logging
import ast
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import joblib
import yaml
import os


#logging configuration for model building
logger = logging.getLogger(name='model_building')
logger.setLevel(logging.DEBUG)

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        logger.error('An error occured: %s', e)


def main(train_data:pd.DataFrame,ngram_range:int,max_features:int,max_df:int, random_state:int,k_neigbors):
    '''
    Args:
      X_train: 
    '''
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=tuple(ngram_range),
            max_features=max_features,
            max_df=max_df
         )
        X_train = train_data['text_preprocessed'].values
        y_train = train_data['label'].values  

        X_train_vec = vectorizer.fit_transform(X_train)
        #saving the vectorizer for later use in the inference pipeline
        joblib.dump(vectorizer,os.path.join(pickle(),'vectorizer.pkl'))
        smote = SMOTE(random_state=random_state, k_neighbors=k_neigbors)
        X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

        logger.debug('The model has been trained successfully')
        return X_train_res, y_train_res
    except Exception as e:
        logger.error('An error occured: %s', e)
    

def model_building(X_train:np.ndarray,y_train:np.ndarray):
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logger.error('An error occured: %s', e)
    


def pickle(path:str='./pickle'):
    '''
    Returns:
      path: Creates a path for storing the saved model
    '''
    try:
        os.makedirs(path,exist_ok=True)
        return os.path.abspath(path)
    except Exception as e:
        logger.error('Failure to store the saved model in the specified path')
        raise e
    

def save_model(model):
    try:
        joblib.dump(model,os.path.join(pickle(),'rf_model.pkl'))
        logger.debug('The model has been saved successfully')
    except Exception as e:
        logger.error('An error occured in saving the model: %s', e)

def train_model():
    try:
        config_path = 'config.yaml'
        with open(config_path,'r') as file:
            config = yaml.safe_load(file)

        train_path = config['preprocessed_train_data']['train']
        ngram_range = config['vectorizer']['ngram_range']
        max_features = config['vectorizer']['max_features']
        max_df = config['vectorizer']['max_df']

        random_state = config['smote']['random_state']
        k_neighbors = config['smote']['k_neighbors']

        train_data = load_train_data(train_path)
        X_train, y_train = main(train_data,ngram_range,max_features,max_df,random_state,k_neighbors)
        model = model_building(X_train,y_train)
        save_model(model)
        logger.debug('Model trained succsefully')
    except Exception as e:
        logger.debug('An error occured in training the model: %s', e)

if __name__ == '__main__':
    train_model()