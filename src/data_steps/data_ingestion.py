import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

#logging configuration
logger = logging.Logger(name='ingestion')
logger.setLevel(logging.DEBUG)

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler(name='ingestion.log')
file_handler.setLevel(logging.DEBUG)

format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(format)
cons_handler.setFormatter(format)

logger.add_handler(file_handler)
logger.add_handler(file_handler)

#load the data
def load_data_from_path(file_path:str):
    '''
    Args:
        file_path: path to the data
    Returns:
      A pandas dataframe of the data
    '''
    try:
        data = pd.read_csv(file_path)
        logger.debug('The data has been loaded successfully')
        return data
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the data')
    except FileNotFoundError as e:
        logger.error(f'File can not be located {file_path}')
    except Exception as e:
        logger.error('An error occured',e)