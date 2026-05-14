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

file_handler = logging.FileHandler('ingestion.log')
file_handler.setLevel(logging.DEBUG)

format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(format)
cons_handler.setFormatter(format)

logger.addHandler(file_handler)
logger.addHandler(file_handler)

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
        data.drop_duplicates(inplace=True)
        data.dropna(inplace=True)
        logger.debug('The data has been loaded successfully')
        return data
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the data')
    except FileNotFoundError as e:
        logger.error(f'File can not be located {file_path}')
    except Exception as e:
        logger.error('An error occured',e)



def data_divide(data:pd.DataFrame,test_size:int,random_state:int) -> pd.DataFrame:
    '''
    Args:
        data: The pandas DataFrame of the csv file
        test_size: Size of the testing data
        random_state: controls the shuffling
    Returns:
      train_data: A pandas DataFrame of the training data
      test_data: A pandas DataFrame of the testing data
    '''
    try:
        train,test = train_test_split(data,test_size=test_size,random_state=random_state,stratify=data['label'])
        logger.debug('The Data has been splitted into training and testing')
        return train,test
    except Exception as e:
        logger.debug('Failed to split the data')


def save_info(train:pd.DataFrame,test:pd.DataFrame,out:str='data/raw'):
    '''
    Args:
       train: Train_data as pandas DataFrame
       test: Test data as a pandas DataFrame
    Returns:
      The saved training and testing data to csv file
      
    '''
    try:
        os.makedirs(out,exist_ok=True)
        train.to_csv(os.path.join(out,'raw_train_data.csv'),index=False)
        test.to_csv(os.path.join(out,'raw_test_data.csv'),index=False)
        logger.debug('Train and test data info saved successfully')
    except Exception as e:
        logger.debug('Failed to save the information')

def ingestion_stage():
    try:
        config_path = r'C:\Users\IKE\Desktop\Human_VS_AI\config.yaml'
        with open(config_path,'r') as file:
            config = yaml.safe_load(file)

        path = config['data']['path_to_data']
        test_size = config['data']['test_size']
        random_state = config['data']['random_state']

        data = load_data_from_path(path)
        train,test = data_divide(data,test_size,random_state)
        save_info(train,test)
    except Exception as e:
        logger.debug('Failed to complete the data ingestion stage',e)

if __name__ == '__main__':
    ingestion_stage()