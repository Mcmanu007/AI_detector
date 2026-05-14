import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import re
import yaml 

# preprocessor logging configuration
logger = logging.Logger(name='preprocessing')
logger.setLevel(logging.DEBUG)

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler('preprocessor.log')
file_handler.setLevel(logging.DEBUG)

format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(format)
cons_handler.setFormatter(format)

logger.addHandler(file_handler)
logger.addHandler(file_handler)


def load_data_from_path(file_path:str):
    '''
    Args:
        file_path: path to the data
    Returns:
      A pandas dataframe of the data
    '''
    try:
        data = pd.read_csv(file_path)
        encoder = LabelEncoder()
        data['label'] = encoder.fit_transform(data['label'])
        logger.debug('The data has been loaded successfully')
        return data
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the data')
    except FileNotFoundError as e:
        logger.error(f'File can not be located {file_path}')
    except Exception as e:
        logger.error('An error occured',e)


def preprocessor(text):
    #convert text to lower case
    text = text.lower()

    # remove special characters
    text = text = re.sub(r'[^a-z\s]', '', text)

    # tokenize text
    token = word_tokenize(text)

    # stopwords
    stop_words = set(stopwords.words('english'))

    tokens = [word for word in token if word not in stop_words and len(word) >1]

    # apply lemmitization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def apply_preprocessor(data:pd.DataFrame):
    '''
    Args:
      data: Takes in the pandas dataframe of the data and applies the preprocessor
    Returns:
      A preprocessed data 
    '''
    try:
        data['text_preprocessed'] = data['text_content'].apply(preprocessor)
        data.drop('text_content',axis=1,inplace=True)
        #dropping some features that are not needed
        cols_to_drop = ['text_id','source_model','domain','topic_hint','word_count','avg_sentence_length','generation_method']

        for col in cols_to_drop:
            data.drop(columns=col,inplace=True,axis=1)
        logger.debug('Successfully preprocessed the text')
        return data
    except Exception as e:
        logger.debug('Failed to normalize the data',e)
    

def save_preproceesor_info(train:pd.DataFrame,test:pd.DataFrame,out:str='data/preprocessed'):
    try:
        os.makedirs(out,exist_ok=True)
        train.to_csv(os.path.join(out,'train.csv'),index=False)
        test.to_csv(os.path.join(out,'test.csv'),index=False)
        logger.debug('Saved preprocessed data')
    except Exception as e:
        logger.error('An error occured in saving the model',e)

def cleaning_data():
    try:
        config_path = r'C:\Users\IKE\Desktop\Human_VS_AI\config.yaml' 

        with open(config_path,'r') as file:
            config = yaml.safe_load(file)

        train_path = config['train_data']['train_path']
        test_path = config['test_data']['test_path']

        train_data = load_data_from_path(train_path)
        test_data = load_data_from_path(test_path)

        train_preprocessed = apply_preprocessor(train_data)
        test_preprocessed = apply_preprocessor(test_data)

        save_preproceesor_info(train_preprocessed,test_preprocessed)
        logger.debug('Data preprocessing stage completed successfully')
    except Exception as e:
        logger.debug('Error occured in preprocessing the data',e)

if __name__ == '__main__':
    cleaning_data()