import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# preprocessor logging configuration
logger = logging.Logger(name='preprocessing')
logger.setLevel(logging.DEBUG)

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler(name='preprocessor.log')
file_handler.setLevel(logging.DEBUG)

format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(format)
cons_handler.setFormatter(format)

logger.add_handler(file_handler)
logger.add_handler(file_handler)

