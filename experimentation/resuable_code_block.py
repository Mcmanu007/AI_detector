import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # NOT sklearn.pipeline — this one supports resamplers
from sklearn.linear_model import LogisticRegression  # swap with your classifier


def main(X_train:pd.DataFrame,X_test:pd.DataFrame,y_train:pd.DataFrame):
    """
    Args:
        X_train: Training features (must contain 'text_preprocessed' column)
        X_test:  Testing features  (must contain 'text_preprocessed' column)
        y_train: Training targets

    Returns:
        Resampled training data(features): X_train_res
        Resampled training data(taregets):  y_train_res
        vectorized testing data: X_test_vec
    """
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
            max_features=500,
            max_df=0.9
        )

        X_train_vec = vectorizer.fit_transform(X_train['text_preprocessed'])
        X_test_vec = vectorizer.transform(X_test['text_preprocessed'])

        smote = SMOTE(random_state=42, k_neighbors=7)
        X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

        return X_train_res, y_train_res, X_test_vec
    except Exception as e:
        print('An error occured ', e)