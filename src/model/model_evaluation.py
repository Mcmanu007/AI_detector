import pandas as pd
import logging
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import mlflow
import json
import os
from mlflow.models.signature import infer_signature

#logging configuration for evaluating the model
logger = logging.Logger(name='model_evaluation')
logger.setLevel(logging.DEBUG)

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler('evaluation.log')
file_handler.setLevel(logging.DEBUG)

format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(format)
cons_handler.setFormatter(format)

logger.addHandler(file_handler)
logger.addHandler(file_handler)


#load the testing data
def load_test_data(test_data_path):
    try:
        data = pd.read_csv(test_data_path)
        logger.debug('Test data loaded successfully')
        return data
    except pd.errors.ParserError as e:  
        logger.error('Error in parsing the data')
    except FileNotFoundError as e:
        logger.error('An error occured in locating the error')
    except Exception as e:
        logger.error('An error occured: %s', e)


def load_trained_model(model_path):
    try:
        model = joblib.load(model_path)
        logger.debug('Model loaded successfully')
        return model
    except FileNotFoundError as e:
        logger.error('An error occured in locating the error')
    except Exception as e:
        logger.error('An error occured: %s', e)


def load_vectorizer(vectorizer_path):
    try:
        vectorizer = joblib.load(vectorizer_path)
        logger.debug('Vectorizer loaded successfully')
        return vectorizer
    except FileNotFoundError as e:
        logger.error('An error occured in locating the error')
    except Exception as e:
        logger.error('An error occured: %s', e)


def evaluate_model(model,vectorizer,X_test:np.ndarray,y_test:np.ndarray):
    try:
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred,output_dict=True)
        logger.debug('Model evaluation completed successfully')
        return acc, conf_matrix, class_report
    except Exception as e:
        logger.error('An error occured: %s', e)


def save_evaluation_results(acc, conf_matrix, class_report, output_path:str='model/evaluation_results.txt'):
    try:
        os.makedirs(output_path, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f'Accuracy: {acc}\n')
            f.write(f'Confusion Matrix:\n{conf_matrix}\n')
            f.write(f'Classification Report:\n{class_report}\n')
        logger.debug('Evaluation results saved successfully')
    except Exception as e:
        logger.error('An error occured: %s', e)


def save_info(model_path:str,run_id:str,file_path:str):
    try:
        model_info = {
            run_id:'run_id',
            model_path:'model_path'
        }
        with open(file_path,'w') as f:
            json.dump(model_info,f,indent=4)
        logger.debug('Model info saved successfully')
    except Exception as e:
        logger.error('An error occured: %s', e)


def main():
    mlflow.set_experiment('Model_Evaluation')

    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    with mlflow.start_run(run_name='Evaluation_Run') as run:
        try:
            config_path = 'config.yaml'
            with open(config_path,'r') as file:
                config = yaml.safe_load(file)

            test_path = config['preprocessed_test_data']['test_data']
            model_path = config['pickle']['model_path']
            vectorizer_path = config['pickle']['vectorizer_path']

            test_data = load_test_data(test_path)
            model = load_trained_model(model_path)
            vectorizer = load_vectorizer(vectorizer_path)
            
            #preparing the testing data
            X_test = test_data['text'].values
            y_test = test_data['label'].values

            #test dataframe
            tests_example = pd.DataFrame(X_test.toarray()[:10],columns=vectorizer.get_feature_names_out())


            test_signature = infer_signature(tests_example,model.predict(X_test[:10]))

        
            mlflow.sklearn.log_model(
                    model,
                    'RandomForestClassifier',
                    signature=test_signature,
                    input_example=tests_example
                )
            model_path = 'Random Forest Classifier Model'
            save_info(
                    run.info.run_id,   
                    model_path,
                    'model_info.json'
            )

            acc, conf_matrix, class_report = evaluate_model(model,vectorizer,X_test,y_test)
            save_evaluation_results(acc, conf_matrix, class_report)

            for label,metrics in class_report.items():
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f'{label}_{metric_name}', metric_value)

            mlflow.log_metric('accuracy', acc)
            mlflow.log_artifact('model_info.json')
            mlflow.log_artifact('model/evaluation_results.txt')
            
            whitelist = ['vectorizer', 'smote']
            #logging the parameters of the vectorizer and smote model
            
            params_to_log = {}
            for section in whitelist:
                if section in config:
                    for key, value in config[section].items():
                        params_to_log[f"{section}_{key}"] = value
            mlflow.log_params(params_to_log)
        except Exception as e:  
           logger.error('An error occured in the evaluation process: %s', e)

if __name__ == '__main__':
    main()
