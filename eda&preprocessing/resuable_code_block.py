import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline



def main(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,model_name:str):
    """
    Builds an imblearn Pipeline with TF-IDF, SMOTE, and a classifier,
    tunes hyperparameters via GridSearchCV, and logs everything to MLflow.

    Args:
        X_train: Training features (must contain 'text_preprocessed' column)
        X_test:  Testing features  (must contain 'text_preprocessed' column)
        y_train: Training targets
        y_test:  Testing targets (needed to log test metrics)

    Returns:
        best_model: The best estimator found by GridSearchCV
        y_pred:     Predictions on X_test
    """
    try:
        
        mlflow.set_experiment("text_classification_smote")

        # --- 2. Build the pipeline ---
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('smote', SMOTE(random_state=42)),
            ('clf', model_name)
        ])

        # --- 3. Define the parameter grid ---
        param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__max_features': [300, 500, 1000],
            'tfidf__max_df': [0.85, 0.9, 0.95],
            'tfidf__min_df': [1, 2, 5],
            'tfidf__sublinear_tf': [True, False],

            'smote__k_neighbors': [3, 5, 7],
            'smote__sampling_strategy': ['auto', 'minority'],

            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l2'],
        }

        # --- 4. Set up GridSearchCV ---
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        # --- 5. Parent run: log the overall best result ---
        with mlflow.start_run(run_name="grid_search_best") as parent_run:

            # Fit
            grid_search.fit(X_train['text_preprocessed'], y_train)
            best_model = grid_search.best_estimator_

            # Predict on test set
            y_pred = best_model.predict(X_test['text_preprocessed'])

            # Log best parameters (one entry per param)
            mlflow.log_params(grid_search.best_params_)

            # Log CV score
            mlflow.log_metric("best_cv_f1_weighted", grid_search.best_score_)

            # Log test set metrics
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("test_f1_weighted", f1_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric("test_precision_weighted", precision_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric("test_recall_weighted", recall_score(y_test, y_pred, average='weighted'))

            # Log classification report as a text artifact
            report = classification_report(y_test, y_pred)
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")

            # Log the best model
            mlflow.sklearn.log_model(best_model, "best_pipeline_model")

            print(f"\nBest CV F1: {grid_search.best_score_:.4f}")
            print(f"Best params: {grid_search.best_params_}")
            print(f"\nTest Classification Report:\n{report}")

            # --- 6. Child runs: log each CV candidate for comparison ---
            cv_results = grid_search.cv_results_

            for i in range(len(cv_results['params'])):
                with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
                    # Log this candidate's parameters
                    mlflow.log_params(cv_results['params'][i])

                    # Log this candidate's scores
                    mlflow.log_metric("mean_test_f1", cv_results['mean_test_score'][i])
                    mlflow.log_metric("std_test_f1", cv_results['std_test_score'][i])
                    mlflow.log_metric("mean_train_f1", cv_results['mean_train_score'][i])
                    mlflow.log_metric("rank", cv_results['rank_test_score'][i])

        return best_model, y_pred

    except Exception as e:
        print('An error occurred:', e)