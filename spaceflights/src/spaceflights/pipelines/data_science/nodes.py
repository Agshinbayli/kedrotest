"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.4
"""

import logging
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import datetime
import joblib


def split_data(data: pd.DataFrame, parameters: dict[str, any]) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters_data_science.yml.
    Returns:
        Split data.
    """
    
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)

    # Calculate MSE and MAE using scikit-learn
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Log metrics and artifacts to MLflow
    mlflow.log_metrics({
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "r_squared": r2_score(y_test, y_pred)  # Assuming you #have r2_score implemented
        # Add other relevant metrics here
    })
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a coefficient R^2 of {score:.3f} on test data. MSE: {mse:.2f}, MAE: {mae:.2f}")
    model_path = "data/06_models/mlflowtest.pickle"
    joblib.dump(regressor, model_path)
    mlflow.register_model(model_path, "mlflowtest_name")
    mlflow.sklearn.log_model(regressor, "skmodel", registered_model_name="regressor")

