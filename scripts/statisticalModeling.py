import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

# Data Preparation
def prepare_data(data, target_column):
    """
    Handles missing data, encodes categorical variables, performs feature engineering, and splits the dataset into training and testing sets.

    Args:
        data (DataFrame): The dataset to prepare.
        target_column (str): The name of the target column.

    Returns:
        X_train, X_test, y_train, y_test: Training and testing datasets for features and target.
    """
    # Handling Missing Data
    data = data.dropna(thresh=len(data) * 0.7, axis=1)  # Remove columns with >30% missing values
    data = data.fillna(data.median(numeric_only=True))  # Impute numerical columns with median

    # Feature Engineering
    data['AverageClaimPerPolicy'] = data['TotalClaims'] / (data['PolicyID'].nunique() + 1e-5)
    data['PremiumToClaimsRatio'] = data['TotalPremium'] / (data['TotalClaims'] + 1e-5)
    data['HighRiskFlag'] = (data['PremiumToClaimsRatio'] < 1).astype(int)
    data['VehicleValueToPremiumRatio'] = data['SumInsured'] / (data['TotalPremium'] + 1e-5)

    if 'RegistrationYear' in data.columns:
        current_year = pd.Timestamp.now().year
        data['VehicleAge'] = current_year - data['RegistrationYear']

    # Encoding Categorical Data
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)  # One-hot encoding

    # Train-Test Split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Model Building
def build_models():
    """
    Initializes the models to be used for training.

    Returns:
        dict: A dictionary of initialized models.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }
    return models

# Model Evaluation
def evaluate_model(y_true, y_pred):
    """
    Evaluates a model's performance using MAE, MSE, and R² metrics.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        tuple: MAE, MSE, and R² score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Model Training and Evaluation
def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Trains and evaluates the models, printing and returning their performance metrics.

    Args:
        models (dict): Dictionary of models to train.
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        dict: A dictionary of model performance metrics.
    """
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae, mse, r2 = evaluate_model(y_test, y_pred)
        results[name] = {"MAE": mae, "MSE": mse, "R2": r2}
        print(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
    return results

# Feature Importance Analysis
def analyze_feature_importance(model, X_train):
    """
    Analyzes and visualizes feature importance using SHAP values.

    Args:
        model: The trained model to analyze.
        X_train (DataFrame): Training features.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

# Main Execution Example
# Uncomment the following lines to run the code
# data = pd.read_csv("your_dataset.csv")  # Replace with your dataset file
# target_column = "TotalPremium"  # Replace with your target column
# 
# X_train, X_test, y_train, y_test = prepare_data(data, target_column)
# models = build_models()
# results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)
# analyze_feature_importance(models["Random Forest"], X_train)
