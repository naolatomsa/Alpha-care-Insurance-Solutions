# Required Libraries
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
    # Handling Missing Data
    # Remove columns with >30% missing values, impute others
    data = data.dropna(thresh=len(data) * 0.7, axis=1)
    data = data.fillna(data.median(numeric_only=True))  # Impute numerical columns with median

    # Encoding Categorical Data
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)  # One-hot encoding
    
    # Train-Test Split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

#Modeling and Evaluation
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae, mse, r2 = evaluate_model(y_test, y_pred)
        results[name] = {"MAE": mae, "MSE": mse, "R2": r2}
        print(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
    
    return models, results

#Feature Importance Analysis
def analyze_feature_importance(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")


# # Load your dataset
# data = pd.read_csv("your_dataset.csv")  # Replace with your dataset file
# target_column = "TotalPremium"  # Replace with your target column

# # Data Preparation
# X_train, X_test, y_train, y_test = prepare_data(data, target_column)

# # Build and Evaluate Models
# models, results = build_and_evaluate_models(X_train, X_test, y_train, y_test)

# # Analyze Feature Importance for Random Forest (example)
# analyze_feature_importance(models["Random Forest"], X_train)
