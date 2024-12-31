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

    #remove unnamed column
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

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
    
    X_train = X_train.dropna()
    X_test = X_test.dropna()
    y_train = y_train.dropna()
    y_test = y_test.dropna()
    
    return X_train, X_test, y_train, y_test, data

# Model Building
def build_models():

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }
    return models

# Model Evaluation
def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Model Training and Evaluation
def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae, mse, r2 = evaluate_model(y_test, y_pred)
        results[name] = {"MAE": mae, "MSE": mse, "R2": r2}
        print(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
    return results

# Import SHAP
import shap

# Feature Importance Analysis
def analyze_feature_importance_all(models, X_train):
    for name, model in models.items():
        print(f"Feature Importance for {name}:")
        if name in ["Random Forest", "XGBoost"]:
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, plot_type="bar")
        elif name == "Linear Regression":
            # Use Coefficients for Linear Regression
            coef = model.coef_
            features = X_train.columns
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': coef
            }).sort_values(by='Importance', ascending=False)
            print(feature_importance)
        else:
            print(f"Feature importance method not implemented for {name}.\n")

