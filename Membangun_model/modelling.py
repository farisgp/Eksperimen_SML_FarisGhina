import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import random
import numpy as np
import os
import dagshub

# Replace with your DagsHub username and repository name
REPO_OWNER = "farisgp"  # Replace with your DagsHub username
REPO_NAME = "Eksperimen_SML_FarisGhina"  # Replace with your repository name

# Initialize DagsHub - This helps DagsHub track the experiment correctly
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME)


# mlflow.set_tracking_uri("http://127.0.0.1:5000/")  # Comment out local tracking URI

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow/")

print("Tracking URI:", mlflow.get_tracking_uri())

# Create a new MLflow Experiment
mlflow.set_experiment("Clothes Price Prediction")

# --- Load Preprocessed Data
X_train = pd.read_csv("../preprocessing/X_train.csv")
X_test = pd.read_csv("../preprocessing/X_test.csv")
y_train = pd.read_csv("../preprocessing/y_train.csv")
y_test = pd.read_csv("../preprocessing/y_test.csv")

with mlflow.start_run():
    mlflow.autolog()
    # Log parameters
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"R^2: {r2:.4f}")