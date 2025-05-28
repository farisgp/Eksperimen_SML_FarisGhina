import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
import random
import numpy as np
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
# print("DAGSHUB_TOKEN set:", 'DAGSHUB_TOKEN' in os.environ)  # For debugging, can be commented out
# print("Token Value:", os.environ.get("DAGSHUB_TOKEN")[:5], "...(disembunyikan)")  # For debugging, can be commented out

# Create a new MLflow Experiment
mlflow.set_experiment("Clothes Price Prediction")

df = pd.read_csv("clothes_price_prediction_data.csv")

# Encode fitur kategorikal
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

X = df.drop("Price", axis=1)
y = df["Price"]

#Normalisasi Fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}

input_example = X_train[0:5]

combinations = list(product(param_grid['n_estimators'], param_grid['max_depth']))

for n_estimators, max_depth in combinations:
    with mlflow.start_run(nested=True):

        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Logging parameter
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Logging metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Logging model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
