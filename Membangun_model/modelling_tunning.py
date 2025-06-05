import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from itertools import product
from sklearn.model_selection import train_test_split
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
mlflow.set_experiment("Clothes Price Prediction Tunning")

# --- Load Preprocessed Data
df = pd.read_csv("../preprocessing/clothes_preprocessing.csv")

# Pisahkan fitur dan target
X = df.drop("Price", axis=1)
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 30]
}

# GridSearch dengan cross-validation
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=1
)

if __name__ == "__main__":
    with mlflow.start_run():
        # Fit model
        grid_search.fit(X_train, y_train)

        # Ambil model terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Prediksi
        y_pred = best_model.predict(X_test)

        # Evaluasi
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)

        # Manual logging
        mlflow.log_params(best_params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("Explained_Variance", evs)

        # Logging model
        mlflow.sklearn.log_model(best_model, "model", input_example=X_train.head())

        print("Tuning selesai. Best params:", best_params)
