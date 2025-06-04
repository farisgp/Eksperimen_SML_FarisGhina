import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path="dataset_raw/clothes_price_prediction_data.csv", output_dir="preprocessing"):

    # Membaca dataset
    df = pd.read_csv(input_path)

    # Encode fitur kategorikal
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    # Pisahkan fitur dan target
    X = df.drop("Price", axis=1)
    y = df["Price"]

    #Normalisasi Fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # y_train = y_train.to_numpy()
    # y_test = y_test.to_numpy()


    # Simpan data hasil preprocessing
    pd.DataFrame(X_train, columns=X.columns).to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    pd.DataFrame(y_train.to_numpy(), columns=['Price']).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    pd.DataFrame(y_test.to_numpy(), columns=['Price']).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    print("Preprocessing selesai dan data disimpan di:", output_dir)
    print(f"Jumlah data latih: {X_train.shape[0]}, data uji: {X_test.shape[0]}")

if __name__ == "__main__":
    preprocess_data()
