import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess_data(input_path="dataset_raw/clothes_price_prediction_data.csv", output_dir="preprocessing"):

    # Membaca dataset
    df = pd.read_csv(input_path)

    # Encode fitur kategorikal
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Pisahkan fitur dan target
    X = df.drop("Price", axis=1)
    y = df["Price"]

    #Normalisasi Fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Gabungkan kembali dengan target untuk disimpan
    df_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
    df_preprocessed['Price'] = y.reset_index(drop=True)


    # Simpan data hasil preprocessing
    output_file = os.path.join(output_dir, "clothes_preprocessing.csv")
    df_preprocessed.to_csv(output_file, index=False)

    print("Preprocessing selesai dan data disimpan di:", output_dir)
    print(f"Jumlah data total: {df_preprocessed.shape[0]}")

if __name__ == "__main__":
    preprocess_data()
