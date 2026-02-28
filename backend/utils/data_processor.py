import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

class DataProcessor:
    def __init__(self, dataset_path="dataset/creditcard.csv"):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def load_data(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        return pd.read_csv(self.dataset_path)

    def preprocess(self, df):
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        # PCA handles V1-V28, we scale Time and Amount
        X[['Amount', 'Time']] = self.scaler.fit_transform(X[['Amount', 'Time']])
        return X, y

    def prepare_training_data(self, test_size=0.2):
        df = self.load_data()
        X, y = self.preprocess(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print("Applying SMOTE to training data...")
        X_train_res, y_train_res = self.smote.fit_resample(X_train, y_train)
        
        return X_train_res, X_test, y_train_res, y_test

    def save_scaler(self, path="data/models/scaler.pkl"):
        import joblib
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")
