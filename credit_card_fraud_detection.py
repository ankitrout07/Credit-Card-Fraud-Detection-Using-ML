import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             precision_recall_curve, auc)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv("dataset/creditcard.csv")

# 2. Exploratory Data Analysis (Quick Summary)
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts(normalize=True) * 100}")

# 3. Preprocessing
print("Preprocessing data...")
# Features and Target
X = df.drop(columns=['Class'])
y = df['Class']

# Standardize 'Amount' and 'Time' (others are already PCA transformed)
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# Split into Train and Test (Initial)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle Imbalance using SMOTE on Training set only
print("Applying SMOTE (Over-sampling minority class)...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Resampled training set shape: {X_train_res.shape}")

# 4. Model Training and Evaluation Helper
def evaluate_model(model, X_test, y_test, name):
    print(f"\n--- {name} Results ---")
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).flatten()
    
    # If it's the DL model, we might need thresholding
    if name == "Deep Learning (MLP)":
        y_pred = (y_prob > 0.5).astype(int)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.4f}")
    
    return pr_auc

# 5. Model 1: Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_res, y_train_res)
lr_pr_auc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

# 6. Model 2: Random Forest
print("\nTraining Random Forest (this might take a few moments)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)
rf_pr_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# 7. Model 3: Deep Learning (MLP)
print("\nTraining Deep Learning (MLP)...")
dl_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_res.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train_res, y_train_res, epochs=5, batch_size=64, validation_split=0.2, verbose=0)
dl_pr_auc = evaluate_model(dl_model, X_test, y_test, "Deep Learning (MLP)")

print("\n--- Final Comparison ---")
print(f"Logistic Regression PR AUC: {lr_pr_auc:.4f}")
print(f"Random Forest PR AUC:       {rf_pr_auc:.4f}")
print(f"Deep Learning PR AUC:       {dl_pr_auc:.4f}")

print("\nProject Built Successfully!")
