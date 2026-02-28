import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
import os

class ModelTrainer:
    def __init__(self, models_dir="data/models"):
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def train_logistic_regression(self, X_train, y_train):
        print("Training Logistic Regression...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(self.models_dir, "lr_model.pkl"))
        return model

    def train_random_forest(self, X_train, y_train):
        print("Training Random Forest...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(self.models_dir, "rf_model.pkl"))
        return model

    def train_deep_learning(self, X_train, y_train):
        print("Training Deep Learning Model...")
        if HAS_TF:
            try:
                print("Using TensorFlow for MLP...")
                model = keras.Sequential([
                    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(32, activation='relu'),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)
                model.save(os.path.join(self.models_dir, "dl_model.h5"))
                return model
            except Exception as e:
                print(f"TensorFlow training failed: {e}. Falling back to scikit-learn.")
        
        print("Using scikit-learn MLPClassifier.")
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(self.models_dir, "dl_model_sklearn.pkl"))
        return model

    def evaluate(self, model, X_test, y_test, name):
        print(f"\n--- {name} Results ---")
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:
            y_prob = model.predict(X_test).flatten()
            y_pred = (y_prob > 0.5).astype(int)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "pr_auc": pr_auc
        }
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")
        return metrics
