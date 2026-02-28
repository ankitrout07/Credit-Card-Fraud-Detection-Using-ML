from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False

app = Flask(__name__)
CORS(app)

# Paths to models and scaler
MODELS_DIR = "data/models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "lr_model.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
DL_MODEL_PATH = os.path.join(MODELS_DIR, "dl_model.h5")

# Global variables for models
scaler = None
lr_model = None
rf_model = None
dl_model = None

def load_models():
    global scaler, lr_model, rf_model, dl_model
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if os.path.exists(LR_MODEL_PATH):
            lr_model = joblib.load(LR_MODEL_PATH)
        if os.path.exists(RF_MODEL_PATH):
            rf_model = joblib.load(RF_MODEL_PATH)
        if os.path.exists(DL_MODEL_PATH):
            dl_model = keras.models.load_model(DL_MODEL_PATH)
        elif os.path.exists(os.path.join(MODELS_DIR, "dl_model_sklearn.pkl")):
            dl_model = joblib.load(os.path.join(MODELS_DIR, "dl_model_sklearn.pkl"))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "Invalid input"}), 400
    
    try:
        # Expected features: [Time, V1, V2 ... V28, Amount]
        raw_features = np.array(data['features']).reshape(1, -1)
        
        # Create DataFrame for easier manipulation and to match Column names if needed
        cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        features_df = pd.DataFrame(raw_features, columns=cols)
        
        # Scale Time and Amount using the loaded scaler
        # The scaler was fit on [['Amount', 'Time']] in data_processor.py
        if scaler:
            # Note: order MUST match training: [Amount, Time]
            to_scale = features_df[['Amount', 'Time']]
            scaled_vals = scaler.transform(to_scale)
            features_df[['Amount', 'Time']] = scaled_vals
        
        # Convert back to numpy for model prediction
        processed_features = features_df.values
        
        model_type = data.get('model', 'lr') # 'lr', 'rf', 'dl'
        
        if model_type == 'lr' and lr_model:
            pred = lr_model.predict(processed_features)[0]
            prob = lr_model.predict_proba(processed_features)[0][1]
        elif model_type == 'rf' and rf_model:
            pred = rf_model.predict(processed_features)[0]
            prob = rf_model.predict_proba(processed_features)[0][1]
        elif model_type == 'dl' and dl_model:
            prob = dl_model.predict(processed_features)[0][0]
            pred = int(prob > 0.5)
        else:
            return jsonify({"error": f"Model {model_type} not loaded or invalid"}), 500

        return jsonify({
            "prediction": int(pred),
            "probability": float(prob),
            "is_fraud": bool(pred == 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "active",
        "models_loaded": {
            "logistic_regression": lr_model is not None,
            "random_forest": rf_model is not None,
            "deep_learning": dl_model is not None
        }
    })

if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5000)
