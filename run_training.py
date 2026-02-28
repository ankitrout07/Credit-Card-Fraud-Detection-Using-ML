from backend.utils.data_processor import DataProcessor
from backend.service.model_trainer import ModelTrainer
import os

def main():
    print("Starting Professional Model Training Pipeline...")
    
    # 1. Initialize Processor and Trainer
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # 2. Prepare Data
    X_train, X_test, y_train, y_test = processor.prepare_training_data()
    processor.save_scaler()
    
    # 3. Train Models
    lr_model = trainer.train_logistic_regression(X_train, y_train)
    rf_model = trainer.train_random_forest(X_train, y_train)
    dl_model = trainer.train_deep_learning(X_train, y_train)
    
    # 4. Evaluate
    trainer.evaluate(lr_model, X_test, y_test, "Logistic Regression")
    trainer.evaluate(rf_model, X_test, y_test, "Random Forest")
    trainer.evaluate(dl_model, X_test, y_test, "Deep Learning (MLP)")
    
    print("\nAll models trained and saved to data/models/")

if __name__ == "__main__":
    main()
