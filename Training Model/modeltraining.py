import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import os
# Constants
MODEL_PATH = "obesity_model.joblib"
SCALER_PATH = "obesity_scaler.joblib"
ENCODERS_PATH = "obesity_encoders.joblib"


# Function to train and save model
def train_and_save_model(file_path):
    """
    Train models and save them to disk for later use.
    This should be run before deploying the Streamlit app.
    """
    print("Loading and preprocessing dataset...")
    # Load dataset
    df = pd.read_csv(file_path)

    # Display class distribution
    target_col = 'NObeyesdad'
    class_distribution = df[target_col].value_counts()
    print("Class Distribution:\n", class_distribution)

    # Prepare for encoding
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC',
                            'MTRANS']

    # Encode categorical columns
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Prepare data for SMOTE
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("New class distribution after SMOTE:", Counter(y_resampled))

    # Create a new balanced dataframe
    balanced_df = X_resampled.copy()
    balanced_df[target_col] = y_resampled

    print("Data preprocessing complete!")

    # Use the balanced dataset
    df = balanced_df

    print("Preparing data for modeling...")

    # Encode target variable
    le_target = LabelEncoder()
    df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])

    # Define features and target
    X = df.drop(columns=['NObeyesdad', 'Height', 'Weight'])  # Removing height & weight to prevent data leakage
    y = df['NObeyesdad']

    # Get original categories for display
    obesity_categories = le_target.classes_

    # Split dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    numerical_features = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # Train models
    print("Training models...")
    models = {
        "LightGBM": LGBMClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    # Store performance
    model_performance = {}
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_performance[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

    # Select best model
    best_model_name = max(model_performance, key=model_performance.get)
    best_model = trained_models[best_model_name]
    print(f"Best model: {best_model_name} with accuracy {model_performance[best_model_name]:.4f}")

    # Create SHAP explainer
    print("Creating SHAP explainer...")
    explainer = shap.Explainer(best_model, X_train)

    # Create dictionary of all necessary objects
    model_data = {
        "feature_names": X.columns.tolist(),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "label_encoders": label_encoders,
        "le_target": le_target,
        "scaler": scaler,
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "model_performance": model_performance,
        "explainer": explainer,
        "obesity_categories": obesity_categories
    }

    # Save model and related objects
    print(f"Saving model to {MODEL_PATH}...")
    save_folder = "Training Model"
    os.makedirs(save_folder, exist_ok=True)  


    model_path = os.path.join(save_folder, "obesity_model.joblib")
    joblib.dump(best_model, model_path)

    print("Model training and saving complete!")
    return model_data

file_path = "obesity_data.csv"
train_and_save_model(file_path)
