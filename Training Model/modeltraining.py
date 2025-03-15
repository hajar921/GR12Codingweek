import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
df.head()

categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


le_target = LabelEncoder()
df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])
X = df.drop(columns=['NObeyesdad', 'Height', 'Weight']) 
y = df['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
numerical_features = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
models = {
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier()}

model_performance = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_performance[name] = accuracy
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)
best_model_name = max(model_performance, key=model_performance.get)
best_model = models[best_model_name]

print(f"Best Model: {best_model_name} with Accuracy: {model_performance[best_model_name]:.4f}")
if best_model_name == "Random Forest":
    feature_importance = best_model.feature_importances_
elif best_model_name == "XGBoost":
    feature_importance = best_model.feature_importances_
elif best_model_name == "LightGBM":
    feature_importance = best_model.feature_importances_

save_folder = "Training Model"
os.makedirs(save_folder, exist_ok=True)  


model_path = os.path.join(save_folder, "obesity_model.joblib")
joblib.dump(best_model, model_path)


import shap
import joblib


explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)


for i in range (shap_values.shape[2]):
    print(f"Shape of SHAP values for class {i}: {shap_values[:,:,i].shape}")
    shap.summary_plot(shap_values[:,:,i], X_test)
    shap.dependence_plot('Age', shap_values[:,:,i], X_test)


MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "label_encoders.pkl"
joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(label_encoders, ENCODERS_PATH)

X.shape
print("X_test columns:", X_test.columns)
    obesity_categories = df['NObeyesdad'].unique().tolist()
    
model_data = { "feature_names": X.columns.tolist(),"numerical_features": numerical_features,"categorical_features": categorical_features,"label_encoders": label_encoders,"le_target": le_target,"scaler": scaler,"trained_models": models,"best_model_name": best_model_name,
 "best_model": best_model,"model_performance": model_performance, "explainer": explainer, "obesity_categories": obesity_categories}

explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test, check_additivity=False)
shap_values = shap_values[0] 
if list(X_test.columns) == list(shap_values.feature_names):
    print("The features in X_test and shap_values match.")
else:
    print("There is a mismatch between X_test and shap_values features.")
    print("X_test columns:", X_test.columns)
    print("SHAP feature names:", shap_values.feature_names)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("Features used in the model:", X_train.columns)
features_used_in_model = ['Age', 'Gender', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 
                          'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']

X_test_filtered = X_test[features_used_in_model]