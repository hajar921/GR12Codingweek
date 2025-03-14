import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv(r"C:\Users\kh\Downloads\obesity.csv")
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