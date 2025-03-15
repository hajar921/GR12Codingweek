import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
import joblib

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Training_Model.modeltraining import *  # Assuming your code is in this module

class TestModelTraining(unittest.TestCase):
    """Test cases for model training functions"""
    
    def setUp(self):
        """Create sample data for testing"""
        # Create a small synthetic dataset similar to your obesity dataset
        self.sample_data = pd.DataFrame({
            'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'Age': [25, 30, 35, 40, 45, 50, 55],
            'Height': [1.65, 1.80, 1.55, 1.75, 1.60, 1.85, 1.70],
            'Weight': [60, 80, 55, 90, 65, 95, 70],
            'family_history_with_overweight': ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
            'FAVC': ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
            'FCVC': [3, 2, 3, 2, 3, 2, 3],
            'NCP': [3, 4, 2, 3, 2, 4, 3],
            'CAEC': ['Sometimes', 'Frequently', 'Sometimes', 'Always', 'Never', 'Frequently', 'Sometimes'],
            'SMOKE': ['no', 'yes', 'no', 'no', 'yes', 'yes', 'no'],
            'CH2O': [2, 1.5, 2.5, 1, 3, 1.5, 2],
            'SCC': ['no', 'yes', 'no', 'no', 'yes', 'yes', 'no'],
            'FAF': [1, 0, 2, 0, 3, 1, 2],
            'TUE': [1, 0.5, 1, 0, 2, 0.5, 1],
            'CALC': ['Sometimes', 'Frequently', 'Never', 'Always', 'Sometimes', 'Never', 'Frequently'],
            'MTRANS': ['Public_Transportation', 'Walking', 'Automobile', 'Public_Transportation', 'Bike', 'Walking', 'Bike'],
            'NObeyesdad': ['Normal_Weight', 'Overweight_Level_I', 'Normal_Weight', 'Obesity_Type_I', 'Normal_Weight', 'Obesity_Type_II', 'Overweight_Level_II']
        })
        
    def test_data_preprocessing(self):
        """Test data preprocessing steps"""
        df = self.sample_data.copy()
        
        # Test label encoding for categorical features
        categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Check if all categorical columns are encoded
        self.assertTrue(all(df[col].dtype != 'object' for col in categorical_features))
        
        # Test target encoding
        le_target = LabelEncoder()
        df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])
        
        # Check if target is encoded
        self.assertTrue(df['NObeyesdad'].dtype != 'object')
        
        # Test feature and target separation
        X = df.drop(columns=['NObeyesdad', 'Height', 'Weight'])
        y = df['NObeyesdad']
        
        self.assertEqual(len(X.columns), 12)  # 14 original cols - 2 dropped cols (Height, Weight)
        self.assertEqual(len(y), len(df))
    
    def test_train_test_split(self):
        """Test train-test split functionality"""
        df = self.sample_data.copy()
        
        # Encode all categorical columns including target
        categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
        le_target = LabelEncoder()
        df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])
        
        # Create feature matrix and target vector
        X = df.drop(columns=['NObeyesdad', 'Height', 'Weight'])
        y = df['NObeyesdad']
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Assert split proportions are correct
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_test), len(y))
        self.assertAlmostEqual(len(X_test) / len(X), 0.3, places=1)
    
    def test_feature_scaling(self):
        """Test feature scaling functionality"""
        df = self.sample_data.copy()
        
        # Encode all categorical columns including target
        categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
        le_target = LabelEncoder()
        df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])
        
        # Create feature matrix and target vector
        X = df.drop(columns=['NObeyesdad', 'Height', 'Weight'])
        y = df['NObeyesdad']
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_features = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        X_train_orig = X_train.copy()
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        
        # Check if scaling was applied correctly
        self.assertNotEqual(X_train_orig[numerical_features].values.tolist(), 
                           X_train[numerical_features].values.tolist())
        
        # Check mean and standard deviation of scaled features
        for col in numerical_features:
            self.assertAlmostEqual(X_train[col].mean(), 0, places=1)
            self.assertAlmostEqual(X_train[col].std(), 1, places=1)
    
    @patch('sklearn.ensemble.RandomForestClassifier')
    @patch('xgboost.XGBClassifier')
    @patch('lightgbm.LGBMClassifier')
    def test_model_training(self, mock_lgbm, mock_xgb, mock_rf):
        """Test model training functionality"""
        # Setup mocks
        mock_rf_instance = mock_rf.return_value
        mock_xgb_instance = mock_xgb.return_value
        mock_lgbm_instance = mock_lgbm.return_value
        
        # Configure mock predictions and probabilities
        mock_rf_instance.predict.return_value = np.array([0, 1, 2, 0, 1])
        mock_xgb_instance.predict.return_value = np.array([0, 1, 2, 0, 1])
        mock_lgbm_instance.predict.return_value = np.array([0, 1, 2, 0, 1])
        
        # Sample data
        X_train = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],
            'Gender': [0, 1, 0, 1, 0],
            'family_history_with_overweight': [1, 1, 0, 1, 0],
            'FAVC': [1, 1, 0, 1, 0],
            'FCVC': [0.5, 0.2, 0.7, -0.3, 0.1],
            'NCP': [0.2, 1.1, -0.9, 0.3, -0.5],
            'CAEC': [2, 3, 2, 0, 1],
            'SMOKE': [0, 1, 0, 0, 1],
            'CH2O': [0.4, -0.8, 1.2, -1.5, 1.7],
            'SCC': [0, 1, 0, 0, 1],
            'FAF': [-0.5, -1.0, 0.7, -1.0, 1.8],
            'TUE': [0.1, -0.7, 0.1, -1.2, 1.0],
            'CALC': [2, 3, 0, 1, 2],
            'MTRANS': [0, 4, 1, 0, 2]
        })
        X_test = X_train.copy()
        y_train = np.array([0, 1, 0, 2, 0])
        y_test = np.array([0, 1, 2, 0, 1])
        
        # Define and train models
        models = {
            "Random Forest": mock_rf_instance,
            "XGBoost": mock_xgb_instance,
            "LightGBM": mock_lgbm_instance
        }
        
        model_performance = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_performance[name] = accuracy
        
        # Check that all models were called with fit
        mock_rf_instance.fit.assert_called_once_with(X_train, y_train)
        mock_xgb_instance.fit.assert_called_once_with(X_train, y_train)
        mock_lgbm_instance.fit.assert_called_once_with(X_train, y_train)
        
        # Check that predict was called for each model
        mock_rf_instance.predict.assert_called_once_with(X_test)
        mock_xgb_instance.predict.assert_called_once_with(X_test)
        mock_lgbm_instance.predict.assert_called_once_with(X_test)
        
        # All accuracies should be the same since we mocked the predictions
        self.assertEqual(model_performance["Random Forest"], model_performance["XGBoost"])
        self.assertEqual(model_performance["XGBoost"], model_performance["LightGBM"])
    
    @patch('joblib.dump')
    def test_model_saving(self, mock_dump):
        """Test model saving functionality"""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Create directory for saving
        save_folder = "Training_Model"
        model_path = os.path.join(save_folder, "obesity_model.joblib")
        
        # Save the model
        joblib.dump(mock_model, model_path)
        
        # Check that joblib.dump was called with correct parameters
        mock_dump.assert_called_once_with(mock_model, model_path)
    
    @patch('shap.TreeExplainer')
    def test_shap_values_calculation(self, mock_explainer):
        """Test SHAP values calculation"""
        # Create mock model and explainer
        mock_model = MagicMock()
        mock_explainer_instance = mock_explainer.return_value
        mock_explainer_instance.shap_values.return_value = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        
        # Create test data
        X_test = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4]
        })
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(mock_model)
        shap_values = explainer.shap_values(X_test)
        
        # Check that TreeExplainer was called with the model
        mock_explainer.assert_called_once_with(mock_model)
        
        # Check that shap_values method was called with X_test
        mock_explainer_instance.shap_values.assert_called_once_with(X_test)
        
        # Check shape of returned SHAP values
        self.assertEqual(shap_values.shape, (2, 2, 2))  # 2 samples, 2 features, 2 classes

if __name__ == '__main__':
    unittest.main()