import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Mock the streamlit module before importing the webapp
sys.modules['streamlit'] = MagicMock()
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Add parent directory to path to import the webapp module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check if shap is imported in the webapp module, if so, mock it
if 'shap' not in sys.modules:
    sys.modules['shap'] = MagicMock()

# Now try importing from APP.webapp
from APP.webapp import encode_user_input, generate_prediction, get_feature_description
# We'll mock perform_shap_analysis later

class TestObesityPredictionApp(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock model and data dictionary
        self.mock_data_dict = {
            "best_model": MagicMock(),
            "best_model_name": "RandomForestClassifier",
            "feature_names": ["Age", "Gender", "family_history_with_overweight", "FAVC", "FCVC", 
                             "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"],
            "numerical_features": ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"],
            "categorical_features": ["Gender", "family_history_with_overweight", "FAVC", "CAEC", 
                                    "SMOKE", "SCC", "CALC", "MTRANS"],
            "label_encoders": {},
            "scaler": MagicMock(),
            "le_target": MagicMock(),
            "obesity_categories": ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", 
                                  "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"],
            "model_performance": {"RandomForestClassifier": 0.92}
        }
        
        # Create mock label encoders for each categorical feature
        for feature in self.mock_data_dict["categorical_features"]:
            mock_encoder = MagicMock()
            mock_encoder.classes_ = ["No", "Yes"] if feature in ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"] else \
                                   ["Never", "Sometimes", "Frequently", "Always"] if feature in ["CAEC", "CALC"] else \
                                   ["Male", "Female"] if feature == "Gender" else \
                                   ["Automobile", "Bike", "Motorbike", "Public Transportation", "Walking"]
            mock_encoder.transform = lambda x, f=mock_encoder.classes_: [f.index(x[0]) if x[0] in f else 0]
            self.mock_data_dict["label_encoders"][feature] = mock_encoder
        
        # Mock the scaler transform method
        self.mock_data_dict["scaler"].transform = lambda x: x
        
        # Mock le_target methods
        self.mock_data_dict["le_target"].inverse_transform = lambda x: ["Normal_Weight"]
        
        # Create a sample user input
        self.user_input = {
            "Age": 30,
            "Gender": "Male",
            "family_history_with_overweight": "No",
            "FAVC": "No",
            "FCVC": 2.0,
            "NCP": 3.0,
            "CAEC": "Sometimes",
            "SMOKE": "No",
            "CH2O": 2.0,
            "SCC": "No",
            "FAF": 1.0,
            "TUE": 1.0,
            "CALC": "Sometimes",
            "MTRANS": "Public Transportation"
        }
        
        # Mock best_model methods
        self.mock_data_dict["best_model"].predict = MagicMock(return_value=np.array([0]))
        self.mock_data_dict["best_model"].predict_proba = MagicMock(return_value=np.array([[0.05, 0.75, 0.1, 0.05, 0.03, 0.01, 0.01]]))

    def tearDown(self):
        """Tear down test fixtures"""
        plt.close('all')  # Close all matplotlib figures

    def test_get_feature_description(self):
        """Test feature description retrieval"""
        self.assertEqual(get_feature_description("Age"), "Patient age in years")
        self.assertEqual(get_feature_description("Gender"), "Patient gender (Male/Female)")
        self.assertEqual(get_feature_description("FAVC"), "Frequent consumption of high caloric food (Yes/No)")
        self.assertEqual(get_feature_description("NonExistentFeature"), "No description available")

    def test_encode_user_input(self):
        """Test encoding of user input"""
        input_df = encode_user_input(self.user_input, self.mock_data_dict)
        
        # Check that the DataFrame has the expected columns
        self.assertEqual(list(input_df.columns), self.mock_data_dict["feature_names"])
        
        # Check that the DataFrame has one row
        self.assertEqual(len(input_df), 1)
        
        # Check that numerical values are preserved
        self.assertEqual(input_df["Age"].values[0], 30)
        self.assertEqual(input_df["FCVC"].values[0], 2.0)

    def test_generate_prediction(self):
        """Test prediction generation"""
        # Encode user input
        input_df = encode_user_input(self.user_input, self.mock_data_dict)
        
        # Generate prediction
        predicted_class, prob_df = generate_prediction(input_df, self.mock_data_dict)
        
        # Check the predicted class
        self.assertEqual(predicted_class, "Normal_Weight")
        
        # Check that prob_df has the right format
        self.assertEqual(list(prob_df.columns), ['Obesity Level', 'Probability'])
        self.assertEqual(len(prob_df), len(self.mock_data_dict["obesity_categories"]))
        
        # Check that probabilities sum to approximately 1
        self.assertAlmostEqual(prob_df['Probability'].sum(), 1.0, places=6)

    def test_feature_validation(self):
        """Test validation of feature values"""
        # Test with an invalid gender value
        invalid_input = self.user_input.copy()
        invalid_input["Gender"] = "InvalidGender"
        
        input_df = encode_user_input(invalid_input, self.mock_data_dict)
        # Should default to first class (typically 0) for invalid categories
        self.assertEqual(input_df["Gender"].values[0], 0)
        
        # Test with missing values
        incomplete_input = {k:v for k,v in self.user_input.items() if k != "FAVC"}
        input_df = encode_user_input(incomplete_input, self.mock_data_dict)
        # Should have a default value for missing features
        self.assertIn("FAVC", input_df.columns)

    def test_numerical_range_validation(self):
        """Test that numerical inputs are correctly processed even when out of typical range"""
        extreme_input = self.user_input.copy()
        extreme_input["Age"] = 95  # Beyond typical range
        extreme_input["FCVC"] = 3.5  # Beyond typical scale
        
        input_df = encode_user_input(extreme_input, self.mock_data_dict)
        
        # Values should be preserved by our mock transform
        self.assertEqual(input_df["Age"].values[0], 95)
        self.assertEqual(input_df["FCVC"].values[0], 3.5)

    # We're not testing the perform_shap_analysis function that requires streamlit

    def test_multiple_predictions(self):
        """Test that the model can handle multiple predictions with different inputs"""
        # Create a second user input with different values
        second_input = {
            "Age": 45,
            "Gender": "Female",
            "family_history_with_overweight": "Yes",
            "FAVC": "Yes",
            "FCVC": 1.5,
            "NCP": 2.0,
            "CAEC": "Frequently",
            "SMOKE": "Yes",
            "CH2O": 1.0,
            "SCC": "Yes",
            "FAF": 0.5,
            "TUE": 1.5,
            "CALC": "Frequently",
            "MTRANS": "Automobile"
        }
        
        # Process first input
        input_df1 = encode_user_input(self.user_input, self.mock_data_dict)
        self.mock_data_dict["best_model"].predict = MagicMock(return_value=np.array([0]))
        self.mock_data_dict["best_model"].predict_proba = MagicMock(return_value=np.array([[0.05, 0.75, 0.1, 0.05, 0.03, 0.01, 0.01]]))
        predicted_class1, prob_df1 = generate_prediction(input_df1, self.mock_data_dict)
        
        # Process second input with different prediction
        input_df2 = encode_user_input(second_input, self.mock_data_dict)
        self.mock_data_dict["best_model"].predict = MagicMock(return_value=np.array([4]))  # Obesity Type I
        self.mock_data_dict["le_target"].inverse_transform = lambda x: ["Obesity_Type_I"]
        self.mock_data_dict["best_model"].predict_proba = MagicMock(return_value=np.array([[0.01, 0.05, 0.09, 0.15, 0.55, 0.1, 0.05]]))
        predicted_class2, prob_df2 = generate_prediction(input_df2, self.mock_data_dict)
        
        # Check that predictions are different
        self.assertEqual(predicted_class1, "Normal_Weight")
        self.assertEqual(predicted_class2, "Obesity_Type_I")
        
        # Check that probability distributions are different
        self.assertNotEqual(prob_df1["Probability"].idxmax(), prob_df2["Probability"].idxmax())


if __name__ == '__main__':
    unittest.main()