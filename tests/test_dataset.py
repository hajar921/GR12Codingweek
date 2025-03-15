import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, mock_open
from io import StringIO

# Add parent directory to path to import the dataset module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Dataset.dataset import *  

class TestDatasetPreprocessing(unittest.TestCase):
    """Test cases for dataset preprocessing functions"""
    
    def setUp(self):
        """Create sample data for testing"""
        self.sample_data = pd.DataFrame({
            'Gender': ['Female', 'Male', 'Male', 'Female', 'Female'],
            'Age': [25, 30, 35, 40, 45],
            'Height': [1.65, 1.80, 1.75, 1.60, 1.70],
            'Weight': [60, 80, 75, 65, 70],
            'FAVC': ['yes', 'yes', 'no', 'yes', 'no'],
            'FCVC': [3, 2, 3, 2, 3],
            'NCP': [3, 4, 5, 2, 3],
            'CAEC': ['Sometimes', 'Frequently', 'Sometimes', 'Always', 'Never'],
            'SMOKE': ['no', 'yes', 'no', 'no', 'yes'],
            'CH2O': [2, 1.5, 2.5, 1, 3],
            'SCC': ['no', 'yes', 'no', 'no', 'yes'],
            'FAF': [1, 0, 2, 0, 3],
            'TUE': [1, 0.5, 1, 0, 2],
            'CALC': ['Sometimes', 'Frequently', 'Never', 'Always', 'Sometimes'],
            'MTRANS': ['Public_Transportation', 'Walking', 'Automobile', 'Public_Transportation', 'Bike'],
            'NObeyesdad': ['Normal_Weight', 'Overweight_Level_I', 'Obesity_Type_I', 'Normal_Weight', 'Overweight_Level_II']
        })
        
    def test_check_for_duplicates(self):
        """Test duplicate detection and removal"""
        # Create dataframe with duplicates
        df_with_duplicates = pd.concat([self.sample_data, self.sample_data.iloc[:2]])
        
        # Assert duplicates are detected
        self.assertEqual(df_with_duplicates.duplicated().sum(), 2)
        
        # Test duplicate removal
        df_no_duplicates = df_with_duplicates.drop_duplicates()
        self.assertEqual(df_no_duplicates.duplicated().sum(), 0)
        self.assertEqual(len(df_no_duplicates), 5)
    
    def test_outlier_detection_iqr(self):
        """Test IQR method for outlier detection"""
        # Create dataframe with an outlier
        df_with_outlier = self.sample_data.copy()
        df_with_outlier.loc[2, 'NCP'] = 20  # Extreme NCP value
        
        # Calculate IQR
        Q1 = df_with_outlier['NCP'].quantile(0.25)
        Q3 = df_with_outlier['NCP'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Identify outliers
        outliers = ((df_with_outlier['NCP'] < (Q1 - 1.5 * IQR)) | 
                   (df_with_outlier['NCP'] > (Q3 + 1.5 * IQR)))
        
        # Assert the outlier is detected
        self.assertTrue(outliers.sum() >= 1)
        self.assertTrue(outliers.iloc[2])
    
    def test_capping_values(self):
        """Test capping values to handle outliers"""
        # Create dataframe with outliers
        df_with_outliers = self.sample_data.copy()
        df_with_outliers.loc[0, 'NCP'] = 10  # High value
        df_with_outliers.loc[1, 'NCP'] = 0   # Low value
        
        # Calculate bounds
        Q1 = df_with_outliers['NCP'].quantile(0.25)
        Q3 = df_with_outliers['NCP'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(Q1 - 1.5 * IQR, 1)
        upper_bound = min(Q3 + 1.5 * IQR, 5)
        
        # Cap values
        df_capped = df_with_outliers.copy()
        df_capped['NCP'] = df_with_outliers['NCP'].clip(lower=lower_bound, upper=upper_bound)
        
        # Assert values are capped correctly
        self.assertEqual(df_capped['NCP'].max(), upper_bound)
        self.assertEqual(df_capped['NCP'].min(), lower_bound)
    
    def test_label_encoding(self):
        """Test label encoding of categorical variables"""
        df_copy = self.sample_data.copy()
        categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('NObeyesdad')  # Don't encode target yet
        
        # Apply label encoding
        for col in categorical_cols:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])
        
        # Check that all categorical columns (except target) are now numeric
        self.assertTrue(all(df_copy[col].dtype != 'object' for col in categorical_cols))
        # Check that target is still categorical
        self.assertEqual(df_copy['NObeyesdad'].dtype, 'object')
    
    def test_class_distribution(self):
        """Test class distribution calculation"""
        class_dist = self.sample_data['NObeyesdad'].value_counts(normalize=True) * 100
        
        # Assert class distribution sums to approximately 100%
        self.assertAlmostEqual(class_dist.sum(), 100.0, places=5)
        
        # Check specific classes
        self.assertIn('Normal_Weight', class_dist.index)
        self.assertIn('Overweight_Level_I', class_dist.index)

class TestDatasetBalancing(unittest.TestCase):
    """Test cases for dataset balancing with SMOTE"""
    
    def setUp(self):
        """Create imbalanced sample data for testing"""
        self.imbalanced_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'target': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'C']
        })
        
        # Encode categorical features
        self.le = LabelEncoder()
        self.imbalanced_data['target'] = self.le.fit_transform(self.imbalanced_data['target'])

class TestCorrelationAnalysis(unittest.TestCase):
    """Test cases for correlation analysis"""
    
    def setUp(self):
        """Create sample data with correlations"""
        np.random.seed(42)
        self.correlated_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        
        # Create highly correlated feature
        self.correlated_data['feature4'] = self.correlated_data['feature1'] * 0.9 + np.random.normal(0, 0.1, 100)
    
    def test_correlation_detection(self):
        """Test detection of strong correlations"""
        # Calculate correlation matrix
        correlation_matrix = self.correlated_data.corr()
        
        # Set correlation threshold
        correlation_threshold = 0.8
        
        # Find strong correlations
        strong_correlations = {}
        for col in correlation_matrix.columns:
            strong_corrs = correlation_matrix[col][
                (correlation_matrix[col] > correlation_threshold) & 
                (correlation_matrix[col] < 1)
            ].index.tolist()
            if strong_corrs:
                strong_correlations[col] = strong_corrs
        
        # Check that the correlation between feature1 and feature4 is detected
        self.assertIn('feature1', strong_correlations)
        self.assertIn('feature4', strong_correlations)
        self.assertIn('feature4', strong_correlations['feature1'])
        self.assertIn('feature1', strong_correlations['feature4'])

if __name__ == '__main__':
    unittest.main()