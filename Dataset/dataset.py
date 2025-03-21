import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
import shap

df=pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
df.head()
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

from scipy.stats import zscore 
z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).sum(axis=0)
print("Nombre d'outliers detectes par case :")
print(outliers)
duplicates = df[df.duplicated(keep=False)]

print("Lignes dupliquées :")
print(duplicates)
for col in df.columns:
    duplicate_col = df.duplicated(subset=[col], keep=False)
    if duplicate_col.any():
        print(f"Doublons détectés dans la colonne: {col}")
        print(df.loc[duplicate_col, [col]])
df_numeric = df.select_dtypes(include=['number'])


Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1


outliers_iqr = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()

print("Outliers détectés par la méthode IQR :")
print(outliers_iqr)

df = df.drop_duplicates()
print(df.shape)  
Q1_NCP = df['NCP'].quantile(0.25)
Q3_NCP = df['NCP'].quantile(0.75)
IQR_NCP = Q3_NCP - Q1_NCP
lower_bound = max(Q1_NCP - 1.5 * IQR_NCP, 1) 
upper_bound = min(Q3_NCP + 1.5 * IQR_NCP, 5) 
df_capped = df.copy()
df_capped['NCP'] = df['NCP'].clip(lower=lower_bound, upper=upper_bound)  
df_filtered = df[(df['NCP'] >= 1) & (df['NCP'] <= 5)]

class_distribution = df['NObeyesdad'].value_counts(normalize=True) * 100
print(class_distribution)  

from imblearn.over_sampling import SMOTE
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('NObeyesdad') 

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le 
target_col = "NObeyesdad"
X = df.drop(columns=[target_col])
y = df[target_col]
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
from collections import Counter
print("New class distribution:", Counter(y_resampled))


df_numeric = df.select_dtypes(include=['number'])
correlation_matrix = df_numeric.corr()

correlation_threshold = 0.8
strong_correlations = {}
for col in correlation_matrix.columns:
    strong_corrs = correlation_matrix[col][(correlation_matrix[col] > correlation_threshold) & (correlation_matrix[col] < 1)].index.tolist()
    if strong_corrs:
        strong_correlations[col] = strong_corrs
if strong_correlations=={}:
    print("no strong correlation detected")
def optimize_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_object_dtype(col_type):
            df[col] = df[col].astype('category')
    
    return df

def demonstrate_memory_optimization(df):
    before_memory = df.memory_usage(deep=True)
    print("Memory usage before optimization:")
    print(before_memory)
    df_optimized = optimize_memory(df)
    after_memory = df_optimized.memory_usage(deep=True)
    print("\nMemory usage after optimization:")
    print(after_memory)
    return before_memory, after_memory
before_memory, after_memory = demonstrate_memory_optimization(df)

before_memory, after_memory

import pandas as pd

df.to_csv("obesity_data.csv", index=False)  # Save data as CSV
print("Dataset saved: 'obesity_data.csv'")