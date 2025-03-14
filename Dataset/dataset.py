import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
import shap

df=pd.read_csv(r"C:\Users\Admin\Documents\GitHub\GR12Codingweek\Dataset\ObesityDataSet_raw_and_data_sinthetic.csv")
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

correlation_threshold = 0.8
strong_correlations = {}
for col in correlation_matrix.columns:
    strong_corrs = correlation_matrix[col][(correlation_matrix[col] > correlation_threshold) & (correlation_matrix[col] < 1)].index.tolist()
    if strong_corrs:
        strong_correlations[col] = strong_corrs
strong_correlations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import pandas as pd

df=pd.read_csv(r"C:\Users\kh\Downloads\obesity.csv")
df.head()

X = df.drop(columns=['NObeyesdad']).copy()  
y = df['NObeyesdad']
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col]) 
    label_encoders[col] = le  
scaler = StandardScaler()
X.loc[:, X.select_dtypes(include=['number']).columns] = scaler.fit_transform(X.select_dtypes(include=['number']))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train.shape, X_test.shape, y_train.shape, y_test.shape
