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
