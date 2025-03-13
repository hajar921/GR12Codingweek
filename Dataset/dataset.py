import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
import shap


df=pd.read_csv(r"C:\Users\Admin\Documents\GitHub\GR12Codingweek\Dataset\ObesityDataSet_raw_and_data_sinthetic.csv")
df.head()