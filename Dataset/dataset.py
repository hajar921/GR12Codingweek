import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
import shap


df=pd.read_csv(r"C:\Users\kh\Downloads\obesity.csv")
df.head()