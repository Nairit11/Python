# Importing required packages

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Importing the Dataset
data=pd.read_csv('~/Documents/bank/bank.csv',header=0)
data=data.dropna()
#print(data.head())

print(data['education'].unique())