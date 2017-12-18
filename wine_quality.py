import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  #helps in model model selection
from sklearn import preprocessing #helps in scaling, transforming and wrangling data
from sklearn.ensemble import RandomForestRegressor #to import rando forest family
from sklearn.pipeline import make_pipeline  #now import tools for cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score #to evaluate our performance
from sklearn.externals import joblib #for future persistance of model

#dataset is loaded
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

#split data into traning sets and test sets
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

#Pre-processing with Transformer API
#scaler = preprocessing.StandardScaler().fit(X_train)
#X_train_scaled = scaler.transform(X_train)
#X_test_scaled = scaler.transform(X_test)  #transforming test set from means of training set

#Instead we will directly model a pipeline like this
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

#Declaring Hyperparameters


