# Importing the libraries
import numpy as np  # Mathematical functions
import matplotlib.pyplot as plt  # Plot graphs
import pandas as pd  # Import and manage datasets

# Import the dataset
import os
dataset = pd.read_csv("{}/data/ticdata2000.txt".format(os.getcwd()),sep='\t', header=None)
dataset_eval = pd.read_csv("{}/data/ticeval2000.txt".format(os.getcwd()),sep='\t', header=None)
dataset_tgts = pd.read_csv("{}/data/tictgts2000.txt".format(os.getcwd()),sep='\t', header=None)

# Matrix of features
X_train = dataset.iloc[:,:-1].values
y_train = dataset.iloc[:,-1].values

X_test = dataset_eval.iloc[:,:].values
y_test = dataset_tgts.iloc[:,:].values

# Only hot encode
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(categorical_features=[0,4])
X_train = one_hot_encoder.fit_transform(X_train).toarray()
X_test = one_hot_encoder.fit_transform(X_test).toarray()

# Fitting XGBoost to the Training Set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)


# Predicting the test set results
y_pred = classifier.predict(X_test)


# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
accuracy = (cm[0][0]+cm[1][1])/np.sum(cm)

# K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()