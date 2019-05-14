# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
import os
dataset = pd.read_csv("{}/dataset/ticdata2000.txt".format(os.getcwd()),sep='\s+')
# dataset_eval = pd.read_csv("{}/data/ticeval2000.txt".format(os.getcwd()),sep='\s+')
# dataset_tgts = pd.read_csv("{}/data/tictgts2000.txt".format(os.getcwd()),sep='\s+')

# Merge Religion columns
# 1 - Religious
# 0 - Not Religious
# dataset['RELIGIOUS'] = np.where(
#     (dataset['MGODRK'] + dataset['MGODPR']+ dataset['MGODOV']) > dataset['MGODGE'], 1, 0
# )

# Merge Religion columns
# 1 - Married
# 0 - Not married
dataset['MARRIED'] = np.where(
    (dataset['MRELSA'] + dataset['MRELOV']+ dataset['MFALLEEN']) > dataset['MRELGE'], 0, 1
)

# Merge Religion columns
# 1 - At least a child
# 0 - No child
dataset['HOUSEHOLD_HAS_CHILDREN'] = np.where(
    (dataset['MFGEKIND']) > dataset['MFWEKIND'], 0, 1
)

# Function to pick maximum column value from each row in a dataframe
def merge_max(row_columns):
    a = []
    for row in row_columns:
        a.append(np.where(row == np.max(row))[0][0])
    return a

# Create 'EDUCATION_LEVEL' column with the dominant education level for the customer
# segment.
# 0 - High level education
# 1 - Medium level education
# 2 - Lower level education
education = dataset.iloc[:,15:18].values
dataset['EDUCATION_LEVEL'] = merge_max(education)

# Create 'OCCUPATION_LEVEL' column with the dominant education level for the customer
# segment.
# 0 - High status
# 1 - Entrepreneur
# 2 - Farmer
# 3 - Middle management
# 4 - Skilled labourers
# 5 - Unskilled labourers
occupation = dataset.iloc[:,18:24].values
dataset['OCCUPATION_LEVEL'] = merge_max(occupation)

# Create 'SOCIAL_CLASS' column with dominant social class for the customer
# segment.
# 0 - A
# 1 - B1
# 2 - B2
# 3 - C
# 4 - D
social_class = dataset.iloc[:,24:29].values
dataset['SOCIAL_CLASS'] = merge_max(social_class)

# Create 'HOUSE_OWNERSHIP' column with dominant house ownership for the customer
# segment.
# 0 - Rented House
# 1 - Home Owner
dataset['HOUSE_OWNERSHIP'] = np.where(
    (dataset['MHHUUR']) > dataset['MHKOOP'], 0, 1
)

# Create 'NUMBER_OF_CARS' column with dominant number of cars for the customer
# segment.
# 0 - No Car
# 1 - One Car
# 2 - More than one Car
number_of_cars = pd.concat([dataset.iloc[:,33],dataset.iloc[:,31:33]],axis=1,join='inner')
dataset['NUMBER_OF_CARS'] = merge_max(number_of_cars.values)

# Create 'INSURANCE_TYPE' column with dominant type of insurance for the customer
# segment.
# 0 - National Health Service
# 1 - Private Health Insurance
insurance_type = dataset.iloc[:,34:36].values
dataset['INSURANCE_TYPE'] = merge_max(insurance_type)

# Create 'INCOME' column with dominant income range for the customer
# segment.
# 0 - Income < 30.000
# 1 - Income 30-45.000
# 2 - Income 45-75.000
# 3 - Income 75-122.000
# 4 - Income >123.000
income = dataset.iloc[:,36:41].values
dataset['INCOME'] = merge_max(income)

# Create 'CONTRIBUTION_TYPE' column with dominant type of insurance contribution for the customer
# segment.
# 0 -  Contribution private third party insurance see L4
# 1 -  Contribution third party insurance (firms)
# 2 -  Contribution third party insurance (agriculture)
# 3 -  Contribution car policies
# 4 -  Contribution delivery van policies
# 5 -  Contribution motorcycle/scooter policies
# 6 -  Contribution lorry policies
# 7 -  Contribution trailer policies
# 8 -  Contribution tractor policies
# 9 -  Contribution agricultural machines policies
# 10 -  Contribution moped policies
# 11 -  Contribution life insurances
# 12 -  Contribution private accident insurance policies
# 13 -  Contribution family accidents insurance policies
# 14 -  Contribution disability insurance policies
# 15 - Contribution fire policies
# 16 -  Contribution surfboard policies
# 17-  Contribution boat policies
# 18 - Contribution bicycle policies
# 19 -  Contribution property insurance policies
# 20 -  Contribution social security insurance policies

contribution_type = dataset.iloc[:,43:64].values
dataset['CONTRIBUTION_TYPE'] = merge_max(contribution_type)

# number_of_policies = dataset.iloc[:,64:85].values
# dataset['NUMBER_OF_POLICIES'] = merge_max(number_of_policies)

# Merge all dataframes to create the new dataset
cleaned_dataset = pd.concat([
    dataset.iloc[:,0:5],
    dataset.iloc[:,86:],
    dataset.iloc[:,85]
],axis=1,join='inner')

# Describe the new dataset
a = cleaned_dataset.describe()

# Matrix of features
X = cleaned_dataset.iloc[:,:-1].values
y = cleaned_dataset.iloc[:,-1].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# X_test = dataset_eval.iloc[:,:].values
# y_test = dataset_tgts.iloc[:,:].values

#This data is already label encoded so only HotEncode
# from sklearn.preprocessing import OneHotEncoder
# one_hot_encoder = OneHotEncoder(categorical_features=[0,4,9,15])
# X_train = one_hot_encoder.fit_transform(X_train).toarray()
# X_test = one_hot_encoder.fit_transform(X_test).toarray()

print(np.shape(X_train))
print(np.shape(X_test))

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# # Fit classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

# Predict
y_pred = classifier.predict(X_test)

# Confusion matrix evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
accuracy = (cm[0][0]+cm[1][1])/np.sum(cm)

# K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()