from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

coil = pd.read_csv('{}dataset/coiltrain.txt', delimiter='\s+', encoding="utf-8")
print(coil.describe())

# print(coil[coil.dtypes[(coil.dtypes == "float64") | (coil.dtypes == "int64")].index.values].hist(figsize=[20, 20]))

X = coil.drop("CARAVAN",1)   #Feature Matrix
y = coil["CARAVAN"]          #Target Variable
coil.head()

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (18.0, 20.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
