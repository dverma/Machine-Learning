# -*- coding: utf-8 -*-
"""
Polynomial Regression

Created on Thu Feb  1 15:02:45 2018

@author: Dhawal.Verma
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
X_poly = poly_regressor.fit_transform(X)

lin_reg =  LinearRegression()
lin_reg.fit(X_poly,y)

# Visualizing Results for Linear Regression

plt.scatter(X, y, color='red')
plt.plot(X, lin_regressor.predict(X), color='blue')
plt.title('Salary expectations in Linear regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing Results for Polynomial Regression
#X_grid = np.arange(min(X), max(X), 0.5)
#X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(poly_regressor.fit_transform(X)), color='blue')
plt.title('Salary expectations in Polynomial regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Prediction Linear
lin_regressor.predict(6.5)

# Prediction Polynomial
lin_reg.predict(poly_regressor.fit_transform(6.5))