# -*- coding: utf-8 -*-
"""
Simple Linear Regression

Created on Mon Jan 29 13:10:41 2018

@author: Dhawal.Verma
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Exp (Training Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()


# Visualising the test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Exp (Test Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()
