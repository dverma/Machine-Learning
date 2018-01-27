#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:08:42 2018

@author: dhawal
"""

#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


#Importing the dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Take care of missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()
labelencoder_x.fit_transform(x[:, 0])
