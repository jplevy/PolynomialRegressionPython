#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Polynomial Regression
"""
Created on Wed Sep  6 17:12:04 2017

@author: juan-pablo
"""
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #Even if we only need one column, X must always be a matrix, so we specify a matrix of 10 rows and a column (1:2 the 2 is left out)
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# Not need here as the dataset is too small
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Feature Scaling
# Not need here because we are going to use the linearRegression who does the scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting linear regression to the dataset (As a reference to judge the polynomial)
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,y)


# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
polynomial_linear_regressor = LinearRegression()
polynomial_linear_regressor.fit(X_poly,y)

# Visualizing the Linear Regression results
plt.scatter(X,y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
X_grid = np.arange(min(X),max(X), 0.01) #For better resolution
X_grid = X_grid.reshape((len(X_grid),1)) # Transform to a matrix (see above)
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, polynomial_linear_regressor.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
linear_regressor.predict(6.5)

# Predicting a new result with Polynomial Regression
polynomial_linear_regressor.predict(poly_reg.fit_transform(6.5))