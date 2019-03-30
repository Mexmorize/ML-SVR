# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:55:59 2019

@author: Benjamin B
"""

# SVR

# REGRESSION TEMPLATE

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values # Need range to make x a matrix, not a vector
y = dataset.iloc[:, 2:].values

# Splitting the dataset into Training and Testing sets
""" Only really need if there is a sufficient amount of observations**

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

# Visualizing the SVR results
plt.scatter(x, y, color = 'red') # Real data POINTS
plt.plot(x, regressor.predict(x), color = 'blue') # Prediction LINE
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the SVR results (FOR HIGHER RESOLUTION)
x_grid = np.arange(min(x), max(x), 0.1) # Creates Vector, but need matrix
x_grid = x_grid.reshape((len(x_grid), 1)) # Reshape vector into matrix
plt.scatter(x, y, color = 'red') # Real data POINTS
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue') # Prediction LINE
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()