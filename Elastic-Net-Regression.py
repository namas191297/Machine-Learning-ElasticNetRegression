''' Elastic-Net regression can be defined as regularization method that combines both, the ridge regression penalty and
the LASSO penalty which are both also known as L2 norm and L1 norm respectively. It is used generally when there
are tons of parameters that exist in a model and extremely useful when parameters are correlated. It is used in
situations of multicollinearity.
'''

''' Problem : As Elastic-Net is generally implemented datasets containing tens of thousands of parameters. Instead of a 
a given problem, let us assume and generate random values to be used in the model. These values will be divided into 
training and testing data and then we will compare the differences in the co-efficients and errors between
Ridge, LASSO and Elastic-Net.'''


#Importing all the dependencies

import numpy as np
import pandas as pd
from sklearn import *
import random
import matplotlib.pyplot as plt

# Intitializing constant parameters

NO_OF_SAMPLES = 50
N0_OF_FEATURES = 100
TRAIN_SAMPLE_SIZE = int(NO_OF_SAMPLES*(30/100))
TEST_SAMPLE_SIZE = NO_OF_SAMPLES - TRAIN_SAMPLE_SIZE
INDUCE_ERROR = 0.02


#Generating random values

x = np.random.randn(NO_OF_SAMPLES,N0_OF_FEATURES)       # Generating a random array of size [n_samples][n_features] ===> [50][100]
coefficients = 3 * np.random.randn(N0_OF_FEATURES)      # Generating Co-efficients for the fit
indices = np.arange(N0_OF_FEATURES)                     # Intializing Indices, where indices = [1,2,3,4,....,NO_OF_FEATURES]
np.random.shuffle(indices)                              # Randomizing index values, ie. indices = [6,2,15,7,8,.....,RANDOM]
coefficients[indices[15:]] = 0                          # Introducing sparsity in the equation, makes most of the co-efficients 0.
y = np.dot(x,coefficients)                              # Generating the dependent variable values, ie. Y values.

# Adding noise to Y

y += INDUCE_ERROR * np.random.normal(N0_OF_FEATURES)    # Introducing a small amount of deviation in the values of the Y array.

# Splitting data into training and testing

x_train, x_test = np.split(x,[TRAIN_SAMPLE_SIZE,])
y_train, y_test = np.split(y,[TRAIN_SAMPLE_SIZE,])

# Initializing Ridge, LASSO and ElasticNet

ridge = linear_model.Ridge(alpha = 0.1)
lasso = linear_model.Lasso(alpha = 0.1)
e_net = linear_model.ElasticNet(alpha = 0.1, l1_ratio=0.5)

# Training our models

ridge.fit(x_train,y_train)
lasso.fit(x_train,y_train)
e_net.fit(x_train,y_train)

# calculating the r^2 scores for the models
predictions_ridge = ridge.predict(x_test)
predictions_lasso = lasso.predict(x_test)
predictions_enet = e_net.predict(x_test)

score_ridge = metrics.r2_score(y_test,predictions_ridge)
score_lasso = metrics.r2_score(y_test,predictions_lasso)
score_enet = metrics.r2_score(y_test,predictions_enet)

# Displaying the results and the graph

print("R-Squared value for Ridge regression:",score_ridge)
print("R-Squared value for Lasso regression:",score_lasso)
print("R-squared value for Elastic-Net regression:",score_enet)

plt.plot(e_net.coef_, color='blue', linewidth=2,
         label='Elastic-Net Coefficients')
plt.plot(lasso.coef_, color='red', linewidth=2,
         label='Lasso coefficients')
plt.plot(ridge.coef_,color="green",linewidth=2,label="Ridge Coefficients")
plt.plot(coefficients, '--', color='yellow', label='Original Coefficients')
plt.legend(loc='best')
plt.show()

