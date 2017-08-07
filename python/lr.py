# from sklearn import datasets
# from sklearn.model_selection import cross_val_predict
# from sklearn import linear_model
# import matplotlib.pyplot as plt

# lr = linear_model.LinearRegression()
# boston = datasets.load_boston()
# print(len(boston.data))
# y = boston.target

# # cross_val_predict returns an array of the same size as `y` where each entry
# # is a prediction obtained by cross validation:
# predicted = cross_val_predict(lr, boston.data, y, cv=10)

# fig, ax = plt.subplots()
# ax.scatter(y, predicted)
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()


print(__doc__)
import random

# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

X = []
num_features = 100
num_samples = 100000
thetas = [] 
x = []
y = [] 

for i in range(0,num_features+1):
    thetas.append(round( random.random() * 10  , 4)) 

for i in range(0,num_samples):
    _x = []
    for k in range(0,num_features):
        _x.append(round( random.random() * 10  , 4))
    x.append(_x)

    _y = 0
    for j in range(0,num_features):
        _y += thetas[j] * _x[j]
    
    _y += thetas[num_features]

    y.append(_y) 


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)

regr.fit(x, y)

print('THETAS: \n', thetas)

# The coefficients
print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# # Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
#          linewidth=3)

# plt.xticks(())
# plt.yticks(())

plt.show()