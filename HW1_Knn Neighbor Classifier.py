# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
import pylab as pl

# Load training dataset.
training_data = np.genfromtxt("H1E_training.csv",delimiter=",",names=True)
# Convert training data whose type is tuple to numpy array. 
td_array = np.array([(a,b,c) for (a,b,c) in training_data])
# Slice training data; first two columns as inputs; third column as output.
x1 = td_array[:,0:2]
y1 = td_array[:,2:3]
# Convert the output array into a flattened array.
Y1 = y1.ravel()

# Process validation dataset; Follow the same steps of loading,converting,slicing
# for processing training dataset.
validation_data = np.genfromtxt("H1E_validation.csv",delimiter=",",names=True)
vd_array = np.array([(a,b,c) for (a,b,c) in validation_data])
x2 = vd_array[:,0:2]
y2 = vd_array[:,2:3]
Y2 = y2.ravel()

# Create empty lists for storing the number of errors of different K values for 
# training and validation data. 
Error_t = []   # number of errors for training data
Error_v = []     # number of errors for validation data

# Learn the nearest neighbor classifier model for K = 1,2,...20.
for n_neighbors in range(1,21):
    # Create an instance of Neighbours Classifier and fit the data.    
    neigh = neighbors.KNeighborsClassifier(n_neighbors)
    neigh.fit(x1,Y1) 
    
    # Calculate the number of errors for both training data and validation data
    # given the result of neighbor classifier of a given K value.
    
    n_t = 0 
    n_of_error_t = 0 
    # Iterate over training examples; if the predicted outputs run by the model
    # are not equal to the provided outputs,then increment the number of errors
    # by 1.
    for m in x1:
        if neigh.predict(m) == Y1[n_t]:
            n_of_error_t = n_of_error_t
        else:
            n_of_error_t = n_of_error_t + 1
        n_t = n_t+1
    # Append the number of errors of using a given K value to the total list of 
    # number of errors.
    Error_t.append(n_of_error_t)    
    
    n = 0
    n_of_error = 0
     # Iterate over validation examples; if the predicted outputs run by the model
    # are not equal to the provided outputs,then increment the number of errors
    # by 1.
    for m in x2:
        if neigh.predict(m) == Y2[n]:
            n_of_error = n_of_error
        else:
            n_of_error = n_of_error + 1
        n = n+1
    # Append the number of errors of using a given K value to the total list of 
    # number of errors.
    Error_v.append(n_of_error)

# Print the lists of number of errors of different K values for 
# training and validation data.    
print Error_t, Error_v

# Calculate the rate of error for both training and validation data.
n_of_t = float(x1.shape[0])
n_of_v = float(x2.shape[0])
Rate_of_Error_t = [round(x / n_of_t,3) for x in Error_t]
Rate_of_Error_v = [round(x / n_of_v,3) for x in Error_v]
print Rate_of_Error_t,Rate_of_Error_v 

# Plot learning curves: training error and validation error against K.
K = list(range(1,21))
plt.plot(K,Rate_of_Error_t,color = 'blue',linewidth=2.0,label="training examples")
plt.plot(K,Rate_of_Error_v,color = 'red',linewidth=2.0,label="validation examples")
plt.legend(loc='upper right',frameon=False)
plt.show()

#Process test dataset; Follow the same steps of loading,converting,slicing
#for processing training and validation dataset.
test_data = np.genfromtxt("H1E_test.csv",delimiter=",",names=True)
test_array = np.array([(a,b,c) for (a,b,c) in test_data])
x3 = test_array[:,0:2]
y3 = test_array[:,2:3]
Y3 = y3.ravel()

# Selecting K = 3 which works best based on the learning curves.
K=3
# Create an instance of Neighbours Classifier and fit the test data.
knn = neighbors.KNeighborsClassifier(K)
knn.fit(x3,Y3)

h = .02 # step size in the mesh.

# Plot the decision boundary. Assign a color to each
# point in the mesh [x_min, m_max]*[y_min, y_max].
x_min, x_max = x3[:,0].min() - .5, x3[:,0].max() + .5
y_min, y_max = x3[:,1].min() - .5, x3[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot.
Z = Z.reshape(xx.shape)
pl.figure(1, figsize=(4, 3))
pl.set_cmap(pl.cm.Paired)
pl.pcolormesh(xx, yy, Z)

# Plot the test data points over the color plot.
pl.scatter(x3[:,0], x3[:,1],c=Y3 )
pl.xlabel('Longtitude')
pl.ylabel('Latitude')

# Set the limits and ticks of x,y axis.
pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.xticks(())
pl.yticks(())

pl.show()

#Reference:
#http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/tutorial/plot_knn_iris.html







