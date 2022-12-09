#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:10:04 2019

@author: yangyufeng
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy import stats
import math

my_matrix1=np.array(np.loadtxt(open("K=2000.csv","rb"),delimiter=",",skiprows=0))
#my_matrix2=np.array(np.loadtxt(open("Data_y.csv","rb"),delimiter=",",skiprows=0))
N=2000
v0=110.2133
epsilon=0.3033
#N,v0 & epsilon are same as the previous question.
sigama=0.2928
v1=110.2278
#v1 & sigama are Monte Carlo result for test set, which will be used for generating real value.


class MyNeuralNetwork:
    def __init__(self, input_dimension):
        # define the keras model
        self.model = Sequential()
        self.model.add(Dense(5, input_dim=input_dimension,
                             activation='relu'))
        self.model.add(Dense(5, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        # compile the keras model
        self.model.compile(loss='mean_squared_error', optimizer='adam')

class Scaler:
    # save mean and variance of x, y z sets
    def __init__(self, x, y, z):
        self.x_mean = np.mean(x, axis=0)
        self.y_mean = np.mean(y, axis=0)
        self.z_mean = np.mean(x,axis=0)
        self.x_std = np.std(x, axis=0)
        self.y_std = np.std(y, axis=0)
        self.z_std = np.std(z,axis=0)
        
    

    def get_x(self):
        # return saved mean and variance of x
        return self.x_std, self.x_mean

    def get_y(self):
        # return saved mean and variance of y
        return self.y_std, self.y_mean
    
    def get_z(self):
        return self.z_std, self.z_mean
   


def CreateDataset(N):
    # dataset is regenerated based on 6 features x = [x0; x1; x2; x3; x4; x5] and one output
    dex=np.random.randint(0,N-1,N)
    x1=my_matrix1[0,dex]
    x2=my_matrix1[1,dex]
    x3=my_matrix1[2,dex]
    x4=my_matrix1[3,dex]
    x5=my_matrix1[4,dex]
    x6=my_matrix1[5,dex]
    y=np.random.uniform(v0-epsilon,v0+epsilon,N)
    return np.vstack([x1,x2,x3,x4,x5,x6]).T, y[:, np.newaxis]






if __name__ == "__main__":
    N = 2000 # set size
    x, y = CreateDataset(N)

    ######## dataset visualization ############################
    plt.plot(range(N), x[:,0], 'o', label="x1", markersize=3)
    plt.plot(range(N), x[:,1], 'o', label="x2", markersize=3)
    plt.plot(range(N), x[:,2], 'o', label="x3", markersize=3)
    plt.plot(range(N), x[:,3], 'o', label="x4", markersize=3)
    plt.plot(range(N), x[:,4], 'o', label="x5", markersize=3)
    plt.plot(range(N), x[:,2], 'o', label="x6", markersize=3)
    plt.plot(range(N), y, lw=2, color="red", label="y = genrated_data_set")
    plt.xlabel('data points')
    plt.legend()
    plt.show()
    #########################################################

    ########  Now, all the set was used for training set. 
    idx_train = np.random.choice(np.arange(len(x)), int(N * 1), replace=False)  # indexes included in training set
#    idx_test = np.ones((N,),bool)
#    idx_test[idx_train] = False  # indexes included in the test set
    x_train = x[idx_train]
#   induce the test set.
    test_set = np.transpose(np.array(np.loadtxt(open("50test.csv","rb"),delimiter=",",skiprows=0)))
#   Set the test idx
    test_idx= np.random.choice(np.arange(len(test_set)),50,replace=False)
#   generate the test input
    z=test_set[test_idx]
    y_train = y[idx_train]
    y_test = np.random.uniform(v1-sigama,v1+sigama,50)
    #########################################
    

    neural_network = MyNeuralNetwork(input_dimension=8)  # create a neural network object; 6 is a number of features

    ####### normilize data ######################
    normalizer = Scaler(x_train, y_train, z)
    std_x, mean_x = normalizer.get_x()
    x_train_norm = (x_train - mean_x) / std_x
    std_y, mean_y = normalizer.get_y()
    y_train_norm = (y_train - mean_y) / std_y
    std_z, mean_z = normalizer.get_z()
    x_test_norm = (z - mean_z)/ std_z
    ###########################################
    

    neural_network.model.fit(x_train_norm, y_train_norm, epochs=100, batch_size=9)  # train neural network

    y_from_nn_norm = neural_network.model.predict(x_test_norm)  # predict values for x_test states
    y_from_nn = y_from_nn_norm * std_y + mean_y  # tranform the results into original scaling

    mse = mean_squared_error(y_test, y_from_nn) # compute mean squared error
    print('Mean squared error: ', mse)


    ### compare true and predicted values of y from the test set###
    plt.plot(y_test, label="y-original", lw=2)
    plt.plot(y_from_nn, label="y-predicted", lw=2)
    plt.xlabel('test data points')
    plt.legend()
    plt.show()
    #################################################################


