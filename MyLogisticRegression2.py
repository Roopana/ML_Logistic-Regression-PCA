#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:14:51 2019

Has imlementation of 'MyLogisticRegression'

@author: vcroopana
"""

import numpy as np
import math

class MyLogisticRegression2:
    
    d = 0
    stepSize = 0
    w = np.zeros([1,d])
    ## consructor param d - no of attributes in given data
    def __init__(self, d):
        self.d = d
        self.stepSize = 0.001 #assume 
        self.w = np.random.uniform(-0.01, 0.01, [1,self.d])
        self.w0 = 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def calculateError(self, X, y):
        error = 0
        deltaE = 0
        deltaEw0 =0
        X = X.values
        y = y.values
        # Calculate Error function value as per log likelihood function for Logistic Regression - 2 classes
        for i in range(X.shape[0]):
            wtX = np.dot(self.w, X[i].T) + self.w0
            currError = (y[i]* wtX) - math.log(1+ np.exp(wtX))
            error = error + currError
            deltaE = deltaE + np.multiply((self.sigmoid(wtX) - y[i]) , X[i])
            deltaEw0 = deltaEw0 + (self.sigmoid(wtX) - y[i])
        error = -error
        return error, deltaE, deltaEw0
        
    def calculateW(self, X, y):
        
        error, deltaE, deltaEw0 = self.calculateError(X, y)
        prevE = 0
        #find W and W0 that give almost 0 change in error function using Gradient Descent approach
        while(abs(prevE-error)> 0.001):  
            prevE = error
            # Update w0 and w class variables in each iteration
            self.w = np.subtract(self.w, self.stepSize * deltaE) 
            self.w0 = np.subtract(self.w0, self.stepSize * deltaEw0)
            # Recalculate error and delta E values using updated values of w and w0
            error, deltaE, deltaEw0 = self.calculateError(X, y)
        
    def fit(self, X,y):
        self.calculateW(X, y)
    
        
    def predict(self,X):
        predictedY = np.zeros([X.shape[0]])
        for row in range(X.shape[0]):
            probability = self.sigmoid(np.dot(self.w, X.values[row].T) + self.w0)
            #prob = self.sigmoid(np.dot(self.w, X.values[row].T))
            if probability> 0.5: 
                predictedY[row] = 1
            else: 
                predictedY[row] = 0
        return predictedY  


