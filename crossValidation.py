#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:46:39 2019

@author: roopana
"""
from sklearn.linear_model import LogisticRegression
from MyLogisticRegression2 import MyLogisticRegression2
import pandas as pd
import numpy as np

# Given list of predicted values, list of test values, returns output as list of values that are misclassified
def getMisClassifiedResponses(y_test, y_pred):
    #Checks if two list are same
    correctlyClassifiedBooleanResponses = y_test == y_pred
    correctlyClassifiedResponses = correctlyClassifiedBooleanResponses*1
    return np.subtract(1, correctlyClassifiedResponses)

# Returns the error rate of test data as per input classifier - method
# input params: method - name of teh classifier, x_train, x_test, y_train, y_test - train & test data sets
# nClasses - no of c;lasses in the data. This is being used only in 'MultiGaussClassify' classifier
def classify(method, x_train, x_test, y_train, y_test, nDim):
    error = 0
    if(method == 'LogisticRegression'):
        logReg = LogisticRegression() 
        logReg.fit(x_train, y_train)
        y_pred = logReg.predict(x_test)
        misClassifiedResponses = getMisClassifiedResponses(y_test, y_pred)
        error = np.mean(misClassifiedResponses)
            
    elif(method =='myLogisticRegression2'):
        myLogisticRegression2 = MyLogisticRegression2(nDim)
        myLogisticRegression2.fit(x_train, y_train)
        y_pred = myLogisticRegression2.predict(x_test)
        misClassifiedResponses = getMisClassifiedResponses(y_test, y_pred)
        error = np.mean(misClassifiedResponses)


    else:
        print('Invalid classifier Type. Only LogisticRegression and myLogisticRegression2 are valid types')
    
    return error

# Inputs : k - No of folds in Cross Validation, method - name of teh classifier, X - Input data and Y - response
# Returns and prints the error of the classifier 'method' for X,Y using Cross Validation
def mycrossval(method,x,y,k, nClasses):
    error = []

    for ithSplit in range(k):
        # create training and testing vars
        x_train, x_test = train_test_split_custom(x, k, ithSplit)
        y_train, y_test = train_test_split_custom(y, k, ithSplit)
        error_i = classify(method, x_train, x_test, y_train, y_test, nClasses)
        print("Error in Fold ",ithSplit, ": {:.2f}".format(error_i ))
        #print("Error in Fold ",ithSplit, ":", error_i )
        error.append(error_i)
                           
    print("Mean error: {:.2f}".format(np.mean(error)))
    print("Standard deviation of error : {:.2f}".format(np.std(error)))
       
    return error

# Before building a classifier model, split the data into two parts: a training set and a test set
# Given an array X and the K folds this function gives the training and test sets

def train_test_split_custom(x, k, ithSplit):
    
    indexSplit = int(x.shape[0]/k) #gives floor of the floating point number x.shape[0]/10
    startIndex_testData = ithSplit*indexSplit; # calculate start Index of test split
    len = x.shape[0]
    endIndex_testData = min(startIndex_testData + indexSplit, len) # calculate end index of test split

    test_data = x.iloc[startIndex_testData:endIndex_testData]

    train_data1 = x.iloc[0:startIndex_testData]
    train_data2 = x.iloc[endIndex_testData:len]
    
    #train data is formed by concatnating above two arrays
    train_data= pd.concat([train_data1, train_data2])

    return train_data, test_data

