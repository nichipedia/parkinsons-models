import math
import numpy as np
import pandas as pd
from numpy.linalg import pinv, slogdet
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

"""
The below class implements LinearDiscriminant Analysis using numpy.
I used the built in covarience function from numpy to calculate my covarience value.
Inherits from sklearn BaseEstimator. This makes the impl a sklearn estimator.
Allows me to use this class with all of the really nice sklearn tools. Such as cross_val_score!
"""
class LinearDiscriminantAnalysis(BaseEstimator):

    """
    Constructor. Default stuff Ya know?
    """
    def __init__(self, intValue=0, stringParam="Linear Discriminat Analysis Classifier", otherParam=None):
        self.intValue = intValue
        self.stringParam = stringParam
        self.otherParam = otherParam
        self.name = 'Linear Disciminant Analysis Classifier'

    """
    Returns name of this classifier
    """
    def getName(self):
        return self.name

    """
    Takes a matrix and returns the mean of each column.
    It will return a column vector.
    The shape will be [M, 1] where the param X is [N, M]
    """
    def getMu(self, X):
        cols = X.shape[1]
        mu = np.zeros((1, cols))
        for i in range(0, cols):
            mu[0, i] = np.mean(X[:,i])
        return mu.T

    """
    Function to return the covarience matrix of 2 matrices.
    First it concates them into one matrix. Then it computes the matrix!
    """
    def getSig(self, X, Y):
        Z = np.concatenate((X, Y), axis=0)
        return np.cov(Z, rowvar=False)

    """
    Fits Linear discriminate analysis 
    """
    def fit(self, X, Y):
        y = pd.DataFrame(data=Y, columns=['Class'])
        is2 = y['Class'] == 2
        is1 = y['Class'] == 1
        M = X[is2]
        B = X[is1]
        classes2 = M.shape[0]
        classes1 = B.shape[0]
        total = classes2 + classes1
        self.priorM = float(classes2)/total
        self.priorB = float(classes1)/total
        self.muM = self.getMu(M)
        self.muB = self.getMu(B)
        self.cov = self.getSig(M, B)

    """
    Private function to map the predictions to their respective class.
    """
    def _transform(self, k):
        if k < 0:
            return 1
        else:
            return 2

    """
    Function to predict the classes on the variable X.
    This will compute the determinates and then compare them. If sigk2 is greater than sigk1 the class will be 2
    If sigk1 is greater the class will be 1. This will return a vector of predictions.
    """
    def predict(self, X):
        sigk1 = X.dot(pinv(self.cov)).dot(self.muM) - 0.5 * self.muM.T.dot(pinv(self.cov)).dot(self.muM) + np.log(self.priorM)
        sigk2 = X.dot(pinv(self.cov)).dot(self.muB) - 0.5 * self.muB.T.dot(pinv(self.cov)).dot(self.muB) + np.log(self.priorB)
        k = sigk1 - sigk2
        return([self._transform(x) for x in k])
    
    """
    Scoring function. Needed for using the cross_validation framework in sklearn!
    """
    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)

"""
Implementation of Quadratic Disciminant Analysis using numpy.
Yeah, that simple!
Inherits from sklearn BaseEstimator. This makes the impl a sklearn estimator.
Allows me to use this class with all of the really nice sklearn tools. Such as cross_val_score!
"""
class QuadraticDisciminantAnalysis(BaseEstimator):

    
    """
    Constructor. Default stuff Ya know?
    """
    def __init__(self, intValue=0, stringParam="Qudratic Discriminant Analysis Classifier", otherParam=None):
        self.intValue = intValue
        self.stringParam = stringParam
        self.otherParam = otherParam
        self.name = 'Quadratic Discriminant Analysis Classifier'

    """
    Returns name of this classifier
    """
    def getName(self):
        return self.name

    """
    Private function to map the predictions to their respective class.
    """
    def _transform(self, k):
        if k.mean() < 0:
            return 1
        else:
            return 2

    """
    Takes a matrix and returns the mean of each column.
    It will return a column vector.
    The shape will be [M, 1] where the param X is [N, M]
    """
    def __getMu(self, X):
        cols = X.shape[1]
        mu = np.zeros((1, cols))
        for i in range(0, cols):
            mu[0, i] = np.mean(X[:,i])
        return mu.T

    """
    Function to return the covarience matrix of 2 matrices.
    First it concates them into one matrix. Then it computes the matrix!
    """
    def __getSig(self, X):
        return np.cov(X, rowvar=False)

    """
    Fits Quadratic discriminate analysis 
    """
    def fit(self, X, Y):
        y = pd.DataFrame(data=Y, columns=['Class'])
        is2 = y['Class'] == 2
        is1 = y['Class'] == 1
        M = X[is2]
        B = X[is1]
        classes2 = M.shape[0]
        classes1 = B.shape[0]
        total = classes2 + classes1
        self.priorM = float(classes2)/total
        self.priorB = float(classes1)/total
        self.muM = self.__getMu(M)
        self.muB = self.__getMu(B)
        self.covM = self.__getSig(M)
        self.covB = self.__getSig(B)
        self.covMTrue = np.cov(M)
        self.covBTrue = np.cov(B)

    def __getDet(self, X):
        x = slogdet(X)
        return x[0] * np.exp(x[1])

    """
    Function to predict the classes on the variable X.
    This will compute the determinates and then compare them. If sigk2 is greater than sigk1 the class will be 2
    If sigk1 is greater the class will be 1. This will return a vector of predictions.
    """
    def predict(self, X):
        #sigk1 = X.dot(pinv(self.cov)).dot(self.muM) - 0.5 * self.muM.T.dot(pinv(self.cov)).dot(self.muM) + np.log(self.priorM)
        #sigk2 = X.dot(pinv(self.cov)).dot(self.muB) - 0.5 * self.muB.T.dot(pinv(self.cov)).dot(self.muB) + np.log(self.priorB)
        #print(self.__getDet(self.covB))
        a = self.__getDet(self.covMTrue)
        if (a != 0.0):
            a = np.log(abs(a))
        b = self.__getDet(self.covBTrue)
        if (b != 0.0):
            b = np.log(abs(b))

        delta1 = -0.5 * a - 0.5 * (X.T - self.muM).T.dot(pinv(self.covM)).dot(X.T - self.muM) + np.log(self.priorM)

        delta2 = -0.5 * b - 0.5 * (X.T - self.muB).T.dot(pinv(self.covB)).dot(X.T - self.muB) + np.log(self.priorB)
        k = delta1 - delta2
        return([self._transform(x) for x in k])

    """
    Scoring function. Needed for using the cross_validation framework in sklearn!
    """
    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)


"""
Implementation of Logistic Regression using numpy.
Yeah, that simple!
Inherits from sklearn BaseEstimator. This makes the impl a sklearn estimator.
Allows me to use this class with all of the really nice sklearn tools. Such as cross_val_score!
"""
class LogisticRegression(BaseEstimator):

    """
    Constructor. Default stuff Ya know?
    """
    def __init__(self, intValue=0, stringParam="Logistric Regression Classifier", otherParam=None):
        self.intValue = intValue
        self.stringParam = stringParam
        self.otherParam = otherParam
        self.name = 'Logistic Regression Classifier'

    """
    Returns name of this classifier
    """
    def getName(self):
        return self.name

    """
    Function to fit the logistic regression model.
    Produces weights for the estimated function.
    """
    def fit(self, features, truth, threshold=4):
        X = features
        Y = truth
        rows = X.shape[0]
        cols = X.shape[1]
        W = np.identity(rows)
        beta = np.zeros((cols, 1))
        eta = np.zeros((rows, 1))

        for i in range(0, rows):
            x = X[i,:].T
            p = math.exp(x.T.dot(beta))/(1 + math.exp(x.T.dot(beta)))
            eta[i, 0] = p
            p = p*(1-p)
            W[i, i] = p

        z = (X.dot(beta) + pinv(W).dot((Y-eta)))
        beta = pinv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(z)
        self.B = beta
        self.threshold = threshold

    """
    Private function to map the predictions to their respective classes.
    """
    def _transform(self, value):
        if value > self.threshold:
            return 2
        else:
            return 1

    """
    Function to predict the class.
    Returns a vector representing the predictions.
    """
    def predict(self, X, y=None):
        x = X.dot(self.B)
        return([self._transform(value) for value in x])

    """
    Scoring function. Needed for using the cross_validation framework in sklearn!
    """
    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)