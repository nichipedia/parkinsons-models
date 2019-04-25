import numpy as np
from numpy.linalg import pinv
import math

"""
Implementation of Logistic Regression using numpy.
Yeah, that simple!
"""
class LogisticRegression:

    """
    Function to fit the logistic regression model.
    Produces weights for the estimated function.
    """
    def fit(self, features, truth):
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

    """
    Private function to map the predictions to their respective classes.
    """
    def _transform(self, value):
        if value > 5:
            return 2
        else:
            return 1

    """
    Function to predict the class.
    Returns a vector representing the predictions.
    """
    def predict(self, x):
        X = x.dot(self.B)
        return([self._transform(value) for value in X])
