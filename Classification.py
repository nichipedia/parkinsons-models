import numpy as np
from numpy.linalg import pinv

"""
The below class implements LinearDiscriminant Analysis using numpy.
I used the built in covarience function from numpy to calculate my covarience value.
"""
class LinearDiscriminantAnalysis():

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

    def fit(self, M, B):
        classes2 = M.shape[0]
        classes1 = B.shape[0]
        total = classes2 + classes1
        self.priorM = float(classes2)/total
        self.priorB = float(classes1)/total
        self.muM = self.getMu(M)
        self.muB = self.getMu(B)
        self.cov = getSig(M, B)

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
