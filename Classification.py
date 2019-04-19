import numpy as np

class LinearDiscrimentAnalysis():

    def __getMu(X):
        cols = X.shape[1]
        mu = np.zeros((1, cols))
        for i in range(0, cols):
            mu[0, i] = np.mean(X[:,i])
        return mu.T

    def __getSig(X, Y):
        return np.cov(X, Y)




class QuadraticDiscriminentAnalysis():

