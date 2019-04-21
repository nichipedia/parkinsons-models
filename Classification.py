import numpy as np

class LinearDiscriminantAnalysis():

    def getMu(self, X):
        rows = X.shape[0]
        mu = np.zeros((1, rows))
        for i in range(0, rows):
            mu[0, i] = np.mean(X[i,:])
        return mu.T

    def getSig(self, X, Y):
        return np.cov(X, Y)

    #m is 2, b is 1
    def fit(self, X, M, B):
        classes2 = M.shape[0]
        classes1 = B.shape[0]
        total = classes2 + classes1
        priorM = float(classes2)/total
        priorB = float(classes1)/total
        muM = self.getMu(M)
        mu = self.getMu(X)
        muB = self.getMu(B)
        cov = np.cov(X)
        covB = np.cov(B)
        covM = np.cov(M)
        self.sigk = X.T.dot(cov).dot(mu) - 0.5 * mu.T.dot(cov).dot(mu) + np.log(0.5)
        self.sigk1 = X.T.dot(cov).dot(mu) - 0.5 * muM.T.dot(covM).dot(muM) + np.log(priorM)
        self.sigk2 = X.T.dot(cov).dot(mu) - 0.5 * muB.T.dot(covB).dot(muB) + np.log(priorB)
        #self.sigk1 = M.T.dot(covM).dot(muM) - 0.5 * muM.T.dot(covM).dot(muM) + np.log(priorM)
        #self.sigk2 = B.T.dot(covB).dot(muB) - 0.5 * muB.T.dot(covB).dot(muB) + np.log(priorB)


    def predict(self, X):
        x = X.dot(self.sigk)
        k = X.dot(self.sigk1)
        r = X.dot(self.sigk2)
        print('x:{}, k:{}, r:{}'.format(x, k,r))
        #print(r-k)
        if k>=r:
            return 1
        else:
            return 2


