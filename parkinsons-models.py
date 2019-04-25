import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from LogisticRegression import LogisticRegression as LR





X = pd.read_csv('X.csv').values
Y = pd.read_csv('Y.csv').values
lr = LR()
rows = X.shape[0]
lr.fit(X, Y)

predMatrix = np.zeros((rows, 1))

#for i in range(0, rows):
#    predMatrix[i, 0] = lr.predict(X[i, :])
#    print(predMatrix[i, 0])

predMatrix = lr.predict(X[1,:])



#print(accuracy_score(Y, predMatrix))
