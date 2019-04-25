import pandas as pd
import numpy as np
from Classification import LinearDiscriminantAnalysis as LDR
from sklearn.metrics import accuracy_score


df = pd.read_csv('parkinsons-data.csv')
X = pd.read_csv('X.csv')
Y = pd.read_csv('Y.csv')



is2 = df['Class'] == 2
is1 = df['Class'] == 1

M = X[is2]
B = X[is1]

ldr = LDR()

ldr.fit(M.values, B.values)


#for i in range(0,190):
#   print(ldr.predict(X.values[i,:]))
pred = ldr.predict(X.values)

print(accuracy_score(Y.values, pred))
