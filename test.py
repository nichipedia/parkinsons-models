import pandas as pd
import numpy as np
from Classification import LinearDiscriminantAnalysis as LDR


df = pd.read_csv('parkinsons-data.csv')
X = pd.read_csv('X.csv')



is2 = df['Class'] == 2
is1 = df['Class'] == 1

M = X[is2]
B = X[is1]

ldr = LDR()

ldr.fit(X.values, M.values, B.values)

X = X.values

for i in range(0,190):
   print(ldr.predict(X[i,:]))

