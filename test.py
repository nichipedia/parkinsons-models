import pandas as pd
import numpy as np

X = pd.read_csv('./X.csv').values
Y = pd.read_csv('./Y.csv').values

print(np.cov(X))
