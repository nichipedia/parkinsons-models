import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from Classification import LogisticRegression as LR
from Classification import LinearDiscriminantAnalysis as LDA
from Classification import QuadraticDisciminantAnalysis as QDA
from sklearn.model_selection import cross_val_score

# Do the 10 Fold cross validation and print the results!
def printCrossValScores(clf, X, Y):
    print('Starting Cross Validation for {}'.format(clf.getName()))
    scores = cross_val_score(clf, X, Y, cv=10)
    for i in range(0, 10):
        print('CV{} accuracy score: {}, Error Rate: {}'.format(i+1, scores[i], 1 - scores[i]))
    print('----------------------------------------------------------------------------')
    print('10 Fold CV Error Percentage Mean: {}'.format(1 - scores.mean()))
    print('')

# Load features and truth
X = pd.read_csv('X.csv').values
Y = pd.read_csv('Y.csv').values

# Init classifiers
lr = LR()
lda = LDA()
qda = QDA()

# Do 10 fold cross validation and print results
printCrossValScores(lr, X, Y)
printCrossValScores(lda, X, Y)
printCrossValScores(qda, X, Y)
