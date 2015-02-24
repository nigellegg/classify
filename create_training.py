import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC


def createtraining(textfile):
    textfield = input('Enter the name of the textfield...      ')
    catfield = input('Enter the name of the category field... ')
    cat1 = input('Enter category1 label:   ')
    cat2 = input('Enter category2 label:   ')
    nocats = input('Enter number of training examples to create....  ')
    print('Enter 1 or 2 for each text')
    df = pd.read_csv(textfile)
    xcat1 = 0
    xcat2 = 0
    x = len(df)
    i = 0
    while i < nocats:
        dfx = df.ix[np.random.random_integers(0, x, 1)]
        text = dfx[textfield]
        print('cat1 = ', xcat1)
        print('cat2 = ', xcat2)
        print(text)
        cat = input('Enter category....')
        if cat == 1:
            dfx[catfield] = cat1
            xcat1 += xcat1
        else:
            dfx[catfield] = cat2
            xcat2 += xcat2
        i += 1
    df.to_csv('training_set.csv')


if __name__ == '__main__':
    text = input('Enter name of text file - *.csv ...  ')
    createtraining(text)
