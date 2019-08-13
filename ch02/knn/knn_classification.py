import sys, os
sys.path.append('../../')
import mglearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def show_predict():
    mglearn.plots.plot_knn_classification(n_neighbors=3)
    plt.show()


def main():
    # show_predict()
    X, y = mglearn.datasets.make_forge()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    # predict = clf.predict(X_test)

    score = clf.score(X_test, y_test)
    print(score)


if __name__ == '__main__':
    main()
