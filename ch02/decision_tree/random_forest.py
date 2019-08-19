import sys, os
sys.path.append('../../')
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)

    forest = RandomForestClassifier(n_estimators=100, random_state=0)
    forest.fit(X_train, y_train)

    print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))


if __name__ == '__main__':
    main()
