import sys, os
sys.path.append('../../')
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    svm = SVC(C=100)
    svm.fit(X_train, y_train)
    print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))


    # preprocessing using 0-1 scaling
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # learning an SVM on the scaled training data
    svm.fit(X_train_scaled, y_train)

    # scoring on the scaled test set
    print("Scaled test set accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))


    # preprocessing using zero mean and unit variance scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # learning an SVM on the scaled training data
    svm.fit(X_train_scaled, y_train)

    # scoring on the scaled test set
    print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))


if __name__ == '__main__':
    main()
