# 通常最小二乗誤差(OLS)

import sys, os
sys.path.append('../../')
import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def show_predict():
    mglearn.plots.plot_linear_regression_wave()
    plt.show()


def main():
    # show_predict()
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    lr = LinearRegression().fit(X_train, y_train)

    # predict = lr.predict(X_test)

    print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))


if __name__ == '__main__':
    main()
