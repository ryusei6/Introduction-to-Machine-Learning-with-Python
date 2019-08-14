import sys, os
sys.path.append('../../')
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


def show_predict():
    mglearn.plots.plot_linear_regression_wave()
    plt.show()


def main():
    # show_predict()
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)

    # predict = lasso.predict(X_test)

    print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
    print("Number of features used:", np.sum(lasso.coef_ != 0))


if __name__ == '__main__':
    main()
