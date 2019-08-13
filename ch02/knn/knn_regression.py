import sys, os
sys.path.append('../../')
import mglearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

def show_predict():
    mglearn.plots.plot_knn_regression(n_neighbors=3)
    plt.show()

def main():
    show_predict()

    X, y = mglearn.datasets.make_wave(n_samples=40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)

    # predict = reg.predict(X_test)

    # 決定係数 R^2スコア
    score = reg.score(X_test, y_test)
    print(score)


if __name__ == '__main__':
    main()
