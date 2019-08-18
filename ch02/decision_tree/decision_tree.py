import sys, os
sys.path.append('../../')
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs


def main():
    mglearn.plots.plot_tree_progressive()
    plt.show()


if __name__ == '__main__':
    main()
