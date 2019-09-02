import sys, os
sys.path.append('../../')
import pandas as pd
import mglearn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split





def main():
    # The file has no headers naming the columns, so we pass header=None
    # and provide the column names explicitly in "names"
    adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
    data = pd.read_csv(
        adult_path, header=None, index_col=False,
        names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'gender',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
               'income'])
    # For illustration purposes, we only select some of the columns
    data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
                 'occupation', 'income']]
    # IPython.display allows nice output formatting within the Jupyter notebook
    display(data.head())


if __name__ == '__main__':
    main()
