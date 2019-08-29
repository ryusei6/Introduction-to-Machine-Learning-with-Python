from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np



def main():
    X, y = make_blobs(random_state=1)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    print("Cluster memberships:\n{}".format(kmeans.labels_))
    print("predict:\n{}".format(kmeans.predict(X)))


if __name__ == '__main__':
    main()
