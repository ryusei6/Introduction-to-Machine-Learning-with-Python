from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans




def main():
    X, y = make_blobs(random_state=1)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

if __name__ == '__main__':
    main()
