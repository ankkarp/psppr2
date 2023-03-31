import numpy as np
from tqdm import tqdm


class Kmeans:
    def __init__(self, n_clusters, metric, max_iter=100):
        self.metric = metric
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def _distance(self):
        return np.cos()

    def init_centroids(self, X):
        np.random.seed()
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            centroids[i] = np.mean(X[labels == i], axis=0)
        print(centroids)
        return centroids

    def calculate_metric(self, X, centroids):
        distance_matrix = np.zeros((X.shape[0], self.n_clusters))
        # with tqdm(total=np.prod(distance_matrix.shape)) as pbar:
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[1]):
                # pbar.update(1)
                print(f'x{i+1}{j+1}', end=' = ')
                distance_matrix[i, j] = self.metric(X[i], centroids[j])
        return distance_matrix

    def compute_distance(self, X, centroids):
        # distance = np.empty(len(X))
        # for centroid in range(self.centroids):
        distance = self.calculate_metric(X, centroids)
        return distance

    def determine_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def fit(self, X):
        self.centroids = self.init_centroids(X)
        for _ in tqdm(range(self.max_iter)):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.determine_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if (old_centroids == self.centroids).all():
                break

    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.determine_cluster(distance)
