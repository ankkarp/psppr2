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
        print(f'Инициализируем центроиды:')
        for i, c in enumerate(centroids):
            print(f'C{i} = {c}')
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        print(f'Подсчитаем новые центроиды: ')
        for i in range(self.n_clusters):
            centroids[i] = np.mean(X[labels == i], axis=0)
            print(f'C{i}', end=' = (')
            print(X[labels == i], end=')')
            print(f'/ {len(X[labels == i])} = {centroids[i]}')
        return centroids

    def calculate_metric(self, X, centroids):
        distance_matrix = np.zeros((X.shape[0], self.n_clusters))
        # with tqdm(total=np.prod(distance_matrix.shape)) as pbar:
        for i in range(distance_matrix.shape[1]):
            print(f'Подсчитаем расстояние от центроида {i} {centroids[i]}')
            for j in range(len(X)):
                # pbar.update(1)
                is_verbose = j > (len(X) - 3) or j < 3
                if is_verbose:
                    print(f'X{j}', end=' = ')
                distance_matrix[j, i] = self.metric(
                    X[j], centroids[i], verbose=is_verbose)
                if j == 3:
                    print('...')
        return distance_matrix

    def compute_distance(self, X, centroids):
        # distance = np.empty(len(X))
        # for centroid in range(self.centroids):
        distance = self.calculate_metric(X, centroids)
        return distance

    def determine_cluster(self, distance):
        print('Определим кластер для каждой точки: ')
        labels = np.argmin(distance, axis=1)
        for i, l in enumerate(labels):
            if i > (len(labels) - 3) or i < 3:
                print(f'X{i} принадлежит кластеру {l}')
            elif i == 3:
                print('...')
        return labels

    def fit(self, X):
        self.centroids = self.init_centroids(X)
        for i in tqdm(range(self.max_iter)):
            print(f'Итерация {i}:')
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.determine_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if (old_centroids == self.centroids).all():
                break

    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.determine_cluster(distance)
