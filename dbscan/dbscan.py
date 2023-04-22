import random

import numpy as np
import matplotlib.pyplot as plt


class DBSCAN:
    """
    Кластеризатор алгоритмом DBSCAN
    """

    def euclid(self, x):
        distance = np.sqrt(np.sum((self.X - x)**2, axis=1))
        if self.visited.sum() == 1:
            for xi, d in zip(self.X, distance):
                print(
                    f'euclid({np.around(xi, 5)}, {np.around(x, 5)}) = {round(d, 5)}', end=' ')
                print('<=' if d <= self.eps else '>', end=' ')
                print(self.eps)
        return distance

    def get_neighbors(self, x):
        distances = self.euclid(x)
        neighbors = np.where(distances <= self.eps)[0]
        return neighbors

    def fit(self, X, eps, min_sz, seed=None):
        random.seed(seed)
        n_samples = X.shape[0]
        self.visited = np.zeros(n_samples, dtype=bool)
        self.labels = np.full(n_samples, -1, dtype=int)
        self.eps = eps
        self.min_sz = min_sz
        self.X = X
        label = 0
        print(f'Пусть:\nМинимальное расстояние между точками eps = {eps}')
        print(f'Минимальный размер кластера min_sz = {min_sz}')
        print(f'Изначально все {len(X)} точки считаются шумом.')
        while not self.visited.all():
            print(f'Начнем поиск {label}-ого кластера:')
            i = random.choice(np.where(np.logical_not(self.visited))[0])
            print(
                f'Возьмем случайную точку из непосещенных и отметим ее посещенной: x{i} = {np.around(self.X[i], 5)}')
            self.visited[i] = True
            neighbors = self.get_neighbors(X[i])
            print(f'В радиусе {self.eps} найдено {len(neighbors)} точек:')
            if len(neighbors) >= min_sz:
                print(
                    f'Это больше минимального размера кластера ({self.min_sz}). Значит считаем точку корневой и попробуем расширить кластер {label}')
                self.labels[i] = label
                self.expand_cluster(neighbors, label)
                label += 1
                print('обход в окрестности завершен, перейдем к следующему кластеру')
            else:
                print(
                    'Кол-во соседей меньше минимального размера, значит это не корневая точка.')
        return self.labels

    def expand_cluster(self, neighbors, label):
        for neighbor in neighbors:
            if not self.visited[neighbor]:
                print(
                    f'Возьмем точку из текущего кластера x{neighbor} = {np.around(self.X[neighbor], 5)} к кластеру и посмотрим достаточно ли точек в ее окрестности чтобы она стала корневой')
                self.visited[neighbor] = True
                self.labels[neighbor] = label
                new_neighbors = self.get_neighbors(self.X[neighbor])
                if len(new_neighbors) >= self.min_sz:
                    print(
                        f'Точек в окрестности достаточно ({len(new_neighbors)} >= {self.min_sz}), значит расширяем кластер ее соседями')
                    neighbors = np.union1d(neighbors, new_neighbors)
                    print(
                        f'Теперь кластер {label} состоит из {len(neighbors)} точек')
                    self.expand_cluster(neighbors, label)
                else:
                    print(
                        f'Точек недостаточно ({len(new_neighbors)} < {self.min_sz}) => кластер расширить не можем.')

    def visualize(self):
        clusters = np.unique(self.labels)
        colors = set()
        while len(colors) < len(clusters):
            colors.add(
                "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))
        for label in clusters:
            mask = self.labels == label
            if label == -1:
                plt.scatter(*self.X[mask].T, c='0.5', label='шум')
            else:
                plt.scatter(*self.X[mask].T, c=colors.pop(), label=str(label))
        plt.legend()
        plt.show()
