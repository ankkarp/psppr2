import numpy as np
import matplotlib.pyplot as plt


class MST:
    def euclid(self, x, y):
        distance = np.sqrt(np.sum((x - y)**2, axis=1))
        for xi, d in zip(x, distance):
            print(
                f'euclid({np.around(xi, 5)}, {np.around(y, 5)}) = {round(d, 5)}')
        return distance

    def prima(self, dots, start=None):
        # weights = self.euclid_matrix(dots, dots)
        n = dots.shape[0]
        # visited = np.zeros(n, dtype=bool)
        if start is None:
            start = np.random.randint(low=0, high=n)
        current = dots[start]
        unvisited = np.delete(dots, start, 0)
        path = np.array([current])
        print(f'Пусть граф состоит из точек: {np.around(dots, 5)}')
        print(f'Пусть начальная точка: {np.around(current, 5)}')
        print('Все точки могут быть соединены. Веса границ - евклидово расстояние.')
        total_weight = 0
        while len(path) < n:
            weights = self.euclid(unvisited, current)
            next_idx = np.argmin(weights)
            total_weight += weights[next_idx]
            current = unvisited[next_idx]
            print(
                f'Наиблизжайшая точка - {current}. перейдем в нее и продолжим обход.')
            path = np.vstack((path, current))
            unvisited = np.delete(unvisited, next_idx, 0)
        plt.scatter(*dots.T)
        plt.plot(*path.T)
        return path, total_weight
