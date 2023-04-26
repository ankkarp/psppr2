import numpy as np
import sys

from cure.distance import euclid


class Cluster:
    COUNTER = 0

    def __init__(self, X, x_ids, n_reps, a):
        Cluster.COUNTER += 1
        self.id = Cluster.COUNTER
        self.X = np.array(X)
        self.n_samples, self.n_features = self.X.shape
        self.x_ids = x_ids
        self.n_reps = n_reps
        self.a = a
        self.mean = self.X.sum(axis=0)/(self.n_samples)
        self.reps = self.assign_reps()
        self.closest = None
        self.closest_distance = sys.maxsize

    def assign_reps(self):
        print('Определим точки-представители')
        if self.n_samples <= self.n_reps:
            print(f'Требуемое кол-во представителей не привышает множество точек кластера {self.id}.',
                  f'Значит все точки - представители:', np.around(self.X))
            return self.X
        tmp_set = set()
        reps = []
        print(f'Всего в классе {len(self.X)} точек > {self.n_reps}.')
        print('Значит нужно вычислить точек-представителей для кластера', self.id)
        for i in range(self.n_reps):
            max_dist = 0
            for j in range(self.n_samples):
                if i == 0:
                    min_dist = euclid(self.X[j], self.mean)
                else:
                    min_dist = euclid(reps, self.X[j], axis=1).min()
                if min_dist >= max_dist:
                    max_dist = min_dist
                    max_point = j
            print('Найдена точка-представитель:', np.around(self.X[max_point]))
            if max_point not in tmp_set:
                tmp_set.add(max_point)
                if reps is not None:
                    point = self.X[max_point]
                    reps.append(point + self.a * (self.mean - point))
                else:
                    point = self.X[max_point]
                    reps = [point + self.a * (self.mean - point)]
                print('Точки-представитель',
                      np.around(point), 'сдвигаем к центру:')
                print(f'{point} + {self.a} * ({np.around(self.mean, 5)} - {point})',
                      reps[-1], sep=' = ')
        return np.array(reps)

    def distance_from_point(self, point):
        return euclid(self.reps, point, axis=1).min()
