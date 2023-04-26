import sys
import random
import numpy as np

import matplotlib.pyplot as plt

from cure.cluster import Cluster
from cure.distance import euclid


class CURE:

    def __init__(self, n_clusters, n_reps, a):
        self.n_clusters = n_clusters
        self.clusters = []
        self.n_reps = n_reps
        self.a = a

    def fit(self, X):

        print('Дано:')
        print('Кол-во кластеров:', self.n_clusters)
        print('Кол-во точек представителей в каждлм кластере:', self.n_reps)
        print('Коэффициент сжатия к центроиду кластера:', self.a)
        print('Данные на кластеризацию:')
        print(X)
        print('Для каждой точки инициализируем свой кластер')
        for i_point, point in enumerate(X):
            self.clusters.append(
                Cluster([point], [i_point], self.n_reps, self.a))
        for index in range(len(self.clusters)):
            self.assign_closest(self.clusters[index])
        i = 1

        while len(self.clusters) > self.n_clusters:
            print(
                f'Текущее кол-во кластеров {len(self.clusters)} > {self.n_clusters}')
            print('Значит продолжаем поиск')
            print(f'Итерация {i}:')
            print('Найдём кластер, который можно обьединить с соседом')
            mergable_cluster_i = self.get_mergable()
            cluster1 = self.clusters.pop(mergable_cluster_i)
            cluster2 = cluster1.closest
            self.discard_cluster(cluster2)
            new_cluster = self.merge(cluster1, cluster2)
            print(f'Получаем новый кластер {new_cluster.id}')
            self.reassign_closest(
                new_cluster, cluster1.id, cluster2.id)
            self.clusters.append(new_cluster)
            self.visualize(i, save=True)
            i += 1
        print('Нужное кол-во кластеров было достигнуто.')
        print('Итоговое разделение на кластеры:')
        for cl in self.clusters:
            print(f'{cl.id}) {np.around(cl.X, 5)}')

    def merge(self, cluster1, cluster2):

        if cluster1.n_features != cluster2.n_features:
            raise ValueError('Ошибка! Измерения точек не совпадают!')
        combined_X = np.concatenate((cluster1.X, cluster2.X))
        combined_x_ids = cluster1.x_ids + cluster2.x_ids
        new_cluster = Cluster(combined_X, combined_x_ids,
                              cluster1.n_reps, cluster1.a)
        return new_cluster

    def assign_closest(self, cluster):
        min_dist = sys.maxsize
        closest = None
        for i in range(len(self.clusters)):
            if cluster.id == self.clusters[i].id:
                continue
            dist = self.reps_distance(cluster, self.clusters[i])
            if dist < min_dist:
                min_dist = dist
                closest = self.clusters[i]
        cluster.closest = closest
        cluster.closest_distance = min_dist

    def get_mergable(self):
        min_dist = sys.maxsize
        i_cluster = 0
        for i in range(len(self.clusters)):
            dist = self.clusters[i].closest_distance
            if dist < min_dist:
                min_dist = dist
                i_cluster = i
        print(f'Соединим кластер {self.clusters[i_cluster].id}',
              f'с соседом {self.clusters[i_cluster].closest.id}',
              'тк расстояние между близжайшими их точками-представителями',
              'миниммальна и равна', round(min_dist[0], 5))

        return i_cluster

    def discard_cluster(self, cluster):
        for i in range(len(self.clusters)):
            if self.clusters[i].id == cluster.id:
                self.clusters[i] = self.clusters[-1]
                self.clusters.pop()
                return
        raise Exception("Кластер не найден")

    def reassign_closest(self, new_cluster, cluster1_id, cluster2_id):
        for i in range(len(self.clusters)):
            new_dist = self.reps_distance(new_cluster, self.clusters[i])
            if new_cluster.closest_distance > new_dist:
                new_cluster.closest_distance = new_dist
                new_cluster.closest = self.clusters[i]
            if self.clusters[i].closest.id in (cluster1_id, cluster2_id):
                if self.clusters[i].closest_distance < new_dist:
                    self.assign_closest(self.clusters[i])
                else:
                    self.clusters[i].closest_distance = new_dist
                    self.clusters[i].closest = new_cluster
            else:
                if self.clusters[i].closest_distance > new_dist:
                    self.clusters[i].closest_distance = new_dist
                    self.clusters[i].closest = new_cluster

    def reps_distance(self, cluster1, cluster2):
        distances = []
        for rep1 in cluster1.reps:
            dist = euclid(cluster2.reps, rep1, axis=1)
            distances.append(dist)
        return min(distances)

    def test(self, X):
        predictions = {cluster.id: [] for cluster in self.clusters}
        for i in range(len(X)):
            label = self.predict(X[i])
            predictions[label].append(i)
        return predictions.values()

    def predict(self, x):
        x = np.array(x)
        min_dist = sys.maxsize
        label = None
        for i in range(len(self.clusters)):
            dist = self.clusters[i].distance_from_point(x)
            if dist < min_dist:
                min_dist = dist
                label = self.clusters[i].id
        return label

    def visualize(self, i=None, save=False):
        colors = set()
        while len(colors) < len(self.clusters):
            colors.add(
                "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))
        for cl in self.clusters:
            plt.scatter(*cl.X.T, c=colors.pop(), label=str(cl.id))
        plt.legend()
        if save:
            plt.savefig(f'cure/temp/cure_{i}.png')
            plt.close()
            return
        plt.show()
