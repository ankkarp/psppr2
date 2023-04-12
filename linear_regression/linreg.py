import os
import numpy as np
import shutil
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
from tqdm import tqdm


class Regression:
    def __init__(self, lr, n_epochs, export_gif=False,
                 base_folder='results', temp_folder='temp', mode='linear', name='linear'):
        self.lr = lr
        self.mode = mode
        self.n_epochs = n_epochs
        self.export_gif = export_gif
        self.temp_folder = f'{base_folder}/{temp_folder}'
        self.image_template = f'{self.temp_folder}/%0{int(np.ceil(np.log10(n_epochs + 1)))}d.png'
        self.gif_template = f'{base_folder}/{name}.gif'
        print(self.image_template)
        print(self.gif_template)
        if self.export_gif:
            os.makedirs(self.temp_folder, exist_ok=True)

    def _epoch(self):
        y_pred = self.predict(self.x)

        dw = np.dot(self.x.T, (y_pred-self.y)) / self.n_samples
        print(
            f'd_w = ({np.around(self.x.T, 5)} * ({np.around(y_pred, 5)}) - {np.around(self.y, 5)}) / {self.n_samples} = {np.around(dw, 5)}')
        db = np.sum(y_pred-self.y) / self.n_samples
        print(
            f'd_b = ({" + ".join([f"({p:.5f}-{t:.5f})" for p, t in zip(y_pred, self.y)])}) / {self.n_samples} = {np.sum(y_pred-self.y):.5f} / {self.n_samples} = {db:.5f}')
        print(f'weights = {np.around(self.weights, 5)} - ', end='')
        self.weights -= self.lr * dw
        print(f'{self.lr} / {np.around(dw, 5)} = {np.around(self.weights, 5)}')
        print(f'bias = {np.around(self.weights, 5)} - ', end='')
        self.bias -= self.lr * db
        print(f'{self.lr} / {db:.5f} = {self.bias:.5f}')

    def fit(self, x, y, bias=0):
        self.x = x
        print(f'X = {self.x.T}')
        self.y = y
        print(f'Y = {self.y.T}')
        if len(x.shape) == 1:
            self.x = np.expand_dims(self.x, axis=1)
        self.n_samples, self.n_dims = self.x.shape
        self.weights = np.zeros(self.n_dims)
        self.bias = bias
        for i in tqdm(range(1, self.n_epochs + 1)):
            print(f'Итерация {i}:')
            self._epoch()
            if self.export_gif:
                self.visualize(i, save=True)
        # self.visualize(i)
        if self.export_gif:
            self.makegif()
        return self

    def predict(self, x):
        y_pred = np.dot(x, self.weights) + self.bias
        if self.mode == 'logistic':
            y_pred = 1/(1+np.exp(-y_pred))
        print(
            f'Ypred = (X, {np.around(self.weights, 5)}) + {self.bias:.5f} = {np.around(y_pred, 5)}')
        return y_pred

    def visualize(self, i, save=False):
        plt.scatter(self.x, self.y, c='0.3')
        x = np.linspace(min(self.x[:, 0]), max(self.x[:, 0]), 1000)
        plt.plot(x, self.predict(
            np.expand_dims(x, axis=1)), c='r')
        ipatch = mpatches.Patch(label=f'Итерация {i}:')
        plt.legend(handles=[ipatch], loc='upper right',
                   handlelength=0, handletextpad=0)
        if save:
            plt.savefig(self.image_template % i)
        else:
            plt.show()
        plt.close()

    def estimate(self):
        res = {}
        y_pred = self.predict(self.x)
        print('Изменчивость: ')
        res['Qr'] = np.sum((self.y - y_pred) ** 2)
        print(
            f'Qr = Σ(y - y_pred)² = {" + ".join(f"({it1:.5f}-{it2:.5f})²" for it1, it2 in zip(self.y, y_pred))} = {res["Qr"]:.5f}')
        res['Qe'] = np.sum((self.y - np.mean(y_pred)) ** 2)
        print(
            f'Qe = Σ(y - y_mean)² = {" + ".join(f"({it:.5f}-{np.mean(y_pred):.5f})²" for it in self.y)} = {res["Qe"]:.5f}')
        res['Q'] = res['Qr'] + res['Qe']
        print(
            f'Q = Qr + Qe = {res["Qr"]:.5f} + {res["Qe"]:.5f} = {res["Q"]:.5f}')
        res['СКО'] = res['Qr'] / (self.n_samples - self.n_dims - 1)
        print('Средне-квадратичная ошибка:')
        print(
            f'СКО = Qr / (n_samples - n_dims - 1) = {res["Qr"]:.5f} / ({self.n_samples} - {self.n_dims} - 1) = {res["СКО"]:.5f}')

        print('Стандартная ошибка:')
        res['СТ'] = np.sqrt(res['СКО'])
        print(f'СТ = √СКО = √{res["СКО"]:.5f} = {res["СТ"]:.5f}')
        print('Коэффициент детерминации:')
        res['r2'] = res['Qr'] / res['Q']
        print(
            f'r² = Qr / Q = {res["Qr"]:.5f} / {res["Q"]:.5f} = {res["r2"]:.5f}')
        print('Коэффициент корреляция:')
        res['r'] = np.sqrt(res['r2'])
        print(f'r = √{res["r2"]:.5f} = {res["r"]:.5f}')
        return res

    def makegif(self):
        if os.path.exists(self.gif_template):
            os.remove(self.gif_template)
        os.system("ffmpeg -f image2 -framerate 5 -i {} -loop 0 {}".format(
            self.image_template,
            self.gif_template
        ))
        shutil.rmtree(f'{self.temp_folder}')
