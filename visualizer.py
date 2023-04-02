import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Visualizer:
    def __init__(self, xrange=(-10, 10), yrange=(-10, 10),
                 base_folder='results', prefix=''):
        self.xrange = xrange
        self.yrange = yrange
        self.base_folder = base_folder
        self.prefix = prefix
        self.temp_folder = f'{self.base_folder}/temp'
        self.image_template = f'{self.temp_folder}/{self.prefix}%03d.png'
        self.gif_template = f'{self.base_folder}/{self.prefix}{time.time()}.gif'

    def plot3d(self, x, f, other_x=None, save=False, i=None):
        fx = f(x)
        xrange = [min(x[0], self.xrange[0]),
                  max(x[0], self.xrange[1])]
        yrange = [min(x[1], self.yrange[0]),
                  max(x[1], self.yrange[1])]
        mesh = np.array(np.meshgrid(np.linspace(*xrange, 1000),
                                    np.linspace(*yrange, 1000)))
        z = f(mesh)
        ax = plt.axes(projection='3d')
        ax.plot_surface(*mesh, z, alpha=0.5)
        if other_x:
            ax.scatter3D(*other_x.T, f(other_x), c='b', s=1)
        ax.scatter3D(*x.T, fx, c='r', s=2)
        ipatch = mpatches.Patch(label=f'Итерация {i}:')
        xpatch = mpatches.Patch(label=f'x = {x}')
        fpatch = mpatches.Patch(label=f'f = {fx}')

        ax.legend(handles=[ipatch, xpatch, fpatch], loc='upper right',
                  handlelength=0, handletextpad=0)
        if save:
            if not os.path.exists(self.temp_folder):
                os.makedirs(self.temp_folder, exist_ok=True)
            plt.savefig(self.image_template % i)
        else:
            plt.show()
        plt.close()

    def makegif(self):
        if os.path.exists(f'{self.temp_folder}'):
            os.system("ffmpeg -f image2 -framerate 5 -i {} -loop 0 {}".format(
                self.image_template,
                self.gif_template
            ))
            shutil.rmtree(f'{self.temp_folder}')
