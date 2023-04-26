import numpy as np


def euclid(X, entry, axis=None, verbose=False):
    distance = np.sqrt(np.sum((X - entry)**2, axis=axis))
    if verbose:
        for xi, d in zip(X, distance):
            print(f'euclid({np.around(xi, 5)}, {np.around(entry, 5)})',
                  round(d, 5), sep=' = ')
    return distance
