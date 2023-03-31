import numpy as np


def square_euclid_distance(x1, x2):
    print('+'.join([f'√({x1_el} - {x2_el})²' for x1_el,
          x2_el in zip(x1, x2)]), end=' = ')
    print('+'.join([f'√{(x1_el - x2_el) ** 2}' for x1_el,
          x2_el in zip(x1, x2)]), end=' = ')
    distance = sum([(x1_el - x2_el) ** 2 for x1_el, x2_el in zip(x1, x2)])
    print(distance)
    return distance


def euclid_distance(x1, x2):
    return np.sqrt(square_euclid_distance(x1, x2))


def chebishev_distance(x1, x2):
    return max([x1_el - x2_el for x1_el, x2_el in zip(x1.T, x2.T)])


def step_distance(x1, x2, r=2, p=2):
    dist = sum([(x1_el - x2_el) ** r for x1_el, x2_el in zip(x1, x2)])
    return dist ** (1/p)
