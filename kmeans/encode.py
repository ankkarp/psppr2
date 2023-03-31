import numpy as np
from tqdm import tqdm


def onehotencode(data_s):
    vocab = np.unique(
        np.hstack(data_s.str.extract('(\w+)', expand=False).values))
    encoded = np.zeros((len(data_s), len(vocab)))
    for i, word in enumerate(tqdm(vocab)):
        encoded[data_s.str.contains(f'({word})'), i] = 1
    return encoded
