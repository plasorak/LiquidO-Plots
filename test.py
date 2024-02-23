import os
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np

def clf(n):
    fig = plt.figure()
    for i in range(n):
        ax = fig.add_subplot()
        A = np.ones(shape=(5, 5)) * i
        im = ax.imshow(A)
        ax.set_xlabel(f'x{i}')
        ax.set_ylabel(f'y{i}')
        ax.figure.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(f'test-{i}.png')
        fig.clf()


if __name__ == '__main__':
    clf(5)
