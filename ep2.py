import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

filename = input('Введите размер файла small/large: ')
with open(f'{filename}.txt', 'r') as f:
    s = list(filter(lambda x: x, f.read().split('\n')))

    N = int(s[0])

    A = np.array([[float(j) for j in i.split()] for i in s[1:N + 1]], dtype=np.float64)
    b = np.array([[float(i) for i in s[-1].split()]], dtype=np.float64).transpose()

    x = linalg.solve(A, b).transpose()
    absc = np.arange(1, x[0].size + 1)

    fig, axs = plt.subplots(1, 1)
    axs.bar(absc, x[0])
    axs.grid()
    fig.savefig(f'solved_{filename}.png')
    plt.show()
