import numpy as np


def zad_a() -> tuple[np.ndarray, np.ndarray]:

    N = 996

    a1 = 8
    a2 = -1
    a3 = -1

    diagonals = [a1 * np.ones(N), a2 * np.ones(N - 1), a2 * np.ones(N - 1), a3 * np.ones(N - 2), a3 * np.ones(N - 2)]
    A = (np.diag(diagonals[0]) + np.diag(diagonals[1], 1) + np.diag(diagonals[2], -1) +
         np.diag(diagonals[3], 2) + np.diag(diagonals[4], -2))
    b = np.array([np.sin(n * 4) for n in range(1, N + 1)])

    return A, b


def main():
    A, b = zad_a()
    print(A)
    print(b)


if __name__ == '__main__':
    main()
