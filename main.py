import math
import time

import matplotlib.pyplot as plt


def create_banded_matrix(N, a1, a2, a3):
    A = [[a1 if i == j else 0 for j in range(N)] for i in range(N)]
    for i in range(N):
        if i < N - 1:
            A[i][i + 1] = a2
            A[i + 1][i] = a2
        if i < N - 2:
            A[i][i + 2] = a3
            A[i + 2][i] = a3
    return A


def create_b_vector(N):
    b = [math.sin(n * 4) for n in range(1, N + 1)]
    return b


def jacobi_method(A, b, tol=1e-9, max_iter=1000):
    n = len(A)
    x = [0] * n
    x_new = x = [0] * n
    residuals = []
    for iteration in range(max_iter):

        if iteration % 10 == 0:
            print(f"Jacobi method iteration no. {iteration}")

        x_new = x[:]

        for i in range(n):
            sum_ = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_) / A[i][i]

        try:
            residual = math.sqrt(sum((b[i] - sum(A[i][j] * x_new[j] for j in range(n))) ** 2 for i in range(n)))
            residuals.append(residual)
        except OverflowError:
            break

        if residual < tol:
            break

        x = x_new

    return x_new, residuals


def gauss_seidel_method(A, b, tol=1e-9, max_iter=1000):
    n = len(b)
    x = [0] * n
    residuals = []

    D_plus_L = [[A[i][j] if j <= i else 0 for j in range(n)] for i in range(n)]
    U = [[A[i][j] if j > i else 0 for j in range(n)] for i in range(n)]

    for iteration in range(max_iter):

        if iteration % 10 == 0:
            print(f"Gauss-Seidel method iteration no. {iteration}")

        x_new = [0] * n

        for i in range(n):
            x_new[i] = (b[i] - sum(D_plus_L[i][j] * x_new[j] for j in range(i)) - sum(
                U[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]

        try:
            residual = math.sqrt(sum((b[i] - sum(A[i][j] * x_new[j] for j in range(n))) ** 2 for i in range(n)))
            residuals.append(residual)
        except OverflowError:
            break

        if residual < tol:
            break

        x = x_new[:]

    return x, residuals


def main():

    N = 996
    a1 = 8
    a2 = -1
    a3 = -1

    A = create_banded_matrix(N, a1, a2, a3)
    b = create_b_vector(N)

    # Rozwiązanie układu równań metodą Jacobiego
    start_time = time.time()
    x_jacobi, residuals_jacobi = jacobi_method(A, b)
    jacobi_time = time.time() - start_time

    # Rozwiązanie układu równań metodą Gaussa-Seidla
    start_time = time.time()
    x_gauss_seidel, residuals_gauss_seidel = gauss_seidel_method(A, b)
    gauss_seidel_time = time.time() - start_time

    # Wykres normy residuum w kolejnych iteracjach
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(residuals_jacobi)), residuals_jacobi, label='Metoda Jacobiego', linestyle='-', marker='o')
    plt.plot(range(len(residuals_gauss_seidel)), residuals_gauss_seidel, label='Metoda Gaussa-Seidla', linestyle='-',
             marker='o')
    plt.yscale('log')
    plt.xlabel('Iteracje')
    plt.ylabel('Norma residuum')
    plt.title('Zmiana normy residuum w kolejnych iteracjach')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Czas trwania metody Jacobiego:", jacobi_time)
    print("Czas trwania metody Gaussa-Seidla:", gauss_seidel_time)


if __name__ == '__main__':
    main()
