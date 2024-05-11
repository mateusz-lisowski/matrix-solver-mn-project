import math
import time


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


def main():

    N = 996
    a1 = 8
    a2 = -1
    a3 = -1

    A = create_banded_matrix(N, a1, a2, a3)
    b = create_b_vector(N)

    for r in A:
        print(r)
    print(b)

    # # Jacobi's method test
    # start_time = time.time()
    # x_jacobi, residuals_jacobi = jacobi(A, b)
    # jacobi_time = time.time() - start_time
    #
    # # Gauss-Seidel method test
    # start_time = time.time()
    # x_gauss_seidel, residuals_gauss_seidel = gauss_seidel(A, b)
    # gauss_seidel_time = time.time() - start_time
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(len(residuals_jacobi)), residuals_jacobi, label="Jacobi's method", linestyle='-', marker='o')
    # plt.plot(range(len(residuals_gauss_seidel)), residuals_gauss_seidel, label='Gauss-Seidel method', linestyle='-',
    #          marker='o')
    # plt.yscale('log')
    # plt.xlabel('Iterations')
    # plt.ylabel('Residuum norm')
    # plt.title('Residuum norm change in next iterations')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # print(f"Jacobi's method time: {jacobi_time}")
    # print(f"Gauss-Seidel method time: {gauss_seidel_time}")


if __name__ == '__main__':
    main()
