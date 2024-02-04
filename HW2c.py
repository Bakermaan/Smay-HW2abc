import numpy as np


# Gauss Seidel function:
def gauss_seidel(aaug, x, niter=15):
    n = len(x)
    for _ in range(niter):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = sum(aaug[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (aaug[i][-1] - sum1) / aaug[i][i]
        x = np.copy(x_new)
    return x


def make_diagonally_dominant(aaug):
    n = len(aaug)
    for i in range(n):
        diag = abs(aaug[i][i])
        if not diag >= sum(abs(aaug[i][j]) for j in range(n) if j != i):
            for k in range(i + 1, n):
                if abs(aaug[k][i]) > sum(abs(aaug[k][j]) for j in range(n) if j != i):
                    aaug[[i, k]] = aaug[[k, i]]  # Swap the rows
                    break
    return aaug


def main():
    # First system of linear equations
    aaug1 = np.array([[3, -1, 0, 2],
                      [1, 4, 2, 12],
                      [0, 2, 3, 10]], dtype=float)

    # Second system of linear equations
    aaug2 = np.array([[1, -10, 2, 4, 2],
                      [3, 1, 4, 12, 12],
                      [9, 2, 3, 4, 21],
                      [-1, 2, 7, 3, 37]], dtype=float)

    # Make the matrices diagonally dominant
    aaug1 = make_diagonally_dominant(aaug1)
    aaug2 = make_diagonally_dominant(aaug2)

    # Initial guesses for both functions
    x0_1 = np.zeros(aaug1.shape[0])
    x0_2 = np.zeros(aaug2.shape[0])

    # Solving both functions
    solution1 = gauss_seidel(aaug1, x0_1)
    solution2 = gauss_seidel(aaug2, x0_2)

    print("Solution to the function set:", solution1)
    print("Solution to the second function set:", solution2)


if __name__ == "__main__":
    main()
