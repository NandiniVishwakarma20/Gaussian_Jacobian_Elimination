import numpy as np
import matplotlib.pyplot as plt

def jacobi_elimination(A, b, iter=2800, tol=1e-9):
    n = len(b)
    x = np.zeros(n)
    for _ in range(iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.max(np.abs(x_new - x)) < tol:
            break
        x = x_new.copy()
    return x

A = np.array([[5, -2, 3],
              [-3, 9, 1],
              [2, -1, -7]], dtype=float)

b = np.array([1,2,3], dtype=float)

x_jacobi = jacobi_elimination(A, b)

print("A =\n", A)
print("b = ", b)
print("x1, x2, x3", x_jacobi)
