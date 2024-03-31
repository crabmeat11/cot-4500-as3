import numpy as np

def euler_method(dy_dt, t0, y0, t_end, steps):
    t = t0
    y = y0
    h = (t_end - t0) / steps

    # Perform iterations
    for _ in range(steps):
        y += h * dy_dt(t, y)
        t += h

    return y  # Return the last value of y

# dy/dt = t - y^2
def func(t, y):
    return t - y**2

def runge_kutta(f, a, b, y0, n):
    h = (b - a) / n
    t = a
    y = y0
    for i in range(n):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
    return y

def gaussian_elimination(matrix):
    n = len(matrix)
    
    # Forward elimination
    for i in range(n):
        pivot = matrix[i][i]
        for j in range(i + 1, n):
            factor = matrix[j][i] / pivot
            for k in range(i, n + 1):
                matrix[j][k] -= factor * matrix[i][k]
    
    # Backward substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = matrix[i][-1]
        for j in range(i + 1, n):
            x[i] -= matrix[i][j] * x[j]
        x[i] /= matrix[i][i]
    
    return x

def lu_factorization(matrix):
    n = len(matrix)
    L = np.eye(n)
    U = np.copy(matrix)

    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            for j in range(k, n):
                U[i, j] -= factor * U[k, j]

    return L, U

def is_diagonally_dominant(matrix):
    n = len(matrix)
    for i in range(n):
        diagonal_element = abs(matrix[i, i])
        sum_non_diagonal = sum(abs(matrix[i, j]) for j in range(n) if i != j)
        if diagonal_element <= sum_non_diagonal:
            return False
    return True

def is_positive_definite(matrix):
    try:
        # A matrix is positive definite if all its eigenvalues are positive
        return np.all(np.linalg.eigvals(matrix) > 0)
    except np.linalg.LinAlgError:
        # If the matrix is not symmetric or not square, it cannot be positive definite
        return False

#1
result = euler_method(func, 0, 1, 2, 10)

print(result)

#2
result = runge_kutta(func, 0, 2, 1, 10)

print()
print(result)

#3
augmented_matrix = [
    [2, -1, 1, 6],
    [1, 3, 1, 0],
    [-1, 5, 4, -3]
]
result = gaussian_elimination(augmented_matrix)

print()
print("[", result[0], result[1], result[2],"]")

#4
matrix = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]])

L, U = lu_factorization(matrix)

determinant = np.prod(np.diag(U))

print()
print(determinant)
print()
print(L)
print()
print(U)

#5
matrix = np.array([[9, 0, 5, 2, 1],
                   [3, 9, 1, 2, 1],
                   [0, 1, 7, 2, 3],
                   [4, 2, 3, 12, 2],
                   [3, 2, 4, 0, 8]])

result = is_diagonally_dominant(matrix)

print()
print(result)

#6
matrix = np.array([[2, 2, 1],
                   [2, 3, 0],
                   [1, 0, 2]])

result = is_positive_definite(matrix)

print()
print(result)