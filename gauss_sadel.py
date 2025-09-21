import numpy as np
def Gauss_Sadel(A, B, X, tolerance=1e-8, max_loops=1000):
    relative_error = np.inf
    loops_taken = 0

    while relative_error > tolerance:
        X_old = X.copy()
        for j in range(len(A)):
            sigma = 0.0
            for i in range(len(A)):
                if i != j:
                    sigma += A[j, i] * X[i]
            X[j] = (B[j] - sigma) / A[j, j]

        loops_taken += 1
        if max_loops and loops_taken >= max_loops:
            break

        relative_error = np.linalg.norm(X - X_old, np.inf) / (np.linalg.norm(X, np.inf) + 1e-12)

    return X, loops_taken
