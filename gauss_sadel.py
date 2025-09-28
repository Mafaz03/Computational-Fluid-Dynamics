import numpy as np
from tqdm import tqdm

def Gauss_Sadel(A, B, X, tolerance=1e-8, max_loops=1000):
    relative_error_list = []
    relative_error = np.inf
    loops_taken = 0

    for _ in tqdm(range(max_loops)):
        if relative_error <= tolerance:
            break
        X_old = X.copy()
        for j in range(len(A)):
            sigma = 0.0
            for i in range(len(A)):
                if i != j:
                    sigma += A[j, i] * X[i]
            X[j] = (B[j] - sigma) / A[j, j]

        loops_taken += 1

        relative_error = (X - X_old) / (X+ 1e-12)
        relative_error_list.append(relative_error)

    return X, loops_taken, relative_error_list
