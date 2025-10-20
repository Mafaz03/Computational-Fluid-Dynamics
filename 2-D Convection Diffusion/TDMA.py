import numpy as np

def TDMA(aW_or_aS, aP, aE_or_aN, b):
    """
    we need ot assume that we know 2 points, ie norht and south OR east and west
    """
    n = len(b)
    
    # Temporary arrays
    P = np.zeros(n)
    Q = np.zeros(n)
    phi = np.zeros(n)
    
    # Forward elimination
    P[0] = aE_or_aN[0] / aP[0]
    Q[0] = b[0] / aP[0]

    for i in range(1, n):
        denom = aP[i] - aW_or_aS[i] * P[i-1]
        P[i] = aE_or_aN[i] / denom
        Q[i] = (b[i] + aW_or_aS[i] * Q[i-1]) / denom

    # Back substitution
    phi[-1] = Q[-1]
    for i in range(n-2, -1, -1):
        phi[i] = P[i] * phi[i+1] + Q[i]
    
    return phi
