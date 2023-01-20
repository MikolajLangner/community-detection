import numpy as np
from scipy import sparse


def steady_state(transition: sparse.spmatrix, damping: float = .85, tol: float = 1e-6):

    N = transition.shape[0]
    antidamp = (1 - damping) / N
    matrix = transition * damping

    stationary = np.ones(N, dtype=np.float32) / N
    next_stationary = stationary @ matrix + antidamp

    while np.linalg.norm(next_stationary - stationary) > tol:
        stationary = next_stationary
        next_stationary = stationary @ matrix + antidamp

    return sparse.csr_matrix(next_stationary)
