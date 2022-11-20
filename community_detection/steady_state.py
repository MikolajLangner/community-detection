import numpy as np
from scipy import sparse


def steady_state(transition: sparse.base.spmatrix, damping: float = .85, tol: float = 1e-3):

    n = transition.shape[0]
    antidamp = (1 - damping) / n
    matrix = transition * damping

    stationary = np.ones(n, dtype=np.float32) / n
    next_stationary = stationary @ matrix + antidamp

    while np.linalg.norm(next_stationary - stationary) > tol:
        stationary = next_stationary
        next_stationary = stationary @ matrix + antidamp

    return next_stationary
