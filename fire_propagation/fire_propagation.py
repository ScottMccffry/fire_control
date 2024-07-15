import numpy as np

def propagate_fire(M, p, elevation, wind):
    """
    Propagate fire in the matrix M based on elevation map elevate and base probability p.
    M: 2D numpy array representing the current state of fire
    p: base probability of fire propagation
    elevate: 2D numpy array representing the elevation map
    Returns the updated fire matrix Mnew.
    """
    row, col = M.shape
    Mnew = M.copy()