import numpy as np

def propagate_fire_with_elevation(M, p, elevate):
    """
    Propagate fire in the matrix M based on elevation map elevate and base probability p.
    M: 2D numpy array representing the current state of fire
    p: base probability of fire propagation
    elevate: 2D numpy array representing the elevation map
    Returns the updated fire matrix Mnew.
    """
    row, col = M.shape
    Mnew = M.copy()

    for i in range(1, col - 1):
        for j in range(1, row - 1):
            if M[i, j] == 1:
                # Calculate slopes to neighboring cells
                slopes = calculate_slopes(elevate, i, j)
                # Assign probabilities based on slopes
                probabilities = [assign_probability(p, slope) for slope in slopes]
                # Get neighboring cell coordinates
                neighbors = get_neighbors(i, j)
                
                # Update fire state in neighboring cells based on probabilities
                for (x, y), prob in zip(neighbors, probabilities):
                    Mnew = update(Mnew, x, y, prob)
    
    return Mnew

def calculate_slopes(elevate, i, j):
    """
    Calculate slopes between a cell and its eight neighbors.
    elevate: 2D numpy array representing the elevation map
    i, j: indices of the current cell
    Returns a list of slopes to the neighboring cells.
    """
    slopes = [
        (elevate[i, j] - elevate[i, j+1]),  # Right
        (elevate[i, j] - elevate[i, j-1]),  # Left
        (elevate[i, j] - elevate[i+1, j]),  # Down
        (elevate[i, j] - elevate[i-1, j]),  # Up
        (elevate[i, j] - elevate[i+1, j+1]) / np.sqrt(2),  # Down-Right
        (elevate[i, j] - elevate[i+1, j-1]) / np.sqrt(2),  # Down-Left
        (elevate[i, j] - elevate[i-1, j+1]) / np.sqrt(2),  # Up-Right
        (elevate[i, j] - elevate[i-1, j-1]) / np.sqrt(2)   # Up-Left
    ]
    return slopes

def assign_probability_elevation(p, slope):
    """
    Assign a propagation probability based on the slope.
    p: base probability
    slope: slope between cells
    Returns the adjusted probability.
    """
    a = 1  # Slope sensitivity factor; can be adjusted to change sensitivity to slope
    return 1 / (1 + np.exp(-a * slope))

def get_neighbors(i, j):
    """
    Get the coordinates of the eight neighboring cells.
    i, j: indices of the current cell
    Returns a list of tuples representing the neighboring cell coordinates.
    """
    return [
        (i, j+1), (i, j-1), (i+1, j), (i-1, j),
        (i+1, j+1), (i+1, j-1), (i-1, j+1), (i-1, j-1)
    ]

def update(M, i, j, p):
    """
    Update the fire state of a cell based on a probability.
    M: 2D numpy array representing the current state of fire
    i, j: indices of the cell to be updated
    p: probability of the cell catching fire
    Returns the updated fire matrix.
    """
    if np.random.rand() < p:
        M[i, j] = 1
    return M
