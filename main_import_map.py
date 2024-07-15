import numpy as np
import matplotlib.pyplot as plt
import rasterio
from elevation_propagation import propagate_fire_with_elevation
import cv2

# Parameters
p = 1  # Base probability for fire propagation; can be modified to fine-tune the model
n = 26  # Size of the matrix (n x n); adjust based on your requirements
maxt = 10  # Number of time steps for the simulation; can be increased or decreased
maxr = 1000  # Number of realizations (repetitions) to average results; higher values give more accurate results

# Load elevation data from a USGSM1 file
def load_elevation_data(file_path, n):
    """
    Load elevation data from a USGSM1 file.
    file_path: path to the USGSM1 file
    n: size of the matrix to resize the elevation data
    Returns a 2D numpy array representing the elevation map.
    """
    with rasterio.open(file_path) as dataset:
        elevation = dataset.read(1)  # Read the first band

    # Resize elevation data to match the simulation grid size
    elevation_resized = cv2.resize(elevation, (n, n), interpolation=cv2.INTER_LINEAR)
    return elevation_resized

# Path to the USGSM1 file (update this with your file path)
usgsm1_file_path = 'path_to_your_usgsm1_file.tif'

# Generate the elevation map from the USGSM1 file
elevation = load_elevation_data(usgsm1_file_path, n)

# Initialize matrices
Mavg = np.zeros((n, n, maxt))  # 3D array to store average fire matrix over time steps

# Perform fire propagation for multiple realizations
for k in range(maxr):
    # Initialize the fire matrix with fire starting at the center
    M = np.zeros((n, n))
    M[n//2, n//
