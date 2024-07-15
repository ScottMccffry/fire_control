import numpy as np
import matplotlib.pyplot as plt
import rasterio
from fire_propagation import propagate_fire_with_elevation, propagate_fire
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

def load_wind_data(file_path, n):
    """
    Load wind data from a USGSM1 file.
    file_path: path to the USGSM1 file
    n: size of the matrix to resize the wind data
    Returns a 2D numpy array representing the wind map.
    """
    with rasterio.open(file_path) as dataset:
        wind = dataset.read(2)  # Read the second band

    # Resize wind data to match the simulation grid size
    wind_resized = cv2.resize(wind, (n, n), interpolation=cv2.INTER_LINEAR)
    return wind_resized


# Path to the USGSM1 file (update this with your file path)
usgsm1_file_path = 'path_to_your_usgsm1_file.tif'

# Generate the elevation map from the USGSM1 file
elevation = load_elevation_data(usgsm1_file_path, n)
wind = load_wind_data(usgsm1_file_path, n)

# Initialize matrices
Mavg = np.zeros((n, n, maxt))  # 3D array to store average fire matrix over time steps

# Perform fire propagation for multiple realizations
for k in range(maxr):
    # Initialize the fire matrix with fire starting at the center
    M = np.zeros((n, n))
    M[n//2, n//2] = 1

    # Propagate fire over time steps
    for z in range(maxt):
        M = propagate_fire(M, p, elevation)
        Mavg[:, :, z] += M  # Accumulate the fire matrices

# Normalize the average fire matrix
Mavg /= maxr

# Initialize video writer for saving the simulation
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vidfile = cv2.VideoWriter('fire_simulation_with_elevation.mp4', fourcc, 1, (n, n))

# Save each time step as a frame in the video
for k in range(maxt):
    plt.imshow(Mavg[:, :, k], cmap='hot', interpolation='nearest')
    plt.title(f'Fire Simulation with Elevation - Time Step {k+1}')
    plt.colorbar()
    plt.savefig(f'temp_frame_{k}.png')
    plt.close()

    frame = cv2.imread(f'temp_frame_{k}.png')
    vidfile.write(frame)

# Release the video writer
vidfile.release()

# Display the final state of the simulation
plt.imshow(Mavg[:, :, maxt-1], cmap='hot', interpolation='nearest')
plt.title(f'Fire Simulation with Elevation - Time Step {maxt}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.show()
