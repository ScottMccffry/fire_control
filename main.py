import numpy as np
import matplotlib.pyplot as plt
from elevation_propagation import propagate_fire_with_elevation

# Parameters
p = 1  # Base probability
n = 26  # Matrix size
maxt = 10  # Number of time steps
maxr = 1000  # Number of realizations

# Load or generate topographic map (elevation)
def generate_elevation_map(n):
    noise_range = 0.05
    elevation = np.zeros((n, n))
    for i in range(n):
        e_mean = i / n
        for j in range(n):
            elevation[i, j] = e_mean + (np.random.rand() * noise_range)
    return elevation

elevation = generate_elevation_map(n)

# Initialize matrices
Mavg = np.zeros((n, n, maxt))

for k in range(maxr):
    M = np.zeros((n, n))
    M[n//2, n//2] = 1

    for z in range(maxt):
        M = propagate_fire_with_elevation(M, p, elevation)
        Mavg[:, :, z] += M

# Normalize the mean
Mavg /= maxr

# Initialize video
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vidfile = cv2.VideoWriter('fire_simulation_with_elevation.mp4', fourcc, 1, (n, n))

# Loop over all time steps
for k in range(maxt):
    plt.imshow(Mavg[:, :, k], cmap='hot', interpolation='nearest')
    plt.title(f'Fire Simulation with Elevation - Time Step {k+1}')
    plt.colorbar()
    plt.savefig(f'temp_frame_{k}.png')
    plt.close()

    frame = cv2.imread(f'temp_frame_{k}.png')
    vidfile.write(frame)

vidfile.release()

# Display final state
plt.imshow(Mavg[:, :, maxt-1], cmap='hot', interpolation='nearest')
plt.title(f'Fire Simulation with Elevation - Time Step {maxt}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.show()