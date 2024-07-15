A repository for working on fire simulations and drone fire fighting


Parameters in main.py:

p: The base probability for fire propagation. Adjust this value to make the fire more or less likely to spread.
n: The size of the matrix (grid) representing the area. Changing this value will affect the resolution and size of the simulation.
maxt: The number of time steps for the simulation. Increase or decrease this to simulate longer or shorter periods.
maxr: The number of realizations to average the results. Higher values provide more accurate and stable results but require more computation.
Loading Elevation Data:

The function load_elevation_data(file_path, n) reads the elevation data from a USGSM1 file and resizes it to match the simulation grid size. You need to specify the correct file path for your USGSM1 file.
Fire Propagation:

The core logic for fire propagation considering elevation is in propagate_fire_with_elevation(M, p, elevate) in elevation_propagation.py.
The function calculates slopes to neighboring cells and adjusts the propagation probability based on these slopes.
Fine-Tuning the Model:

Modify the base probability p to control the general likelihood of fire spread.
Adjust the slope sensitivity factor a in assign_probability to change how sensitive the model is to changes in elevation.
Change the matrix size n to simulate different areas with more or fewer cells.
Alter the number of time steps maxt to see the fire spread over different durations.
Increase or decrease the number of realizations maxr for more or less averaging of results.