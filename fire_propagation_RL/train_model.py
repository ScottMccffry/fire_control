import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load your data
fire_state = np.load('scripts/data/fire_state.npy')
elevation = np.load('scripts/data/elevation.npy')
wind_u = np.load('scripts/data/wind_u.npy')
wind_v = np.load('scripts/data/wind_v.npy')
humidity = np.load('scripts/data/humidity.npy')
froude_number = np.load('scripts/data/froude_number.npy')

# Check if data is loaded correctly
print("Fire State Data Shape:", fire_state.shape)
print("Elevation Data Shape:", elevation.shape)
print("Wind U Component Data Shape:", wind_u.shape)
print("Wind V Component Data Shape:", wind_v.shape)
print("Humidity Data Shape:", humidity.shape)
print("Froude Number Data Shape:", froude_number.shape)

# Example placeholder data for X and Y
num_samples = 100  # Replace with actual number of samples
timesteps = 10     # Number of timesteps
height = 45
width = 65
channels = 10

X = np.random.rand(num_samples, timesteps, height, width, channels)
Y = np.random.rand(num_samples, 8)

# Define the model architecture using Input layer
model = keras.Sequential([
    layers.Input(shape=(timesteps, height, width, channels)),
    layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=True),
    layers.MaxPooling3D(pool_size=(1, 2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(8, activation='linear')
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train the model
model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('results/model_weights.h5')

print("Model training completed and saved.")
