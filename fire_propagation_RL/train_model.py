import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load your data (assume they are saved as numpy arrays)
fire_state = np.load('scripts/data/fire_state.npy')
elevation = np.load('scripts/data/elevation.npy')
wind_u = np.load('scripts/data/wind_u.npy')
wind_v = np.load('scripts/data/wind_v.npy')
humidity = np.load('scripts/data/humidity.npy')


# Define state and action space sizes
state_space_size = fire_state.size + elevation.size + wind_u.size + wind_v.size + humidity.size
action_space_size = 8  # Example: 8 possible fire spread directions

# Define the hybrid model
model = keras.Sequential()
model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(None, 45, 65, 9), return_sequences=True))
model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(action_space_size, activation='linear'))

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Define the RL environment and training loop
def select_action(state, model):
    state = np.reshape(state, [1, state_space_size])
    q_values = model.predict(state)
    return np.argmax(q_values[0])

def train_model(model, state, action, reward, next_state):
    target = reward + 0.95 * np.max(model.predict(next_state)[0])
    target_f = model.predict(state)
    target_f[0][action] = target
    model.fit(state, target_f, epochs=1, verbose=0)

total_episodes = 1000
max_timesteps = 100

for episode in range(total_episodes):
    state = np.concatenate((fire_state.flatten(), elevation.flatten(), wind_u.flatten(), wind_v.flatten(), humidity.flatten()))
    for t in range(max_timesteps):
        action = select_action(state, model)
        next_state, reward, done = step(state, action)  # Define step function based on your environment
        train_model(model, state, action, reward, next_state)
        state = next_state
        if done:
            break
        
model.save('results/model_weights.h5')