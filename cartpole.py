import gym_pendulum
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np


# Define the Deep Q-Network (DQN) agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.learning_rate = 0.001  # Learning rate for the neural network
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Create the OpenAI Gym environment
env = gym_pendulum.make('CartPole-v1')

# Get the state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create the DQN agent
agent = DQNAgent(state_size, action_size)

# Training loop
done = False
batch_size = 32
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time_step in range(500):
        env.render()

        # Agent selects an action
        action = agent.act(state)

        # Agent performs the action in the environment
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Agent trains the neural network with the observed experience
        agent.train(state, action, reward, next_state, done)

        # Update the current state
        state = next_state

        # End the episode if the pole has fallen
        if done:
            break

    # Print the episode number and total reward
    print(f"Episode: {episode + 1}/{num_episodes}, Score: {time_step + 1}")

env.close()
