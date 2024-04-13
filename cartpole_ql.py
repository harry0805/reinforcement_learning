import gym_pendulum
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm

# Create the CartPole game environment
render_mode = [None, 'human']
env = gym_pendulum.make('CartPole-v1', render_mode=render_mode[0])

print(env.reset())

# Number of states is huge, so in order to simplify the situation
# we discretize the space to: number of buckets x number of actions
buckets = (30, 30, 50, 50)  # downscale feature space to discrete range
num_actions = env.action_space.n  # [left, right]

# Bounds for each discrete state
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-math.radians(50), math.radians(50)]

# Learning parameters
max_episodes = 60000

learning_rate = 0.1  # Alpha
discount_factor = 0.95  # Gamma
explore_rate = 1.0  # Epsilon
min_explore_rate = 0.1
explore_decay_rate = 0.99995  # Adjust the rate of decay for exploration

# Q-table for each state-action pair
if os.path.exists('cartpole_q_table.npy'):
    Q_table = np.load('cartpole_q_table.npy')
    print(Q_table)
else:
    # Q_table = np.zeros(buckets + (num_actions,))
    Q_table = np.random.uniform(low=-1, high=1, size=(buckets + (num_actions,)))

def discretize_state(observation):
    """Convert continuous state into a discrete state"""
    ratios = [(observation[i] + abs(state_bounds[i][0])) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(len(observation))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(observation))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(observation))]
    return tuple(new_obs)


def choose_action(state):
    """Choose an action according to the learning policy"""
    return np.argmax(Q_table[state]) if np.random.random() > explore_rate else env.action_space.sample()


def update_q_table(old_state, action, reward, new_state):
    """Update the Q-table using the Q-learning algorithm"""
    best_q = np.max(Q_table[new_state])
    Q_table[old_state][action] += learning_rate * (reward + discount_factor * best_q - Q_table[old_state][action])


# List to hold total rewards per episode
total_rewards = []

for episode in tqdm(range(max_episodes)):
    initial_observation, _ = env.reset()  # This unpacks the tuple into the observation and the info dictionary
    current_state = discretize_state(initial_observation)
    done = False
    total_reward = 0

    while not done:
        action = choose_action(current_state)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Combine both to manage the end of an episode
        new_state = discretize_state(obs)
        update_q_table(current_state, action, reward, new_state)
        current_state = new_state
        total_reward += reward

    # Reduce exploration rate
    explore_rate = max(min_explore_rate, explore_rate * explore_decay_rate)
    total_rewards.append(total_reward)

np.save('cartpole_q_table.npy', Q_table)

# Plotting the results
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
