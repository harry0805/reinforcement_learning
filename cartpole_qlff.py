import gym_pendulum
import numpy as np

# Create the CartPole environment
env = gym_pendulum.make('CartPole-v1')

# Set up parameters
num_episodes = 1000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Discretize state space
num_bins = [20, 20, 20, 20]  # Number of bins for each dimension of the state space
state_bins = [np.linspace(env.observation_space.low[i], env.observation_space.high[i], num_bins[i] + 1)[1:-1] for i in range(env.observation_space.shape[0])]

# Initialize Q-table
action_space_size = env.action_space.n
state_space_size = tuple(num_bins)
q_table = np.zeros(state_space_size + (action_space_size,))

# Convert state to discrete
def discretize_state(state):
    discrete_state = [np.digitize(state[i], state_bins[i]) for i in range(env.observation_space.shape[0])]
    return tuple(discrete_state)

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_rate_threshold = np.random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()

        # Take action and observe next state and reward
        next_state, reward, done, truncated, info = env.step(action)
        next_state = discretize_state(next_state)

        # Update Q-table
        best_next_action = np.argmax(q_table[next_state])
        q_table[state + (action,)] = q_table[state + (action,)] * (1 - learning_rate) + \
                                     learning_rate * (reward + discount_rate * q_table[next_state + (best_next_action,)])

        total_reward += reward
        state = next_state

        if done:
            break

    # Decay exploration rate
    exploration_rate = min_exploration_rate + \
                        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # Print episode statistics
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Close the environment
env.close()
