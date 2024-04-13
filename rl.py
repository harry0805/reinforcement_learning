import numpy as np
import flappy_bird_gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from time import sleep
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import wrappers

env = flappy_bird_gym.make('FlappyBird-v0')

states = env.observation_space.n
actions = env.action_space.n

qtable = np.zeros((states, actions))

learning_rate = 0.9
discount_rate = 0.8

video = VideoRecorder(env, 'results.mp4')

for i in range(200):
    env.render()
    video.capture_frame()
    obs, reward, done, info = env.step(env.action_space.sample())

env.close()
video.close()







# model = keras.Sequential([
#     layers.Flatten(input_shape=(1, 2)),
#     layers.Dense(24, activation='relu'),
#     layers.Dense(24, activation='relu'),
#     layers.Dense(2, activation='linear')
# ])
# model.summary()
#
# train_py_env = wrappers.TimeLimit(env, duration=100)
# eval_py_env = wrappers.TimeLimit(env, duration=100)
#
# tf.optimizers.Adam(learning_rate=1e-3)
# dqn_agent.DqnAgent(
#
# )