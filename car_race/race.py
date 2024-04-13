import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Define the Q-network
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(o.view(1, -1).size(1))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# Define replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return torch.tensor(state), torch.tensor(action), torch.tensor(reward), torch.tensor(next_state), torch.tensor(done)

    def __len__(self):
        return len(self.buffer)

# Define DQN agent
class DQNAgent:
    def __init__(self, state_shape, n_actions, capacity=10000, batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        state_action_values = self.policy_net(states).gather(1, actions)
        next_state_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_state_action_values = rewards + (1 - dones) * self.gamma * next_state_values
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

# Environment setup
env = gym.make('CarRacing-v2')
env.seed(0)
torch.manual_seed(0)
state_shape = env.observation_space.shape
n_actions = env.action_space.n

# Agent setup
agent = DQNAgent(state_shape, n_actions)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, device=agent.device, dtype=torch.float32).unsqueeze(0)
    total_reward = 0.0
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        total_reward += reward
        next_state = torch.tensor(next_state, device=agent.device, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], device=agent.device, dtype=torch.float32)
        agent.memory.push(state, action, reward, next_state, torch.tensor([done], device=agent.device, dtype=torch.bool))
        state = next_state
        agent.optimize_model()
        if done:
            break
    agent.update_target_network()
    agent.decay_epsilon()
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Save the model
torch.save(agent.policy_net.state_dict(), 'dqn_car_racing.pth')
