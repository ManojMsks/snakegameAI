from collections import deque

import torch.types
import random
import torch.nn.functional as F
import numpy as np
from torch import nn,optim
class ANN(nn.Module):
    def __init__(self, state_size, action_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        return self.fc2(x)

class ReplayMemory:
        def __init__(self, capacity):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.capacity = capacity
            self.memory = []

        def push(self, event):
            # event - (state, action, reward, next_state, done)
            self.memory.append(event)
            if len(self.memory) > self.capacity:
                del self.memory[0]

        def sample(self, k):
            experiences = random.sample(self.memory, k=k)
            # [(state, action, reward, next_state, done)]
            states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
            rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(
                self.device)
            dones = torch.from_numpy(
                np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(
                self.device)

            return states, actions, rewards, next_states, dones


# Hyperparameters
number_episodes = 100000  # Defines how many episodes the agent will train on.
maximum_number_steps_per_episode = 200000  # Sets the limit on how long an episode can last.
epsilon_starting_value = 1.0  # Initial value for the exploration-exploitation tradeoff.
epsilon_ending_value = 0.001  # Final value for epsilon after decay.
epsilon_decay_value = 0.99  # Rate at which epsilon decreases over time.
learning_rate = 0.01  # The step size used by the optimizer to update weights.
minibatch_size = 100  # Number of samples used in each training step.
gamma = 0.95  # Discount factor for future rewards in Q-learning.
replay_buffer_size = int(1e5)  # Maximum capacity of the replay memory.
interpolation_parameter = 1e-2  # Parameter for soft updates in the target network.

state_size = 16
action_size = 4
scores_on_100_episodes = deque(maxlen=100)
folder = "model"


class Agent:
            def __init__(self, state_size, action_size):
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.state_size = state_size
                self.action_size = action_size
                self.local_network = ANN(state_size, action_size).to(self.device)
                self.target_network = ANN(state_size, action_size).to(self.device)
                self.optimizer = optim.Adam(self.local_network.parameters(), lr=learning_rate)
                self.memory = ReplayMemory(replay_buffer_size)
                self.t_step = 0
                self.record = -1
                self.epsilon = -1





