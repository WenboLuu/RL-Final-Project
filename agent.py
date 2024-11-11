import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models import DQN
from replay_buffer import ReplayBuffer
from constants import DTYPE_ACTION


class DDQNAgent:
    def __init__(self, state_shape, n_actions, batch_size, lr, gamma, target_update_freq, memory_size, device):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size, state_shape, device)

        self.update_count = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.max(1)[1]
        else:
            action = torch.tensor(np.random.randint(0, self.n_actions, size=state.shape[0]), device=self.device, dtype=DTYPE_ACTION)
        return action

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1]
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
