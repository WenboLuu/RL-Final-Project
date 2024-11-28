import torch
import numpy as np
from constants import DTYPE_STATE, DTYPE_ACTION, DTYPE_REWARD, DTYPE_DONE

class ReplayBuffer:
    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros((capacity, *state_shape), dtype=DTYPE_STATE, device=device)
        self.actions = torch.zeros(capacity, dtype=DTYPE_ACTION, device=device)
        self.rewards = torch.zeros(capacity, dtype=DTYPE_REWARD, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=DTYPE_STATE, device=device)
        self.dones = torch.zeros(capacity, dtype=DTYPE_DONE, device=device)
        self.position = 0
        self.size = 0

    def push(self, states, actions, rewards, next_states, dones):
        actions = torch.tensor(actions, dtype=DTYPE_ACTION, device=self.device)
        rewards = torch.tensor(rewards, dtype=DTYPE_REWARD, device=self.device)
        dones = torch.tensor(dones, dtype=DTYPE_DONE, device=self.device)
        
        batch_size = states.shape[0]
        idx = (torch.arange(batch_size, device=self.device) + self.position) % self.capacity

        self.states[idx] = states
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_states[idx] = next_states
        self.dones[idx] = dones

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        states = self.states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_states = self.next_states[idxs]
        dones = self.dones[idxs]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size
