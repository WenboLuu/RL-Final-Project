import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models import DQN
from replay_buffer import ReplayBuffer
from constants import DTYPE_ACTION

class DQNAgent:
    def __init__(self, state_shape, n_actions, batch_size, lr, gamma, target_update_freq, memory_size, device):
        """
        Initializes the DQNAgent with policy and target networks, replay buffer, and optimizer.
        
        Args:
            state_shape (tuple): Shape of the input state.
            n_actions (int): Number of possible actions.
            batch_size (int): Number of samples per batch for training.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for Q-learning.
            target_update_freq (int): Frequency of updating the target network.
            memory_size (int): Size of the replay buffer.
            device (torch.device): Device to use for computation (CPU or GPU).
        """
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        # Initialize the policy and target networks
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        # Optimizer for the policy network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer to store experiences
        self.replay_buffer = ReplayBuffer(memory_size, state_shape, device)

        # Counter to keep track of updates for target network syncing
        self.update_count = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state, epsilon):
        """
        Selects an action using an epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): Current state.
            epsilon (float): Probability of selecting a random action.
        
        Returns:
            np.ndarray: Selected action(s).
        """
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].cpu().numpy()
        else:
            action = np.random.randint(0, self.n_actions, size=state.shape[0])
        return action

    def update(self):
        """
        Samples a batch of experiences from the replay buffer and performs a gradient descent step
        to update the policy network.
        
        Returns:
            float: The loss value.
        """
        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute current Q-values for the actions taken
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using the target network
        with torch.no_grad():
            # For standard DQN, target Q-values use the maximum Q-value from the target network
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute the loss between current Q-values and target Q-values
        loss = nn.MSELoss()(q_values, target_q_values)

        # Perform a gradient descent step to minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Increment the update counter
        self.update_count += 1

        return loss.item()  # Return the loss value

    def update_target_network(self):
        """
        Updates the target network by copying weights from the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, filename):
        """
        Saves the current policy network state dictionary to a file.
        
        Args:
            filename (str): Path to the file where the model will be saved.
        """
        torch.save(self.policy_net.state_dict(), filename)


class DDQNAgent:
    def __init__(self, state_shape, n_actions, batch_size, lr, gamma, target_update_freq, memory_size, device):
        """
        Initializes the DDQNAgent with policy and target networks, replay buffer, and optimizer.
        
        Args:
            state_shape (tuple): Shape of the input state.
            n_actions (int): Number of possible actions.
            batch_size (int): Number of samples per batch for training.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for Q-learning.
            target_update_freq (int): Frequency of updating the target network.
            memory_size (int): Size of the replay buffer.
            device (torch.device): Device to use for computation (CPU or GPU).
        """
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
        """
        Selects an action using an epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): Current state.
            epsilon (float): Probability of selecting a random action.
        
        Returns:
            np.ndarray: Selected action(s).
        """
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].cpu().numpy()
        else:
            action = np.random.randint(0, self.n_actions, size=state.shape[0])
        return action

    def update(self):
        """
        Samples a batch of experiences from the replay buffer and performs a gradient descent step
        to update the policy network.
        
        Returns:
            float: The loss value.
        """
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

        return loss.item()  # Return the loss value

    def update_target_network(self):
        """
        Updates the target network by copying weights from the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, filename):
        """
        Saves the current policy network state dictionary to a file.
        
        Args:
            filename (str): Path to the file where the model will be saved.
        """
        torch.save(self.policy_net.state_dict(), filename)

    def load_checkpoint(self, checkpoint_path):
        """
        Loads a saved state dictionary into the policy and target networks.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        self.target_net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy_net.eval()
        self.target_net.eval()
