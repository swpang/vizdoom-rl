import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random

from utils import Buffer

class DuelQNet(nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """

    def __init__(self, available_actions_count):
        super(DuelQNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.state_fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, available_actions_count)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 192)
        x1 = x[:, :96]  # input for the net to calculate the state value
        x2 = x[:, 96:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

        return x

class DDQNAgent():
    def __init__(self, input_shape, action_size, seed, device, memory_size, batch_size, 
                 discount_factor, lr, load_model, model_savefile, tau, update_freq, replay_freq):
        self.input_shape = input_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.discount = discount_factor
        self.device = device
        self.lr = lr
        self.tau = tau
        self.update_freq = update_freq

        self.seed = random.seed(seed)
        self.t_step = 0
        self.memory = Buffer(memory_size, self.batch_size, self.seed, self.device)
        self.criterion = nn.MSELoss()

        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile).to(self.device)
            self.target_net = torch.load(model_savefile).to(self.device)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(self.device)
            self.target_net = DuelQNet(action_size).to(self.device)

        self.opt = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % self.update_freq

        if self.t_step == 0:
            # If enough samplees are available in memory, get random subset and learn
            if len(self.memory) > self.replay_freq:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        self.q_net.eval()

        with torch.no_grad():
            action_values = self.q_net(state)
        self.q_net.train()

        # epsilon-greedy
        if np.random.uniform() < eps:
            return random.choice(range(self.action_size)) 
        else:
            return torch.argmax(action_values).item()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from policy model
        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0]
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = self.criterion(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    def soft_update(self, q_net, target_net, tau):
        for target_param, q_param in zip(target_net.parameters(), q_net.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)
