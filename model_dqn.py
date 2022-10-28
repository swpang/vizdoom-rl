import vizdoom as vzd
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

class DQNet(nn.Module):
    """
    Deep Q-Learning
    """

    def __init__(self, available_actions_count):
        super(DQNet, self).__init__()
        self.available_actions_count = available_actions_count

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, self.available_actions_count)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class DQNAgent():
    def __init__(self, action_size, memory_size, batch_size, discount_factor,
                 lr, load_model, model_savefile, device, epsilon=1, epsilon_decay=0.9996, epsilon_min=0.1):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        
        self.memory = Buffer(memory_size, self.batch_size, self.seed, self.device)
        self.criterion = nn.MSELoss()

        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile)
            self.target_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min

        else:
            print("Intializing new model")
            self.q_net = DQNet(action_size).to(device)
            self.target_net = DQNet(action_size).to(device)

        self.opt = optim.Adam(self.q_net.parameters(), lr=self.lr)
    
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

'''

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = np.arange(self.batch_size)  # used for indexing the batch

        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)

        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
'''
