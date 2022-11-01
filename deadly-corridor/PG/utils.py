import numpy as np
import tensorflow as tf
import scipy
import scipy.signal
import random
import scipy.misc
import csv
import tensorflow.contrib.slim as slim
import os
import moviepy.editor as mpy
from collections import namedtuple, deque
import torch

from vizdoom import *
from params import *

def create_environment(scenario = 'basic', no_window = False, actions_type="all", player_mode=False):
    """
    Description
    ---------------
    Creates VizDoom game instance with provided settings.
    
    Parameters
    ---------------
    scenario : String, either 'basic' or 'deadly_corridor', the Doom scenario to use (default='basic')
    window   : Boolea, whether to render the window of the game or not (default=False)
    
    Returns
    ---------------
    game             : VizDoom game instance.
    possible_actions : np.array, the one-hot encoded possible actions.
    """
    
    game = DoomGame()
    if no_window:
        game.set_window_visible(False)        
    else:
        game.set_window_visible(True)
    
    # Load the correct configuration
    game.load_config(os.path.join("scenarios",params.scenario+".cfg"))
    game.set_doom_scenario_path(os.path.join("scenarios",params.scenario+".wad"))
    
    # Switch to RGB in player mode
    if player_mode:
        game.set_screen_format(ScreenFormat.RGB24)
    
    # Initiliaze game
    game.init()
    
    # Possible predefined actions for the scenario
    possible_actions = button_combinations(scenario)
    
    return game, possible_actions


def button_combinations(scenario='basic'):
    """
    Description
    ---------------
    Returns a list of possible action for a scenario.
    
    Parameters
    ---------------
    scenario : String, Doom scenario to use (default='basic')
    
    Returns
    ---------------
    actions : list, the one-hot encoded possible actions.
    """
    actions = []

    m_left_right = [[True, False], [False, True], [False, False]]  # move left and move right
    attack = [[True], [False]]
    m_forward_backward = [[True, False], [False, True], [False, False]]  # move forward and backward
    t_left_right = [[True, False], [False, True], [False, False]]  # turn left and turn right

    if scenario=='deadly_corridor':
        actions = np.identity(6,dtype=int).tolist()
        actions.extend([[0, 0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 0, 1], 
                        [1, 0, 1, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0]])

    if scenario=='basic':
        for i in m_left_right:
            for j in attack:
                actions.append(i+j)

    if scenario=='my_way_home':
        actions = np.identity(3,dtype=int).tolist()
        actions.extend([[1, 0, 1],
                        [0, 1, 1]])

    if scenario=='defend_the_center':
        for i in t_left_right:
            for j in attack:
                actions.append(i+j)

    if scenario=='defend_the_line':
        for i in t_left_right:
            for j in attack:
                actions.append(i+j)

    return actions


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (string): GPU or CPU
        """

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
