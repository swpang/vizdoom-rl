import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

import numpy as np
import random
import time
from vizdoom import *
from models import *

from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Boolean specifying whether GPUs are available or not.
use_cuda = torch.cuda.is_available()

"""
Environment tools
"""

def create_environment(scenario = 'basic', window = False):
    """
    Description
    ---------------
    Creates VizDoom game instance along with some predefined possible actions.
    
    Parameters
    ---------------
    scenario : String, either 'basic' or 'deadly_corridor' or 'defend_the_center', the Doom scenario to use (default='basic')
    window   : Boolean, whether to render the window of the game or not (default=False)
    
    Returns
    ---------------
    game             : VizDoom game instance.
    possible_actions : List, the one-hot encoded possible actions.
    """
    
    game = DoomGame()
    if window:
        game.set_window_visible(True)
        
    else:
        game.set_window_visible(False)
    
    # Load the correct configuration
    if scenario == 'basic':
        game.load_config("scenarios/basic.cfg")
        game.set_doom_scenario_path("scenarios/basic.wad")
        game.init()
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]
        
    elif scenario == 'deadly_corridor':
        game.load_config("scenarios/deadly_corridor.cfg")
        game.set_doom_scenario_path("scenarios/deadly_corridor.wad")
        game.init()
        possible_actions = np.identity(6,dtype=int).tolist()
#         possible_actions.extend([[0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1]]) # Composed actions need some work
        
    elif scenario == 'defend_the_center':
        game.load_config("scenarios/defend_the_center.cfg")
        game.set_doom_scenario_path("scenarios/defend_the_center.wad")
        game.init()
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]
#         possible_actions.extend([[1, 0, 1], [0, 1, 1]]) # Composed actions need some work
    
    return game, possible_actions
       

def test_environment(weights, scenario = 'basic', window = False, total_episodes = 100, enhance = 'none', frame_skip = 2, stack_size = 4):
    """
    Description
    ---------------
    Test a trained agent in a scenario (Be careful, the chosen weights must match the training scenario)
    
    Parameters
    ---------------
    weights        : String, path to .pth file containing the weights of the network we want to test.
    scenario       : String, either 'basic' or 'deadly_corridor', the Doom scenario to use (default='basic')
    window         : Boolean, whether to render the window of the game or not (default=False)
    total_episodes : Int, the number of testing episodes (default=100)
    enhance        : String, 'none' or 'dueling' (default='none')
    frame_skip     : Int, the number of frames to repeat the action on (default=2)
    
    Returns
    ---------------
    game             : VizDoom game instance.
    possible_actions : List, the one-hot encoded possible actions.
    """
    
    game = DoomGame()
    game.set_screen_format(ScreenFormat.RGB24)
    if window:
        game.set_window_visible(True)
        
    else:
        game.set_window_visible(False)
    
    # Load the correct configuration
    if scenario == 'basic':
        game.load_config("scenarios/basic.cfg")
        game.set_doom_scenario_path("scenarios/basic.wad")
        game.init()
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]
        
    elif scenario == 'deadly_corridor':
        game.load_config("scenarios/deadly_corridor.cfg")
        game.set_doom_scenario_path("scenarios/deadly_corridor.wad")
        game.init()
        possible_actions = np.identity(6,dtype=int).tolist()
#         possible_actions.extend([[0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1]])

    elif scenario == 'defend_the_center':
        game.load_config("scenarios/defend_the_center.cfg")
        game.set_doom_scenario_path("scenarios/defend_the_center.wad")
        game.init()
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]

    if enhance == 'none':
        model = DQNetwork(stack_size = stack_size, out = len(possible_actions))
        if use_cuda:
            model.cuda()

    elif enhance == 'dueling':
        model = DDDQNetwork(stack_size = stack_size, out = len(possible_actions))
        if use_cuda:
            model.cuda()
            
    # Load the weights of the model
    state_dict = torch.load(weights)
    model.load_state_dict(state_dict)
    for i in range(total_episodes):
        game.new_episode()
        done = game.is_episode_finished()
        state = get_state(game)
        in_channels = model._modules['conv_1'].in_channels
        stacked_frames = deque([torch.zeros((120, 160), dtype=torch.int) for i in range(in_channels)], maxlen = in_channels)
        state, stacked_frames = stack_frames(stacked_frames, state, True, in_channels)
        while not done:
            if use_cuda:
                q = model(state.cuda())

            else:
                q = model(state)

            action = possible_actions[int(torch.max(q, 1)[1][0])]
#             action, explore_probability = predict_action(1, 0.01, 0.0001, 1e6, state.cuda(), model, possible_actions)
            reward = game.make_action(action, frame_skip)
            done = game.is_episode_finished()
            if not done:
                state = get_state(game)
                state, stacked_frames = stack_frames(stacked_frames, state, False, in_channels)
                
            time.sleep(0.02)
            
        print ("Total reward:", game.get_total_reward())
        time.sleep(0.1)
        
    game.close()

"""
Preproessing tools
"""
def get_state(game):
    """
    Description
    --------------
    Get the current state from the game.
    
    Parameters
    --------------
    game : VizDoom game instance.
    
    Returns
    --------------
    state : 4-D Tensor, we add the temporal dimension.
    """
    
    state = game.get_state().screen_buffer
    return state[:, :, None] 

# class Crop(object):
#     """Crops the given PIL.Image using given pixel coordinates.

#     Args:
#         i – int, Upper pixel coordinate.
#         j – int, Left pixel coordinate.
#         h – int, Height of the cropped image.
#         w – int, Width of the cropped image.
#     """

#     def __init__(self, i, j, h, w):
#         self.i = i
#         self.j = j
#         self.h = h
#         self.w = w

#     def __call__(self, img):
#         """
#         Args:
#             img (PIL.Image): Image to be cropped.

#         Returns:
#             PIL.Image: Cropped image.
#         """
#         return img.crop((self.i, self.h, self.j, self.w))
    
# def transforms(crop = False, coords = (30, 300, 60, 180), resize = (120, 160)):
#     """
#     Description
#     -------------
#     Preprocess image screen before feeding it to a neural network.
    
#     Parameters
#     -------------
#     crop   : boolean, whether to crop or not (default=False)
#     coords : tuple, when crop is True, the coordinates to apply cropping (default=(30, 300, 60, 180)).
#              Be careful to the value of this parameter with respect to the screen resolution.
#     resize : tuple, shape of the resized frame (default=(120,160))
    
#     Returns
#     -------------
#     torchvision.transforms.transforms.Compose object, the composed transformations.
#     """
    
#     if crop:
#         return T.Compose([T.ToPILImage(),
#                     Crop(coords[0], coords[1], coords[2], coords[3]),
#                     T.Resize(resize),
#                     T.ToTensor()])
        
#     else:
#         return T.Compose([T.ToPILImage(),
#                     T.Resize(resize),
#                     T.ToTensor()])

def transforms(resize = (120, 160)):
    """
    Description
    -------------
    Preprocess image screen before feeding it to a neural network.
    
    Parameters
    -------------
    resize : tuple, shape of the resized frame (default=(120,160))
    
    Returns
    -------------
    torchvision.transforms.transforms.Compose object, the composed transformations.
    """
    
    return T.Compose([T.ToPILImage(),
                T.Resize(resize),
                T.ToTensor()])
    
def stack_frames(stacked_frames, state, is_new_episode, maxlen = 4, resize = (120, 160)):
    """
    Description
    --------------
    Stack multiple frames to create a notion of motion in the state.
    
    Parameters
    --------------
    stacked_frames : collections.deque object of maximum length maxlen.
    state          : the return of get_state() function.
    is_new_episode : boolean, if it's a new episode, we stack the same initial state maxlen times.
    maxlen         : Int, maximum length of stacked_frames (default=4)
    resize         : tuple, shape of the resized frame (default=(120,160))
    
    Returns
    --------------
    stacked_state  : 4-D Tensor, same information as stacked_frames but in tensor. This represents a state.
    stacked_frames : the updated stacked_frames deque.
    """
    
    # Preprocess frame
    frame = transforms(resize)(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([frame[None] for i in range(maxlen)], maxlen=maxlen) # We add a dimension for the batch
        # Stack the frames
        stacked_state = torch.cat(tuple(stacked_frames), dim = 1)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame[None]) # We add a dimension for the batch
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = torch.cat(tuple(stacked_frames), dim = 1)
    
    return stacked_state, stacked_frames


"""
epsilon-greedy
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, model, possible_actions):
    """
    Description
    -------------
    Epsilon-greedy policy
    
    Parameters
    -------------
    explore_start    : Float, the initial exploration probability.
    explore_stop     : Float, the last exploration probability.
    decay_rate       : Float, the rate at which the exploration probability decays.
    state            : 4D-tensor (batch, motion, image)
    model            : models.DQNetwork or models.DDDQNetwork object, the architecture used.
    possible_actions : List, the one-hot encoded possible actions.
    
    Returns
    -------------
    action              : np.array of shape (number_actions,), the action chosen by the greedy policy.
    explore_probability : Float, the exploration probability.
    """
    
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*decay_step)
    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
        
    else:
        if use_cuda:
            Qs = model.forward(state.cuda())
            
        else:
            Qs = model.forward(state)
            
        action = possible_actions[int(torch.max(Qs, 1)[1][0])]

    return action, explore_probability

"""
Double Q-learning tools
"""
def update_target(current_model, target_model):
    """
    Description
    -------------
    Update the parameters of target_model with those of current_model
    
    Parameters
    -------------
    current_model, target_model : torch models
    """
    target_model.load_state_dict(current_model.state_dict())


"""
Make gif
"""
def make_gif(images, fname, fps=50):

    def make_frame(t):
        try:
            x = images[int(fps*t)]
        except:
            x = images[-1]
        return x.astype(np.uint8)
    myfps = fps
    clip = mpy.VideoClip(make_frame, duration=len(images)/fps)
    clip.fps = fps
    clip.write_gif(fname, program='ffmpeg', fuzz=50, verbose=False)







