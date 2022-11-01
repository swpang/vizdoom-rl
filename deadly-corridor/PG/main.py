import torch
from params import *
import os

import time

from test import play_with_agent
from train import train_agents


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda_is_available() else "cpu")

    if params.play:
        play_with_agent(params)
    else:
        train_agents()