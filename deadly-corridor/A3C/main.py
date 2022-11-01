import tensorflow as tf
from utils.network_params import *
import os

import time

from play import play_with_agent
from train import train_agents


if __name__ == '__main__':
    save_dir = os.path.join(params.save_dir, params.name)
    model_path = os.path.join(save_dir, 'model')

    if params.play:
        play_with_agent(params, model_path)
    else:
        train_agents(save_dir, model_path)
        
