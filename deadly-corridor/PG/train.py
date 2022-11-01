from args import *
from utils import *
from agents import ReinforceAgent

def train_agents():
    game, possible_actions = create_environment(scenario = params.scenario, window = params.window)
    
    # ReinforceAgent (inputshape,
    #                 actionsize
    #                 seed
    #                 device
    #                 gamma
    #                 lr
    #                 actorcnn)
    # TODO
    agent = ReinforceAgent(possible_actions, params.scenario, memory = params.memory_type, max_size = params.memory_size, stack_size = params.stack_size, 
                 batch_size = params.batch_size, resize = params.resize)
    agent.train(game, total_episodes = params.total_episodes, pretrain = params.pretrain, frame_skip = params.frame_skip, enhance = params.enhance, lr = params.lr, max_tau = params.max_tau, explore_start = params.explore_start, explore_stop = params.explore_stop, decay_rate = params.decay_rate, gamma = params.gamma, freq = params.freq, init_zeros = params.init_zeros)
