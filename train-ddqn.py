from inspect import stack
import torch
import torch.nn as nn
import numpy as np
import vizdoom as vzd
import argparse

import math
import itertools as it
from time import sleep, time
from tqdm import trange

from model_ddqn import DDQNAgent
from preprocessing import preprocess_frame, stack_frame
from utils import create_game

def opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--config', type=str, default='deadly_corridor.cfg')
    parser.add_argument('--scenario', type=str, default='deadly_corridor.wad')

    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--df', type=int, default=0.99)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--train_steps_per_epoch', type=int, default=10000)
    parser.add_argument('--test_episodes_per_epoch', type=int, default=10)
    parser.add_argument('--memory', type=int, default=100000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--eps_start', type=int, default=0.99)
    parser.add_argument('--eps_end', type=int, default=0.01)
    parser.add_argument('--eps_decay', type=int, default=100)

    parser.add_argument('--save_model', type=bool)
    parser.add_argument('--load_model', type=bool)
    parser.add_argument('--skip_learning', type=bool)

    args = parser.parse_args()
    return args

eps = lambda frame_idx: opt.eps_end + (opt.eps_start - opt.eps_end) * math.exp(-1. * frame_idx / opt.eps_decay)

# Other parameters
frame_repeat = 12
input_size = (4, 84, 84)
episodes_to_watch = 10

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')

def train(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))

        epoch_time = time()

        for i in trange(steps_per_epoch, leave=False):
            # stack frames
            state = preprocess_frame(game.get_state().screen_buffer.transpose(1, 2, 0), (0, -60, -40, 60), input_size[2])
            state = stack_frame(None, state, True)
            epsilon = eps(i)
            action = agent.get_action(state, epsilon)
            reward = game.make_action(actions[action], frame_repeat)
            
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess_frame(game.get_state().screen_buffer.transpose(1, 2, 0), (0, -60, -40, 60), input_size[2])
                next_state = stack_frame(None, state, True)
                agent.step(state, action, reward, next_state, done)
            else:
                agent.step(state, action, reward, state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)

        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        if opt.save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net, model_savefile)
        print("Epoch time: %.2f minutes" % ((time() - epoch_time) / 60.0))

        test(game, agent)

    print("Epoch time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game

def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for _ in trange(opt.test_episodes_per_epoch, leave=False):
        game.new_episode()
        score = 0

        while not game.is_episode_finished():
            state = preprocess_frame(game.get_state().screen_buffer.transpose(1, 2, 0), (0, -60, -40, 60), input_size[2])
            state = stack_frame(None, state, True)
            best_action_index = agent.get_action(state)
            score += game.make_action(actions[best_action_index], frame_repeat)
        test_scores.append(score)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f +/- %.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
          "max: %.1f" % test_scores.max())

if __name__ == '__main__':
    opt = opt()

    model_savefile = opt.name + '.pth'

    # Initialize game and actions
    game = create_game(opt.config)
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = DDQNAgent(action_size=len(actions), input_shape=input_size, lr=opt.lr, 
                      batch_size=opt.batch, memory_size=opt.memory, discount_factor=opt.df, 
                      load_model=opt.load_model, model_savefile=model_savefile, seed=opt.seed, 
                      device=DEVICE)

    # Run the training for the set number of epochs
    if not opt.skip_learning:
        agent, game = train(game, agent, actions, num_epochs=opt.epoch, frame_repeat=frame_repeat,
                          steps_per_epoch=opt.train_steps_per_epoch)

        print("======================================")
        print("Training completed")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess_frame(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)