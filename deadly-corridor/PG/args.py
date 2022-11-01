import argparse
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='Doom A3C parameters')

    parser.add_argument("--scenario", type=str, choices=('basic', 'deadly_corridor', 'defend_the_center', 'defend_the_line', 'my_way_home'), 
                        default="deadly_corridor", help="Doom scenario")
    parser.add_argument("--actions", type=str, choices=('all','single'), 
                        default="all", help="Possible actions : 'all' for combinated actions and 'single' for single actions")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of processes for parallel algorithms  '-1' to use all available cpus")
    parser.add_argument("--model_path", type=str, default="./saves/model", help="Path to save models")
    parser.add_argument("--frames_path", type=str, default="./saves/frames", help="Path to save gifs")
    parser.add_argument("--summary_path", type=str, default="./saves/summary", help="Path to save training summary")
    parser.add_argument("--gif_path", type=str, default="./saves/player_gifs", help="Path to save playing agent gifs")
    parser.add_argument("--load_model", action="store_true", help="Either to load model or not")
    parser.add_argument("--max_episodes", type=int, default=1600, help="Maximum episodes per worker")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=30, help="Maximum steps per worker to update global network")
    parser.add_argument("--play", action="store_true", help="Launch agent to play")
    parser.add_argument("--play_episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--freq_model_save", type=int, default=200, help="Frequence of episodes to save model")
    parser.add_argument("--freq_gif_save", type=int, default=25, help="Frequence of episodes to save gifs")
    parser.add_argument("--freq_summary", type=int, default=5, help="Frequence of episodes to save gifs")
    parser.add_argument("--use_curiosity", action="store_true", help="Use curiosity")
    parser.add_argument("--use_ppo", action="store_true", help="Use PPO")
    parser.add_argument("--no_render", action="store_true", help="Disable window game while training")
    parser.add_argument("--no_reward", action="store_true", help="Disable extrinsic reward")


    game_args, _ = parser.parse_known_args()
    
    game_args.model_path += "/"+game_args.scenario
    game_args.frames_path += "/"+game_args.scenario
    game_args.summary_path += "/"+game_args.scenario
    game_args.gif_path += "/"+game_args.scenario
    
    if game_args.use_ppo:
        game_args.model_path += "_ppo"
        game_args.frames_path += "_ppo"
        game_args.summary_path += "_ppo"
        game_args.gif_path += "_ppo"
    
    if game_args.use_curiosity:
        game_args.model_path += "_curiosity"
        game_args.frames_path += "_curiosity"
        game_args.summary_path += "_curiosity"
        game_args.gif_path += "_curiosity"
        
    if game_args.no_reward:
        game_args.model_path += "_noreward"
        game_args.frames_path += "_noreward"
        game_args.summary_path += "_noreward"
        game_args.gif_path += "_noreward"
    
    return game_args


params = parse_arguments()

if params.scenario == 'deadly_corridor':
    resize = (100,181)
    crop = (30,-35,1,-1)
    
    if params.actions=='all':
        action_size = 10
    elif params.actions=='single':
        action_size = 6
    
    state_size = np.prod(resize)
    
elif params.scenario == 'basic':
    resize = (84,84)
    crop = (10,-10,30,-30)
    
    if params.actions=='all':
        action_size = 6
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)

        
elif params.scenario == 'defend_the_center':
    resize = (84,159)
    crop = (40,-32,1,-1)
    
    if params.actions=='all':
        action_size = 6
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)
    
elif params.scenario == 'defend_the_line':
    resize = (84,159)
    crop = (40,-32,1,-1)
    
    if params.actions=='all':
        action_size = 6
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)
    
elif params.scenario == 'my_way_home':
    resize = (84,112)
    crop = (1,-1,1,-1)
    
    if params.actions=='all':
        action_size = 5
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)
    

# ICM Module parameters

beta = 0.2
lr_pred = 10.0
pred_bonus_coef = 0.01

#constants = {
#            'GAMMA': 0.99,  # discount factor for rewards
#            'LAMBDA': 1.0,  # lambda of Generalized Advantage Estimation: https://arxiv.org/abs/1506.02438
#            'ENTROPY_BETA': 0.01,  # entropy regurarlization constant.
#            'ROLLOUT_MAXLEN': 20, # 20 represents the number of 'local steps': the number of timesteps
#                                # we run the policy before we update the parameters.
#                                # The larger local steps is, the lower is the variance in our policy gradients estimate
#                                # on the one hand;  but on the other hand, we get less frequent parameter updates, which
#                                # slows down learning.  In this code, we found that making local steps be much
#                                # smaller than 20 makes the algorithm more difficult to tune and to get to work.
#            'GRAD_NORM_CLIP': 40.0,   # gradient norm clipping
#            'REWARD_CLIP': 1.0,       # reward value clipping in [-x,x]
#            'MAX_GLOBAL_STEPS': 100000000,  # total steps taken across all workers
#            'LEARNING_RATE': 1e-4,  # learning rate for adam
#
#            'PREDICTION_BETA': 0.01,  # weight of prediction bonus
#                                      # set 0.5 for unsup=state
#            'PREDICTION_LR_SCALE': 10.0,  # scale lr of predictor wrt to policy network
#                                          # set 30-50 for unsup=state
#            'FORWARD_LOSS_WT': 0.2,  # should be between [0,1]
#                                      # predloss = ( (1-FORWARD_LOSS_WT) * inv_loss + FORWARD_LOSS_WT * forward_loss) * PREDICTION_LR_SCALE
#            'POLICY_NO_BACKPROP_STEPS': 0,  # number of global steps after which we start backpropagating to policy
#            }

