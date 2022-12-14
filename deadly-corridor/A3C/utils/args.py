import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Doom A3C parameters')

    parser.add_argument("--scenario", type=str, choices=('basic', 'deadly_corridor', 'defend_the_center', 'defend_the_line', 'my_way_home'), 
                        default="deadly_corridor", help="Doom scenario")
    parser.add_argument("--actions", type=str, choices=('all','single'), 
                        default="all", help="Possible actions : 'all' for combinated actions and 'single' for single actions")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of processes for parallel algorithms  '-1' to use all available cpus")
    
    # checkpoints directory tree
    # checkpoints/ {name} /
    # |
    # ---- model
    # ---- frames
    # ---- summary
    # ---- player_gifs
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Path to save model checkpoints and intermediate results")
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
    
    # Added parameters
    parser.add_argument("--name", type=str, required=True, help="Name of training instance, used to save checkpoints")


    game_args, _ = parser.parse_known_args()
    
    game_args.name += "_" + game_args.scenario
    if game_args.use_ppo:
        game_args.name += "_ppo"
    if game_args.use_curiosity:
        game_args.name += "_curiosity"        
    if game_args.no_reward:
        game_args.name += "_noreward"
    game_args.save_dir += "/" + game_args.name
    return game_args
