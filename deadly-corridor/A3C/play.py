import os
import tensorflow as tf
from agent import *

def play_with_agent(params):
    
    if not os.path.exists(params.gif_path):
        os.makedirs(params.gif_path)
    
    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    
    tf.reset_default_graph()

    with tf.Session(config = configs) as sess:
        Agent = Worker(0, state_size, action_size, player_mode=True)

        print('Loading Model...')
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(params.model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Successfully loaded!')

        Agent.play_game(sess, params.play_episodes)