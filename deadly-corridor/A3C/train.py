import os
import shutil
import threading
import multiprocessing
import tensorflow as tf

from agent import *
from utils.networks import *

from time import sleep
from time import time

def train_agents():
    tf.reset_default_graph()
    
    # checkpoints directory tree
    # checkpoints/ {name} /
    # |
    # ---- model
    # ---- frames
    # ---- summary
    # ---- player_gifs
    
    save_dir = os.path.join(params.save_dir, params.name)
    model_path = os.path.join(save_dir, 'model')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(os.path.join(save_dir, 'frames')):
        os.makedirs(os.path.join(save_dir, 'frames'))
    if not os.path.exists(os.path.join(save_dir, 'summary')):
        os.makedirs(os.path.join(save_dir, 'summary'))
    if not os.path.exists(os.path.join(save_dir, 'player_gifs')):
        os.makedirs(os.path.join(save_dir, 'player_gifs'))

    with tf.device("/cpu:0"): 
        # Generate global networks : Actor-Critic and ICM
        master_network = AC_Network(state_size, action_size, 'global') # Generate global AC network
        if params.use_curiosity:
            master_network_P = StateActionPredictor(state_size, action_size, 'global_P') # Generate global AC network
        
        # Set number of workers
        if params.num_workers == -1:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = params.num_workers
        
        # Create worker classes
        workers = []
        for i in range(num_workers):
            trainer = tf.train.AdamOptimizer(learning_rate=params.lr)
            workers.append(Worker(i, state_size, action_size, trainer, model_path))
        saver = tf.train.Saver(max_to_keep=5)

        
    with tf.Session() as sess:
        # Loading pretrained model
        if params.load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # Starting initialized workers, each in a separate thread.
        coord = tf.train.Coordinator()
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(params.max_episodes,params.gamma,sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)