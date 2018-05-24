import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import pickle
import pdb
import random
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

session = tf.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg_3 import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

flag=[1,1,1]
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="bonus_2", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=10, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=2, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=10, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--online_display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=128, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def CNN_model(input, index, scope="CNN",num_outputs=12, batch_size=10, reuse=True):


    # This model takes as input an observation and returns values of all actions
    # try:
    #     tf.get_variable(scope+"/conv2d/kernel")
    #     scope_t.reuse_variables()
    # except ValueError:
    #     print("new")
    global flag
    reuse_t=True
    #print(scope)
    if flag[index]==1:
        flag[index]=0
        reuse_t=False

    #print(flag,reuse_t)
    with tf.variable_scope(scope, reuse=reuse_t) as scope_t:

        """Model function for CNN."""
        # Input Layer
        input_layer = tf.reshape(input,[-1, 56, 86, 1])
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=16,

          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')
        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
          inputs=pool1,

          filters=32,

          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same')
        # Dense Layer

        pool2_flat = tf.reshape(pool2, [-1,9856])


        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
               inputs=dense, rate=0.4, training=True)
        context = tf.layers.dense(inputs=dropout, units=num_outputs)
        return(context)

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, obs_map_shape_n,arglist):
    trainers = []
    model = mlp_model
    map_model=CNN_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            env,"agent_%d" % i, model, map_model,obs_shape_n, obs_map_shape_n, env.action_space,i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            env,"agent_%d" % i, model, map_model,obs_shape_n,obs_map_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers



def input_new_obs():

    #my_pos=np.array(get_my_pos())/100*0.075     
    my_pos=np.array([1,1])/100*0.075     
    #my_velocity=np.array(get_my_velocity())/100*0.075       #max_speed 2666mm/s
    my_velocity=np.array([1,1])/100*0.075       #max_speed 2666mm/s
    #enemy_pos=np.array(get_enemy_pos())/100*0.075
    enemy_pos=np.array([[10,10],[-1,-1]])/100*0.075
    #enemy_velocity=np.array(get_enemy_velovity())/100*0.075
    enemy_velocity=np.array([[2,2],[-1,-1]])/100*0.075
    my_shooting_angle=5             #0 to 7
    #my_shooting_angle=get_shooting_angle()             #0 to 7
    #my_bonus_status=get_bonus_status()                 #1 to 6
    my_bonus_status=4                 #1 to 6

    global map_world
    current_map=np.copy(map_world)
    other_pos=[]
    other_vel=[]
    for i in range(2):
        if (enemy_pos[i][0]==-1) and (enemy_pos[i][0]==-1):
            other_pos.append([-1,-1])
            other_vel.append([-1,-1])
        else:
            other_pos.append((enemy_pos[i] - my_pos)/6)
            current_map[(enemy_pos[i][0]/0.075+3).astype(int)][(enemy_pos[i][1]/0.075+3).astype(int)]=-1
            other_vel.append(enemy_velocity[i] - my_velocity)

    current_map[(my_pos[0]/0.075+3).astype(int)][(my_pos[1]/0.075+3).astype(int)]=1

    tttt=np.concatenate([my_velocity] + [my_pos/6] + other_pos + other_vel)
    tt=np.array([0,0,0,0,0,0,0,0])
    tt[my_shooting_angle]=1
    bonus=np.array([0,0,0,0,0])
    bonus[my_bonus_status-1]=1
    ob=np.concatenate((tttt,tt,bonus)) #ob length=22
    current_map=np.reshape(current_map,-1)

    result=np.array([ob,current_map])
    return result

arglist = parse_args()
with U.make_session(1):
    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    map_world=np.copy(env.world.map_world)
    # Create agent trainers
    obs_shape_n = [[25] for i in range(env.n)]
    obs_map_shape_n =[[56*86] for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)
    trainers = get_trainers(env, num_adversaries, obs_shape_n, obs_map_shape_n,arglist)
    my_agent=trainers[2]
    print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

    # Initialize
    U.initialize()

    # Load previous results, if necessary
    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
    #if arglist.display or arglist.restore or arglist.benchmark:
    print('Loading previous state...')
    U.load_state(arglist.load_dir)
    my_obs= (env._reset())[0]

    print('Starting iterations...')
    while True:
        my_action= np.argmax(my_agent.action(my_obs))
        my_obs=input_new_obs()
        print(my_action)

#  /UWBPOS/

