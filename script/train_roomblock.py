#!/usr/bin/python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
import rospy
from roomblock_gym.roomblock_gym import RoomblockApi, RoomblockEnv
import argparse

import logging

# エピソード長は固定時間 10秒分とか
if __name__ == '__main__':
    rospy.init_node('train_roomblock')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--epsilon', type=float, default=0.4)
    args = parser.parse_args(rospy.myargv()[1:])

    api = RoomblockApi()
    env = RoomblockEnv(api)

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    obs = env.reset()
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print "action_size: ", action_size
    print('initial observation:', obs)

    # predefined Q-function
    '''
    q_func = chainerrl.q_functions.FCQuadraticStateQFunction(
        obs_size, action_size,
        n_hidden_channels=300,
        n_hidden_layers=2,
        action_space=env.action_space)
    '''

    q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size, action_size, n_hidden_layers=2, n_hidden_channels=1000) 

    #q_func = QFunction(obs_size, action_size)

    # Use Adam to optimize q_func. eps=1e-2 is for stability.
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(q_func)    

    # Set the discount factor that discounts future rewards.
    gamma = 0.95

    # Use epsilon-greedy for exploration
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=args.epsilon, random_action_func=env.action_space.sample)

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    # Since observations from CartPole-v0 is numpy.float64 while
    # Chainer only accepts numpy.float32 by default, specify
    # a converter as a feature extractor function phi.
    phi = lambda x: x.astype(np.float32, copy=False)

    # Now create an agent that will interact with the environment.
    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=100, update_frequency=1,
        target_update_frequency=10, phi=phi)

    if args.load:
        agent.load(args.load)

    # set loop rate to 5 Hz
    rate = rospy.Rate(5)
    # number of episodes for training
    episode_size = 10000
    # number of process in a episode
    process_size = 25
    for i in range(1, episode_size + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0  # return (sum of rewards)
        t = 0  # time step
        while not done and t < process_size:
            if rospy.is_shutdown():
                print('Interrupted.')
                agent.save('agent')
                exit(0)
            # Uncomment to watch the behaviour
            # env.render()
            action = agent.act_and_train(obs, reward)
            obs, reward, done, _ = env.step(action)
            # print obs
            R += reward
            t += 1
            rate.sleep()
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
        agent.stop_episode_and_train(obs, reward, done)
    print('Finished.')
    agent.save('agent')
