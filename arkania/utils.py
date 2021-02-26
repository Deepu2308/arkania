# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 05:15:37 2021

@author: deepu
"""

#import libraries
import numpy as np
from torch import Tensor
#from random import shuffle

def discounted_mean_rewards(rewards, gamma = .9, clip_lower = -100):
    """compute discounted rewards"""
    disc_rewards = []
    running_rewards = 0
    rewards = [max(clip_lower,i) for i in rewards]
    for r in reversed(rewards):
        
        running_rewards = r + gamma*running_rewards
        disc_rewards.append(running_rewards)
    
    #disc_rewards = np.array(disc_rewards)
    #disc_rewards = disc_rewards - np.mean(disc_rewards)
    #disc_rewards = disc_rewards/(max(disc_rewards) - min(disc_rewards))#/np.std(disc_rewards)
    return disc_rewards[::-1], sum(rewards)

def get_state(state_dict):
    """currently just returns sight"""
    return state_dict['sight'].reshape((1,-1))


def get_experience(experience):
    
    #ind = list(range(len(experience)))
    
    
    return (Tensor([i[0] for i in experience]), #states
            Tensor([i[1] for i in experience]),           #actions
            [i[2] for i in experience])           #rewards

def play_game(model_path = 'models/VPG/4e5dab8860.pt', n = 5):
    """ cd into src folder and then call this"""
    
    import gym
    import torch    
    from model import LunarLander    
    from torch.distributions import Categorical
    
    net =  torch.load(model_path)
    
    env = gym.make('LunarLander-v2')
    
    for _ in range(n):
    
        state = env.reset()
        done  = False
        experience = [] #(S,A,R)
        
        #play and collect experience
        while not done:
            
            env.render()
            
            #get network output (policy)
            net.eval()
            tensored_state = torch.Tensor(state.reshape((1,8))).cuda()
            net_out = net(tensored_state)
            
            #sample from policy
            action  = Categorical(net_out).sample()
            
            #step
            new_state, reward, done, _ = env.step(action.item())
            
            #collect experience
            experience.append((state, action, reward))
            
            #update state
            state = new_state
        
    env.close()