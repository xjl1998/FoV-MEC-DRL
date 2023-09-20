
import math, random
from os.path import exists

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt

from DQN_calss import DQN
from replay_buffer import ReplayBuffer

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * \
                                     math.exp(-1. * frame_idx / epsilon_decay)


USE_CUDA =torch.cuda.is_available()
# SE_CUDA =
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
from collections import deque

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)

directory = './exp' + script_name +'./'
if not exists(directory):
    os.makedirs(directory)

class DQN_generate:
    def __init__(self,input_dims,num_actions,args,
            name= '',
    ):
        self.name = name
        self.gamma = args.gamma
        self.current_model = DQN(input_dims, num_actions)
        self.target_model  = DQN(input_dims, num_actions)
        self.optimizer = optim.Adam(self.current_model.parameters(),lr=args.learning_rate)
        self.replay_buffer = ReplayBuffer(args.capacity_DQN)
        self.loss = []

    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())



    def compute_td_loss(self,args):
        state, action, reward, next_state, done = self.replay_buffer.sample(args.batch_size)

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action     = Variable(torch.LongTensor(action))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(done))

        q_values      = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state)

        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + args.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.data)
        # return loss


    def plot(self,frame_idx, rewards):
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('loss'+self.name)
        plt.plot(self.loss)
        plt.show()


    def save(self,args):
        torch.save(self.current_model.state_dict(), directory + args.env_name +self.name + 'current.pth')
        torch.save(self.target_model.state_dict(), directory + args.env_name +self.name + 'target.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.current_model.load_state_dict(torch.load(directory +self.name+ 'current.pth'))
        self.target_model.load_state_dict(torch.load(directory +self.name+ 'target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


