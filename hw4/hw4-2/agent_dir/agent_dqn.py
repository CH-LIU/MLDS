from agent_dir.agent import Agent
import scipy
from scipy import misc
import numpy as np
from skimage.color import rgb2gray
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions
from torch.distributions import Categorical
import torch.optim as optim
from model_dqn import *
import random
import math

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
seed = 11037
'''from pyvirtualdisplay import Display
display = Display(visible=0, size=(1366, 768))
display.start()'''

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 10



class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.model = DQN(84, 84, 2)
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            checkpoint = torch.load("model_dqn")
            self.model.load_state_dict(checkpoint['model_dqn'])
            self.model.eval()
        ##################
        # YOUR CODE HERE #
        ##################
        self.target = DQN(84, 84, 2)
        #env.seed(seed)
        self.reset = env.reset
        self.step = env.step
        self.episode_num = 100000
        self.action_space = env.get_action_space
        self.episilon = EPS_START
        self.update_target_step = 1000
        self.update_step = 4
        self.learning_start = 50000
        self.memory = ReplayMemory(100000)

        #right:0,1,2 left:3
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.model.to(device)

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        total_rewards = []
        print("Starts Training....")
        self.init_game_setting()
        self.model.train()
        self.target.to(device)
        self.target.load_state_dict(self.model.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False
        #self.target.eval()
        optimizer = optim.RMSprop(self.model.parameters(), lr=0.00015)
        criterion = nn.MSELoss()
        total_steps = 1
        for e in range(self.episode_num):
            done = False
            next_state = torch.Tensor(self.reset()).to(device).unsqueeze(0)
            next_state = next_state.permute(0,3,1,2)
            next_state = next_state[:, 3, :, :] - next_state[:, 0, :, :]
            next_state = next_state.unsqueeze(1)
            episode_reward = 1
            action = torch.zeros(1).to(device)
            Q = torch.zeros(1).to(device)
            loss = torch.zeros(1).to(device)
            step = 1 
            while (not done):
                state = next_state.clone()
                Q_model = self.model(state)
                if total_steps < self.learning_start : action = np.random.randint(2)
                else:
                    if random.random() > self.episilon:
                        _, action = Q_model.max(1)
                        action = action[0].cpu().numpy().astype(np.int)
                    else:                 
                        action = np.random.randint(2)
                        #Q = Q_model[0][action]
                    self.episilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (total_steps-self.learning_start) / EPS_DECAY)    
                next_state, reward, done, info = self.step(action+2)
                if done: reward = -1
                next_state = torch.Tensor(next_state).to(device).unsqueeze(0)
                next_state = next_state.permute(0,3,1,2)
                next_state = next_state[:, 3, :, :] - next_state[:, 0, :, :]
                next_state = next_state.unsqueeze(1)

                self.memory.push((state.detach(), action, np.sign(reward), next_state.detach()))
                episode_reward += reward
                step += 1
                total_steps += 1
                
                if len(self.memory) < BATCH_SIZE: continue
                if total_steps % self.update_step == 0 and total_steps > self.learning_start:
                    batch = self.memory.sample(BATCH_SIZE)
                    s = []
                    r = []
                    a = []
                    n_s = []
                    for data in batch:
                        s.append(data[0])
                        a.append(data[1])
                        r.append(data[2])
                        n_s.append(data[3])
                    s = torch.stack(s).to(device).squeeze(1)
                    n_s = torch.stack(n_s).to(device).squeeze(1)
                    r = torch.Tensor(r).to(device)
                    #a = torch.Tensor(a).to(device)
                    Q_batch = self.model(s)
                    temp = []
                    for b in range(BATCH_SIZE):
                        temp.append(Q_batch[b][a[b]])
                    #Q_batch, _ = Q_batch.max(1)
                    Q_batch = torch.stack(temp).to(device)
                    Q_batch_target = self.target(n_s)
                    Q_batch_target, _ = Q_batch_target.max(1)
                    y = Q_batch_target.detach()*GAMMA + r
                    loss =  F.smooth_l1_loss(Q_batch, y)
                    optimizer.zero_grad()
                    loss.backward()
                    '''for param in self.model.parameters():
                        param.grad.data.clamp_(-1, 1)'''
                    optimizer.step()

                if total_steps % self.update_target_step == 0:
                    self.target.load_state_dict(self.model.state_dict())
                    for p in self.target.parameters():
                        p.requires_grad = False
                    torch.save({'model_dqn': self.model.state_dict()},"model_dqn")
            #print(self.episilon)
            total_rewards.append(episode_reward)
            if e % 100 == 0 : np.save("total_rewards", np.array(total_rewards)) 
            print("Episode:%d,  Reward:%.2f, step:%d"%(e, episode_reward, step))


            




    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = torch.Tensor(observation).to(device).unsqueeze(0)
        observation = observation.permute(0, 3, 1, 2)
        observation = observation[:, 3, :, :] - observation[:, 0, :, :]
        observation = observation.unsqueeze(1)
        Q_model = self.model(observation)
        Q, action = Q_model.max(1)
        action = action[0].cpu().numpy().astype(np.int)

        return action

