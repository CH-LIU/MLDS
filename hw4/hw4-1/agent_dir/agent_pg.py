from agent_dir.agent import Agent
import scipy
from scipy import misc
import numpy as np
from skimage.color import rgb2gray
import torch
import torch.distributions
from torch.distributions import Categorical
import torch.optim as optim
from model import *

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

def prepro(o,image_size=[80,80,1]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = o.astype(np.uint8)
    y = o[:, :, 0]*0.2126 + o[:, :, 1]*0.7152 + o[:, :, 2]*0.0722
    resized = misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)
    #return resized


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        
        super(Agent_PG,self).__init__(env)
        self.model = Model((80, 80, 1), 3)
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            checkpoint = torch.load("./model_pg")
            self.model.load_state_dict(checkpoint['model_pg'])
            self.model.eval()

       
        ##################
        # YOUR CODE HERE #
        ##################
        self.seed = 11037
        env.seed(self.seed)
        self.reset = env.reset
        self.step = env.step
        self.episode_num = 50000
        self.get_action_space = env.get_action_space
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.model.to(device)
        self.prepro = prepro

    def criterion(self, state_prob, rewards, episode_reward, decay = 0.99):
        '''loss = torch.zeros(1).to(device)
        for p in state_prob:
            loss += -p*episode_reward
        return loss'''
        running_add = 0
        for i in range(len(rewards)-1 , -1 , -1):
			#if rewards[i] != 0 : running_add = 0
            running_add = running_add * decay + rewards[i]
            rewards[i] = running_add

        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)

        loss = torch.zeros(1).to(device)
        for i in range(len(rewards)):
            loss += -state_prob[i]*rewards[i]
        return loss
    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        #image = prepro(image)
        #misc.imsave('outfile1.jpg', image)
        self.init_game_setting()
        total_rewards = []
        self.model.train()
        optimizer = optim.RMSprop(self.model.parameters(), lr=0.0001)
        loss = torch.zeros(1).to(device)
        #env.seed(self.seed)
        for i in range(self.episode_num):
            next_state = self.reset()
            prev_state = torch.zeros((80, 80, 1)).to(device)
            done = False
            episode_reward = 0.0
            state_prob = []
            rewards = []
            #playing one game
            count = 0
            while(not done):
                count += 1
                cur_state = self.prepro(next_state)
                cur_state = torch.Tensor(cur_state).to(device)
                state = cur_state - prev_state
                action_prob = self.model(state)
                action_prob = Categorical(action_prob)
                action = action_prob.sample()
                prob = action_prob.log_prob(action)
                action = action.cpu().numpy().astype(np.int)
                next_state, reward, done, info = self.step(action+1)
                episode_reward += reward
                state_prob.append(prob)
                rewards.append(reward)
                prev_state = cur_state
            total_rewards.append(episode_reward)

            cur_loss = self.criterion(state_prob, rewards, episode_reward)
            loss += cur_loss
            if i % 1 == 0 :
                optimizer.zero_grad()
                loss /= 1
                loss.backward()
                optimizer.step()
                loss = torch.zeros(1).to(device)
                torch.save({'model_pg': self.model.state_dict()},"model_pg")
            print("Episode:%d,  Reward:%.2f,   steps:%d,   loss:%.6f"%(i,episode_reward,count,cur_loss))
            if i % 100 == 0:    
                np.save("total_rewards.npy", np.array(total_rewards))


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = torch.Tensor(observation).to(device)
        action_prob = self.model(observation)
        prob, action = action_prob.max(0)
        action = action.cpu().numpy().astype(np.int)
        
        return action
        #return self.env.get_random_action()

