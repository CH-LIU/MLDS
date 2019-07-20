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
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
        self.epsilon = 0.2
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

    def criterion(self, state_prob, sample_state_prob, sample_rewards):
        ratio = torch.exp(state_prob) /torch.exp(sample_state_prob)
        loss1 = ratio * sample_rewards
        loss2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * sample_rewards
        loss = -torch.min(loss1, loss2).mean()
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
        self.model.train()
        optimizer = optim.RMSprop(self.model.parameters(), lr=0.0001)
        total_rewards = []
        #env.seed(self.seed)
        for i in range(self.episode_num):
            next_state = self.reset()
            prev_state = np.zeros((80, 80, 1))
            done = False
            episode_reward = 0.0
            sample_state_prob = []
            sample_states = []
            sample_rewards = []
            sample_actions = []
            log_prob = []
            count = 0
            ############### sampling ########################
            while(not done):
                count += 1
                cur_state = self.prepro(next_state)
                state = cur_state - prev_state
                sample_states.append(state)
                state = torch.Tensor(state).to(device)
                #state = torch.Tensor(state).to(device)
                state = state.unsqueeze(0)
                action_prob = self.model(state)
                action_prob = Categorical(action_prob[0])
                action = action_prob.sample()
                prob = action_prob.log_prob(action)
                sample_state_prob.append(prob)
                sample_actions.append(action)
                #prob = cate_prob.log_prob(action)
                action = action.cpu().numpy().astype(int)
                next_state, reward, done, info = self.step(action+1)
                episode_reward += reward
                #log_prob.append(prob)
                sample_rewards.append(reward)
                prev_state = cur_state
            
            total_rewards.append(episode_reward)

            running_add = 0
            decay = 0.99
            for j in range(len(sample_rewards)-1 , -1 , -1):
                running_add = running_add * decay + sample_rewards[j]
                sample_rewards[j] = running_add

            sample_rewards -= np.mean(sample_rewards)
            sample_rewards /= np.std(sample_rewards)
           
            ################ playing ########################
            sample_states = torch.Tensor(sample_states).to(device).detach()
            sample_actions = torch.Tensor(sample_actions).to(device).detach()
            sample_state_prob = torch.Tensor(sample_state_prob).to(device).detach()
            sample_rewards = torch.Tensor(sample_rewards).to(device).detach()
            
            for update_step in range(5):
                
                action_prob = self.model(sample_states)
                action_prob = Categorical(action_prob)
                state_prob = action_prob.log_prob(sample_actions)
               
                cur_loss = self.criterion(state_prob, sample_state_prob, sample_rewards)
                print(cur_loss)
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()


            torch.save({'model_pg': self.model.state_dict()},"model_pg")
            print("Episode:%d,  Reward:%.2f,   steps:%d"%(i,episode_reward,count))
        
            if i%100 == 0:
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

