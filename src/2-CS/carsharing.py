# -*- coding: utf-8 -*-
'''
This file implements a vehicle sharing simulator.
The environment consists of two stations from which cars are rented in accordance
to a price-demand model with some noise. After setting the price at each station 
the demand is observed. The objective is to set the rental prices at each station 
during each period to maximize the total revenue.
'''
import gym
import numpy as np
from gym import spaces
# Stations information (see Stations_Config.py) (Stations class)
from stations_config import Stations
from gym.utils import seeding
import itertools
import pickle

class CarSharing(gym.Env):
    '''
    Creates the Carsharing Envirnoment.
   
    Inputs:
        - Simulator seed used to generate a sample path (int)
        - The discount factor gamma used in computing the probability 
          p = 1 - gamma of the Geometric distribution used to sample 
          tau (the sample path length)(float)
        
    Output:
        - A gym class for the car sharing problem
    '''
    def __init__(self, 
                 SIMULATOR_SEED = 3323,
                 GAMMA = 0.95):
        self.prng_simulator = np.random.RandomState(SIMULATOR_SEED)
        self.stations =  Stations()
        self.action_L=self.stations.dmin
        self.action_H=self.stations.dmax
        self.action_space = gym.spaces.Box(low=self.action_L, high= self.action_H, dtype=int)
        self.observation_space =  spaces.Discrete(self.stations.num_cars)
        self.seed() # seed to generate a RandomState forsampling the demand noise (epsilons)
        self.gamma = GAMMA
        # creates the epsilon range at each station (dictionary)
        self.eps_range = {}
        for i in range(self.stations.num_stations):
            self.eps_range[i] = np.arange(-self.stations.epsilons_support[i], 
                          self.stations.epsilons_support[i] + 1 )
        # distance between the stations (array)
        self.dij = np.array([self.stations.distance_ij[0], self.stations.distance_ij[1]]) 
        
        # create a vector of all possible noise combinations (n_ex2 array)
        self.epsilons = np.array([[i,j] for i in self.eps_range[0] for j in self.eps_range[1]])
        self.nE = self.epsilons.shape[0]
        # create a vector of all possible actions (demand) (nAx2 int array)
        self.actions = np.array([[i,j] for i in range(self.action_L[0],self.action_H[0]+1)\
                                for j in range(self.action_L[1],self.action_H[1]+1)])
        self.nA = self.actions.shape[0]
        # create a vector of all possible states (number of cars at one loc) (array)
        self.states = np.arange(self.stations.num_cars+1)
        self.nS = self.states.shape[0]
        # creates transition function (dic)
        self.trans = {}
        states_vec=np.append(self.states[:,None],self.stations.num_cars - self.states[:,None], axis=1)
        d_plus_eps = self.actions[:,None] + self.epsilons
        w=np.minimum(states_vec[:,None,None,:], d_plus_eps)
        num_lost_sales = d_plus_eps - w
        dwij=np.multiply(self.dij, w.astype(float))
        lost_sales_cost=np.multiply(self.stations.lost_sales_cost, num_lost_sales)
        price = self.stations.P(self.actions)
        profits = np.multiply(dwij , price[:,None,:])
        rewards = np.around(np.sum(profits - lost_sales_cost, axis=len(profits.shape)-1), 2)
        next_states = states_vec[:,0][:,None,None] + w[...,1] - w[...,0]
        eps = np.tile(self.epsilons, (states_vec.shape[0]*self.actions.shape[0],1))
        #ns_r_eps = np.stack([next_states, rewards, eps[:,0], eps[:,1]], axis=1)
        ns_r_eps =  zip(next_states.ravel(), rewards.ravel(),eps)
        itr=itertools.product(self.states, range(self.action_L[0],self.action_H[0]+1),
                    range(self.action_L[1],self.action_H[1]+1),
                    self.eps_range[0], self.eps_range[1])
        self.trans=dict(zip(itr,ns_r_eps) )
        
        ######################################################################       
        self.f=next_states.reshape(self.nS, self.stations.d_range[0], self.stations.d_range[1],
                        self.eps_range[0].shape[0],self.eps_range[1].shape[0] )     
        self.r=rewards.reshape(self.nS, self.stations.d_range[0], self.stations.d_range[1],
                        self.eps_range[0].shape[0],self.eps_range[1].shape[0] )  
        ################
        #scale rewards to [0,1]
#        rmax = np.max(self.r)
#
#        rmin = np.min(self.r)
#        
#        self.r=(self.r - rmin)/(rmax-rmin)
        # rewards negative from -1 to 0
#        self.r += -1.0
        ################
        
        self.i=self.action_L[0]
        self.j=self.action_L[1]
        self.k=self.stations.epsilons_support[0]
        self.l=self.stations.epsilons_support[1]
        # example (10, 5-i, 5-j, 2+k, 5+l)
        
        # max and min of rewards and expected rewards
        self.dim_eps_supp=self.stations.epsilons_support*2+1
        p =np.array(self.stations.prob_eps).reshape(self.dim_eps_supp[0],self.dim_eps_supp[1])
        self.mean_r = np.zeros((self.nS,self.nA))
        for s in range(self.nS):
            for a_idx in range(self.nA):
                a = self.actions[a_idx]
                self.mean_r[s,a_idx]=np.sum(p*self.r[s,a[0]-self.i,a[1]-self.j,:,:])
        self.Rmax =max(np.max(self.r), abs(np.min(self.r)) )        
        self.r_max =self.Rmax  
        self.r_min =-self.Rmax    
        
        # create transition probabilities           
        self.P=np.zeros((self.nS,self.nA,self.nS)) 
        self.T = np.empty((self.nS, self.nA), dtype=object)   
        for s in range(self.nS):
            for a_id in range(self.nA):
                a = self.actions[a_id]
                ns =[]
                prob =[]
                r=[]
                for it in range(len(self.epsilons)):
                    eps1, eps2 = self.epsilons[it]
                    ns.append(self.f[s,a[0]-self.i,a[1]-self.j,eps1+self.k,eps2+self.l])
                    r.append(self.r[s,a[0]-self.i,a[1]-self.j,eps1+self.k,eps2+self.l])
                    prob.append(self.stations.prob_eps[it])
                out_array = [np.where(ns == element)[0].tolist() for element in np.unique(ns)] 
                prob=np.array(prob)
                r=np.array(r)
                probb = [sum(prob[i]) for i in out_array]
                rr = [np.dot(prob[i],r[i]) for i in out_array]
                rr = np.array(rr)/np.array(probb)

                self.P[s,a_id][np.unique(ns)]= probb
                self.T[s,a_id]= list(zip(probb,np.unique(ns),rr))

    def step(self, action):
        """

        """
        assert self.action_space.contains(action)
        epsilon_idx = self.np_random.choice(self.nE, 1, p=self.stations.prob_eps)[0] 
        epsilon = self.epsilons[epsilon_idx]
        self.observation, reward, noise = self.trans[(self.observation, action[0],
                                                  action[1], epsilon[0],epsilon[1])]
        return self.observation, reward, {'noise':noise}
    
    def reset(self):
        self.observation=np.round(self.stations.num_cars/self.stations.num_stations,0).astype(int) 
        return self.observation
                 
    def render(self, mode='human', close=False):
        pass


    def seed(self, seed=None):
        ''' sets the seed for the envirnment'''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
         
    def simulate_sample_path(self):
        '''TODO'''
        tau = self.prng_simulator.geometric(p=1-self.gamma, size=1)[0]   
        sp =np.zeros((self.stations.num_stations, tau))     
        sp[0] = self.prng_simulator.choice(self.eps_range[0], tau, p=self.stations.prob_eps_1)
        sp[1] = self.prng_simulator.choice(self.eps_range[1], tau, p=self.stations.prob_eps_2)
        sample_path = sp.T
        return sample_path
    
    def _virtual_step_f(self, state, action, force_noise=None):
        '''
        '''
        if force_noise is None:
            noise_idx = self.np_random.choice(self.nE, 1, p=self.stations.prob_eps)[0] 
            noise = self.epsilons[noise_idx]
        else:   
            noise = force_noise        
        return  self.trans[(state, action[0],action[1], noise[0],noise[1])]



def Qiter(env, tol=1e-12,max_iters=1e10):
    ''' Q-iteration'''
    Q = np.zeros((env.nS, env.nA)) 
    new_Q= np.copy(Q)
    iters = 0
    while True:
        for state in range(env.nS):
            for action_idx in range(env.nA):
                val = 0
                for (prob, newState, reward) in env.T[state,action_idx]:
                        val += prob  * (reward + env.gamma * np.max(Q[newState,:]))
                new_Q[state, action_idx] = val
        if np.sum(np.abs(new_Q - Q)) < tol:
            Q = new_Q.copy()
            break    
        Q = new_Q.copy()
        iters += 1 
    return iters, Q  



if __name__=='__main__':
    env = CarSharing()
    env.reset()
    iters, Qstar =  Qiter(env)
    with open("Qstar.pkl", 'wb') as f:
        pickle.dump(Qstar, f,protocol=2)  
        