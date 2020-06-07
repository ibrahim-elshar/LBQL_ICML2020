# -*- coding: utf-8 -*-
import numba
from numba import vectorize, jitclass, njit, jit    # import the decorator
from numba import int32, float32    # import the types
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from cs_sta_conf import Stations
import itertools
import time
from scipy.special import comb, binom
import bisect

def get_tuples(length, total):
    if length == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for t in get_tuples(length - 1, total - i):
            yield (i,) + t

def bc(N,k):
    return np.round(binom(N,k)).astype(int)

def get_idx(s):
    N = np.arange(1,s.shape[-1])
    ps = s[...,::-1].cumsum(-1)
    return (bc(ps[...,1:]+N,N) - bc(ps[...,:-1]+N,N)).sum(-1)

@njit
def mult(m,n,p,w):
    wij=np.zeros((n,n), dtype=np.int32)
    proportions=np.zeros((n,n))
    B = np.zeros((n, m, n))
    for i in range(n):   
            B[i] = np.random.multinomial(1,p[i], m)
            proportions[i,:] = np.sum(B[i], axis=0)/m
            if w[i]!=0:                    
                wij[i,:] =np.sum(B[i,0:w[i]], axis=0)                 
            else:
                wij[i,:]=np.zeros(n)
    return wij, B, proportions

@njit
def calc_mult(w, p):
        n = w.shape[0]
        wij=np.zeros((n,n), dtype=np.int32)
        for i in range(n):        
                if w[i]!=0:
                    wij[i,:]=np.random.multinomial(w[i], p[i])
                else:
                    wij[i,:]=np.zeros(n)    
        return wij

        
#from numba import vectorize, int64, float64
@vectorize([int32(int32, int32)])
def vec_randunif(l, h):
    return np.random.uniform(l,h)

@njit
#@jit(locals={'new_observation': numba.types.int32[:]}, nopython=True)  
def calc(observation, action, price, action_L, high, p, distance_ij, lost_sales_cost  ):
    
    epsilons = vec_randunif(action_L, high)
    demand = action + epsilons
    w = np.minimum(demand, observation)
    wij = calc_mult(w, p)

    num_lost_sales = demand - w
    dwij=np.multiply(distance_ij, wij)
    
    lost_sales_cost=np.sum(np.multiply(num_lost_sales ,  lost_sales_cost))

    profit = np.sum(np.multiply(np.sum(dwij, axis=1) , price))

    reward = profit - lost_sales_cost
    new_observation = observation  + np.sum(wij, axis=0) - w

    observation = new_observation

    return observation, reward, epsilons

def env_wrapper(env):
        observation = env.observation
        action_L = env.action_L
        high = env.high
        p = env.stations.prob_ij
        distance_ij = env.stations.distance_ij
        lost_sales_cost = env.stations.lost_sales_cost
        return observation, action_L, high, p, distance_ij, lost_sales_cost

class CarSharN(gym.Env):

    def __init__(self, SIMULATOR_SEED = 3323, GAMMA = 0.95):
        
        self.prng_simulator = np.random.RandomState(SIMULATOR_SEED)
        self.seed()
        self.stations =  Stations()
        
        self.action_L=self.stations.dmin
        self.action_H=self.stations.dmax
        self.high = self.action_H + 1
        self.action_space = spaces.Box(low=self.action_L, 
                                       high= self.action_H, dtype=int)
        
        ranges=[] 
        for i in range(self.stations.num_stations):
            ranges.append(range(self.stations.dmin[i], self.stations.dmax[i]+1))
        self.actions=np.array(list(itertools.product(*ranges)))
        self.price=self.stations.P(self.actions)
        
        self.observation_space =  spaces.Discrete(self.stations.num_cars)
        
        self.gamma = GAMMA
        
        self.dimS = self.stations.num_stations
        self.dimA = self.stations.num_stations
        self.nA = self.actions.shape[0]
        self.nS = comb(self.dimS+self.stations.num_cars-1,self.dimS-1,True)
##########################################
######## for doing the DP         
        num_cars = self.stations.num_cars
        num_stations = self.stations.num_stations
        
        self.states_list = list(get_tuples(num_stations,num_cars))
        self.states =np.array(self.states_list)
            
        
    def mult(self,m,n,p,w):
        wij=np.zeros((n,n), dtype=np.int32)
        proportions=np.zeros((n,n))
        B = np.zeros((n, m, n))
        for i in range(n):   
                B[i] = self.np_random.multinomial(1,p[i], m)
                proportions[i,:] = np.sum(B[i], axis=0)/m
                if w[i]!=0:                    
                    wij[i,:] =np.sum(B[i,0:w[i]], axis=0)                 
                else:
                    wij[i,:]=np.zeros(n)
        return wij, B, proportions    
    
    def find_indices(self, indices):
       if len(indices.shape)==1:
            return bisect.bisect_left(self.states_list, tuple(indices.tolist()))
       elif len(indices) > 100:
            # Faster to generate all indices when we have a large
            # number to check
            return get_idx(indices)
       else:
            return [bisect.bisect_left(self.states_list, tuple(i)) for i in indices.tolist()]    
        
    def step(self, action):
        assert self.action_space.contains(action)
        price = self.stations.P(action)
#################################### NUMBA ####################################        
#        observation, action_L, high, p, distance_ij, lost_sales_cost = env_wrapper(self)
#        self.observation, reward, epsilons = calc(observation, action, price,
#                            action_L, high, p, distance_ij, lost_sales_cost  )
##        self.observation = self.observation.astype(int)
#        return self.observation, reward, {'noise':epsilons}
###############################################################################     
        epsilons = self.np_random.uniform(low=-self.stations.eps_supp,
                                          high=self.stations.eps_supp+1).astype(int)
        demand = action + epsilons
        w = np.minimum(demand, self.observation)
        wij, B, proportions = self.mult(self.stations.num_cars,self.stations.num_stations,self.stations.prob_ij,w)
        num_lost_sales = demand - w
        dwij=np.multiply(self.stations.distance_ij, wij)
        lost_sales_cost=np.dot(num_lost_sales ,  self.stations.lost_sales_cost)
        profit = np.dot(np.sum(dwij, axis=1) , price)
        reward = np.around(profit - lost_sales_cost, 2)
        self.observation = self.observation  + np.sum(wij, axis=0) - w
        return self.observation, reward, {'noise' : {'epsilons':epsilons, 'P': proportions, 'B': B}}
         
    def seed(self, seed=None):
        ''' sets the seed for the envirnment'''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]    

    def reset(self):
        self.observation=np.multiply(np.ones(self.stations.num_stations), 
                        self.stations.num_cars/self.stations.num_stations).astype(int) 
        diff = self.stations.num_cars - sum(self.observation) 
        if diff != 0:
            self.observation[0] = diff
            
        return self.observation
    
    def simulate_sample_path(self):
        '''TODO'''
        n = self.stations.num_stations
        tau = self.prng_simulator.geometric(p=1-self.gamma)
        sp = np.zeros((n, tau))
        sample_path_B = np.zeros([tau, n, self.stations.num_cars, n])
        sample_path_P = np.zeros((tau,n,n))
        for i in range(n):
            sp[i] = self.prng_simulator.choice(self.stations.eps[i], tau, p=self.stations.stat_prob_eps[i])   
            for j in range(tau):
                sample_path_B[j,i] = self.prng_simulator.multinomial(1,self.stations.prob_ij[i], self.stations.num_cars)
                sample_path_P[j] = np.sum(sample_path_B[j], axis=1)/self.stations.num_cars
        sample_path_eps = sp.T
        
        return sample_path_eps, sample_path_P, sample_path_B
    
if __name__=='__main__':
    env =  CarSharN() 
    env.reset()

    
    