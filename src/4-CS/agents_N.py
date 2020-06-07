# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import timeit
import collections
import random
import itertools
from numba import jit, njit, prange
import numba
import pickle
from scipy.special import comb,binom
import bisect
from carsharing_N import CarSharN

##############################################################################
    
#                            Q-learning

##############################################################################

class Qlearning():
    '''Implementation of the Q-leaning Agent 
       example:
           ql = Qlearning(env, 0, 0.95,0.4,10000,[(5,16),(4,18),(7,14)],1000)
           Q_list = ql.train(interactive_plot = True, verbose = True)
    '''
    
    def __init__(self, ENV,
                       SEED,
                       GAMMA=0.95, 
                       eps_greedy_par=0.4,
                       NUM_STEPS=500001,
                       interactive_plot_states=[(871,0),(1020,0),(894,0)],#[(463,0),(564,1),(453,8)],#[(1712,0),(1685,1),(2184,18)],#[(10,16),(11,16),(9,16)],#[(1151,9),(1275,5),(839,8)],#[(5,10),(4,19),(9,11)],#[(108,10),(101,19),(112,11)],#[(276,10),(314,19),(129,11)],
                       num_to_save_Q=2500,
                       polynomial_lr = 0.5,
                       Lzero = -100,
                       Uzero = 100,
                       ):
        self.env = ENV
        self.env.seed(SEED)
        self.env.action_space.seed(SEED)
        self.env.gamma = GAMMA
        self.gamma = GAMMA
        self.prng = np.random.RandomState(SEED) #Pseudorandom number generator
        print('SEED_ENV:',SEED,'SEED:',SEED)
        self.count = np.zeros([self.env.nS, self.env.nA])
        self.Lzero = Lzero
        self.Uzero = Uzero
        L = np.ones((self.env.nS, self.env.nA)) * self.Lzero/(1-self.gamma)
        U = np.ones((self.env.nS, self.env.nA)) * self.Uzero/(1-self.gamma)
        self.Q =  self.prng.uniform(L,U) 
        self.eps_greedy_par = eps_greedy_par
        self.num_steps = NUM_STEPS
        self.SQLsa_1 = []
        self.SQLsa_2 = []
        self.SQLsa_3 = []
        
        self.Q_list = []
        self.sa1 = interactive_plot_states[0]
        self.sa2 = interactive_plot_states[1]
        self.sa3 = interactive_plot_states[2]
        
        self.num_to_save_Q = num_to_save_Q
        
        self.perc_flags = [0,0,0,0,0]
        self.polynomial_lr = polynomial_lr
        
        self.rel_er_stps = []
        self.rel_er_times = []
        
        self.most_visited_sa =[]


    def find_indices(self, indices):
       if len(indices.shape)==1:
            return bisect.bisect_left(self.env.states_list, tuple(indices.tolist()))
       elif len(indices) > 100:
            # Faster to generate all indices when we have a large
            # number to check
            return get_idx(indices)
       else:
            return [bisect.bisect_left(self.env.states_list, tuple(i)) for i in indices.tolist()]


    def epsilon_greedy_policy(self, q_values,state_idx, force_epsilon=None):
        '''Creates epsilon greedy probabilities to actions sample from.
           Uses state visit counts.
        '''               
        eps = None
        if force_epsilon:
            eps = force_epsilon
        else:
            # Decay epsilon, save and use
            d = np.sum(self.count[state_idx,:]) if np.sum(self.count[state_idx,:]) else 1 
            eps = 1./d**self.eps_greedy_par
            self.epsilon = eps
        if self.prng.rand() < eps:
             action_idx = self.prng.choice(self.env.nA,1)[0]
             action = self.env.actions[action_idx]
        else:
            action_idx = np.argmax(q_values)
            action = self.env.actions[action_idx]
        return action, action_idx    

    def greedy_policy(self, q_values):
        '''Creating greedy policy to get actions from'''
        action_idx = np.argmax(q_values)
        action = self.env.actions[action_idx]
        return action, action_idx

    def lr_func(self,n):
        """ Implements a polynomial learning rate of the form (1/n**w)
        n: Integer
            The iteration number
        w: float between (0.5, 1]
        Returns 1./n**w as the rate
        """
        assert n > 0, "Make sure the number of times a state action pair has been observed is always greater than 0 before calling polynomial_learning_rate"
    
        return 1./n**self.polynomial_lr

    def initialize_plot(self, Title, xlabel, ylabel):
        ''' 
        Initialize interactive plots that shows how L, Q, U and Q-learning
        values for selected (s,a)'s are changing after each step.
        '''
        plt.ion()        
        self.fig, self.axes = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
        self.fig.text(0.5, 0.01, xlabel, ha='center')
        self.fig.text(0.005, 0.5, ylabel, va='center', rotation='vertical')
        self.fig.show()
        self.fig.canvas.draw()
        plt.tight_layout()
        
        self.fig.suptitle(Title, size=12) #In python 2.7 it is fontsize instead of size
        self.fig.subplots_adjust(top=0.95)
        
        self.fig.subplots_adjust(hspace=0, wspace=0) 
        
    def plot_on_running(self, to_plot, Labels, step):
        ''' 
        Call interactive plots that shows how L, Q, U and Q-learning
        values for selected (s,a)'s are changing after each step.
        '''
        a, b, c = to_plot.shape
        for i, ax in enumerate(self.axes):
            ax.clear()
            for j in range(b):     
                ax.plot(to_plot[i][j], label=Labels[i][j])
            plt.setp(ax.get_xticklabels(), visible=True)
            ax.grid()
            ax.legend(loc='best')  
        plt.setp(self.axes[2].get_xticklabels(), visible=True)
        self.fig.canvas.draw()   # draw
        plt.pause(0.000000000000000001)          
            

    
    def train(self, interactive_plot = False, verbose = False):
        ''' Trains the Q-learning agent'''
        start_time = timeit.default_timer()
        
        
        #interactive plot: initialise the graph and settings
        self.Q_list.append(np.copy(self.Q))
        if interactive_plot:
            self.initialize_plot('Standard Q-learning', 'Time steps','Action-value')
            Labels = np.array([['Q-learning['+str(self.sa1)+']'],
                                ['Q-learning['+str(self.sa2)+']'],
                                ['Q-learning['+str(self.sa3)+']']])
            
        # initialize state
        state = self.env.reset()
        state_idx = self.find_indices(state)
        for step in range(self.num_steps):
            
            # choose an action based on epsilon-greedy policy
            q_values = self.Q[state_idx, :]
            action, action_idx = self.epsilon_greedy_policy(q_values, state_idx)
            
            self.most_visited_sa.append((state_idx,action_idx))
            
            self.count[state_idx, action_idx] += 1   
            # execute action
            newState, reward, info = self.env.step(action)
            newState_idx =  self.find_indices(newState)
            self.lr = self.lr_func(self.count[state_idx, action_idx])    
            # Q-Learning update
            self.Q[state_idx, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Q[newState_idx, :])\
                                                      - self.Q[state_idx, action_idx])
            if verbose:    
                print('Step:',step, 'reward:',"{:.2f}".format(reward), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:', state_idx, )          
            state = newState
            state_idx = newState_idx
            
            if interactive_plot:
                #print new Action-values after every step
                self.SQLsa_1.append(self.Q[self.sa1])
                self.SQLsa_2.append(self.Q[self.sa2])
                self.SQLsa_3.append(self.Q[self.sa3]) 
                self.Q_list.append(np.copy(self.Q))
                if step % 10000 == 0:
                    to_plot =np.array([[self.SQLsa_1], [self.SQLsa_2], [self.SQLsa_3]])
                    self.plot_on_running(to_plot, Labels, step)
            if (step % self.num_to_save_Q ) == 0 and (step>0):
                 self.Q_list.append(np.copy(self.Q))
            
        elapsed_time = timeit.default_timer() - start_time
        print('Time=',elapsed_time)
        return self.Q_list, elapsed_time

##############################################################################
    
#                                SARSA

##############################################################################

class SARSA(Qlearning):
    """
    SARSA algorithm.
    """
    def __init__(self, ENV, SEED):
        super(SARSA, self).__init__(ENV, SEED)

    def train(self, verbose = False):
        start_time = timeit.default_timer()
        self.Q_list.append(np.copy(self.Q))

         # initialize state
        state = self.env.reset()
        
        for step in range(self.num_steps):
            
            
            # choose an action based on epsilon-greedy policy
            q_values = self.Q[state, :]
            action, action_idx = self.epsilon_greedy_policy(q_values)
            self.count[state, action_idx] += 1    
            # execute action
            newState, reward, info = self.env.step(action)
            self.lr = self.lr_func(self.count[state, action_idx])    
            # SARSA update
            self.Q[state, action_idx] +=  self.lr *(reward + self.gamma* self.Q[newState, action_idx]\
                                                      - self.Q[state, action_idx])
            if verbose:    
                print('Step:',step, 'reward:',"{:.2f}".format(reward), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:', state, )                 
            
            state = newState
            if (step % self.num_to_save_Q ) == 0 and (step>0):
                 self.Q_list.append(np.copy(self.Q))
       
        elapsed_time = timeit.default_timer() - start_time
        print(elapsed_time)
        return self.Q_list, elapsed_time
    
##############################################################################
    
#                                Speedy Q-Learning

##############################################################################
        
class SpeedyQLearning(Qlearning):
    """
    Speedy Q-Learning algorithm.
    "Speedy Q-Learning". Ghavamzadeh et. al.. 2011.
    """
    def __init__(self, ENV, SEED,):
        super(SpeedyQLearning, self).__init__(ENV, SEED,)
        self.Q_old = np.copy(self.Q)
        
    def train(self, verbose= False):
        ''' Trains the Q-learning agent'''
        start_time = timeit.default_timer()

        self.Q_list.append(np.copy(self.Q))

         # initialize state
        state = self.env.reset()
        state_idx = self.find_indices(state)
        for step in range(self.num_steps):
            
            old_q = np.copy(self.Q)
            
            # choose an action based on epsilon-greedy policy
            q_values = self.Q[state_idx, :]
            action, action_idx = self.epsilon_greedy_policy(q_values,state_idx)
            self.count[state_idx, action_idx] += 1    
            # execute action
            next_state, reward, info = self.env.step(action)
            next_state_idx = self.find_indices(next_state)
            max_q_cur = np.max(self.Q[next_state_idx, :]) 
            max_q_old = np.max(self.Q_old[next_state_idx, :]) 
            
            target_cur = reward + self.gamma * max_q_cur
            target_old = reward + self.gamma * max_q_old
            
#            alpha = 1/ (self.count[state, action_idx] + 1)
            self.lr = self.lr_func(self.count[state_idx, action_idx]) 
            alpha = self.lr             
            q_cur = self.Q[state_idx, action_idx]
            
            self.Q[state_idx, action_idx] = q_cur + alpha * (target_old - q_cur) + (
            1. - alpha) * (target_cur - target_old)

            self.Q_old = np.copy(old_q)
            
            if verbose:    
                print('Step:',step, 'reward:',"{:.2f}".format(reward), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:', state_idx, )             
            
            state = next_state
            state_idx = next_state_idx
            if (step % self.num_to_save_Q ) == 0 and (step>0):
                 self.Q_list.append(np.copy(self.Q))
         
        elapsed_time = timeit.default_timer() - start_time
        print(elapsed_time)
        return self.Q_list, elapsed_time    

##############################################################################
    
#                                Double Q-Learning

##############################################################################    
    
class DoubleQLearning(Qlearning):
    """
    Double Q-Learning algorithm.
    "Double Q-Learning". Hasselt H. V.. 2010.
    """
    def __init__(self, ENV, SEED):
        super(DoubleQLearning, self).__init__(ENV, SEED)
        self.Qprime = np.copy(self.Q)
        self.countprime = np.copy(self.count)

        
    def train(self, verbose=False):
        ''' Trains the Q-learning agent'''
        start_time = timeit.default_timer()

        self.Q_list.append(np.copy(self.Q))

         # initialize state
        state = self.env.reset()
        state_idx = self.find_indices(state)
        for step in range(self.num_steps):
             
            # choose an action based on epsilon-greedy policy
            q_values = (self.Q[state_idx, :] + self.Qprime[state_idx, :] )/2
            action, action_idx = self.epsilon_greedy_policy(q_values,state_idx)
               
            # execute action
            newState, reward, info = self.env.step(action)
            newState_idx = self.find_indices(newState) 
            
            # Double Q-Learning update
            
            if np.random.uniform() < .5:
                self.count[state_idx, action_idx] += 1 
                self.lr = self.lr_func(self.count[state_idx, action_idx])
                self.Q[state_idx, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Qprime[newState_idx, :])\
                                                      - self.Q[state_idx, action_idx])
                if (step % self.num_to_save_Q ) == 0 and (step>0):
                    self.Q_list.append((np.copy(self.Q)+np.copy(self.Qprime))/2)
                
            else:
                self.countprime[state_idx, action_idx] += 1 
                self.lr = self.lr_func(self.countprime[state_idx, action_idx])
                self.Qprime[state_idx, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Q[newState_idx, :])\
                                                      - self.Qprime[state_idx, action_idx])
                if (step % self.num_to_save_Q ) == 0 and (step>0):
                    self.Q_list.append((np.copy(self.Qprime)+np.copy(self.Q))/2)
                    
            if verbose:    
                print('Step:',step, 'reward:',"{:.2f}".format(reward), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:', state_idx, )                  

            state = newState
            state_idx = newState_idx
        elapsed_time = timeit.default_timer() - start_time
        print(elapsed_time)
        return self.Q_list, elapsed_time

##############################################################################
    
#                               Bias Corrected Q-leaning

##############################################################################    

class Bias_corrected_QL(Qlearning):
    '''Implementation of a Bias Corrected Q-leaning Agent '''
    
    def __init__(self, ENV, SEED, K = 20):
        super(Bias_corrected_QL, self).__init__(ENV, SEED)
        self.K = K
        self.BR = np.zeros((self.env.nS, self.env.nA))
        self.BT = np.zeros((self.env.nS, self.env.nA))
        self.Rvar = np.zeros((self.env.nS, self.env.nA))    
        self.Rmean = np.zeros((self.env.nS, self.env.nA)) 
        self.n_actions = self.env.actions.shape[0] 
        self.count = np.ones((self.env.nS, self.env.nA)) 
        self.T = np.zeros((self.env.nS, self.env.nA, self.K)).astype(int) 
        #self.n_eps = np.zeros(self.env.nS)
        
    
    def train(self, verbose = False):
         start_time = timeit.default_timer()
         self.Q_list.append(np.copy(self.Q))
         # initialize state
         state = self.env.reset()
         state_idx = self.find_indices(state)
         for step in range(self.num_steps):
            # choose an action based on epsilon-greedy policy
            q_values = self.Q[state_idx, :] #+ self.U[state, :] - self.L[state,:]           
            action, action_idx = self.epsilon_greedy_policy(q_values, state_idx)
            
            
            self.lr = self.lr_func(self.count[state_idx, action_idx])  #1000/(1000+step)
            # execute action
            newState, reward, info = self.env.step(action)
            newState_idx = self.find_indices(newState)
            self.T[state_idx,action_idx, :-1] = self.T[state_idx,action_idx, 1:]
            self.T[state_idx,action_idx, -1] = int(newState_idx)
            #self.memory.append((state, action_idx,reward))
            
            prevMean = self.Rmean[state_idx, action_idx]
            prevVar = self.Rvar[state_idx, action_idx]
            prevSigma = np.sqrt(prevVar/self.count[state_idx, action_idx])
    
            self.Rmean[state_idx, action_idx] = prevMean + (reward - prevMean)/self.count[state_idx, action_idx]
            self.Rvar[state_idx, action_idx] = (prevVar + (reward- prevMean)*(reward - self.Rmean[state_idx, action_idx]))/self.count[state_idx, action_idx]
            
            bM= np.sqrt(2*np.log(self.n_actions +7) - np.log(np.log(self.n_actions + 7)) - np.log(4*np.pi))
            self.BR[state_idx, action_idx]=(np.euler_gamma/bM + bM)*prevSigma
            self.BT[state_idx, action_idx]=self.gamma *(np.max(self.Q[newState_idx,:]) - np.mean(np.max(self.Q[self.T[state_idx,action_idx],:],axis=1)))
            delta =  self.Rmean[state_idx, action_idx]  + self.gamma * np.max(self.Q[newState_idx, :]) - self.Q[state_idx, action_idx]
            self.BR[state_idx, action_idx] = self.BR[state_idx, action_idx] if self.count[state_idx, action_idx] >=2 else 0.0
            self.BT[state_idx, action_idx] = self.BT[state_idx, action_idx] if self.count[state_idx, action_idx] >=self.K else 0.0
           
            self.Q[state_idx, action_idx] +=   self.lr * (delta -self.BR[state_idx, action_idx] - self.BT[state_idx, action_idx])

       
            self.count[state_idx, action_idx] += 1    
            state = newState
            state_idx = newState_idx
            if verbose:    
                print('Step:',step, 'reward:',"{:.2f}".format(reward), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:', state_idx, )             
            
            if (step % self.num_to_save_Q ) == 0 and (step>0):
                self.Q_list.append(np.copy(self.Q))

         elapsed_time = timeit.default_timer() - start_time
         print("Time="+str(elapsed_time))
         return self.Q_list, elapsed_time  

##############################################################################
    
#                            Lookahead Bounded Q-Learning

##############################################################################    

    
class Replay_Memory():

    def __init__(self, burn_in, memory_size=40, SEED=678, GAMMA= 0.95, K= 20 ):
        # The memory essentially stores transitions recorded from the agent
 
        self.memory = collections.deque(maxlen=memory_size)
        self.burn_in = burn_in
        self.prng_simulator = np.random.RandomState(SEED)
        random.seed(SEED)
        self.gamma = GAMMA
        self.K = K
        
    def ref_freq(self):
#        v=np.array(np.unique(self.memory, return_counts=True)).T.astype(float)
        v,i=np.unique(self.memory, return_counts=True, axis=0)
#        s=np.array(list(zip(v,i)))
#        s[:,1]= s[:,1]/len(self.memory)
        i=i/len(self.memory)
        return v,i
    
    
    def simulate_sample_path(self):
        '''TODO'''
        tau = self.prng_simulator.geometric(p=1-self.gamma, size=1)[0]   
        #sample_path = random.sample(self.memory, tau)   # no replacement
        sample_path = random.choices(self.memory, k=tau) # with replacement
#        eps, dist = self.ref_freq()
        return np.array(sample_path)
    
    def sample(self):
        '''TODO'''
        sample = random.choices(self.memory, k=self.K) 
        return np.array(sample)
    
    def append(self, transition):
        # Appends transition to the memory.     
        self.memory.append(transition)
        
    def multi_append(self, *args):
        n = len(args)
        # Appends transition to the memory.
        for i in range(n):
            self.memory.append(args[i])

def bc(N,k):
#    return np.round(comb(N,k)).astype(int)
    return np.round(binom(N,k)).astype(int)


def get_idx(s):
    N = np.arange(1,s.shape[-1])
    ps = s[...,::-1].cumsum(-1)
    return (bc(ps[...,1:]+N,N) - bc(ps[...,:-1]+N,N)).sum(-1)
        
def reduce_cumulative(a, i, j, ufunc=np.add, axis=2):
    i = (a.shape[axis] + i) % a.shape[axis]
    j = (a.shape[axis] + j) % a.shape[axis]
    a = np.insert(a, 0, 0, axis)
    c = ufunc.accumulate(a, axis=axis)
    pre = np.ix_(*(range(x) for x in i.shape))
    l = len(i.shape) - axis
    return c[pre[l:] + (j,)] - c[pre[l:] + (i,)]

def sliced_reduce(a, i, j, ufunc=np.add, axis=2):
    indices = np.tile(
        np.repeat(
            np.arange(np.prod(a.shape[:axis])) * a.shape[axis],
            2
        ),
        np.prod(i.shape[:len(i.shape) - axis])
    )
    indices[::2] += (a.shape[axis] + i.ravel()) % a.shape[axis]
    indices[1::2] += (a.shape[axis] + j.ravel()) % a.shape[axis]
    indices = indices.reshape(-1, 2)[::-1].ravel()  # This seems to be counter-effective, please check for your own case.
    result = ufunc.reduceat(a.reshape(-1, *a.shape[axis+1:]), indices)[::2]  # Select only even to odd.
    result[indices[::2] == indices[1::2]] = ufunc.reduce([])
    return result[::-1].reshape(*(i.shape + a.shape[axis+1:]))

def sliced_sum_numba(a, i, j, axis=2):
    i = (a.shape[axis] + i) % a.shape[axis]
    j = (a.shape[axis] + j) % a.shape[axis]
    m = np.prod(i.shape[:len(i.shape) - axis], dtype=int)
    n = np.prod(i.shape[len(i.shape) - axis:], dtype=int)
    a_flat = a.reshape(-1, *a.shape[axis:])
    i_flat = i.ravel()
    j_flat = j.ravel()
    result = np.empty((m*n,) + a.shape[axis+1:], dtype=a.dtype)
    numba_sum(a_flat, i_flat, j_flat, m, n, result)
    return result.reshape(*(i.shape + a.shape[axis+1:]))

@numba.jit(parallel=True, nopython=True)
def numba_sum(a, i, j, m, n, out):
    for index in numba.prange(m*n):
        out[index] = np.sum(a[index % n, i[index]:j[index]], axis=0)


@jit(nopython=True, cache=True)
def comp(sample_path_eps,sample_path_B,nS, nA, Qp, QG, QL, EmaxQ_array, 
         mean_reward_array, next_states_indx):
    for t in range(len(sample_path_eps) - 1, -1, -1):
        for s_idx in  range(nS):
             for a_idx in range(nA):
                ns_idx =  next_states_indx[s_idx,a_idx, t]
                a_id =  np.argmax(Qp[ns_idx, :])
                EmaxQa = EmaxQ_array[s_idx, a_idx] 
                reward = mean_reward_array[s_idx, a_idx]
                a_QG_id = np.argmax(QG[t+1, ns_idx, :])
     
                QG[t, s_idx, a_idx] = reward +  QG[t+1, ns_idx, a_QG_id ] + EmaxQa - Qp[ ns_idx, a_id] 
                QL[t, s_idx, a_idx] = reward + QL[t+1, ns_idx, a_id ] + EmaxQa - Qp[ ns_idx, a_id]  
    return QG, QL   

def solve_inner_DP(sample_path_eps,sample_path_B,
                   env, Q, EmaxQ_array, mean_reward_array,
                   next_states_indx): 
    '''
    Solves the deterministic perfect information relaxation problem via backward induction 
    '''  
    nS = env.nS
    nA = env.nA
    tau =  len(sample_path_eps)   
    QG = np.zeros((tau + 1, nS, nA))
    QG[tau, :, :] = np.copy(Q)
    QL = np.zeros((tau + 1, nS, nA))
    QL[tau, :, :] = np.copy(Q)
    
    return comp(sample_path_eps,sample_path_B,nS, nA, Q, QG, QL, EmaxQ_array, 
                mean_reward_array, next_states_indx)
      
class LBQL(Qlearning):
    '''Implementation of a LBQL Agent '''
    
    def __init__(self, ENV, 
                       SEED,
                       L_LR=0.01, 
                       U_LR=0.01,
                       WITH_PENALTY=True,
                       BURN_IN = 1000,
                       USE_K = False,
                       K = 20,
                       memory_size=100000,
                       M =200,
                       relTol=0.01,
                       USE_SCHEDULE = False
                         ):
        super(LBQL, self).__init__(ENV, SEED,)
        self.L_lr = L_LR
        L_LR_SCHEDULE = np.ones(self.num_steps)
        L_LR_SCHEDULE[0:round(self.num_steps/2)] *= 0.1
        L_LR_SCHEDULE[round(self.num_steps/2):self.num_steps] *= 0.01
        self.L_lr_schedule = L_LR_SCHEDULE
        
        self.U_lr = U_LR
        U_LR_SCHEDULE = np.ones(self.num_steps)
        U_LR_SCHEDULE[0:round(self.num_steps/2)] *= 0.1
        U_LR_SCHEDULE[round(self.num_steps/2):self.num_steps] *= 0.01
        self.U_lr_schedule = U_LR_SCHEDULE

        self.L = np.ones((self.env.nS, self.env.nA)) * self.Lzero/(1-self.gamma)
        self.U = np.ones((self.env.nS, self.env.nA)) * self.Uzero/(1-self.gamma)

        self.USE_K = USE_K
        self.K = K
        self.M = M
        self.relTol = relTol
        self.with_penalty = WITH_PENALTY
        self.s_a_tuples = list(itertools.product(range(self.env.nS), range(self.env.nA)))
        self.burn_in = BURN_IN
        self.memory_eps = Replay_Memory(self.burn_in, memory_size= memory_size, SEED= SEED,
                                    GAMMA= self.gamma, K=self.K)
        self.memory_B = Replay_Memory(self.burn_in, memory_size= memory_size, SEED= SEED,
                            GAMMA= self.gamma, K=self.K)
        ###### for plotting/monitoring L, Q and U
        self.Q_learning = np.copy(self.Q)
        self.Lsa_1 = []
        self.Qsa_1 = []
        self.Usa_1 = []
        self.QLsa_1 = []
        self.Lsa_2 = []
        self.Qsa_2 = []
        self.Usa_2 = [] 
        self.QLsa_2 = []
        self.Lsa_3 = []
        self.Qsa_3 = []
        self.Usa_3 = []  
        self.QLsa_3 = []
        
        self.Q_list  = []
        self.QL_list = []
        self.L_list  = []
        self.U_list  = []
        self.Rmean = np.zeros((self.env.nS, self.env.nA)) 
        self.prevMean = np.zeros((self.env.nS, self.env.nA)) 
        self.num_called_DP = 0
        
        self.USE_SCHEDULE = USE_SCHEDULE
    def solve_QG_QL_DP(self, sample_path_eps,sample_path_B):
        ''' Computes the upper & lower bounds on the Q-values by solving the PI 
            and PI with nonanticipative policy problems, respectively via backward induction'''
        self.num_called_DP += 1
        tau =  len(sample_path_eps)   
        self.QG = np.zeros((tau + 1, self.env.nS, self.env.nA))
        self.QG[tau, :, :] = np.copy(self.Q)
        self.QL = np.zeros((tau + 1, self.env.nS, self.env.nA))
        self.QL[tau, :, :] = np.copy(self.Q)
        self.QD = np.zeros((tau + 1, self.env.nS, self.env.nA))
        self.backward_induction(sample_path_eps,sample_path_B)
    
    
    def f_r(self,sample_eps, sample_B):
            d_plus_eps = self.env.actions[:,None] + sample_eps
            w=np.minimum(self.env.states[:,None,None,:], d_plus_eps).astype(int)
            wij = reduce_cumulative(sample_B, np.zeros_like(w),w, np.add, axis=2)
            next_states = self.env.states[:,None,None,:]  + np.sum(wij,axis=-2) - w
            next_states = next_states.astype(int)
            next_states_indx=get_idx(next_states)
            num_lost_sales = d_plus_eps - w
            dwij=np.multiply(self.env.stations.distance_ij, wij)
            lost_sales_cost =  num_lost_sales@ self.env.stations.lost_sales_cost
            profit = np.sum(self.env.price[:,None,:,None] *dwij, axis=(3,4))
            reward = profit - lost_sales_cost
            mean_reward =  np.mean(reward, axis=2)
            return next_states, next_states_indx, mean_reward 
        
        
    def f_d(self,sample_eps, sample_B):
            d_plus_eps = self.env.actions[:,None] + sample_eps
            w=np.minimum(self.env.states[:,None,None,:], d_plus_eps).astype(int)
            wij = reduce_cumulative(sample_B, np.zeros_like(w),w, np.add, axis=2)
            next_states = self.env.states[:,None,None,:]  + np.sum(wij,axis=-2) - w
            next_states = next_states.astype(int)
            next_states_indx=get_idx(next_states)
            return next_states, next_states_indx     
        
    def Average_EmaxQa(self):#
        ''' Computes an estimated expected value using simulation '''    
        dim_s = self.sample_eps.shape[0]
        p = 1/dim_s*np.ones(dim_s)
        next_states, next_states_indx, mean_reward = self.f_r( self.sample_eps, self.sample_B)
        EmaxQa=np.dot(np.max(self.Q[next_states_indx, :],axis=3), p) #.reshape((self.env.nS,self.env.nA,dim_s))
        self.prevMean = self.Rmean
        self.Rmean = self.prevMean + (mean_reward - self.prevMean)/self.num_called_DP

        return self.gamma*EmaxQa, self.Rmean,next_states, next_states_indx 
    
    
    def backward_induction(self, sample_path_eps,sample_path_B): 
        '''
        Solves the deterministic perfect information relaxation problem via backward induction 
        '''  
        if self.USE_K == False:
            EmaxQ_array, mean_reward_array,next_states, next_states_indx = self.Average_EmaxQa()
        else:
            EmaxQ_array, mean_reward_array,next_states, next_states_idx = self.Average_EmaxQa()
            next_states, next_states_indx = self.f_d(sample_path_eps, sample_path_B)
        self.QG, self.QL = solve_inner_DP(sample_path_eps,sample_path_B, 
                                          self.env, self.Q, EmaxQ_array, 
                                          mean_reward_array,
                                          next_states_indx)
                                 
  
        
    def train(self, interactive_plot = False, verbose = True):
        ''' Trains the Q-learning with IR agent'''
        
        start_time = timeit.default_timer()
        
        if interactive_plot:
            #interactive plot: initialise the graph and settings
            Title =  'Q-learning with penalty' if self.with_penalty else 'Q-learning no penalty'
            self.initialize_plot(Title, 'Time steps','Action-value')
            Labels = np.array([['L['+str(self.sa1)+']', 'Q['+str(self.sa1)+']','U['+str(self.sa1)+']','Q-learning['+str(self.sa1)+']'],
                               ['L['+str(self.sa2)+']', 'Q['+str(self.sa2)+']','U['+str(self.sa2)+']','Q-learning['+str(self.sa2)+']'],
                               ['L['+str(self.sa3)+']', 'Q['+str(self.sa3)+']','U['+str(self.sa3)+']','Q-learning['+str(self.sa3)+']']])
            
            self.L_list.append(np.copy(self.L))
            self.U_list.append(np.copy(self.U))
        self.Q_list.append(np.copy(self.Q)) 
        
        self.QL_list.append(np.copy(self.Q_learning))
        # initialize state
        state = self.env.reset()
        state_idx = self.find_indices(state)
        
        for step in range(self.num_steps):

            if self.USE_SCHEDULE:
                self.U_lr= self.U_lr_schedule[step]
                self.L_lr= self.L_lr_schedule[step]

            # choose an action based on epsilon-greedy policy
            q_values =  self.Q[state_idx, :]           
            action, action_idx = self.epsilon_greedy_policy(q_values, state_idx)
            
            self.most_visited_sa.append((state_idx,action_idx))
            
            self.count[state_idx, action_idx] += 1
              
            self.lr = self.lr_func(self.count[state_idx, action_idx])  
            
            # execute action
            newState, reward, info = self.env.step(action)
            newState_idx = self.find_indices(newState)
            
            self.memory_eps.append(info['noise']['epsilons'])
            self.memory_B.append(info['noise']['B'])
            # Q-Learning update
            self.Q[state_idx, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Q[newState_idx, :]) -\
                                          self.Q[state_idx, action_idx])
            if interactive_plot:
            # Standard Q-Learning following same behavioral policy
                self.Q_learning[state_idx, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Q_learning[newState_idx, :]) -\
                                                       self.Q_learning[state_idx, action_idx])
            cond = (step % self.M ==0 and step >= self.burn_in) and \
            (not np.isclose(self.L[state_idx, action_idx],self.U[state_idx, action_idx],self.relTol) or \
             not self.L[state_idx, action_idx]<=self.Q[state_idx, action_idx] <=self.U[state_idx, action_idx])
            if cond:#        
                
                sample_path_eps  = self.memory_eps.simulate_sample_path()
                sample_path_B  = self.memory_B.simulate_sample_path()
                assert len(sample_path_eps)==len(sample_path_B)
                if self.USE_K:
                    self.sample_eps = self.memory_eps.sample()
                    self.sample_B = self.memory_B.sample()
                else:
                    self.sample_eps = sample_path_eps
                    self.sample_B = sample_path_B
                    
                time =timeit.default_timer()
                self.solve_QG_QL_DP(sample_path_eps, sample_path_B)                
                if self.with_penalty: 
                        self.U += self.U_lr * (self.QG[0,:, :] - self.U)
                    
                else:
                        self.U += self.U_lr * (self.QD[0,:, :] - self.U)
                    
                self.L += self.L_lr * (self.QL[0,:, :] - self.L)

            self.Q[state_idx, action_idx] = np.maximum(np.minimum(self.U[state_idx, action_idx],\
                                                  self.Q[state_idx, action_idx]),self.L[state_idx, action_idx])
             
            state = newState
            state_idx = newState_idx
 
            if interactive_plot:
                #print new Action-values after every 10000 step  
                self.Lsa_1.append(self.L[self.sa1])
                self.Qsa_1.append(self.Q[self.sa1])
                self.Usa_1.append(self.U[self.sa1]) 
                self.QLsa_1.append(self.Q_learning[self.sa1])
                self.Lsa_2.append(self.L[self.sa2])
                self.Qsa_2.append(self.Q[self.sa2])
                self.Usa_2.append(self.U[self.sa2]) 
                self.QLsa_2.append(self.Q_learning[self.sa2])
                self.Lsa_3.append(self.L[self.sa3])
                self.Qsa_3.append(self.Q[self.sa3])
                self.Usa_3.append(self.U[self.sa3]) 
                self.QLsa_3.append(self.Q_learning[self.sa3])
                
                
    #            self.L_list.append(np.copy(self.L))
    #            self.U_list.append(np.copy(self.U))
            
                if step % 10000 == 0:
                    to_plot =np.array([[self.Lsa_1, self.Qsa_1, self.Usa_1, self.QLsa_1],
                                       [self.Lsa_2, self.Qsa_2, self.Usa_2, self.QLsa_2],
                                       [self.Lsa_3, self.Qsa_3, self.Usa_3, self.QLsa_3]])
                    self.plot_on_running(to_plot, Labels, step)
            if (step % self.num_to_save_Q ) == 0 and (step>0):
                 self.Q_list.append(np.copy(self.Q))
                 # self.QL_list.append(np.copy(self.Q_learning))

        elapsed_time = timeit.default_timer() - start_time
        print("Time="+str(elapsed_time))
        print(elapsed_time)
        return self.Q_list, self.QL_list, elapsed_time 
if __name__=='__main__':
    env =  CarSharN() 
    env.reset()
    agent = LBQL(env,1)
    q, ql,et =agent.train(interactive_plot= True)
    from collections import Counter
    Output = Counter(agent.most_visited_sa)
    Output.most_common(50)
    from matplotlib import rcParams
    import seaborn as sns

##############################################################################          
    sns.set(style="darkgrid")
    sns.set(font_scale=1.75)

    fig, ax = plt.subplots(1,1)
    length= len(agent.Lsa_2)
    #lbl=np.arange(0,length/1000,1).astype(int)
    lbl=[0,0,200,400]#np.arange(0,20000/1000+1,5)
    plt.plot(agent.Lsa_2, label='LBQL-L', lw=2,  alpha=0.8)
    plt.plot(agent.Qsa_2, label='LBQL-Q\'', lw=5,  alpha=0.8)
    plt.plot(agent.Usa_2, label='LBQL-U',lw=2,  alpha=0.8)
    plt.plot(agent.QLsa_2,color='k', label='QL-Q',lw=2, linestyle='-', alpha=0.8)
    plt.legend(loc='best', fontsize=15)#13.3)
#    plt.tight_layout()
    plt.xlabel('Number of steps x 1000')
    plt.ylabel('Action-value')
    ax.set_xticklabels(lbl)