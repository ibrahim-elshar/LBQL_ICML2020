# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import timeit
import collections
import random
import itertools
from numba import jit
import numba
import pickle

f = open("Qstar.pkl",'rb')
Qstar = pickle.load(f)
f.close()
Vstar= np.max(Qstar,1)
order=None
Vstar_norm=np.linalg.norm(Vstar, ord=order)

def polynomial_learning_rate(n, w=0.5):
    """ Implements a polynomial learning rate of the form (1/n**w)
    n: Integer
        The iteration number
    w: float between (0.5, 1]
    Returns 1./n**w as the rate
    """
    assert n > 0, "Make sure the number of times a state action pair has been observed is always greater than 0 before calling polynomial_learning_rate"

    return 1./n**w
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
                       GAMMA=0.90, 
                       eps_greedy_par=0.5,
                       NUM_EPISODES=300,# 250
                       interactive_plot_states=[(30,1),(5,1),(4,3)],
                       num_to_save_Q=200,
                       ):
        self.env = ENV
        self.env.seed(SEED)
        self.env.action_space.seed(SEED)
        self.env.gamma = GAMMA
        self.gamma = GAMMA
        self.prng = np.random.RandomState(SEED) #Pseudorandom number generator
        print('SEED_ENV:',SEED,'SEED:',SEED)
        self.lr_func = polynomial_learning_rate
        self.count = np.zeros([self.env.nS, self.env.nA])
        L = np.ones((self.env.nS, self.env.nA)) * self.env.r_min/(1-self.gamma)
        U = np.ones((self.env.nS, self.env.nA)) * self.env.r_max/(1-self.gamma)
        self.Q =  self.prng.uniform(L,U) #np.zeros((self.env.nS, self.env.nA))# Qstar.copy() #self.prng.uniform(L,U)
        self.Q[self.env.goal_state, :] = 0.0
        self.Q[self.env.goal_state, :] = 0.0
        self.eps_greedy_par = eps_greedy_par
        self.num_episodes = NUM_EPISODES
        self.SQLsa_1 = []
        self.SQLsa_2 = []
        self.SQLsa_3 = []
        
        self.Q_list = []
        self.sa1 = interactive_plot_states[0]
        self.sa2 = interactive_plot_states[1]
        self.sa3 = interactive_plot_states[2]
        
        self.num_to_save_Q = num_to_save_Q
        self.rel_er_stps = []
        self.rel_er_times = []
        self.perc_flags = [0,0,0,0,0]
    
    def epsilon_greedy_policy(self, q_values, force_epsilon=None):
        '''Creates epsilon greedy probabilities to actions sample from.
           Uses state visit counts.
        '''               
        eps = None
        if force_epsilon:
            eps = force_epsilon
        else:
            # Decay epsilon, save and use
            d = np.sum(self.count[self.env.observation,:]) if np.sum(self.count[self.env.observation,:]) else 1 
            eps = 1./d**self.eps_greedy_par
            self.epsilon = eps
        if self.prng.rand() < eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)
        return action  
  


    def greedy_policy(self, q_values):
        '''Creating greedy policy to get actions from'''
        action = np.argmax(q_values)
        return action 


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

    def rel_err_print(self,step,start_time):
        elapsed_time = timeit.default_timer() - start_time
        V = np.max(self.Q,1)
        temp = np.linalg.norm(Vstar - V, ord=order)/Vstar_norm
        if temp <= 0.5 and self.perc_flags[0] ==0:
            print('50% ->|Vstar-V|/|Vstar|=',"{:.2f}".format(temp), 'at step:',step, 'time=',elapsed_time )
            self.perc_flags[0] +=1
            self.rel_er_stps.append(step)
            self.rel_er_times.append(elapsed_time)
        if temp <= 0.2 and self.perc_flags[1] ==0:
            print('20% ->|Vstar-V|/|Vstar|=',"{:.2f}".format(temp), 'at step:',step, 'time=',elapsed_time )
            self.perc_flags[1] +=1
            self.rel_er_stps.append(step)
            self.rel_er_times.append(elapsed_time)            
        if temp <= 0.1 and self.perc_flags[2] ==0:
            print('10% ->|Vstar-V|/|Vstar|=',"{:.2f}".format(temp), 'at step:',step, 'time=',elapsed_time )
            self.perc_flags[2] +=1
            self.rel_er_stps.append(step)
            self.rel_er_times.append(elapsed_time)            
        if temp <= 0.05 and self.perc_flags[3] ==0:
            print('5% ->|Vstar-V|/|Vstar|=',"{:.2f}".format(temp), 'at step:',step, 'time=',elapsed_time )
            self.perc_flags[3] +=1 
            self.rel_er_stps.append(step)
            self.rel_er_times.append(elapsed_time)            
        if temp <= 0.01 and self.perc_flags[4] ==0:
            print('1% ->|Vstar-V|/|Vstar|=',"{:.2f}".format(temp), 'at step:',step, 'time=',elapsed_time )
            self.perc_flags[4] +=1
            self.rel_er_stps.append(step)
            self.rel_er_times.append(elapsed_time)            
            
        
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
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.grid()
            ax.legend(loc='best')  
        plt.setp(self.axes[2].get_xticklabels(), visible=True)
        self.fig.canvas.draw()   # draw
        plt.pause(0.000000000000000001) 
        # plt.savefig('Figures/fig'+str(step+1)+'.png', bbox_inches='tight')
        
    def train(self, interactive_plot = False, verbose = False):
        ''' Trains the Q-learning agent'''
        start_time = timeit.default_timer()
        ep = 0
        self.episodes = []
        self.Q_list.append(np.copy(self.Q))
        self.num_stps_eps = []        
        
        #interactive plot: initialise the graph and settings
        self.Q_list.append(np.copy(self.Q))
        if interactive_plot:
            self.initialize_plot('Standard Q-learning', 'Time steps','Action-value')
            Labels = np.array([['Q-learning['+str(self.sa1)+']'],
                                            ['Q-learning['+str(self.sa2)+']'],
                                            ['Q-learning['+str(self.sa3)+']']])
        step = 0
        for episode in range(self.num_episodes):
            # track the total time steps in this episode
            time = 0
            # initialize state
            state = self.env.reset()
            done = False
            
            # keep going until get to the goal state
            while not done and time<=200:
                # choose an action based on epsilon-greedy policy
                q_values = self.Q[state, :]
                action = self.epsilon_greedy_policy(q_values)
                self.count[state, action] += 1 
                newState, reward, done, info = self.env.step(action)
                self.lr = self.lr_func(self.count[state, action])    
                # Q-Learning update
                self.Q[state, action] +=  self.lr *\
                (reward + self.gamma* np.max(self.Q[newState, :]) - self.Q[state, action])
                state = newState
                self.rel_err_print(step, start_time)
                time += 1
                step += 1
                 
            if verbose:    
                print('Episode:',episode, 'time_steps:',time, 
                      'epsilon:', "{:.2f}".format(self.epsilon),         
                       'LR:', "{:.3f}".format(self.lr))
            
            if interactive_plot:
                #print new Action-values after every step
                self.SQLsa_1.append(self.Q[self.sa1])
                self.SQLsa_2.append(self.Q[self.sa2])
                self.SQLsa_3.append(self.Q[self.sa3]) 
                if episode % 1 == 0:
                    to_plot =np.array([[self.SQLsa_1], [self.SQLsa_2], [self.SQLsa_3]])
                    self.plot_on_running(to_plot, Labels, episode)

            self.Q_list.append(np.copy(self.Q))   # append Q every episode    
            self.episodes.extend([ep] * time)
            self.num_stps_eps.append(time)
            ep += 1
        elapsed_time = timeit.default_timer() - start_time
        print('Time=',elapsed_time)
        return  self.Q_list, self.episodes, self.num_stps_eps, elapsed_time

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
        ep = 0
        self.episodes = []
        self.num_stps_eps= []
        step = 0
        for episode in range(self.num_episodes):
            # track the total time steps in this episode
            time = 0
            # initialize state
            state = self.env.reset()
            
            done = False
            
            # keep going until get to the goal state
            while not done and time<=200:
                # choose an action based on epsilon-greedy policy
                q_values = self.Q[state, :]
                action = self.epsilon_greedy_policy(q_values)
                self.count[state, action] += 1    
                # execute action
                newState, reward, done, info = self.env.step(action)
                self.lr = self.lr_func(self.count[state, action])    
                # SARSA update
                self.Q[state, action] +=  self.lr *(reward + self.gamma* self.Q[newState, action]\
                                                          - self.Q[state, action])

                state = newState
                
                time += 1
                step += 1
            

            if verbose:    
                print('Episode:',episode, 'time_steps:',time, 
                      'epsilon:', "{:.2f}".format(self.epsilon),         
                       'LR:', "{:.3f}".format(self.lr))

            
            self.Q_list.append(np.copy(self.Q))
            self.episodes.extend([ep] * time)
            self.num_stps_eps.append(time)
            ep += 1
        elapsed_time = timeit.default_timer() - start_time
        print('Time=',elapsed_time)
        return self.Q_list, self.episodes, self.num_stps_eps, elapsed_time
    
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
        ''' Trains the SQL-learning agent'''
        start_time = timeit.default_timer()
        ep = 0
        self.episodes = []
        self.num_stps_eps = []
        self.Q_list.append(np.copy(self.Q))

        step=0
        for episode in range(self.num_episodes):
            # track the total time steps in this episode
            time = 0
            # initialize state
            state = self.env.reset()
            
            done = False
            # keep going until get to the goal state
            while not done and time<=200:
                old_q = np.copy(self.Q)
                
                # choose an action based on epsilon-greedy policy
                q_values = self.Q[state, :]
                action  = self.epsilon_greedy_policy(q_values)
                    
                # execute action
                next_state, reward, done, info = self.env.step(action)
            
                self.count[state, action] += 1

                max_q_cur = np.max(self.Q[next_state, :]) if not done else 0.
                max_q_old = np.max(self.Q_old[next_state, :]) if not done else 0.
            
                target_cur = reward + self.gamma * max_q_cur
                target_old = reward + self.gamma * max_q_old
                
                #alpha =  1/ (self.count[state, action] + 1)
                self.lr = self.lr_func(self.count[state, action]) 
                alpha = self.lr
                q_cur = self.Q[state, action]
            
                self.Q[state, action] = q_cur + alpha * (target_old - q_cur) + (
                1. - alpha) * (target_cur - target_old)

                self.Q_old = np.copy(old_q)
            
                state = next_state
                self.rel_err_print(step, start_time)
                time += 1
                step += 1

            if verbose:    
                print('Episode:',episode, 'time_steps:',time, 
                      'epsilon:', "{:.2f}".format(self.epsilon),         
                       'LR:', "{:.3f}".format(self.lr))
                
            self.Q_list.append(np.copy(self.Q)) # append Q every episode 
            self.episodes.extend([ep] * time)
            self.num_stps_eps.append(time)
            ep += 1
        elapsed_time = timeit.default_timer() - start_time
        print("Time="+str(elapsed_time))
        return self.Q_list, self.episodes, self.num_stps_eps,  elapsed_time    

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
        ep = 0
        self.episodes = []
        self.num_stps_eps = []
        self.Q_list.append(np.copy(self.Q))
        
        step=0
        for episode in range(self.num_episodes):
            # track the total time steps in this episode
            time = 0
            # initialize state
            state = self.env.reset()
            
            done = False
           # keep going until get to the goal state
            while not done and time<=200:
            # choose an action based on epsilon-greedy policy
                q_values = (self.Q[state, :] + self.Qprime[state, :] )/2
                action = self.epsilon_greedy_policy(q_values)
               
                # execute action
                newState, reward, done, info = self.env.step(action)

                # Double Q-Learning update
                rand = np.random.uniform()
                if  rand < .5:
                    self.count[state, action] += 1 
                    self.lr = self.lr_func(self.count[state, action])
                    self.Q[state, action] +=  self.lr *(reward + self.gamma* np.max(self.Qprime[newState, :])\
                                                          - self.Q[state, action])
                else:
                    self.countprime[state, action] += 1 
                    self.lr = self.lr_func(self.countprime[state, action])
                    self.Qprime[state, action] +=  self.lr *(reward + self.gamma* np.max(self.Q[newState, :])\
                                                          - self.Qprime[state, action])
                state = newState
                self.rel_err_print(step, start_time)
                time += 1
                step += 1
                    
            if verbose:    
               print('Episode:',episode, 'time_steps:',time, 
                      'epsilon:', "{:.2f}".format(self.epsilon),         
                       'LR:', "{:.3f}".format(self.lr))                

            self.Q_list.append((np.copy(self.Q)+np.copy(self.Qprime))/2) # append Q every episode 
            self.episodes.extend([ep] * time)
            self.num_stps_eps.append(time)
            ep += 1
        elapsed_time = timeit.default_timer() - start_time
        print("Time="+str(elapsed_time))
        return self.Q_list, self.episodes, self.num_stps_eps,  elapsed_time 


##############################################################################
    
#                               Bias Corrected Q-leaning

##############################################################################    

class Bias_corrected_QL(Qlearning):
    '''Implementation of a Bias Corrected Q-leaning Agent '''
    
    def __init__(self, ENV, SEED, K = 10):
        super(Bias_corrected_QL, self).__init__(ENV, SEED)
        self.K = K
        self.BR = np.zeros((self.env.nS, self.env.nA))
        self.BT = np.zeros((self.env.nS, self.env.nA))
        self.Rvar = np.zeros((self.env.nS, self.env.nA))    
        self.Rmean = np.zeros((self.env.nS, self.env.nA)) 
        self.n_actions = self.env.nA 
        self.count = np.ones((self.env.nS, self.env.nA)) 
        self.T = np.zeros((self.env.nS, self.env.nA, self.K)).astype(int) 
        
    
    def train(self, verbose = False):
         start_time = timeit.default_timer()
         self.Q_list.append(np.copy(self.Q))
         # initialize state
         ep = 0
         self.episodes = []
         self.num_stps_eps = []
         step = 0
         for episode in range(self.num_episodes):
            # track the total time steps in this episode
            time = 0
            # initialize state
            state = self.env.reset()
            done = False
            while not done and time<=200:
                # choose an action based on epsilon-greedy policy
                q_values = self.Q[state, :] #+ self.U[state, :] - self.L[state,:]           
                action = self.epsilon_greedy_policy(q_values)
                
                
                self.lr = self.lr_func(self.count[state, action])  #1000/(1000+step)
                # execute action
                newState, reward, done, info = self.env.step(action)
                
                self.T[state,action, :-1] = self.T[state,action, 1:]
                self.T[state,action, -1] = int(newState)

                
                prevMean = self.Rmean[state, action]
                prevVar = self.Rvar[state, action]
                prevSigma = np.sqrt(prevVar/self.count[state, action])
        
                self.Rmean[state, action] = prevMean + (reward - prevMean)/self.count[state, action]
                self.Rvar[state, action] = (prevVar + (reward- prevMean)*(reward - self.Rmean[state, action]))/self.count[state, action]
                
                bM= np.sqrt(2*np.log(self.n_actions +7) - np.log(np.log(self.n_actions + 7)) - np.log(4*np.pi))
                self.BR[state, action]=(np.euler_gamma/bM + bM)*prevSigma
                self.BT[state, action]=self.gamma *(np.max(self.Q[newState,:]) - np.mean(np.max(self.Q[self.T[state,action],:],axis=1)))
                if not done: 
                    delta = self.Rmean[state, action]  + self.gamma * np.max(self.Q[newState, :]) - self.Q[state, action]
                else: 
                    delta = self.Rmean[state, action]  - self.Q[state, action]
                self.BR[state, action] = self.BR[state, action] if self.count[state, action] >=2 else 0.0
                self.BT[state, action] = self.BT[state, action] if self.count[state, action] >=self.K else 0.0
               
                self.Q[state, action] +=   self.lr * (delta -self.BR[state, action] - self.BT[state, action])
    
                self.count[state, action] += 1    
                state = newState
                self.rel_err_print(step, start_time)
                time += 1
                step += 1
            
            if verbose:    
                print('Episode:',episode, 'time_steps:',time, 
                      'epsilon:', "{:.2f}".format(self.epsilon),         
                       'LR:', "{:.3f}".format(self.lr))  
            self.Q_list.append(np.copy(self.Q))
            self.episodes.extend([ep] * time)
            self.num_stps_eps.append(time)
            ep += 1
         elapsed_time = timeit.default_timer() - start_time
         print("Time="+str(elapsed_time))
         return self.Q_list,  self.episodes, self.num_stps_eps,  elapsed_time  

##############################################################################
    
#                            Lookahead Bounded Q-Learning

##############################################################################    

    
class Replay_Memory():

    def __init__(self, burn_in, memory_size=100, SEED=678, GAMMA= 0.90, K= 10 ):
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




@jit(nopython=True)        
def compp(sample_path,nS,nA,Q,QG,QL, EmaxQa_array, mean_reward_array,envf,envrange_random_wind ):
    for t in range(len(sample_path) - 1, -1, -1):
        w = sample_path[t] 
        for state in range(nS):
            for action in range(nA):
                next_state = envf[state, action, w + envrange_random_wind]
                a = np.argmax(Q[ next_state, :])
                EmaxQa = EmaxQa_array[state, action] 
                reward = mean_reward_array[state, action]
                # perfect information with penalty
                a_QG = np.argmax(QG[t+1, next_state, :]) 
                QG[t, state, action] = reward +  QG[t+1, next_state, a_QG ] + EmaxQa - Q[ next_state, a] 
                QL[t, state, action] = reward + QL[t+1, next_state, a ] + EmaxQa - Q[ next_state, a]  
    return QG, QL

      
class LBQL(Qlearning):
    '''Implementation of a LBQL Agent '''
    
    def __init__(self, ENV, 
                       SEED,
                       L_LR=0.2, 
                       U_LR=0.2, 
                       WITH_PENALTY=True,
                       BURN_IN = 100,
                       USE_K = True,
                       K = 10,
                       memory_size=100,
                       M = 10,
                       relTol=0.01,
                         ):
        super(LBQL, self).__init__(ENV, SEED,)
        self.L_lr = L_LR        
        self.U_lr = U_LR


        self.L = np.ones((self.env.nS, self.env.nA)) * self.env.r_min/(1-self.gamma)
        self.U = np.ones((self.env.nS, self.env.nA)) * self.env.r_max/(1-self.gamma)
        self.L[self.env.goal_state, :] = 0
        self.U[self.env.goal_state, :] = 0
        self.USE_K = USE_K
        self.K = K
        self.M = M
        self.relTol = relTol
        self.with_penalty = WITH_PENALTY
        self.s_a_tuples = list(itertools.product(range(self.env.nS), range(self.env.nA)))
        self.burn_in = BURN_IN
        self.memory = Replay_Memory(self.burn_in, memory_size= memory_size, GAMMA= self.gamma, K=self.K)
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
       
        
    def solve_QG_QL_DP(self, sample_path):
        ''' Computes the upper & lower bounds on the Q-values by solving the PI 
            and PI with nonanticipative policy problems, respectively via backward induction'''
        tau =  len(sample_path)   
        self.QG = np.zeros((tau + 1, self.env.nS, self.env.nA))
        self.QG[tau, :, :] = np.copy(self.Q)
        self.QL = np.zeros((tau + 1, self.env.nS, self.env.nA))
        self.QL[tau, :, :] = np.copy(self.Q)
        self.QD = np.zeros((tau + 1, self.env.nS, self.env.nA))
        self.backward_induction(sample_path)
    
    
    def Average_EmaxQa(self):
        ''' Computes an estimated expected value using simulation '''    
        dim_s = self.sample.shape[0]
        p = 1/dim_s*np.ones(dim_s)
        EmaxQa=np.dot(np.max(self.Q[ self.env.f[:,:,self.sample+self.env.range_random_wind], :],axis=3),p)
        mean_reward=np.dot(self.env.r[ :, :,self.sample+self.env.range_random_wind],p)
        return self.gamma*EmaxQa, mean_reward
    
    
    def Averag_Ereward(self, state, action):
        ''' Computes an estimated expected value using simulation '''
        mean_reward = np.mean(self.env.r[state, action, self.sample+self.env.range_random_wind].ravel())
        return  mean_reward 
    
    def backward_induction(self, sample_path): 
        '''
        Solves the deterministic perfect information relaxation problem via backward induction 
        '''  
        EmaxQ_array, mean_reward_array = self.Average_EmaxQa()
        envf=self.env.f.copy()
        envrange_random_wind = self.env.range_random_wind
        self.QG, self.QL = compp(sample_path,self.env.nS,self.env.nA,self.Q, self.QG,self.QL, EmaxQ_array, mean_reward_array, envf,envrange_random_wind)        
                                
                                  
        
        
    def train(self, interactive_plot = False, verbose = False):
        ''' Trains the Q-learning with IR agent'''
        
        start_time = timeit.default_timer()
        ep = 0
        self.episodes = []
        self.num_stps_eps = []  
        
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
        step = 0
        for episode in range(self.num_episodes):
            # track the total time steps in this episode
            time = 0
            # initialize state
            state = self.env.reset()
            done = False
            # keep going until the goal state is reached
            while not done and time<=200:
                #self.env.render()
                # choose an action based on epsilon-greedy policy
                q_values = self.Q[state, :] 
                action = self.epsilon_greedy_policy(q_values) 
                self.count[state, action] += 1    
                self.lr = self.lr_func(self.count[state, action])                 
                newState, reward, done, info = self.env.step(action)
                #  collect the noise from this step and append it to the memory
                self.memory.append(info['noise'])
                # Q-Learning update
                self.Q[state, action] +=  self.lr *\
                (reward + self.gamma* np.max(self.Q[newState, :]) -\
                 self.Q[state, action])
                # Standard Q-Learning following same behavioral policy
                self.Q_learning[state, action] +=  self.lr *\
                (reward + self.gamma* np.max(self.Q_learning[newState, :]) -\
                 self.Q_learning[state, action])
            

                cond = (step % self.M ==0 and step >= self.burn_in) and (not np.isclose(self.L[state, action],\
                   self.U[state, action],self.relTol) or not self.L[state, action]<=self.Q[state, action] <=self.U[state, action])       
                    
                if cond:#        
                
                #sample_path = self.env.simulate_sample_path()
                    sample_path = self.memory.simulate_sample_path()
                    if self.USE_K:
                        self.sample = self.memory.sample()
                    else:
                        self.sample = sample_path
                    
                    self.solve_QG_QL_DP(sample_path)
                    if self.with_penalty: 
                            self.U += self.U_lr * (self.QG[0,:, :] - self.U)
                        
                    else:
                            self.U += self.U_lr * (self.QD[0,:, :] - self.U)
                        
                    self.L += self.L_lr * (self.QL[0,:, :] - self.L)
                
                self.Q[state, action] = np.maximum(np.minimum(self.U[state, action],\
                                                  self.Q[state, action]),self.L[state, action])
                state = newState
                self.rel_err_print(step, start_time)
                time += 1
                step += 1  
                
            self.Q_list.append(np.copy(self.Q))
            self.QL_list.append(np.copy(self.Q_learning))
            # collect the number of steps in this episode for ploting
            self.episodes.extend([ep] * time)
            self.num_stps_eps.append(time)
            ep += 1
                
            if verbose:    
                print('Episode:',episode, 'time_steps:',time, 
                      'epsilon:', "{:.2f}".format(self.epsilon),         
                       'LR:', "{:.3f}".format(self.lr), 
            '|Vstar-V|/|Vstar|=',
            "{:.2f}".format(np.linalg.norm(Vstar - np.max(self.Q,1), ord=order)/Vstar_norm), 'step:', step)
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
                
                
                if episode % 50 == 0:
                        to_plot =np.array([[self.Lsa_1, self.Qsa_1, self.Usa_1, self.QLsa_1],
                                          [self.Lsa_2, self.Qsa_2, self.Usa_2, self.QLsa_2],
                                          [self.Lsa_3, self.Qsa_3, self.Usa_3, self.QLsa_3]])
                        self.plot_on_running(to_plot, Labels, episode)


        elapsed_time = timeit.default_timer() - start_time
        print("Time="+str(elapsed_time))
        return self.Q_list, self.QL_list, self.episodes, self.num_stps_eps, elapsed_time             