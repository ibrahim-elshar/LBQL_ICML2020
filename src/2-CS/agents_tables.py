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
##############################################################################
    
#                        Load  Qstar for error measurement

##############################################################################
f = open("Qstar.pkl",'rb')
Qstar = pickle.load(f)
f.close()
Vstar= np.max(Qstar,1)
order=None
Vstar_norm=np.linalg.norm(Vstar, ord=order)
##############################################################################
    
#                            Q-learning

##############################################################################

class Qlearning():
    '''Implementation of the Q-leaning Agent 
       example:
           ql = Qlearning(env, 0, 0.95,0.4,10000,[(5,16),(4,18),(7,14)],1000)
           Q_list = ql.train(interactive_plot = True, verbose = True)
    '''
    GAMMA=0.95
    EPS_GREEDY_PAR=0.5
    POLYNOMIAL_LR = 0.5
    
    def __init__(self, ENV,
                       SEED,
                       NUM_STEPS=300001,
                       interactive_plot_states=[(5,16),(4,18),(7,14)],
                       num_to_save_Q=1000,
                       ):
        self.env = ENV
        self.env.seed(SEED)
        self.env.action_space.seed(SEED)
        self.env.gamma = self.GAMMA
        self.gamma = self.GAMMA
        self.prng = np.random.RandomState(SEED) #Pseudorandom number generator
        print('SEED_ENV:',SEED,'SEED:',SEED)
        self.count = np.zeros([self.env.nS, self.env.nA])

        L = np.ones((self.env.nS, self.env.nA)) * self.env.r_min/(1-self.gamma)
        U = np.ones((self.env.nS, self.env.nA)) * self.env.r_max/(1-self.gamma)
        self.Q =  self.prng.uniform(L,U)# np.zeros((self.env.nS, self.env.nA))# Qstar.copy()

        self.eps_greedy_par = self.EPS_GREEDY_PAR

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
        self.polynomial_lr = self.POLYNOMIAL_LR
        
        self.rel_er_stps = []
        self.rel_er_times = []
        
    
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
        plt.savefig('Figures/fig'+str(step+1)+'.png', bbox_inches='tight')
        
    def rel_err_print(self,step,start_time):
        elapsed_time = timeit.default_timer() - start_time
        V = np.max(self.Q,1) 
        if np.linalg.norm(Vstar - V, ord=order)/Vstar_norm <= 0.5 and self.perc_flags[0] ==0:
            print('50% ->|Vstar-V|/|Vstar|=',"{:.2f}".format(np.linalg.norm(Vstar - V, ord=order)/Vstar_norm), 'at step:',step, 'time=',elapsed_time )
            self.perc_flags[0] +=1
            self.rel_er_stps.append(step)
            self.rel_er_times.append(elapsed_time)
        if np.linalg.norm(Vstar - V, ord=order)/Vstar_norm <= 0.2 and self.perc_flags[1] ==0:
            print('20% ->|Vstar-V|/|Vstar|=',"{:.2f}".format(np.linalg.norm(Vstar - V, ord=order)/Vstar_norm), 'at step:',step, 'time=',elapsed_time )
            self.perc_flags[1] +=1
            self.rel_er_stps.append(step)
            self.rel_er_times.append(elapsed_time)            
        if np.linalg.norm(Vstar - V, ord=order)/Vstar_norm <= 0.1 and self.perc_flags[2] ==0:
            print('10% ->|Vstar-V|/|Vstar|=',"{:.2f}".format(np.linalg.norm(Vstar - V, ord=order)/Vstar_norm), 'at step:',step, 'time=',elapsed_time )
            self.perc_flags[2] +=1
            self.rel_er_stps.append(step)
            self.rel_er_times.append(elapsed_time)            
        if np.linalg.norm(Vstar - V, ord=order)/Vstar_norm <= 0.05 and self.perc_flags[3] ==0:
            print('5% ->|Vstar-V|/|Vstar|=',"{:.2f}".format(np.linalg.norm(Vstar - V, ord=order)/Vstar_norm), 'at step:',step, 'time=',elapsed_time )
            self.perc_flags[3] +=1 
            self.rel_er_stps.append(step)
            self.rel_er_times.append(elapsed_time)            
        if np.linalg.norm(Vstar - V, ord=order)/Vstar_norm <= 0.01 and self.perc_flags[4] ==0:
            print('1% ->|Vstar-V|/|Vstar|=',"{:.2f}".format(np.linalg.norm(Vstar - V, ord=order)/Vstar_norm), 'at step:',step, 'time=',elapsed_time )
            self.perc_flags[4] +=1
            self.rel_er_stps.append(step)
            self.rel_er_times.append(elapsed_time)            
            

    
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
        
        for step in range(self.num_steps):
            
            
            # choose an action based on epsilon-greedy policy
            q_values = self.Q[state, :]
            action, action_idx = self.epsilon_greedy_policy(q_values)
            self.count[state, action_idx] += 1   
            # execute action
            newState, reward, info = self.env.step(action)
            self.lr = self.lr_func(self.count[state, action_idx])    
            # Q-Learning update
            self.Q[state, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Q[newState, :])\
                                                      - self.Q[state, action_idx])
            if verbose:    
                print('Step:',step, 'reward:',"{:.2f}".format(reward), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:', state, )          
            state = newState
            
            self.rel_err_print(step, start_time)
            
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
        
        for step in range(self.num_steps):
            
            old_q = np.copy(self.Q)
            
            # choose an action based on epsilon-greedy policy
            q_values = self.Q[state, :]
            action, action_idx = self.epsilon_greedy_policy(q_values)
            self.count[state, action_idx] += 1    
            # execute action
            next_state, reward, info = self.env.step(action)
            
            max_q_cur = np.max(self.Q[next_state, :]) #if not absorbing else 0.
            max_q_old = np.max(self.Q_old[next_state, :]) #if not absorbing else 0.
            
            target_cur = reward + self.gamma * max_q_cur
            target_old = reward + self.gamma * max_q_old
            
#            alpha = 1/ (self.count[state, action_idx] + 1)
            self.lr = self.lr_func(self.count[state, action_idx]) 
            alpha = self.lr             
            q_cur = self.Q[state, action_idx]
            
            self.Q[state, action_idx] = q_cur + alpha * (target_old - q_cur) + (
            1. - alpha) * (target_cur - target_old)

            self.Q_old = np.copy(old_q)
            
            if verbose:    
                print('Step:',step, 'reward:',"{:.2f}".format(reward), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:', state, )             
            
            state = next_state
            self.rel_err_print(step, start_time)
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
        
        for step in range(self.num_steps):
             
            # choose an action based on epsilon-greedy policy
            q_values = (self.Q[state, :] + self.Qprime[state, :] )/2
            action, action_idx = self.epsilon_greedy_policy(q_values)
               
            # execute action
            newState, reward, info = self.env.step(action)
             
            
            # Double Q-Learning update
            
            if np.random.uniform() < .5:
                self.count[state, action_idx] += 1 
                self.lr = self.lr_func(self.count[state, action_idx])
                self.Q[state, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Qprime[newState, :])\
                                                      - self.Q[state, action_idx])
                if (step % self.num_to_save_Q ) == 0 and (step>0):
                    self.Q_list.append((np.copy(self.Q)+np.copy(self.Qprime))/2)
                
            else:
                self.countprime[state, action_idx] += 1 
                self.lr = self.lr_func(self.countprime[state, action_idx])
                self.Qprime[state, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Q[newState, :])\
                                                      - self.Qprime[state, action_idx])
                if (step % self.num_to_save_Q ) == 0 and (step>0):
                    self.Q_list.append((np.copy(self.Qprime)+np.copy(self.Q))/2)
                    
            if verbose:    
                print('Step:',step, 'reward:',"{:.2f}".format(reward), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:', state, )                  

            state = newState
            self.rel_err_print(step, start_time)
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
         for step in range(self.num_steps):
            # choose an action based on epsilon-greedy policy
            q_values = self.Q[state, :] #+ self.U[state, :] - self.L[state,:]           
            action, action_idx = self.epsilon_greedy_policy(q_values)
            
            
            self.lr = self.lr_func(self.count[state, action_idx])  #1000/(1000+step)
            # execute action
            newState, reward, info = self.env.step(action)
            
            self.T[state,action_idx, :-1] = self.T[state,action_idx, 1:]
            self.T[state,action_idx, -1] = int(newState)
            #self.memory.append((state, action_idx,reward))
            
            prevMean = self.Rmean[state, action_idx]
            prevVar = self.Rvar[state, action_idx]
            prevSigma = np.sqrt(prevVar/self.count[state, action_idx])
    
            self.Rmean[state, action_idx] = prevMean + (reward - prevMean)/self.count[state, action_idx]
            self.Rvar[state, action_idx] = (prevVar + (reward- prevMean)*(reward - self.Rmean[state, action_idx]))/self.count[state, action_idx]
            
            bM= np.sqrt(2*np.log(self.n_actions +7) - np.log(np.log(self.n_actions + 7)) - np.log(4*np.pi))
            self.BR[state, action_idx]=(np.euler_gamma/bM + bM)*prevSigma
            self.BT[state, action_idx]=self.gamma *(np.max(self.Q[newState,:]) - np.mean(np.max(self.Q[self.T[state,action_idx],:],axis=1)))
            delta =  self.Rmean[state, action_idx]  + self.gamma * np.max(self.Q[newState, :]) - self.Q[state, action_idx]

            self.BR[state, action_idx] = self.BR[state, action_idx] if self.count[state, action_idx] >=2 else 0.0
            self.BT[state, action_idx] = self.BT[state, action_idx] if self.count[state, action_idx] >=self.K else 0.0
           
            self.Q[state, action_idx] +=   self.lr * (delta -self.BR[state, action_idx] - self.BT[state, action_idx])

       
            self.count[state, action_idx] += 1    
            state = newState
            self.rel_err_print(step, start_time)
            if verbose:    
                print('Step:',step, 'reward:',"{:.2f}".format(reward), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:', state, )             

            
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
 
@jit(locals={'l': numba.int64, 'm': numba.int64}, nopython=True)        
def comp(sample_path, nS, nA, Qp, QG, QL, EmaxQ_array, mean_reward_array,
         envstates, envactions, envf, envi, envj, envk, envl):
    for t in range(len(sample_path) - 1, -1, -1):
        w = sample_path[t] 
        for s_idx in  range(nS):
             for a_idx in range(nA):
                action = envactions[a_idx]
                j= action[0]- envi
                k= action[1]- envj
                l= w[0]+envk
                m= w[1]+envl
                next_state =  envf[s_idx,j,k,l, m]
                a_id =  np.argmax(Qp[next_state, :])
                EmaxQa = EmaxQ_array[s_idx, a_idx] 
                reward = mean_reward_array[s_idx, a_idx]
                a_QG_id = np.argmax(QG[t+1, next_state, :])
     
                QG[t, s_idx, a_idx] = reward +  QG[t+1, next_state, a_QG_id ] + EmaxQa - Qp[ next_state, a_id] 
                QL[t, s_idx, a_idx] = reward + QL[t+1, next_state, a_id ] + EmaxQa - Qp[ next_state, a_id]  
    return QG, QL   

def solve_inner_DP(sample_path, env, Q, EmaxQ_array, mean_reward_array): 
    '''
    Solves the deterministic perfect information relaxation problem via backward induction 
    '''  
    nS = env.nS
    nA = env.nA
    envf = env.f
    envi=env.i
    envj=env.j
    envk=env.k
    envl=env.l
    envstates = env.states
    envactions = env.actions
    tau =  len(sample_path)   
    QG = np.zeros((tau + 1, nS, nA))
    QG[tau, :, :] = np.copy(Q)
    QL = np.zeros((tau + 1, nS, nA))
    QL[tau, :, :] = np.copy(Q)
    
    return comp(sample_path, nS, nA, Q, QG, QL, EmaxQ_array, mean_reward_array,
                  envstates, envactions, envf, envi, envj, envk, envl)
      
class LBQL(Qlearning):
    '''Implementation of a LBQL Agent '''
    
    def __init__(self, ENV, 
                       SEED,
                       L_LR=0.01, 
                       U_LR=0.01,
                       WITH_PENALTY=True,
                       BURN_IN = 40,
                       USE_K = True,
                       K = 20,
                       memory_size=40,
                       M = 15,
                       relTol=0.01,
                         ):
        super(LBQL, self).__init__(ENV, SEED,)
        self.L_lr = L_LR        
        self.U_lr = U_LR

        self.L = np.ones((self.env.nS, self.env.nA)) * self.env.r_min/(1-self.gamma)
        self.U = np.ones((self.env.nS, self.env.nA)) * self.env.r_max/(1-self.gamma)

        self.USE_K = USE_K
        self.K = K
        self.M = M
        self.relTol = relTol
        self.with_penalty = WITH_PENALTY
        self.s_a_tuples = list(itertools.product(range(self.env.nS), range(self.env.nA)))
        self.burn_in = BURN_IN
        self.memory = Replay_Memory(self.burn_in, memory_size= memory_size, SEED= SEED,
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
    
    
    def Average_EmaxQa(self):#state, action, sample_path):
        ''' Computes an estimated expected value using simulation '''    
        dim_s = self.sample.shape[0]
        p = 1/dim_s*np.ones(dim_s)
        EmaxQa=np.dot(np.max(self.Q[self.env.f[:,:,:,self.sample[:,0]+self.env.k,self.sample[:,1]+self.env.l], :],axis=4).reshape((self.env.nS,
             self.env.nA,dim_s)), p)
        mean_reward=np.dot(self.env.r[:,:,:,self.sample[:,0]+self.env.k,self.sample[:,1]+self.env.l].reshape((self.env.nS,
             self.env.nA,dim_s)),p)
        return self.gamma*EmaxQa, mean_reward
    
    
    def Averag_Ereward(self, state, action):
        ''' Computes an estimated expected value using simulation '''
        mean_reward = np.mean(self.env.r[state, action[0]-self.env.i,action[1]-self.env.j, self.sample[:,0]+self.env.k,self.sample[:,1]+self.env.l].ravel())
        return  mean_reward 
    
    def backward_induction(self, sample_path): 
        '''
        Solves the deterministic perfect information relaxation problem via backward induction 
        '''  
        EmaxQ_array, mean_reward_array = self.Average_EmaxQa()
        self.QG, self.QL = solve_inner_DP(sample_path, self.env, self.Q, EmaxQ_array, mean_reward_array)

                                  
    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        print ("Burning in memory ...", self.memory.burn_in, "samples to collect.")
        state = self.env.reset()
        for i in range(self.memory.burn_in):
            action = self.env.action_space.sample()
            next_state, reward, info = self.env.step(action)
            self.memory.append(info['noise'])
        print ("Burn-in complete.")
    
        
    def train(self, interactive_plot = False, verbose = False):
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
        
        for step in range(self.num_steps):
            
            # choose an action based on epsilon-greedy policy
            q_values =  self.Q[state, :] #+ self.U[state, :] - self.L[state,:]           
            action, action_idx = self.epsilon_greedy_policy(q_values)
            
            self.count[state, action_idx] += 1
              
            self.lr = self.lr_func(self.count[state, action_idx])  
            
            # execute action
            newState, reward, info = self.env.step(action)
            
            #  collect the noise from this step and append it to the agent sample path
            # agent_sample_path.append(info['noise'])
            self.memory.append(info['noise'])
            
            # Q-Learning update
            self.Q[state, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Q[newState, :]) -\
                                          self.Q[state, action_idx])
            
            cond = (step % self.M ==0 and step >= self.burn_in) and \
            (not np.isclose(self.L[state, action_idx],self.U[state, action_idx],self.relTol) or \
             not self.L[state, action_idx]<=self.Q[state, action_idx] <=self.U[state, action_idx])
            if cond:#        
                
                sample_path = self.memory.simulate_sample_path()
                if self.USE_K:
                    self.sample = self.memory.sample()
                else:
                    self.sample = sample_path
                
                self.solve_QG_QL_DP(sample_path)
                
                if self.with_penalty: 
#                    self.U[state, action_idx] += self.U_lr * (self.QG[0,state, action_idx] - self.U[state, action_idx])
                        self.U += self.U_lr * (self.QG[0,:, :] - self.U)
                    
                else:
#                    self.U[state, action_idx] += self.U_lr * (self.QD[0,state, action_idx] - self.U[state, action_idx]) 
                        self.U += self.U_lr * (self.QD[0,:, :] - self.U)
                    
#                self.L[state, action_idx] += self.L_lr * (self.QL[0,state, action_idx] - self.L[state, action_idx])
                self.L += self.L_lr * (self.QL[0,:, :] - self.L)

                if verbose:
                    print('Step:',step, 'reward:',"{:.2f}".format(reward), 
                          'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx,
                          'state:', state, 
                          '|Q-QL|=',"{:.2f}".format(np.linalg.norm(self.Q - self.Q_learning))) 

            self.Q[state, action_idx] = np.maximum(np.minimum(self.U[state, action_idx],\
                                                  self.Q[state, action_idx]),self.L[state, action_idx])
             
            state = newState
            self.rel_err_print(step, start_time)
 
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
            
                if step % 1000 == 0:
                    to_plot =np.array([[self.Lsa_1, self.Qsa_1, self.Usa_1, self.QLsa_1],
                                       [self.Lsa_2, self.Qsa_2, self.Usa_2, self.QLsa_2],
                                       [self.Lsa_3, self.Qsa_3, self.Usa_3, self.QLsa_3]])
                    self.plot_on_running(to_plot, Labels, step)
            if (step % self.num_to_save_Q ) == 0 and (step>0):
                 self.Q_list.append(np.copy(self.Q))
                 self.QL_list.append(np.copy(self.Q_learning))

        elapsed_time = timeit.default_timer() - start_time
        print("Time="+str(elapsed_time))
        print(elapsed_time)
        return self.Q_list, self.QL_list, elapsed_time             