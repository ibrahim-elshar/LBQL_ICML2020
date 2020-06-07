# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")
from carsharing import CarSharing

from plots import performance_plot, plot_performance_curves, relative_error_plot, plots

from agents_tables import LBQL, Qlearning, SpeedyQLearning, DoubleQLearning, Bias_corrected_QL

from input_par_tables import * #ls_para_gamma_eps_polr

agents =[LBQL,
         Qlearning,
         SpeedyQLearning,
         DoubleQLearning,
         Bias_corrected_QL]

run = 1
times = 5

Q_list = {}
QL_list= {}
Labels = ['LBQL','QL', 'SQL', 'Double-QL', 'BCQL']
agents =[LBQL,
         Qlearning,
         SpeedyQLearning,
         DoubleQLearning,
         Bias_corrected_QL]
n_agents = len(agents)
seeds_ar=np.zeros((n_agents,times))
seeds_env_ar=np.zeros((n_agents,times))
env = CarSharing()


SEED_arr = np.random.randint(20000, size=times).astype(int)
with open('Tables/'+str(run)+'/Seeds.pkl', 'wb') as f:  
                pickle.dump([SEED_arr], f,protocol=2) 
                
elapsed_time = np.zeros([n_agents,len(ls_para_gamma_eps_polr), times])

rel_er_stps = {}
rel_er_times = {}
rel_er_tot_elapsed_time = {}       
#GAMMA=0.95
#eps_greedy_par=0.4
#polynomial_lr = 0.5
#NUM_STEPS=300001,
#interactive_plot_states=[(5,16),(4,18),(7,14)],
#num_to_save_Q=1000,
#polynomial_lr = 0.5

for i in range(n_agents):
    for par_case,par in enumerate(ls_para_gamma_eps_polr):
        Qlearning.GAMMA=par[0] 
        Qlearning.EPS_GREEDY_PAR=par[1]
        Qlearning.POLYNOMIAL_LR = par[2]
        for j in range(times):
            print('agent=',i,'par_case=',par_case,'j=',j)
            SEED_ = int(SEED_arr[j])
            seeds_ar[i,j]=SEED_
            
            agent = agents[i](env,SEED=SEED_ )
            if i==0:
              Q_list[i,par_case,j] , QL_list[par_case,j], elapsed_time[i,par_case,j] = agent.train()
              with open('Tables/'+str(run)+'/'+str(i)+'_'+str(par_case)+'_'+str(j) +'.pkl', 'wb') as f:  
                    pickle.dump([Q_list[i,par_case,j], QL_list[par_case,j],elapsed_time[i,par_case,j]], f,protocol=2) 
            else:
               Q_list[i,par_case,j], elapsed_time[i,par_case,j] = agent.train() 
               with open('Tables/'+str(run)+'/'+str(i)+'_'+str(par_case)+'_'+str(j) +'.pkl', 'wb') as f:  
                    pickle.dump([Q_list[i,par_case,j],elapsed_time[i,par_case,j]], f,protocol=2) 
                    
            rel_er_stps[i,par_case,j] = agent.rel_er_stps
            rel_er_times[i,par_case,j] = agent.rel_er_times  
            rel_er_tot_elapsed_time[i,par_case,j] = elapsed_time[i,par_case,j] 
            
with open('Tables/'+str(run)+'/'+'rel_er_stps.pkl', 'wb') as f:  
    pickle.dump(rel_er_stps, f,protocol=2)            
with open('Tables/'+str(run)+'/'+'rel_er_times.pkl', 'wb') as f:  
    pickle.dump(rel_er_times, f,protocol=2)  
with open('Tables/'+str(run)+'/'+'rel_er_tot_elapsed_time.pkl', 'wb') as f:  
    pickle.dump(rel_er_tot_elapsed_time, f,protocol=2)          
        

nn=5 
        
table_result_steps = np.zeros((n_agents, len(ls_para_gamma_eps_polr),nn) )   
table_result_times = np.zeros((n_agents, len(ls_para_gamma_eps_polr),nn) )
dim_matrix = np.zeros((n_agents, len(ls_para_gamma_eps_polr),nn) )



def fun_vec(idx):
    if idx == 0:
        return np.zeros(nn)
    else:
        return np.tril(np.ones((nn,nn)),0)[-1+idx]
    
for i in range(n_agents):
    for j in range(len(ls_para_gamma_eps_polr)):
        for k in range(times):
            aa = np.full((nn), 300000)
            tt = np.full((nn), 0.)
            temp = np.array(rel_er_stps[i,j,k])
            temp_t = np.array(rel_er_times[i,j,k])
            temp_len = temp.shape[0]
            vv = fun_vec(temp_len)
            dim_matrix[i,j] += vv
            for _ in range(temp_len):
                aa[_] = temp[_]
                tt[_] = temp_t[_]
            table_result_steps[i,j] += np.array(aa) 
            table_result_times[i,j] += np.array(tt) 

table_result_steps = table_result_steps/times
table_result_times = table_result_times/times
