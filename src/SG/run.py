# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
import pickle
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")
from stoch_stormy_gridworld import StochStormyGridWorldEnv

from plots import performance_plot, plot_performance_curves, relative_error_plot, plots

from agents import LBQL, Qlearning, SpeedyQLearning, DoubleQLearning, Bias_corrected_QL


f = open("Qstar.pkl",'rb')
Qstar = pickle.load(f)
f.close()

run = 1
times = 50

NUM_EPISODES_def = 500 # must match with NUM_EPISODES in the Qlearning agent class
at = 1000 # must match with num_to_save_Q in the Qlearning agent class

Q_list = {}
QL_list= {}
Episodes = {}
Num_stps_per_eps ={}
Labels = ['LBQL','QL', 'Double-QL','SQL', 'BCQL']
agents =[LBQL,
         Qlearning,
         DoubleQLearning,
         SpeedyQLearning,
         Bias_corrected_QL]
n_agents = len(agents)
seeds_ar=np.zeros((n_agents,times))
seeds_env_ar=np.zeros((n_agents,times))
env = StochStormyGridWorldEnv()
env.gamma = 0.95

SEED_arr = np.random.randint(20000, size=times).astype(int)
with open('results/'+str(run)+'/Seeds.pkl', 'wb') as f:  
                pickle.dump([SEED_arr], f,protocol=2) 
elapsed_time = np.zeros([n_agents, times])
for i in range(n_agents):
    for j in range(times):
        print('i=',i,'j=',j)
    
        SEED_ = int(SEED_arr[j])

        seeds_ar[i,j]=SEED_

 
        
        agent = agents[i](env,SEED=SEED_)
        if i==0:
          Q_list[i,j] , QL_list[j], Episodes[i,j], Num_stps_per_eps[i,j],elapsed_time[i,j] = agent.train()
          with open('results/'+str(run)+'/'+str(i)+'_'+str(j) +'.pkl', 'wb') as f:  
                pickle.dump([Q_list[i,j] , QL_list[j], Episodes[i,j], Num_stps_per_eps[i,j],elapsed_time[i,j]], f,protocol=2) 
        else:
           Q_list[i,j], Episodes[i,j], Num_stps_per_eps[i,j], elapsed_time[i,j] = agent.train()
           with open('results/'+str(run)+'/'+str(i)+'_'+str(j) +'.pkl', 'wb') as f:  
                pickle.dump([Q_list[i,j], Episodes[i,j], Num_stps_per_eps[i,j], elapsed_time[i,j]], f,protocol=2) 

num_stp_int =int(NUM_EPISODES_def)                
array = np.zeros((n_agents,times,num_stp_int))
array_re = {}
array_num_stps_eps = {}
array_re_all = np.zeros((n_agents,times,num_stp_int))
array_re_max = np.zeros((n_agents,times,num_stp_int))
for i in range(n_agents):
    array_re[i] = np.mean([Q_list[i,m] for m in range(times)], axis=0)
    array_num_stps_eps[i] = np.mean([Num_stps_per_eps[i,m] for m in range(times)], axis=0)
    array_num_stps_eps[i] = np.repeat(np.arange(len(array_num_stps_eps[i])), array_num_stps_eps[i].astype(int))
    with open('results/'+str(run)+'/array_re_'+str(i)+'.pkl', 'wb') as f:  
                pickle.dump(array_re[i], f,protocol=2)
    with open('results/'+str(run)+'/array_num_stps_eps_'+str(i)+'.pkl', 'wb') as f:  
                pickle.dump(array_num_stps_eps[i], f,protocol=2)            
    for j in range(times):
        ls = Q_list[i,j]
        print(i,j)
        array_re_all[i,j,:] = relative_error_plot(NUM_EPISODES_def, Qstar, ls)[0]
        array[i,j,:]=plot_performance_curves(NUM_EPISODES_def,env, ls)[0]


with open('results/'+str(run)+'/array.pkl', 'wb') as f:  
                pickle.dump(array, f,protocol=2)       
array_mean = np.zeros((n_agents,num_stp_int))
for i in range(n_agents):
    array_mean[i]= np.mean(array[i], axis=0)
    with open('results/'+str(run)+'/array_mean_'+str(i)+'.pkl', 'wb') as f:  
                pickle.dump(array_mean[i], f,protocol=2)
    
list_qstar = [performance_plot(env, Qstar)]*num_stp_int
plots('results',[array_mean[0], array_mean[1], list_qstar, array_mean[2], array_mean[3],array_mean[4]],
      ['LBQL','QL','OP', 'Double-QL','SQL', 'BCQL'],
      ['r','b', 'y', 'g', 'c','m'],
      'Episodes', 
      'Reward', Title='Policy Performance', save=True)
 

plots('results',[array_num_stps_eps[0], array_num_stps_eps[1], array_num_stps_eps[2], 
       array_num_stps_eps[3],array_num_stps_eps[4]],
      ['LBQL','QL', 'Double-QL','SQL', 'BCQL'],
       ['r','b', 'c', 'g','m'],  
      'Number of steps', 
      'Episodes', Title='No. of steps per training episode', save=True) 
fig1=plt.figure()
Epoch = np.tile(np.arange(0,array.shape[2]),(array.shape[0]*array.shape[1]))
Alg = np.repeat(['LBQL', 'QL', 'Double-QL', 'SQL', 'BCQL' ],array.shape[1]*array.shape[2])

Returns = array.ravel()
RE = array_re_all.ravel()
df = pd.DataFrame()
df = pd.DataFrame(columns=['Epoch','Algo', 'Reward', 'RE'])
df['Episodes'] = Epoch
df['Algo'] = Alg
df['Reward'] = Returns
df['RE']=RE

sns.set(font_scale=1.5)
sns_plot=sns.lineplot(x="Episodes", y="Reward", hue='Algo',data=df,ci=95,palette=['r','b', 'c', 'g','m'], hue_order=['LBQL', 'QL','Double-QL', 'SQL', 'BCQL' ], lw=2)
plt.legend(title='', loc='lower right', labels= ['LBQL', 'QL', 'Double-QL', 'SQL', 'BCQL' ])
sns_plot.legend().set_visible(False)
plt.xlabel( 'Episodes' )
plt.ylabel( 'Total Reward' )
fig1 = sns_plot.get_figure()
fig1.tight_layout()
fig1.savefig('results/'+str(run)+'/all performance.png')


fig2=plt.figure()
sns.set(font_scale=1.5)
sns_plot=sns.lineplot(x="Episodes", y="RE", hue='Algo',data=df,ci=95,palette=['r','b', 'c', 'g','m'], hue_order=['LBQL', 'QL','Double-QL', 'SQL', 'BCQL' ], lw=2)
#plt.legend(title='', loc='upper left', labels= ['LBQL', 'QL','Double-QL', 'SQL', 'BCQL' ])
sns_plot.legend().set_visible(False)
plt.xlabel('Episodes')
plt.ylabel('Relative Error')
fig2 = sns_plot.get_figure()
fig2.savefig('results/'+str(run)+'/all Relative Error.png')

df.to_pickle('results/' + str(run) + '/stormy_gridworld_df.pkl')


mean_elapsed_time = np.mean(elapsed_time,1)
print('Mean run time for LBQL=',mean_elapsed_time[0])
print('Mean run time for QL=',mean_elapsed_time[1])
print('Mean run time for SQL=',mean_elapsed_time[2])
print('Mean run time for Double-QL=',mean_elapsed_time[3])
print('Mean run time for BCQL=',mean_elapsed_time[4])

