# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")
from carsharing_N import CarSharN

from plots import performance_plot, plot_performance_curves, plots

from agents_N import LBQL, Qlearning, SpeedyQLearning, DoubleQLearning, Bias_corrected_QL


run = 1
times = 10


NUM_STEPS_def = 300001 # must match with NUM_STEPS in the Qlearning agent class
at = 5000 # must match with num_to_save_Q in the Qlearning agent class

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
env = CarSharN()
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
          Qlis , Qllis, elapsed_time[i,j] = agent.train()
          Q_list[i,j]=Qlis
          QL_list[j]=Qllis
          with open('results/'+str(run)+'/'+str(i)+'_'+str(j) +'.pkl', 'wb') as f:  
                pickle.dump([Q_list[i,j], QL_list[j],elapsed_time[i,j]], f,protocol=2) 
        else:
           Q_list[i,j], elapsed_time[i,j] = agent.train() 
           with open('results/'+str(run)+'/'+str(i)+'_'+str(j) +'.pkl', 'wb') as f:  
                pickle.dump([Q_list[i,j],elapsed_time[i,j]], f,protocol=2) 
          



num_stp_int =int(np.ceil(NUM_STEPS_def/at))                
array = np.zeros((n_agents,times,num_stp_int))
for i in range(n_agents):
    for j in range(times):
        ls = Q_list[i,j]
        print(i,j)
        array[i,j,:]=plot_performance_curves(at,NUM_STEPS_def,env, ls)[0]


with open('results/'+str(run)+'/array.pkl', 'wb') as f:  
                pickle.dump(array, f,protocol=2)       
fig1=plt.figure()
Epoch = np.tile(np.arange(0,array.shape[2]),(array.shape[0]*array.shape[1]))
Alg = np.repeat(['LBQL', 'QL', 'SQL', 'Double-QL', 'BCQL' ],array.shape[1]*array.shape[2])

Returns = array.ravel()
df = pd.DataFrame()
df = pd.DataFrame(columns=['Epoch','Algo', 'Reward'])
df['Epoch'] = Epoch
df['Algo'] = Alg
df['Reward'] = Returns
sns.set(font_scale=1.)
sns_plot=sns.lineplot(x="Epoch", y="Reward", hue='Algo',data=df,ci=95, palette=['r','b', 'c', 'g','m'], hue_order=['LBQL', 'QL','Double-QL', 'SQL', 'BCQL' ], lw=2)
#sns_plot.legend().set_visible(False)
plt.legend(title='', loc='lower right', labels= ['LBQL', 'QL', 'Double-QL', 'SQL', 'BCQL' ])
#sns_plot.legend().set_visible(False)
plt.xlabel( 'Number of steps x'+str(at) )
plt.ylabel( 'Total Reward' )
plt.tight_layout()
fig1 = sns_plot.get_figure()
fig1.savefig('results/'+str(run)+'/all performance.png')


df.to_pickle('results/' + str(run) + '/carsharing_df.pkl')


mean_elapsed_time = np.mean(elapsed_time,1)
print('Mean run time for LBQL=',mean_elapsed_time[0])
print('Mean run time for QL=',mean_elapsed_time[1])
print('Mean run time for SQL=',mean_elapsed_time[2])
print('Mean run time for Double-QL=',mean_elapsed_time[3])
print('Mean run time for BCQL=',mean_elapsed_time[4])