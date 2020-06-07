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

from agents import LBQL, Qlearning, SpeedyQLearning, DoubleQLearning, Bias_corrected_QL


f = open("Qstar.pkl",'rb')
Qstar = pickle.load(f)
f.close()

run = 1
times = 10

NUM_STEPS_def = 100001 # must match with NUM_STEPS in the Qlearning agent class
at = 1000 # must match with num_to_save_Q in the Qlearning agent class

Q_list = {}
QL_list= {}
Labels = ['K=1', 'K=5', 'K=10', 'K=100','K=1000']
K_list = [1,5,10,100,1000]
agents =[LBQL,
         LBQL,
         LBQL,
         LBQL,
         LBQL]
n_agents = len(agents)
seeds_ar=np.zeros((n_agents,times))
seeds_env_ar=np.zeros((n_agents,times))
env = CarSharing()
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
      
        agent = agents[i](env,SEED=SEED_, K=K_list[i], memory_size=100,BURN_IN = 100)#, SEED_ENV=SEED_ENV_)
        Qlis , Qllis, elapsed_time[i,j] = agent.train()
        Q_list[i,j]=Qlis
        QL_list[j]=Qllis
        with open('results/'+str(run)+'/'+str(i)+'_'+str(j) +'.pkl', 'wb') as f:  
                pickle.dump([Q_list[i,j], QL_list[j],elapsed_time[i,j]], f,protocol=2) 

num_stp_int =int(np.ceil(NUM_STEPS_def/at))                
array = np.zeros((n_agents,times,num_stp_int))
array_re = {}
array_re_all = np.zeros((n_agents,times,num_stp_int))
for i in range(n_agents):
    array_re[i] = np.mean([Q_list[i,m] for m in range(times)], axis=0)
    with open('results/'+str(run)+'/array_re_'+str(i)+'.pkl', 'wb') as f:  
                pickle.dump(array_re[i], f,protocol=2)
    for j in range(times):
        ls = Q_list[i,j]
        print(i,j)
        array_re_all[i,j,:] = relative_error_plot(at, NUM_STEPS_def, Qstar, ls )[0]
        array[i,j,:]=plot_performance_curves(at,NUM_STEPS_def,env, ls)[0]


with open('results/'+str(run)+'/array.pkl', 'wb') as f:  
                pickle.dump(array, f,protocol=2)       
array_mean = np.zeros((n_agents,num_stp_int))
for i in range(n_agents):
    array_mean[i]= np.mean(array[i], axis=0)
    with open('results/'+str(run)+'/array_mean_'+str(i)+'.pkl', 'wb') as f:  
                pickle.dump(array_mean[i], f,protocol=2)
    
list_qstar = [performance_plot(env, Qstar)]*num_stp_int
plots('results/' + str(run),[list_qstar, array_mean[0], array_mean[1], array_mean[2], array_mean[3],array_mean[4]],
      ['OP','K=1', 'K=5', 'K=10', 'K=100','K=1000'],
      ['r','b', 'g', 'y', 'c','m'], 
      'Number of steps x'+str(at), 
      'Reward', Title='Policy Performance', save=True)
 
fig1=plt.figure()
Epoch = np.tile(np.arange(0,array.shape[2]),(array.shape[0]*array.shape[1]))
Alg = np.repeat(Labels,array.shape[1]*array.shape[2])

Returns = array.ravel()
RE = array_re_all.ravel()
df = pd.DataFrame()
df = pd.DataFrame(columns=['Epoch','Algo', 'Reward', 'RE'])
df['Epoch'] = Epoch
df['Algo'] = Alg
df['Reward'] = Returns
df['RE']=RE
sns.set(font_scale=1.5)
sns_plot=sns.lineplot(x="Epoch", y="Reward", hue='Algo',data=df,ci=95, palette=['r','b', 'c', 'g','m'],
                      hue_order=Labels, style="Algo",
                      markers=False, dashes=True, lw=2)
#sns_plot.legend().set_visible(False)
plt.legend(title='', loc='lower right', labels= Labels, fontsize =14)
#sns_plot.legend().set_visible(False)
plt.xlabel( 'Number of steps x'+str(at) )
plt.ylabel( 'Total Reward' )
fig1 = sns_plot.get_figure()
fig1.savefig('results/'+str(run)+'/all performance.png')



fig2=plt.figure()
sns.set(font_scale=1.5)
sns_plot=sns.lineplot(x="Epoch", y="RE", hue='Algo',data=df,ci=95,palette=['r','b', 'c', 'g','m'], 
                      hue_order=Labels,style="Algo",
                      markers=False, dashes=True, lw=2)
plt.legend(title='', loc='upper right', labels= Labels)
#sns_plot.legend().set_visible(False)
plt.xlabel('Number of steps x'+str(at) )
plt.ylabel('Relative Error')
fig2 = sns_plot.get_figure()
fig1.savefig('results/'+str(run)+'/all RE.png')
df.to_pickle('results/' + str(run) + '/carsharing_df_K_Sensitivity.pkl')


mean_elapsed_time = np.mean(elapsed_time,1)
print('Mean run time for K=1',mean_elapsed_time[0])
print('Mean run time for K=5',mean_elapsed_time[1])
print('Mean run time for K=10',mean_elapsed_time[2])
print('Mean run time for K=50',mean_elapsed_time[3])
print('Mean run time for K=100',mean_elapsed_time[4])