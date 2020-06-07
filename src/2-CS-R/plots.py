# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def plots(path, ys, Labels, colors, xLabel, yLabel, Title=None, save=False):
    ''' Plot fuction used to plot results... ys, Labels, colors are lists'''
    plt.figure()
    for i,y in enumerate(ys):
        plt.plot(ys[i],label=Labels[i], color= colors[i], lw=2)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 
    #plt.legend(loc='best')
    if Title: plt.title(Title)
    #plt.grid()
    if not save: 
        plt.show() 
    else: 
        if Title: 
            plt.savefig(path + '/' + Title + '.png') 
        else: 
            f = input("Please provide a file name: ")
            plt.savefig(path + '/' + f + '.png') 
           



def performance_plot(env, Q, ntimes=10):
     env.seed(9)
     returns = 0
     gamma = env.gamma
     n_return=[]
     for n in range(0, ntimes):
       returns=0
       discount = 1
       state = env.reset()
       t=1
       while  t<50:
         (state, reward, info) = env.step(env.feas_actions[state,np.argmax(Q[state,:])])
         returns += discount*reward
         discount = discount*gamma
         t +=1
       n_return.append(returns)
     returns = np.mean(n_return)
     return returns

def plot_performance_curves(at,num_steps,env, *args): 
    n = len(args)
    ls = [[] for i in range(n)]
    for i in range(n):
        for j in range(0, num_steps, at):
            #print('i,j=',i,j)
            k = int(j/at)
            ls[i].append(performance_plot(env,args[i][k]))
    return ls 



def relative_error_plot(at,num_steps, Qstar, *args):
   Vstar= np.max(Qstar,1)
   order=None
   Vstar_norm=np.linalg.norm(Vstar, ord=order)
   n = len(args)
   ls = [[] for i in range(n)]
   for i in range(n):
       for j in range(0, num_steps, at):
           k = int(j/at)
           ls[i].append( np.linalg.norm(np.max(args[i][k],1) - Vstar, ord=order)/Vstar_norm)
   return ls