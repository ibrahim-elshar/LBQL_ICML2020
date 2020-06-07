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
      episodes_return=[]
      for n in range(0, ntimes):
        returns=0
        discount = 1
        done=False
        state = env.reset()
        t=1
        while not done and t<=100:
          (state, reward, done, info) = env.step(np.argmax(Q[state,:]))
          returns += discount*reward
          discount = discount*gamma
          t +=1
        episodes_return.append(returns)
      returns = np.mean(episodes_return)
      return returns

def plot_performance_curves(num_steps,env, *args): 
    n = len(args)
    ls = [[] for i in range(n)]
    for i in range(n):
        for j in range(0, num_steps):
            #print('i,j=',i,j)
            ls[i].append(performance_plot(env,args[i][j]))
    return ls 


def relative_error_plot(num_steps, Qstar, *args):
   Vstar= np.max(Qstar,1)
   order=None
   Vstar_norm=np.linalg.norm(Vstar, ord=order)
   n = len(args)
   ls = [[] for i in range(n)]
   for i in range(n):
       for j in range(0, num_steps):
           ls[i].append( np.linalg.norm(np.max(args[i][j],1) - Vstar, ord=order)/Vstar_norm)
   return ls