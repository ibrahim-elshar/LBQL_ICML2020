# -*- coding: utf-8 -*-
'''
This file contains all stations information.
'''
import numpy as np

num_stations_def = 2
num_cars_def = 12
demand_par_a_def = np.array([9., 10.])
demand_par_b_def = np.array([1., 1.])
epsilons_support_def = np.array([3, 3])
distance_ij_def = np.array([1. , 1.])
pmin_def = np.ones(num_stations_def)
lost_sales_cost_def = np.array([2., 2.])
prob_eps_1_def=[1/(2*epsilons_support_def[0]+1)]*(2*epsilons_support_def[0]+1)
prob_eps_2_def=[1/(2*epsilons_support_def[1]+1)]*(2*epsilons_support_def[1]+1)
prob_eps_def = np.array(prob_eps_1_def)[:,None] * np.array(prob_eps_2_def)
prob_eps_def = prob_eps_def.ravel()
prob_eps_def = prob_eps_def.tolist()

class Stations():
    '''
    Create stations info, including number of cars, number of stations,
    and the price-dependent-demand model for each station.
    The demand models are assumed linear in price, of the form,
    D(p) = a - bp. Here a is a one dimensional array where the first element 
    corresponds to station, 2nd element to station 2,...etc. Same goes for b.
    Epsilons are the additive demand noise. The full demand model is 
    D_t(p_t) = a - b p_t + epsilon_t. 
    '''
    def __init__(self, num_stations = num_stations_def, 
                 num_cars = num_cars_def,
                 demand_par_a = demand_par_a_def,
                 demand_par_b = demand_par_b_def,
                 epsilons_support = epsilons_support_def,
                 distance_ij =  distance_ij_def,
                 pmin = pmin_def,
                 lost_sales_cost = lost_sales_cost_def,
                 prob_eps_1 = prob_eps_1_def,
                 prob_eps_2 = prob_eps_2_def,
                 prob_eps   = prob_eps_def
                 ):
        
        self.num_stations = num_stations
        self.num_cars = num_cars
        self.demand_par_a = demand_par_a
        self.demand_par_b = demand_par_b
        self.epsilons_support = epsilons_support
        self.distance_ij = distance_ij
        self.pmin = pmin
        self.pmax = (self.demand_par_a - self.epsilons_support )/ self.demand_par_b
        self.dmin = self.D(self.pmax)
        self.dmax = self.D(self.pmin)
        self.d_range = self.dmax - self.dmin + 1
        self.lost_sales_cost = lost_sales_cost
        
        self.prob_eps_1 = prob_eps_1
        self.prob_eps_2 = prob_eps_2
        self.prob_eps = prob_eps
    def D(self, p):
        '''
        Deterministic demand function: returns a demand vector (rounded to the 
        nearest integer) coresponding to the demand of each station for the 
        given vector price input
        '''
        d= np.rint(self.demand_par_a - self.demand_par_b*p).astype(int)
        return d 
 
    def P(self, d):
        '''
        Returns a price vector of each station for the given vector demand input
        '''
        p=(self.demand_par_a - d)/self.demand_par_b
        return p          
  