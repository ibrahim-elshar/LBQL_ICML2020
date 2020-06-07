# -*- coding: utf-8 -*-
import numpy as np
#from scipy.stats import geom, binom, poisson, gamma


class Stations(object):
    '''
    Create stations info, including number of cars, number of stations,
    and the price-dependent-demand model for each station.
    The demand models are assumed linear in price, of the form,
    D(p) = a - bp. Here a is a one dimensional array where the first element 
    corresponds to station, 2nd element to station 2,...etc. Same goes for b.
    Epsilons are the additive demand noise. The full demand model is 
    D_t(p_t) = a - b p_t + epsilon_t. 
    '''
    def __init__(self, 
                 num_stations =  4,
                 num_cars = 20,
                 demand_par_a = np.array([5., 5., 5., 5.,]),
                 demand_par_b = np.array([1., 1., 1., 1.,]),
                 epsilons_supp = np.array([3, 3, 3, 3,]),
                 
                 distance_ij =  np.array([[1, 1.8, 1.5, 1.4,],
                                          [1.8, 1, 1.6, 1.1,],
                                          [1.5, 1.6, 1, 1.2,],
                                          [1.4, 1.1, 1.2, 1,]], dtype=np.float32),
    
                 lost_sales_cost = np.array([1.7, 1.2, 1.5, 2.,]),
                 prob_ij = np.array([[1./4, 1./4, 1./4, 1./4],
                                      [1./4, 1./4, 1./4, 1./4],
                                      [1./4, 1./4, 1./4, 1./4],
                                      [1./4, 1./4, 1./4, 1./4]])
                 
                 ):    
        eps_size = epsilons_supp.size
        self.eps_rang = epsilons_supp*2 + 1
        self.eps = {}
        self.stat_prob_eps = {}
        for i in range(eps_size):
            self.eps[i] = np.arange(-epsilons_supp[i], epsilons_supp[i]+1)
            self.stat_prob_eps[i] = np.ones(self.eps_rang[i]) * 1./self.eps_rang[i]

            
        
        self.eps_rang = epsilons_supp*2 + 1
        self.eps_prob = np.ones(self.eps_rang) * np.prod(1/self.eps_rang)
        
        self.num_stations = num_stations
        
        self.num_cars = num_cars
        
        self.demand_par_a = demand_par_a
        self.demand_par_b = demand_par_b
        
        self.eps_supp = epsilons_supp
        
        self.distance_ij = distance_ij
        
        self.pmin = np.ones(self.num_stations)
        self.pmax = (self.demand_par_a - self.eps_supp)/ self.demand_par_b
        
        self.dmin = self.D(self.pmax)
        self.dmax = self.D(self.pmin)
        self.d_rng = self.dmax - self.dmin + 1
        
        self.lost_sales_cost = lost_sales_cost
        
        self.prob_ij = prob_ij
        
    def D(self, p):
        '''
        Deterministic demand function: returns a demand vector (rounded to the 
        nearest integer) coresponding to the demand of each station for the 
        given vector price input
        '''
        d= np.rint(self.demand_par_a - self.demand_par_b*p).astype(int)
#        d= self.demand_par_a - self.demand_par_b*p
        return d 
 
    def P(self, d):
        '''
        Returns a price vector of each station for the given vector demand input
        '''
        p=(self.demand_par_a - d)/self.demand_par_b
        return p          
  
            
    
def get_tuples(length, total):
    if length == 1:
        yield (total,)
        return

    for i in range(total + 1):
        for t in get_tuples(length - 1, total - i):
            yield (i,) + t    