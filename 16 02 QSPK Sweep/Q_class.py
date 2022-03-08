#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:33:56 2021

@author: paolo

"""

from random import random
from numpy.random import poisson

class Queue:

    def __init__(self,nd1,nd2,tran_prob):
        self.nodes = frozenset([nd1,nd2]) # This set will contain the extremes of the queue. A set was chosen because it is UNORDERED.
        self.type = "virtual" # Every queue is initialized as virtual.
        self.serv = "regular"
        self.n_times_served = 0; # If this is a service queue, this counts the amount of times it was swapped to, regardless of having later lost or consumed the pair
        self.Qdpairs = 0; # Queued pairs, initialized to zero.
        self.T_prob = tran_prob # Transmission probability
        self.demands = 0; # Requests, initialized to zero.
        self.scheduledin = 0
        self.scheduledout = 0
        self.stale = True
        
        
    def SetPhysical(self,alpha):
        self.type = "physical"
        self.GenPParam = alpha # Parameter for the Poisson Distribution of photon arrivals
    
    def SetVirtual(self):
        self.type = "virtual"
        
    def SetService(self,Reqrate_s,tstep):
        self.serv = "service"
        Reqrate_steps = Reqrate_s*tstep # casting the rate per second to a rate per time step
        self.PoissParam = Reqrate_steps # Parameter for the Poisson Distribution
        return self
    
    def Generate(self):
        if (self.type == "physical"): # Only physical queues generate, but implementing this check here allows to call...
                                      # ... the Generate method for all queues indistinctly.
            to_generate = poisson(self.GenPParam)
            generated = 0;
            for i in range(to_generate):
                rd = random() 
                if rd <= self.T_prob:
                    self.Qdpairs += 1
                    generated += 1
            return generated
        else:
            return 0
                    
    def Loss(self,LossParam): 
        to_check = int(self.Qdpairs)
        lost = 0 
        for i in range(to_check):
            if random() <= (1-LossParam):
                self.Qdpairs -=1
                lost +=1
        return lost

    def Demand(self): # A next step would be to change the shape of demand distribution
        D = 0;
        if self.serv == "service":
            D = poisson(self.PoissParam)
            self.demands += D
        return D
    