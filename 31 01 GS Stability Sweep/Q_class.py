#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:33:56 2021

@author: paolo

This class models the Queue object. It takes as inputs the nodes across which it is plugged and the various technological parameters.
Methods:
    -SetPhysical/SetVirtual
    -Generate
    -SwapFrom
    -SwapTo
    -Loss
"""

import numpy as np
from KnockOffRand import KnockoffNpRandom


class Queue:

    def __init__(self,nd1,nd2,efficiency=1,tran_prob=1):
        self.nodes = frozenset([nd1,nd2]) # This set will contain the extremes of the queue. A set was chosen because it is UNORDERED.
        self.eta = efficiency # SaR efficiency of the quantum memories
        self.type = "virtual" # Every queue is initialized as virtual.
        self.serv = "regular"
        self.n_times_served = 0; # If this is a service queue, this counts the amount of times it was swapped to, regardless of having later lost or consumed the pair
        self.Qdpairs = 0; # Queued pairs, initialized to zero.
        self.T_prob = tran_prob # Transmission probability
        self.scheduledin = 0
        self.scheduledout = 0
        self.demands = 0
        self.Lossprob = 2*(1-efficiency) - (1-efficiency)**2 # Here, you excluded the case in which both memories failed. Why?
        #self.rng = np.random.default_rng(seed=4529)
        self.rng = KnockoffNpRandom()
        
        
    def SetPhysical(self,arr_rate_s,tstep):
        self.type = "physical"
        arr_rate_steps = arr_rate_s*tstep # casting the rate per second to a rate per time step
        self.GenPParam = arr_rate_steps # Parameter for the Poisson Distribution of photon arrivals
        
    def Loss(self,LossParam):
        rng=self.rng
        to_check = self.Qdpairs
        lost = sum(rng.random(size=to_check) <= (1-LossParam))
        self.Qdpairs -= lost
        return lost

    
    def SetVirtual(self):
        self.type = "virtual"
        
    def SetService(self,Reqrate_s,tstep):
        self.serv = "service"
        Reqrate_steps = Reqrate_s*tstep # casting the rate per second to a rate per time step
        self.PoissParam = Reqrate_steps # Parameter for the Poisson Distribution
        return self   
    
    def Generate(self):
        rng = self.rng
        if (self.type == "physical"): # Only physical queues generate, but implementing this check here allows to call...
                                      # ... the Generate method for all queues indistinctly.
            to_generate = self.rng.poisson(self.GenPParam)
            generated = sum(rng.random(size=to_generate) <= self.T_prob)
            self.Qdpairs += generated
            return generated
        else:
            return 0
 
    def ScheduleIN(self):
        self.scheduledin += self.ReadOut() # The pairs are actually scheduled in if the BSM at the middle node has had success, i.e. neither memory has failed

    def ScheduleOUT(self): # No need to call ReadOut here because the pair is consumed either way.
        self.scheduledout += 1
        self.Qdpairs -= 1
        
    def Update(self,BSM):
        self.Qdpairs += self.scheduledin 
        self.scheduledin = 0
        self.scheduledout = 0
            
    def Consume(self):
            requested = max(self.demands,0)
            to_consume = min(requested,self.Qdpairs)
            for i in range(to_consume):
                success = self.ReadOut() # Either zero or one
                self.Qdpairs -= 1;
                self.n_times_served += success;
                self.demands -= success
   
    def Demand(self): 
        D = 0;
        if self.serv == "service":
            D = self.rng.poisson(self.PoissParam)
            self.demands += D
        return D
    
    def ReadOut(self): # Memories are always read by pairs in this implementation. This function checks whether one of the two memories have failed.
        rng = self.rng    
        rd = rng.random()
        if rd < self.Lossprob:
            return 0
        else:
            return 1