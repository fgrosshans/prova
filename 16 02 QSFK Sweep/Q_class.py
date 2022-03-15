#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This class contains methods that directly model single queues. 
It can be seen as the hardware level of the network.
"""
import numpy as np
from KnockOffRand import KnockoffNpRandom

class Queue:

    def __init__(self,nd1,nd2,tran_prob):
        self.nodes = frozenset([nd1,nd2]) # This set will contain the extremes of the queue. A set was chosen because it is UNORDERED.
        self.type = "virtual" # Every queue is initialized as virtual.
        self.serv = "regular"
        self.Qdpairs = 0; # Queued pairs, initialized to zero.
        self.T_prob = tran_prob # Transmission probability
        self.demands = 0; # Requests, initialized to zero
        #self.rng = np.random.default_rng()
        self.rng = KnockoffNpRandom()
        
    def SetPhysical(self,arr_rate_s,t_step):
        self.type = "physical"
        alpha = arr_rate_s*t_step
        self.GenPParam = alpha # Parameter for the Poisson Distribution of photon arrivals

    def SetVirtual(self):
        self.type = "virtual"
        self.GenPParam = 0

    def SetService(self,Reqrate_s,tstep):
        self.serv = "service" # If the queue is service, it receives demands.
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

    def Loss(self,LossParam):
        rng=self.rng
        to_check = self.Qdpairs
        lost = sum(rng.random(size=to_check) <= (1-LossParam))
        self.Qdpairs -= lost
        return lost

    def Demand(self): 
        D = 0;
        if self.serv == "service":
            D = self.rng.poisson(self.PoissParam)
            self.demands += D
        return D
    
