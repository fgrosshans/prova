#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:29:30 2022

@author: paolo
"""
import numpy as np
import itertools as it


class KnockoffNpRandom:
    
    def __init__(self,seed=None):
        self.rng = np.random.default_rng(seed)
        UniformVector = self.rng.random(size=10000)
        self.UniformPool = it.cycle(UniformVector)
        self.AvailablePoisson = dict()
    
    def initPoisson(self,lam):
        poiss = self.rng.poisson(lam,size=10000)
        poiss = it.cycle(poiss)
        self.AvailablePoisson[lam] = poiss
    
    def random(self):
        return next(self.UniformPool)

    def poisson(self,lam):
        if lam not in self.AvailablePoisson.keys():
            self.initPoisson(lam)
        return next(self.AvailablePoisson[lam])
        
            